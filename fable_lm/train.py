"""Training loop for the miniature (DESIGN.md §4).

Usage:
    python -m fable_lm.train --steps 600 --run-name run1

Includes: AdamW with weight decay on matrix params, linear warmup + cosine
decay to 10% of peak, global-norm gradient clipping, periodic and best
checkpoints (model + optimizer state, so --resume continues training rather
than restarting the optimizer), CSV + stdout logging, and early stopping on
val loss.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from fable_lm.data import build_token_stream, is_dialogue, load_documents, split_docs
from fable_lm.model import FableLM, ModelConfig
from fable_lm.tokenizer import SPECIAL_IDS, ByteBPETokenizer

REPO = Path(__file__).resolve().parents[1]


class TokenWindowDataset(Dataset):
    """Fixed-length windows; targets are inputs shifted left, pad masked."""

    def __init__(self, windows: list[list[int]]):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int):
        w = self.windows[i]
        x = torch.tensor(w, dtype=torch.long)
        y = torch.tensor(w, dtype=torch.long)
        y = torch.roll(y, shifts=-1, dims=0)
        # the shifted-in last position and any pad target are not supervised
        y[y == SPECIAL_IDS["<|pad|>"]] = -100
        y[-1] = -100
        return x, y


def get_lr(step: int, warmup: int, total_steps: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    frac = (step - warmup) / max(1, total_steps - warmup)
    frac = min(1.0, frac)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))


@torch.no_grad()
def evaluate(model: FableLM, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses += loss.item()
        n += 1
    model.train()
    return losses / max(1, n)


def save_ckpt(
    path: Path,
    model: FableLM,
    opt: torch.optim.Optimizer,
    cfg: ModelConfig,
    step: int,
    best_val: float,
    tok_path: Path,
) -> None:
    """One checkpoint format for best/step/final: model, optimizer state, config."""
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "cfg": asdict(cfg),
            "step": step,
            "best_val": best_val,
            "tokenizer": str(tok_path),
        },
        path,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Fable miniature")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--patience", type=int, default=5,
                    help="early stop after N vals w/o improvement")
    ap.add_argument("--run-name", type=str, default="run1")
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--fresh-tokenizer", action="store_true",
                    help="retrain the BPE tokenizer even if data/tokenizer.json exists "
                         "(use after adding corpus documents)")
    ap.add_argument("--data-mult", type=int, default=1, help="repeat corpus N times (more windows)")
    ap.add_argument("--dialogue-mult", type=int, default=1,
                    help="extra repetition for dialogue docs (upweights chat style)")
    ap.add_argument("--resume", type=str, default="", help="checkpoint path to resume from")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    torch.set_num_threads(2)

    # ---------- data ----------
    docs = load_documents()
    if not docs:
        raise SystemExit("No corpus documents found — run from the repo root.")
    train_docs, val_docs = split_docs(docs)
    train_texts = [t for _, t in train_docs]
    val_texts = [t for _, t in val_docs]
    if args.dialogue_mult > 1:
        # upweight chat-format data: repeat dialogue docs from the TRAIN split
        # only — val must stay fully held-out (no copies of val docs in train)
        extra = [t for n, t in train_docs if is_dialogue(n)]
        train_texts = train_texts + extra * (args.dialogue_mult - 1)
    print(f"corpus: {len(docs)} documents ({len(train_texts)} train / {len(val_texts)} val texts)")

    tok_path = REPO / "data" / "tokenizer.json"
    if tok_path.is_file() and not args.fresh_tokenizer:
        tokenizer = ByteBPETokenizer.load(tok_path)
        print(f"tokenizer: loaded {tok_path.name} (vocab {tokenizer.vocab_size})")
    else:
        if tok_path.is_file():
            print("tokenizer: retraining (the old one is in git history)")
        print("training tokenizer on train split ...")
        tokenizer = ByteBPETokenizer()
        tokenizer.train(train_texts, vocab_size=args.vocab_size, verbose=True)
        tok_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(tok_path)
        print(f"tokenizer: saved {tok_path} (vocab {tokenizer.vocab_size})")

    ctx = 512
    train_windows = build_token_stream(train_texts, tokenizer, ctx) * args.data_mult
    val_windows = build_token_stream(val_texts, tokenizer, ctx)
    train_ds = TokenWindowDataset(train_windows)
    val_ds = TokenWindowDataset(val_windows)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    print(f"windows: {len(train_ds)} train x{args.data_mult}, {len(val_ds)} val (ctx {ctx})")

    # ---------- model ----------
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, context_len=ctx)
    model = FableLM(cfg).to(device)
    print(f"model: {model.num_params():,} params ({cfg.n_layers}L x d{cfg.d_model} x ff{cfg.d_ff})")

    # ---------- optimizer (before resume: the moments are restored into it) ----------
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    start_step, best_val = 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        start_step, best_val = ck["step"] + 1, ck.get("best_val", float("inf"))
        if ck.get("opt"):
            opt.load_state_dict(ck["opt"])  # AdamW moments survive a resume
        print(f"resumed from {args.resume} at step {start_step} "
              f"(optimizer state {'restored' if ck.get('opt') else 'reset'})")

    run_dir = REPO / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = REPO / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    log_path = run_dir / "log.csv"
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="")
    logger = csv.writer(log_f)
    if new_log:
        logger.writerow(["step", "lr", "train_loss", "val_loss", "val_ppl",
                         "tokens_per_s", "elapsed_s"])

    # ---------- loop ----------
    print(f"training {args.steps} steps: lr {args.lr}, warmup {args.warmup}, clip {args.clip}")
    model.train()
    t0 = time.time()
    step = start_step
    it = iter(train_loader)
    bad_vals = 0
    while step < args.steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)

        lr = get_lr(step, args.warmup, args.steps, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr

        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        tokens_per_s = (step - start_step + 1) * args.batch_size * ctx / max(1e-9, time.time() - t0)

        if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
            val_loss = evaluate(model, val_loader, device)
            val_ppl = math.exp(min(20.0, val_loss))
            logger.writerow([step + 1, f"{lr:.2e}", f"{loss.item():.4f}",
                             f"{val_loss:.4f}", f"{val_ppl:.2f}", f"{tokens_per_s:.0f}",
                             f"{time.time()-t0:.1f}"])
            log_f.flush()
            star = ""
            if val_loss < best_val:
                best_val, bad_vals, star = val_loss, 0, " *best*"
                save_ckpt(ckpt_dir / "best.pt", model, opt, cfg, step, best_val, tok_path)
            else:
                bad_vals += 1
            print(f"step {step+1:5d} | lr {lr:.2e} | train {loss.item():.4f} | "
                  f"val {val_loss:.4f} (ppl {val_ppl:.2f}) | gnorm {gnorm:.2f} "
                  f"| {tokens_per_s:.0f} tok/s{star}")
            if bad_vals >= args.patience:
                print(f"early stop: {args.patience} vals without improvement")
                break
        step += 1

        if (step + 1) % args.ckpt_every == 0:
            save_ckpt(ckpt_dir / f"step{step+1:06d}.pt", model, opt, cfg,
                      step, best_val, tok_path)

    save_ckpt(ckpt_dir / "final.pt", model, opt, cfg, step, best_val, tok_path)
    log_f.close()
    print(f"done: best val loss {best_val:.4f} (ppl {math.exp(min(20, best_val)):.2f}); "
          f"checkpoints in {ckpt_dir}; log {log_path}")


if __name__ == "__main__":
    main()
