"""Scale-up automation: train progressively larger models toward a parameter
target — the 30B preset included — until the ladder reaches the target or the
hardware ceiling, whichever comes first.

Usage:
    python -m fable_lm.scale --target 30b             # plan + train all feasible rungs
    python -m fable_lm.scale --target 30b --dry-run   # plan only (never instantiates)
    python -m fable_lm.scale --target 30b --safety 0.85   # squeeze: biggest trainable rungs
    python -m fable_lm.scale --target 30b --only 99m --steps 120 --save-ckpt
    fable-scale --target 30b                           # installed entry point

Each rung is trained for a smoke budget of steps (the per-rung loss curve
proves the pipeline scales); rungs whose estimated peak training memory does
not fit are marked blocked with the exact requirement, and the run stops
there. The 30B rung cannot be trained on this class of machine: 30.2B
parameters need ~483 GB of RAM for AdamW fp32 training and cluster-scale
compute — the config itself is arithmetic-verified (param_count()) and
cross-checked against real instantiation on the small rungs.

Results: results/scale_ladder.csv + results/scale_ladder.json (a single-rung
--only run writes results/scale_<name>.* instead, so it never clobbers the
full-ladder record).
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fable_lm.data import build_token_stream, load_documents, split_docs
from fable_lm.model import FableLM, ModelConfig
from fable_lm.tokenizer import ByteBPETokenizer
from fable_lm.train import TokenWindowDataset

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# The ladder: curated rungs from the shipped 3.3M miniature to the 30B target.
# Params are verified by ModelConfig.param_count() — names are rounded labels.
# ---------------------------------------------------------------------------
LADDER: dict[str, ModelConfig] = {
    "3.3m":  ModelConfig(n_layers=4,  d_model=256,  n_heads=4,   d_ff=768),    # shipped miniature
    "15m":   ModelConfig(n_layers=8,  d_model=384,  n_heads=6,   d_ff=1536),
    "59m":   ModelConfig(n_layers=8,  d_model=768,  n_heads=12,  d_ff=3072),
    "99m":   ModelConfig(n_layers=10, d_model=896,  n_heads=14,  d_ff=3584),   # squeeze rung
    "154m":  ModelConfig(n_layers=12, d_model=1024, n_heads=16,  d_ff=4096),   # squeeze rung
    "344m":  ModelConfig(n_layers=12, d_model=1536, n_heads=24,  d_ff=6144),
    "1.2b":  ModelConfig(n_layers=24, d_model=2048, n_heads=32,  d_ff=8192),
    "4.1b":  ModelConfig(n_layers=36, d_model=3072, n_heads=48,  d_ff=12288),
    "30b":   ModelConfig(n_layers=49, d_model=7168, n_heads=56,  d_ff=28672),  # the target
}

TARGET_30B = 30_000_000_000


@dataclass
class Rung:
    name: str
    cfg: ModelConfig
    params: int

    @property
    def params_hr(self) -> str:
        n = self.params
        return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


def build_ladder(target_params: int) -> list[Rung]:
    """Rungs from the smallest preset up to the first one reaching the target."""
    rungs = [Rung(name, cfg, cfg.param_count()) for name, cfg in LADDER.items()]
    rungs = sorted(rungs, key=lambda r: r.params)
    reachable = [r for r in rungs if r.params < target_params]
    ending = [r for r in rungs if r.params >= target_params]
    if not ending:
        raise SystemExit(f"no preset reaches {target_params:,} params — extend LADDER")
    return reachable + [min(ending, key=lambda r: r.params)]


# ---------------------------------------------------------------------------
# Memory / compute estimates (heuristics, labeled as such in every output)
# ---------------------------------------------------------------------------

def training_peak_bytes(cfg: ModelConfig, batch: int, ctx: int = 512) -> int:
    """Heuristic peak RAM for AdamW fp32 training of one rung.

    weights + grads + 2 Adam moments = 16 bytes/param; activations roughly
    30 * B * T * d * L * 4 bytes. The 30 coefficient is CALIBRATED, not
    textbook: it back-predicts all four observations of the v0.4.0 runs —
    15m@b8 measured ~2.2 GB RSS (est 1.75), 59m@b4 measured 2.97 GB and
    survived (est 2.45), 59m@b8 was SIGKILLed by the OOM killer (est 3.94).
    The old 12 coefficient passed 59m@b8 through a 75% gate and got the
    process killed; this one errs on the side of the OOM killer.
    """
    p = cfg.param_count()
    d, L = cfg.d_model, cfg.n_layers
    return 16 * p + 30 * batch * ctx * d * L * 4


def _meminfo(key: str) -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(f"{key}:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def available_bytes() -> int:
    return _meminfo("MemAvailable") or 4 * 1024**3


def machine_total_bytes() -> int:
    return _meminfo("MemTotal") or 4 * 1024**3


def estimate_steps_per_second(cfg: ModelConfig, base_tps: float = 4200.0) -> float:
    """Rough tokens/s scaling: throughput ~ 1/params for CPU matmul-bound work,
    anchored to the measured 3.3M rung (~4.2k tok/s on 2 threads)."""
    return base_tps * (LADDER["3.3m"].param_count() / cfg.param_count())


def fit_batch(cfg: ModelConfig, mem_limit: int, ctx: int = 512,
              max_batch: int = 8) -> int | None:
    """Largest power-of-two batch (max_batch down to 1) whose estimated peak
    fits mem_limit. --batch sets max_batch, so `--batch 1` pins the batch."""
    batch = 1
    while batch * 2 <= max_batch:
        batch *= 2
    while batch >= 1:
        if training_peak_bytes(cfg, batch, ctx) <= mem_limit:
            return batch
        batch //= 2
    return None


# ---------------------------------------------------------------------------
# Training one rung (reuses the run3 pipeline: same data, tokenizer, recipe)
# ---------------------------------------------------------------------------

def train_rung(rung: Rung, steps: int, batch: int, tok, save_ckpt: bool = False) -> dict:
    """Train one rung for a smoke budget. Returns metrics; never pretends a
    scale smoke is a converged model. With save_ckpt=True the final weights
    land in checkpoints/scale/<name>-final.pt (fp32 state_dict, no optimizer
    — the rung artifacts are measurements, not shipped models)."""
    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = torch.device("cpu")

    docs = load_documents()
    train_docs, val_docs = split_docs(docs)
    train_windows = build_token_stream([t for _, t in train_docs], tok, 512)
    val_windows = build_token_stream([t for _, t in val_docs], tok, 512)
    # TokenWindowDataset provides (x, y) with the shift and -100 masking done
    train_loader = DataLoader(TokenWindowDataset(train_windows),
                              batch_size=batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(TokenWindowDataset(val_windows), batch_size=batch)

    cfg = ModelConfig(vocab_size=tok.vocab_size, context_len=rung.cfg.context_len,
                      n_layers=rung.cfg.n_layers, d_model=rung.cfg.d_model,
                      n_heads=rung.cfg.n_heads, d_ff=rung.cfg.d_ff)
    model = FableLM(cfg).to(device)
    assert model.num_params() == rung.params, "param_count() disagrees with instantiation"

    loader = train_loader
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    model.train()
    losses, t0, tokens = [], time.perf_counter(), 0
    it = iter(loader)
    for step in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        tokens += x.numel()

    # quick held-out pass (loss only — this is a scale smoke, not an eval)
    model.eval()
    with torch.no_grad():
        val_losses = []
        for x, y in val_loader:
            _, vl = model(x, y)
            val_losses.append(vl.item())

    elapsed = time.perf_counter() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KiB
    if save_ckpt:
        ckpt_dir = REPO / "checkpoints" / "scale"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{rung.name}-final.pt"
        torch.save({"model": model.state_dict(), "cfg": model.config_dict(),
                    "steps": steps, "batch": batch,
                    "note": "scale-ladder rung checkpoint; measurement, not a shipped model"},
                   ckpt_path)
    return {
        "name": rung.name, "params": rung.params,
        "config": {"n_layers": cfg.n_layers, "d_model": cfg.d_model,
                   "n_heads": cfg.n_heads, "d_ff": cfg.d_ff, "context_len": cfg.context_len},
        "steps": steps, "batch": batch,
        "train_loss_first": round(losses[0], 4) if losses else None,
        "train_loss_last": round(losses[-1], 4) if losses else None,
        "val_loss": round(sum(val_losses) / max(len(val_losses), 1), 4),
        "tokens_per_s": round(tokens / elapsed, 1),
        "elapsed_s": round(elapsed, 1),
        "measured_peak_rss_mb": round(peak_mb, 1),
    }


# ---------------------------------------------------------------------------
# The automation
# ---------------------------------------------------------------------------

def parse_target(s: str) -> int:
    """'30b'/'15m' (case-insensitive), scientific ('3e9'), or a plain integer."""
    t = s.strip().lower()
    if t.endswith("b"):
        return int(float(t[:-1]) * 1e9)
    if t.endswith("m"):
        return int(float(t[:-1]) * 1e6)
    return int(float(t))


def plan_rungs(target: int, steps: int, mem_limit: int, max_batch: int = 8) -> list[dict]:
    """Plan every rung to the target WITHOUT instantiating any model."""
    plan = []
    for rung in build_ladder(target):
        batch = fit_batch(rung.cfg, mem_limit, max_batch=max_batch)
        est = training_peak_bytes(rung.cfg, batch or 1)
        plan.append({
            "name": rung.name, "params": rung.params, "params_hr": rung.params_hr,
            "estimated_peak_gb": round(est / 1e9, 2),
            "batch": batch,
            "feasible": batch is not None,
            "est_tokens_per_s": round(estimate_steps_per_second(rung.cfg), 1),
            "steps": steps,
        })
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Automated scale-up: train rungs toward a parameter target "
                    "until the target or the hardware ceiling is reached")
    ap.add_argument("--target", type=str, default="30b",
                    help="parameter target: '30b', or a raw count like 3000000000")
    ap.add_argument("--steps", type=int, default=40,
                    help="optimizer steps per rung (the scale smoke budget)")
    ap.add_argument("--batch", type=int, default=8, help="preferred batch (auto-halved to fit)")
    ap.add_argument("--safety", type=float, default=0.5,
                    help="fraction of available RAM the estimate may use "
                         "(0.5 default: the estimate is a heuristic and the Linux "
                         "OOM killer SIGKILLs — uncatchable — when it is wrong)")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan only; never instantiate a model (safe for 30B)")
    ap.add_argument("--only", type=str, default=None, metavar="RUNG",
                    help="train only this rung (a LADDER name, e.g. 99m); results go to "
                         "results/scale_<name>.* so the full-ladder record is kept")
    ap.add_argument("--save-ckpt", action="store_true",
                    help="save the trained rung's weights to checkpoints/scale/<name>-final.pt")
    args = ap.parse_args()

    target = parse_target(args.target)

    ram_start, ram_total = available_bytes(), machine_total_bytes()
    mem_limit = int(ram_start * args.safety)
    print(f"target: {target:,} params | machine RAM: {ram_total / 1e9:.2f} GB total, "
          f"{ram_start / 1e9:.2f} GB available | usable budget: {mem_limit / 1e9:.2f} GB "
          f"(x {args.safety:.0%})")

    plan = plan_rungs(target, args.steps, mem_limit, max_batch=args.batch)
    if args.only:
        if args.only not in LADDER:
            raise SystemExit(f"--only {args.only}: not a LADDER rung "
                             f"({', '.join(LADDER)})")
        plan = [r for r in plan if r["name"] == args.only]
        if not plan:
            raise SystemExit(f"--only {args.only}: not below the target {target:,}")
    print("\nladder plan (param_count() arithmetic — no model is built for this):")
    for r in plan:
        status = f"trainable @ batch {r['batch']}" if r["feasible"] else \
            f"BLOCKED: needs ~{r['estimated_peak_gb']:.1f} GB peak"
        print(f"  {r['name']:>5s}  {r['params_hr']:>7s}  ~{r['estimated_peak_gb']:6.2f} GB peak  "
              f"~{r['est_tokens_per_s']:7.0f} tok/s est.  {status}")

    if args.dry_run:
        print("\ndry run: no training. The plan above is the full automation path "
              "to the target; rungs marked BLOCKED exceed this machine's RAM.")
        return

    # -------- train every feasible rung until blocked or target --------
    tok_path = REPO / "data" / "tokenizer.json"
    if not tok_path.is_file():
        raise SystemExit("data/tokenizer.json missing — train the miniature first")
    tok = ByteBPETokenizer.load(tok_path)

    # -------- persist (incrementally: a hard OOM kill mid-rung must not lose
    # the rungs that already trained) --------
    stem = f"scale_{args.only}" if args.only else "scale_ladder"

    def persist() -> None:
        out_dir = REPO / "results"
        out_dir.mkdir(exist_ok=True)
        csv_path = out_dir / f"{stem}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "params", "status", "steps", "batch",
                                              "train_loss_first", "train_loss_last", "val_loss",
                                              "tokens_per_s", "elapsed_s", "measured_peak_rss_mb",
                                              "estimated_peak_gb", "reason"])
            w.writeheader()
            for row in results:
                w.writerow({k: row.get(k, "") for k in w.fieldnames})
        (out_dir / f"{stem}.json").write_text(json.dumps(
            {"target": target, "ram_total_bytes": ram_total,
             "ram_available_at_start_bytes": ram_start,
             "mem_limit_bytes": mem_limit, "rungs": results},
            indent=2))

    results = []
    for r in plan:
        if not r["feasible"]:
            results.append({**r, "status": "blocked_ram",
                            "reason": f"estimated peak {r['estimated_peak_gb']:.2f} GB > "
                                      f"{mem_limit / 1e9:.2f} GB usable"})
            print(f"\n[{r['name']}] BLOCKED — {results[-1]['reason']}")
            break
        rung = Rung(r["name"], LADDER[r["name"]], r["params"])
        print(f"\n[{r['name']}] training {rung.params_hr} params "
              f"({r['steps']} steps, batch {r['batch']}) ...")
        try:
            metrics = train_rung(rung, r["steps"], r["batch"], tok, save_ckpt=args.save_ckpt)
            metrics["status"] = "trained"
            results.append(metrics)
            persist()
            print(f"  train loss {metrics['train_loss_first']} -> {metrics['train_loss_last']} | "
                  f"val {metrics['val_loss']} | {metrics['tokens_per_s']} tok/s | "
                  f"peak RSS {metrics['measured_peak_rss_mb']} MB")
        except MemoryError:
            results.append({**r, "status": "blocked_oom", "reason": "MemoryError"})
            persist()
            print("  OOM: MemoryError")
            break
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise  # a real bug must not hide behind the OOM label
            results.append({**r, "status": "blocked_oom", "reason": str(e)[:200]})
            persist()
            print(f"  OOM: {str(e)[:200]}")
            break
        if r["params"] >= target:
            print("target reached")
            break

    persist()
    print(f"\nwritten: {stem}.csv, {stem}.json")

    trained = [r for r in results if r["status"] == "trained"]
    blocked = [r for r in results if r["status"] != "trained"]
    if blocked:
        b = blocked[0]
        print(f"\nSTOPPED at the hardware ceiling: rung {b['name']} ({b['params']:,} params) "
              f"needs ~{b.get('estimated_peak_gb', 0):.1f} GB of RAM for fp32 AdamW training; "
              f"this machine has {ram_total / 1e9:.1f} GB total.")
    biggest = max((r["params"] for r in trained), default=0)
    print(f"trained rungs: {len(trained)} (largest {biggest / 1e6:.1f}M params) | "
          f"target {target / 1e9:.0f}B — the 30B rung needs a multi-GPU cluster "
          f"(~483 GB of fp32 AdamW state alone). The 30B CONFIG is in LADDER and "
          f"arithmetic-verified; it cannot be trained on this hardware, by design of physics.")


if __name__ == "__main__":
    main()
