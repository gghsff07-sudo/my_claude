"""The 30B construction attempt, for the record.

"Train it to 30B" starts with "hold a 30B model in memory", so this module
attempts exactly that, using the real model classes (Block, the real
initializers — every page touched), and records where the machine stops it.
The Linux OOM killer sends SIGKILL, which no Python handler can catch, so the
attempt runs as a child process and the parent records the death: exit code,
last completed layer, RSS at each layer.

Usage:
    python -m fable_lm.attempt            # parent: run + record + summarize
    python -m fable_lm.attempt --child    # child: the attempt itself
    fable-attempt                         # installed entry point

Results: results/attempt_30b.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import torch.nn as nn

from fable_lm.model import Block, ModelConfig
from fable_lm.scale import LADDER, available_bytes, machine_total_bytes

REPO = Path(__file__).resolve().parents[1]
CHILD_TIMEOUT_S = 300


def rss_mb() -> tuple[float, float]:
    """(current RSS, high-water RSS) in MB, from /proc."""
    cur = hw = 0.0
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                cur = int(line.split()[1]) / 1024
            elif line.startswith("VmHWM:"):
                hw = int(line.split()[1]) / 1024
    except OSError:
        pass
    return cur, hw


def attempt_child() -> int:
    """Build the 30B parameter volume layer by layer with the real classes.

    Constructs exactly what FableLM(cfg) would construct — token embedding,
    learned positions, n_layers Blocks (real Block: LN, causal attention,
    GELU feed-forward, real initializers touching every page), final LN —
    and prints progress after each layer so the parent can see how far it
    got even if the OOM killer SIGKILLs mid-layer.
    """
    cfg: ModelConfig = LADDER["30b"]
    total = cfg.param_count()
    print(f"constructing LADDER['30b']: {cfg.n_layers} layers x d{cfg.d_model}, "
          f"target {total:,} params ({total / 1e9:.4f} B)", flush=True)
    print(f"machine: {machine_total_bytes() / 1e9:.2f} GB total, "
          f"{available_bytes() / 1e9:.2f} GB available at start", flush=True)

    built: list[nn.Module] = []
    built_params = 0

    def report(tag: str) -> None:
        cur, hw = rss_mb()
        print(f"{tag} | rss={cur:.0f} MB | peak={hw:.0f} MB | "
              f"params_built={built_params:,} ({built_params / total:.2%})", flush=True)

    tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
    tok.weight.data.normal_(mean=0.0, std=0.02)
    pos = nn.Embedding(cfg.context_len, cfg.d_model)
    pos.weight.data.normal_(mean=0.0, std=0.02)
    built += [tok, pos]
    built_params += tok.weight.numel() + pos.weight.numel()
    report("embeddings allocated")

    for k in range(cfg.n_layers):
        blk = Block(cfg)  # real module, real init — touches every parameter page
        built.append(blk)
        built_params += sum(p.numel() for p in blk.parameters())
        report(f"layer {k + 1}/{cfg.n_layers} allocated")

    ln_f = nn.LayerNorm(cfg.d_model)
    built.append(ln_f)
    built_params += sum(p.numel() for p in ln_f.parameters())
    report("final layernorm allocated")

    assert built_params == total, f"parameter parity failed: {built_params:,} != {total:,}"
    print("CONSTRUCTED: full 30B parameter volume resident — "
          "training would still need 16 bytes/param of AdamW state "
          "(~484 GB), so it dies here too.", flush=True)
    return 0


LINE_RE = re.compile(
    r"^(?:embeddings allocated|final layernorm allocated|"
    r"layer (?P<layer>\d+)/(?P<n>\d+) allocated|CONSTRUCTED)"
    r".*?params_built=(?P<params>[\d,]+).*?$")


def parse_child_progress(output: str) -> list[dict]:
    """Parse the child's progress lines into structured records (pure)."""
    records = []
    for line in output.splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        rss_m = re.search(r"rss=(\d+) MB", line)
        peak_m = re.search(r"peak=(\d+) MB", line)
        records.append({
            "event": line.split(" | ")[0].strip(),
            "layer": int(m.group("layer")) if m.group("layer") else None,
            "params_built": int(m.group("params").replace(",", "")),
            "rss_mb": int(rss_m.group(1)) if rss_m else None,
            "peak_mb": int(peak_m.group(1)) if peak_m else None,
        })
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="Attempt to construct the 30B model, for the record")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        try:
            return attempt_child()
        except (MemoryError, RuntimeError) as e:
            # malloc itself refused (possible without overcommit)
            print(f"ALLOC-FAILED: {e}", flush=True)
            return 3

    # ---------------- parent ----------------
    print("attempt: constructing the full 30B parameter volume in a child process")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-u", str(Path(__file__).resolve()), "--child"],
        capture_output=True, text=True, timeout=CHILD_TIMEOUT_S)
    elapsed = time.perf_counter() - t0
    records = parse_child_progress(proc.stdout)

    layers_done = sum(1 for r in records if r["layer"] is not None)
    last = records[-1] if records else None
    params_built = last["params_built"] if last else 0
    total = LADDER["30b"].param_count()

    if proc.returncode < 0:  # killed by signal
        outcome = f"SIGKILLed by the OOM killer (signal {-proc.returncode})"
    elif proc.returncode == 0:
        outcome = "constructed fully (unexpected on this machine)"
    else:
        outcome = f"allocator refused (exit {proc.returncode})"

    result = {
        "target_params": total,
        "config": {k: getattr(LADDER["30b"], k) for k in
                   ("n_layers", "d_model", "n_heads", "d_ff", "context_len", "vocab_size")},
        "machine_ram_total_bytes": machine_total_bytes(),
        "ram_available_at_start_bytes": available_bytes(),
        "outcome": outcome,
        "exit_code": proc.returncode,
        "elapsed_s": round(elapsed, 1),
        "layers_completed": layers_done,
        "params_built": params_built,
        "fraction_of_target": round(params_built / total, 6),
        "last_event": last["event"] if last else None,
        "progress": records,
        "child_stderr_tail": proc.stderr.strip()[-400:],
    }
    out = REPO / "results" / "attempt_30b.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"\nchild {outcome} after {elapsed:.1f}s")
    if last:
        print(f"last completed step: {last['event']} "
              f"(rss {last['rss_mb']} MB, {params_built:,} params = "
              f"{params_built / total:.2%} of the 30B target)")
    per_layer = (total - 2048 * LADDER['30b'].d_model
                 - LADDER['30b'].context_len * LADDER['30b'].d_model) // LADDER['30b'].n_layers
    print(f"each of the {LADDER['30b'].n_layers} layers alone is ~{per_layer / 1e6:.0f}M params "
          f"(~{per_layer * 16 / 1e9:.2f} GB of AdamW state); this machine has "
          f"{machine_total_bytes() / 1e9:.1f} GB total.")
    print(f"written: {out}")
    print("recorded honestly: the 30B model cannot exist on this machine, let alone train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
