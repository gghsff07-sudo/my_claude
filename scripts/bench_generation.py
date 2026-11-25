"""Reproducible generation-speed benchmark (thin wrapper).

Usage:
    python scripts/bench_generation.py [checkpoint]

The implementation lives in fable_lm.bench so the installed package can expose
the same benchmark as the `fable-bench` console entry point.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fable_lm.bench import main  # noqa: E402

if __name__ == "__main__":
    main()
