"""Make pytest work from any cwd, installed or not: put the repo on sys.path."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
