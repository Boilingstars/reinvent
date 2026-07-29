"""
Run all epoch-evaluation scripts in order, then build the summary table.

Usage (from repo root):
  python epoch_eval/run_all.py
  python epoch_eval/run_all.py --max-ref 1500
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


SCRIPTS = [
    "eval_tanimoto.py",
    "eval_fcd.py",
    "eval_mahalanobis.py",
    "eval_hitrate.py",
    "eval_novelty_diversity.py",
    "eval_pca.py",
    "eval_summary_table.py",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-ref", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None, help="chromophore-like hit threshold")
    args, extra = p.parse_known_args()

    for name in SCRIPTS:
        cmd = [sys.executable, str(ROOT / name), *extra]
        if args.max_ref is not None and name != "eval_summary_table.py":
            cmd.extend(["--max-ref", str(args.max_ref)])
        if args.threshold is not None and name == "eval_hitrate.py":
            cmd.extend(["--threshold", str(args.threshold)])
        print("\n" + "=" * 60)
        print("Running:", " ".join(cmd))
        print("=" * 60)
        subprocess.run(cmd, check=True, cwd=str(ROOT.parent))


if __name__ == "__main__":
    main()
