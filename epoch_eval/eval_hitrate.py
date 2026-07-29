"""
eval_hitrate.py — hit-rate of generated molecules across TL epochs.

Hit definitions (oracle proxies for chromophore TL without a trained λ_max model):
  1. rdkit_valid          — RDKit can parse & sanitize the SMILES
  2. chromophore_like     — max Tanimoto to chromophore train ≥ --threshold
  3. reinvent_valid       — SMILES_state == 1 in REINVENT CSV (if present)

hit_rate = n_hits / n_generated  (over the raw sample set of the epoch)

Usage:
  python epoch_eval/eval_hitrate.py
  python epoch_eval/eval_hitrate.py --threshold 0.4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    DEFAULT_OUT,
    DEFAULT_SAMPLES,
    DEFAULT_TRAIN,
    canonicalize,
    discover_epoch_files,
    ensure_dir,
    max_sims_to_ref,
    read_smiles,
    smiles_to_fps,
)


def load_epoch_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    smi_col = next((c for c in ("SMILES", "smiles") if c in df.columns), df.columns[0])
    df = df.rename(columns={smi_col: "SMILES"})
    return df


def compute_hitrate(
    samples_dir: Path,
    train_smiles: list[str],
    threshold: float,
    radius: int,
    n_bits: int,
) -> pd.DataFrame:
    _, train_fps, _ = smiles_to_fps(train_smiles, radius, n_bits)
    rows = []

    for ep, path in discover_epoch_files(samples_dir).items():
        df = load_epoch_raw(path)
        n_gen = len(df)
        raw = df["SMILES"].astype(str).tolist()

        valid_flags = []
        canons = []
        for smi in raw:
            c = canonicalize(smi)
            valid_flags.append(c is not None)
            canons.append(c)

        n_rdkit = sum(valid_flags)
        reinvent_col = "SMILES_state" if "SMILES_state" in df.columns else None
        if reinvent_col:
            n_reinvent = int((df[reinvent_col] == 1).sum())
        else:
            n_reinvent = n_rdkit

        # chromophore-like among RDKit-valid unique
        valid_smi = [c for c in canons if c is not None]
        _, fps, _ = smiles_to_fps(valid_smi, radius, n_bits, unique=False)
        # map max-sim for each valid molecule (aligned with fps from unique=False order)
        # recompute with unique=False preserving order of valid_smi
        mx = max_sims_to_ref(fps, train_fps) if fps else np.array([])
        n_chromo = int((mx >= threshold).sum()) if mx.size else 0

        rows.append(
            {
                "epoch": ep,
                "n_generated": n_gen,
                "n_rdkit_valid": n_rdkit,
                "hitrate_rdkit_valid": n_rdkit / n_gen if n_gen else np.nan,
                "n_reinvent_valid": n_reinvent,
                "hitrate_reinvent_valid": n_reinvent / n_gen if n_gen else np.nan,
                "n_chromophore_like": n_chromo,
                "hitrate_chromophore_like": n_chromo / n_gen if n_gen else np.nan,
                "chromophore_threshold": threshold,
            }
        )
        print(
            f"  epoch {ep:>3}: rdkit={rows[-1]['hitrate_rdkit_valid']:.3f}  "
            f"chromo@{threshold}={rows[-1]['hitrate_chromophore_like']:.3f}"
        )
    return pd.DataFrame(rows)


def plot_hitrate(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ep = df["epoch"]
    thr = df["chromophore_threshold"].iloc[0] if len(df) else 0.4
    ax.plot(ep, df["hitrate_rdkit_valid"], "o-", label="RDKit valid")
    ax.plot(ep, df["hitrate_reinvent_valid"], "s--", label="REINVENT SMILES_state=1")
    ax.plot(ep, df["hitrate_chromophore_like"], "^-", label=f"chromophore-like (Tc≥{thr})")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hit-rate")
    ax.set_title("Hit-rate across transfer-learning epochs")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Hit-rate evaluation across TL epochs")
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT / "hitrate")
    p.add_argument("--threshold", type=float, default=0.4,
                   help="Max-Tanimoto threshold for chromophore-like oracle")
    p.add_argument("--max-ref", type=int, default=5000)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = ensure_dir(args.out_dir)
    train = read_smiles(args.train, max_n=args.max_ref, seed=args.seed)
    print(f"[INFO] Samples dir: {args.samples_dir}")
    df = compute_hitrate(args.samples_dir, train, args.threshold, args.radius, args.n_bits)
    if df.empty:
        raise SystemExit("No epoch files found")
    csv_path = out / "hitrate_by_epoch.csv"
    df.to_csv(csv_path, index=False)
    plot_hitrate(df, out / "hitrate_curves.png")
    print(f"[OK] {csv_path}")
    print(f"[OK] {out / 'hitrate_curves.png'}")


if __name__ == "__main__":
    main()
