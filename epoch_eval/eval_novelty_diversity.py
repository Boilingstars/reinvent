"""
eval_novelty_diversity.py — novelty & internal diversity across TL epochs.

Definitions (set-level):
  * uniqueness   = #unique valid / #valid
  * novelty_vs_chromophores = fraction of unique molecules absent from train set
  * novelty_vs_chembl       = fraction of unique molecules absent from ChEMBL set
  * internal_diversity      = 1 − mean pairwise Tanimoto inside the epoch set
  * mean_intra_tanimoto     = mean of all unique pairs (same as Tanimoto script)

Plots diversity trajectory vs epoch.

Usage:
  python epoch_eval/eval_novelty_diversity.py
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
    DEFAULT_CHEMBL,
    DEFAULT_OUT,
    DEFAULT_SAMPLES,
    DEFAULT_TRAIN,
    ensure_dir,
    load_epoch_smiles,
    mean_intra,
    read_smiles,
    smiles_to_fps,
)


def compute(
    epochs: dict[int, list[str]],
    train_set: set[str],
    chembl_set: set[str],
    radius: int,
    n_bits: int,
) -> pd.DataFrame:
    rows = []
    for ep in sorted(epochs):
        raw = epochs[ep]
        valid, fps, invalid = smiles_to_fps(raw, radius, n_bits, unique=False)
        unique_smi, unique_fps, _ = smiles_to_fps(raw, radius, n_bits, unique=True)

        n_valid = len(valid)
        n_unique = len(unique_smi)
        uniqueness = n_unique / n_valid if n_valid else np.nan

        novel_c = sum(1 for s in unique_smi if s not in train_set)
        novel_d = sum(1 for s in unique_smi if s not in chembl_set)
        novelty_c = novel_c / n_unique if n_unique else np.nan
        novelty_d = novel_d / n_unique if n_unique else np.nan

        intra = mean_intra(unique_fps)
        diversity = 1.0 - intra if not np.isnan(intra) else np.nan

        rows.append(
            {
                "epoch": ep,
                "n_generated": len(raw),
                "n_valid": n_valid,
                "n_invalid": invalid,
                "n_unique": n_unique,
                "uniqueness": uniqueness,
                "novelty_vs_chromophores": novelty_c,
                "novelty_vs_chembl": novelty_d,
                "mean_intra_tanimoto": intra,
                "internal_diversity": diversity,
            }
        )
        print(
            f"  epoch {ep:>3}: uniq={uniqueness:.3f}  "
            f"nov_chrom={novelty_c:.3f}  div={diversity:.3f}"
        )
    return pd.DataFrame(rows)


def plot_trajectories(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    ep = df["epoch"]

    axes[0].plot(ep, df["internal_diversity"], "o-", color="#17becf")
    axes[0].set_title("Diversity trajectory")
    axes[0].set_ylabel("Internal diversity (1 − mean Tc)")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep, df["novelty_vs_chromophores"], "o-", label="vs chromophores")
    axes[1].plot(ep, df["novelty_vs_chembl"], "s-", label="vs ChEMBL")
    axes[1].set_title("Novelty")
    axes[1].set_ylabel("Fraction novel")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].plot(ep, df["uniqueness"], "o-", color="#9467bd")
    axes[2].set_title("Uniqueness")
    axes[2].set_ylabel("#unique / #valid")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Novelty & diversity across TL epochs")
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--chembl", type=Path, default=DEFAULT_CHEMBL)
    p.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT / "novelty_diversity")
    p.add_argument("--max-per-epoch", type=int, default=None)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = ensure_dir(args.out_dir)
    # full sets for novelty (membership); canonicalize
    train_raw = read_smiles(args.train)
    chembl_raw = read_smiles(args.chembl)
    train_set = set(smiles_to_fps(train_raw, args.radius, args.n_bits)[0])
    chembl_set = set(smiles_to_fps(chembl_raw, args.radius, args.n_bits)[0])

    epochs = load_epoch_smiles(args.samples_dir, max_per_epoch=args.max_per_epoch, seed=args.seed)
    if not epochs:
        raise SystemExit(f"No epoch CSVs in {args.samples_dir}")

    print(f"[INFO] Epochs: {list(epochs)}")
    df = compute(epochs, train_set, chembl_set, args.radius, args.n_bits)
    csv_path = out / "novelty_diversity_by_epoch.csv"
    df.to_csv(csv_path, index=False)
    plot_trajectories(df, out / "diversity_novelty_trajectories.png")
    print(f"[OK] {csv_path}")
    print(f"[OK] {out / 'diversity_novelty_trajectories.png'}")


if __name__ == "__main__":
    main()
