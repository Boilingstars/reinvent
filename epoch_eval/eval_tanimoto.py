"""
eval_tanimoto.py — set-level Tanimoto metrics across TL epochs.

Compares distributions as wholes (mean over all pairs), not only nearest neighbours:
  * intra-epoch mean Tanimoto
  * consecutive-epoch cross-mean Tanimoto
  * cross-mean / max / median Tanimoto vs chromophore train set
  * cross-mean / max / median Tanimoto vs ChEMBL drug-like set

Writes CSV tables and plots under --out-dir.

Usage (from repo root):
  python -m epoch_eval.eval_tanimoto
  python epoch_eval/eval_tanimoto.py --train data/train.smi --chembl data/chembl_drugs.smi
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
    max_sims_to_ref,
    mean_intra,
    mean_pairwise,
    read_smiles,
    smiles_to_fps,
    tanimoto_matrix,
)


def compute_tanimoto_metrics(
    epochs: dict[int, list[str]],
    train_smiles: list[str],
    chembl_smiles: list[str],
    radius: int,
    n_bits: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, np.ndarray], dict[int, np.ndarray]]:
    _, train_fps, _ = smiles_to_fps(train_smiles, radius, n_bits)
    _, chembl_fps, _ = smiles_to_fps(chembl_smiles, radius, n_bits)

    epoch_fps: dict[int, list] = {}
    rows = []
    max_to_chrom: dict[int, np.ndarray] = {}
    max_to_drug: dict[int, np.ndarray] = {}

    for ep in sorted(epochs):
        valid, fps, invalid = smiles_to_fps(epochs[ep], radius, n_bits)
        epoch_fps[ep] = fps
        mx_c = max_sims_to_ref(fps, train_fps)
        mx_d = max_sims_to_ref(fps, chembl_fps)
        max_to_chrom[ep] = mx_c
        max_to_drug[ep] = mx_d

        rows.append(
            {
                "epoch": ep,
                "n_raw": len(epochs[ep]),
                "n_valid_unique": len(valid),
                "n_invalid": invalid,
                "intra_mean_tanimoto": mean_intra(fps),
                "cross_mean_vs_chromophores": mean_pairwise(fps, train_fps),
                "cross_mean_vs_chembl": mean_pairwise(fps, chembl_fps),
                "mean_max_tanimoto_vs_chromophores": float(mx_c.mean()) if mx_c.size else np.nan,
                "median_max_tanimoto_vs_chromophores": float(np.median(mx_c)) if mx_c.size else np.nan,
                "mean_max_tanimoto_vs_chembl": float(mx_d.mean()) if mx_d.size else np.nan,
                "median_max_tanimoto_vs_chembl": float(np.median(mx_d)) if mx_d.size else np.nan,
            }
        )
        print(
            f"  epoch {ep:>3}: intra={rows[-1]['intra_mean_tanimoto']:.4f}  "
            f"cross_chrom={rows[-1]['cross_mean_vs_chromophores']:.4f}  "
            f"cross_chembl={rows[-1]['cross_mean_vs_chembl']:.4f}"
        )

    # consecutive epoch cross-mean (full bipartite mean, not NN)
    pair_rows = []
    eps = sorted(epoch_fps)
    for a, b in zip(eps, eps[1:]):
        mat = tanimoto_matrix(epoch_fps[a], epoch_fps[b])
        pair_rows.append(
            {
                "epoch_prev": a,
                "epoch_next": b,
                "cross_mean_tanimoto": float(mat.mean()) if mat.size else np.nan,
                "mean_max_prev_to_next": float(mat.max(axis=1).mean()) if mat.size else np.nan,
                "mean_max_next_to_prev": float(mat.max(axis=0).mean()) if mat.size else np.nan,
            }
        )
        print(
            f"  epochs {a}->{b}: cross_mean={pair_rows[-1]['cross_mean_tanimoto']:.4f}"
        )

    return pd.DataFrame(rows), pd.DataFrame(pair_rows), max_to_chrom, max_to_drug


def plot_max_and_median(
    summary: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    ep = summary["epoch"]

    axes[0].plot(ep, summary["mean_max_tanimoto_vs_chromophores"], "o-", label="mean max → chromophores")
    axes[0].plot(ep, summary["median_max_tanimoto_vs_chromophores"], "s--", label="median max → chromophores")
    axes[0].plot(ep, summary["mean_max_tanimoto_vs_chembl"], "^-", label="mean max → ChEMBL")
    axes[0].plot(ep, summary["median_max_tanimoto_vs_chembl"], "v--", label="median max → ChEMBL")
    axes[0].set_title("Max Tanimoto to reference sets")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Tanimoto")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(ep, summary["cross_mean_vs_chromophores"], "o-", label="cross-mean → chromophores")
    axes[1].plot(ep, summary["cross_mean_vs_chembl"], "s-", label="cross-mean → ChEMBL")
    axes[1].plot(ep, summary["intra_mean_tanimoto"], "D-", label="intra-epoch mean")
    axes[1].set_title("Set-level (all-pairs) mean Tanimoto")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Tanimoto")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cross_epoch(pairs: pd.DataFrame, out_path: Path) -> None:
    if pairs.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [f"{int(a)}→{int(b)}" for a, b in zip(pairs["epoch_prev"], pairs["epoch_next"])]
    ax.plot(labels, pairs["cross_mean_tanimoto"], "o-", color="#1f77b4", label="cross-mean (all pairs)")
    ax.plot(labels, pairs["mean_max_prev_to_next"], "s--", color="#ff7f0e", label="mean max prev→next")
    ax.set_title("Consecutive-epoch Tanimoto (set-level)")
    ax.set_xlabel("Epoch pair")
    ax.set_ylabel("Tanimoto")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_max_distributions(
    max_to_chrom: dict[int, np.ndarray],
    max_to_drug: dict[int, np.ndarray],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, data, title in (
        (axes[0], max_to_chrom, "Max Tanimoto → chromophores"),
        (axes[1], max_to_drug, "Max Tanimoto → ChEMBL"),
    ):
        epochs = sorted(data)
        vals = [data[e] for e in epochs]
        ax.boxplot(vals, tick_labels=epochs, showfliers=False)
        means = [float(v.mean()) if len(v) else np.nan for v in vals]
        medians = [float(np.median(v)) if len(v) else np.nan for v in vals]
        ax.plot(range(1, len(epochs) + 1), means, "o-", label="mean")
        ax.plot(range(1, len(epochs) + 1), medians, "s--", label="median")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Max Tanimoto")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Set-level Tanimoto evaluation across TL epochs")
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--chembl", type=Path, default=DEFAULT_CHEMBL)
    p.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT / "tanimoto")
    p.add_argument("--max-ref", type=int, default=3000, help="Subsample size for reference sets")
    p.add_argument("--max-per-epoch", type=int, default=None)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = ensure_dir(args.out_dir)

    print("[INFO] Loading reference sets …")
    train = read_smiles(args.train, max_n=args.max_ref, seed=args.seed)
    chembl = read_smiles(args.chembl, max_n=args.max_ref, seed=args.seed)
    epochs = load_epoch_smiles(args.samples_dir, max_per_epoch=args.max_per_epoch, seed=args.seed)
    if not epochs:
        raise SystemExit(f"No epoch CSVs found in {args.samples_dir}")

    print(f"[INFO] Epochs: {list(epochs)}")
    summary, pairs, mx_c, mx_d = compute_tanimoto_metrics(
        epochs, train, chembl, args.radius, args.n_bits
    )

    summary_path = out / "tanimoto_by_epoch.csv"
    pairs_path = out / "tanimoto_consecutive_epochs.csv"
    summary.to_csv(summary_path, index=False)
    pairs.to_csv(pairs_path, index=False)

    plot_max_and_median(summary, out / "tanimoto_max_median_and_crossmean.png")
    plot_cross_epoch(pairs, out / "tanimoto_cross_epoch.png")
    plot_max_distributions(mx_c, mx_d, out / "tanimoto_max_boxplots.png")

    print(f"[OK] Tables → {summary_path}, {pairs_path}")
    print(f"[OK] Plots  → {out}")


if __name__ == "__main__":
    main()
