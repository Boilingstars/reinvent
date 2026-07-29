"""
eval_mahalanobis.py — Mahalanobis distance of generated molecules to reference
chemical spaces (chromophores & ChEMBL) across TL epochs.

Embedding: PCA of Morgan fingerprints fit on the pooled reference sets.
For each epoch reports set-level statistics (mean / median / p90 of per-molecule
Mahalanobis distances) — distributional comparison, not single-neighbour matching.

Usage:
  python epoch_eval/eval_mahalanobis.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    DEFAULT_CHEMBL,
    DEFAULT_OUT,
    DEFAULT_SAMPLES,
    DEFAULT_TRAIN,
    ensure_dir,
    fps_to_numpy,
    load_epoch_smiles,
    read_smiles,
    smiles_to_fps,
)


def fit_pca_and_stats(
    X_ref: np.ndarray,
    pca_dim: int,
    seed: int,
    eps: float = 1e-5,
) -> tuple[PCA, np.ndarray, np.ndarray]:
    dim = min(pca_dim, X_ref.shape[0] - 1, X_ref.shape[1])
    pca = PCA(n_components=dim, random_state=seed)
    Z = pca.fit_transform(X_ref)
    mu = Z.mean(axis=0)
    cov = np.cov(Z, rowvar=False) + np.eye(dim) * eps
    # store inverse covariance for Mahalanobis
    cov_inv = np.linalg.pinv(cov)
    return pca, mu, cov_inv


def mahalanobis_distances(Z: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    diff = Z - mu
    # (x-μ)^T Σ^{-1} (x-μ)
    left = diff @ cov_inv
    return np.sqrt(np.clip(np.sum(left * diff, axis=1), 0.0, None))


def compute(
    epochs: dict[int, list[str]],
    train: list[str],
    chembl: list[str],
    radius: int,
    n_bits: int,
    pca_dim: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[int, np.ndarray]]:
    _, train_fps, _ = smiles_to_fps(train, radius, n_bits)
    _, chembl_fps, _ = smiles_to_fps(chembl, radius, n_bits)
    X_train = fps_to_numpy(train_fps)
    X_chembl = fps_to_numpy(chembl_fps)

    pca_c, mu_c, inv_c = fit_pca_and_stats(X_train, pca_dim, seed)
    pca_d, mu_d, inv_d = fit_pca_and_stats(X_chembl, pca_dim, seed)

    rows = []
    dist_c: dict[int, np.ndarray] = {}
    dist_d: dict[int, np.ndarray] = {}

    for ep in sorted(epochs):
        _, fps, _ = smiles_to_fps(epochs[ep], radius, n_bits)
        X = fps_to_numpy(fps)
        if len(X) == 0:
            continue
        Zc = pca_c.transform(X)
        Zd = pca_d.transform(X)
        dc = mahalanobis_distances(Zc, mu_c, inv_c)
        dd = mahalanobis_distances(Zd, mu_d, inv_d)
        dist_c[ep] = dc
        dist_d[ep] = dd
        rows.append(
            {
                "epoch": ep,
                "n_molecules": len(dc),
                "mean_mahal_vs_chromophores": float(dc.mean()),
                "median_mahal_vs_chromophores": float(np.median(dc)),
                "p90_mahal_vs_chromophores": float(np.percentile(dc, 90)),
                "mean_mahal_vs_chembl": float(dd.mean()),
                "median_mahal_vs_chembl": float(np.median(dd)),
                "p90_mahal_vs_chembl": float(np.percentile(dd, 90)),
            }
        )
        print(
            f"  epoch {ep:>3}: mean_M_chrom={rows[-1]['mean_mahal_vs_chromophores']:.3f}  "
            f"mean_M_chembl={rows[-1]['mean_mahal_vs_chembl']:.3f}"
        )
    return pd.DataFrame(rows), dist_c, dist_d


def plot_curves(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    ep = df["epoch"]
    axes[0].plot(ep, df["mean_mahal_vs_chromophores"], "o-", label="mean")
    axes[0].plot(ep, df["median_mahal_vs_chromophores"], "s--", label="median")
    axes[0].plot(ep, df["p90_mahal_vs_chromophores"], "^:", label="p90")
    axes[0].set_title("Mahalanobis → chromophore space")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Distance (↓ closer to ref)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(ep, df["mean_mahal_vs_chembl"], "o-", label="mean")
    axes[1].plot(ep, df["median_mahal_vs_chembl"], "s--", label="median")
    axes[1].plot(ep, df["p90_mahal_vs_chembl"], "^:", label="p90")
    axes[1].set_title("Mahalanobis → ChEMBL drug space")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Distance (↓ closer to ref)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_boxplots(
    dist_c: dict[int, np.ndarray],
    dist_d: dict[int, np.ndarray],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, data, title in (
        (axes[0], dist_c, "Mahalanobis → chromophores"),
        (axes[1], dist_d, "Mahalanobis → ChEMBL"),
    ):
        eps = sorted(data)
        ax.boxplot([data[e] for e in eps], tick_labels=eps, showfliers=False)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mahalanobis distance")
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Mahalanobis distance evaluation across TL epochs")
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--chembl", type=Path, default=DEFAULT_CHEMBL)
    p.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT / "mahalanobis")
    p.add_argument("--max-ref", type=int, default=3000)
    p.add_argument("--max-per-epoch", type=int, default=None)
    p.add_argument("--pca-dim", type=int, default=64)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = ensure_dir(args.out_dir)
    train = read_smiles(args.train, max_n=args.max_ref, seed=args.seed)
    chembl = read_smiles(args.chembl, max_n=args.max_ref, seed=args.seed)
    epochs = load_epoch_smiles(args.samples_dir, max_per_epoch=args.max_per_epoch, seed=args.seed)
    if not epochs:
        raise SystemExit(f"No epoch CSVs in {args.samples_dir}")

    print(f"[INFO] Epochs: {list(epochs)}")
    df, dc, dd = compute(epochs, train, chembl, args.radius, args.n_bits, args.pca_dim, args.seed)
    csv_path = out / "mahalanobis_by_epoch.csv"
    df.to_csv(csv_path, index=False)
    plot_curves(df, out / "mahalanobis_curves.png")
    plot_boxplots(dc, dd, out / "mahalanobis_boxplots.png")
    print(f"[OK] {csv_path}")
    print(f"[OK] plots → {out}")


if __name__ == "__main__":
    main()
