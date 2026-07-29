"""
eval_pca.py — 2D PCA projection of Morgan fingerprints.

Pools:
  * ChEMBL drug-like (prior chemical space)
  * chromophore fine-tuning set
  * generated molecules from every TL epoch

Fits one PCA on the combined fingerprint matrix so axes are comparable.

Usage:
  python epoch_eval/eval_pca.py
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


def build_matrix(
    chembl: list[str],
    train: list[str],
    epochs: dict[int, list[str]],
    radius: int,
    n_bits: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return X, max_tanimoto_placeholder unused, labels."""
    blocks: list[np.ndarray] = []
    labels: list[str] = []

    _, fps, _ = smiles_to_fps(chembl, radius, n_bits)
    X = fps_to_numpy(fps)
    blocks.append(X)
    labels.extend(["chembl"] * len(X))

    _, fps, _ = smiles_to_fps(train, radius, n_bits)
    X = fps_to_numpy(fps)
    blocks.append(X)
    labels.extend(["chromophore"] * len(X))

    for ep in sorted(epochs):
        _, fps, _ = smiles_to_fps(epochs[ep], radius, n_bits)
        X = fps_to_numpy(fps)
        if len(X) == 0:
            continue
        blocks.append(X)
        labels.extend([f"epoch_{ep}"] * len(X))

    return np.vstack(blocks), np.array(labels)


def plot_pca(Z: np.ndarray, labels: np.ndarray, out_path: Path, explained: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    # background references first
    for name, color, size, alpha, marker in (
        ("chembl", "#bbbbbb", 12, 0.35, "."),
        ("chromophore", "#2ca02c", 18, 0.45, "."),
    ):
        mask = labels == name
        if mask.any():
            ax.scatter(
                Z[mask, 0],
                Z[mask, 1],
                c=color,
                s=size,
                alpha=alpha,
                marker=marker,
                label=name,
                linewidths=0,
            )

    epoch_labels = sorted(
        {lb for lb in np.unique(labels) if str(lb).startswith("epoch_")},
        key=lambda s: int(str(s).split("_")[1]),
    )
    cmap = plt.get_cmap("plasma")
    for i, lb in enumerate(epoch_labels):
        mask = labels == lb
        color = cmap(i / max(len(epoch_labels) - 1, 1))
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            c=[color],
            s=36,
            alpha=0.85,
            marker="o",
            edgecolors="k",
            linewidths=0.3,
            label=lb.replace("epoch_", "ep "),
        )

    ax.set_xlabel(f"PC1 ({100 * explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * explained[1]:.1f}%)")
    ax.set_title("PCA of Morgan fingerprints: ChEMBL + chromophores + epochs")
    ax.legend(markerscale=1.4, fontsize=8, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_centroids(Z: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    """Centroid trajectory of each epoch in PCA space."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, color, marker in (("chembl", "#888888", "s"), ("chromophore", "#2ca02c", "D")):
        mask = labels == name
        if mask.any():
            c = Z[mask].mean(axis=0)
            ax.scatter(c[0], c[1], c=color, s=120, marker=marker, label=f"{name} centroid", zorder=5)

    eps = sorted(
        {lb for lb in np.unique(labels) if str(lb).startswith("epoch_")},
        key=lambda s: int(str(s).split("_")[1]),
    )
    cents = []
    for lb in eps:
        mask = labels == lb
        cents.append(Z[mask].mean(axis=0))
    cents = np.asarray(cents)
    if len(cents):
        ax.plot(cents[:, 0], cents[:, 1], "o--", color="#d62728", label="epoch centroids")
        for lb, c in zip(eps, cents):
            ax.annotate(lb.replace("epoch_", "e"), (c[0], c[1]), fontsize=8)

    ax.set_title("Centroid trajectory in PCA space")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="PCA 2D projection of chemical spaces")
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--chembl", type=Path, default=DEFAULT_CHEMBL)
    p.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT / "pca")
    p.add_argument("--max-ref", type=int, default=2000, help="Subsample refs for density")
    p.add_argument("--max-per-epoch", type=int, default=None)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = ensure_dir(args.out_dir)
    chembl = read_smiles(args.chembl, max_n=args.max_ref, seed=args.seed)
    train = read_smiles(args.train, max_n=args.max_ref, seed=args.seed + 1)
    epochs = load_epoch_smiles(args.samples_dir, max_per_epoch=args.max_per_epoch, seed=args.seed)
    if not epochs:
        raise SystemExit(f"No epoch CSVs in {args.samples_dir}")

    print("[INFO] Building fingerprint matrix …")
    X, labels = build_matrix(chembl, train, epochs, args.radius, args.n_bits)
    print(f"[INFO] Matrix shape: {X.shape}")

    pca = PCA(n_components=2, random_state=args.seed)
    Z = pca.fit_transform(X)

    coords = pd.DataFrame(
        {"pc1": Z[:, 0], "pc2": Z[:, 1], "label": labels}
    )
    coords.to_csv(out / "pca_coordinates.csv", index=False)

    plot_pca(Z, labels, out / "pca_2d_overview.png", pca.explained_variance_ratio_)
    plot_centroids(Z, labels, out / "pca_centroid_trajectory.png")
    print(f"[OK] plots → {out}")


if __name__ == "__main__":
    main()
