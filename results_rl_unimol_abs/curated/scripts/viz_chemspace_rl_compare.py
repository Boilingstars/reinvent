"""
Chemical space compare: RL TL sweep vs RL UniMol abs.

For each of PCA / UMAP / t-SNE:
  - joint fit on ChEMBL + chromophores + both RL sets
  - plot 01: ChEMBL + chromophores + rl_tl_sweep (Score color)
  - plot 02: ChEMBL + chromophores + rl_unimol_abs (Score color)
  - optional side-by-side

Outputs → results_rl_unimol_abs/chemspace_rl_compare/

Usage:
  .\\.venv_eval\\Scripts\\python.exe results_rl_unimol_abs\\viz_chemspace_rl_compare.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "epoch_eval"))
sys.path.insert(0, str(ROOT / "results_rl_unimol_abs"))
from utils import (  # noqa: E402
    DEFAULT_CHEMBL,
    DEFAULT_TRAIN,
    ensure_dir,
    fps_to_numpy,
    read_smiles,
    smiles_to_fps,
)
from viz_chemspace_compare import load_rl_smiles  # noqa: E402

try:
    from umap import UMAP

    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

OUT = Path(__file__).resolve().parent / "chemspace_rl_compare"
SWEEP_CSV = ROOT / "rl_tl_sweep_20260731_004743_ep00_1.csv"
UNIMOL_CSV = ROOT / "rl_unimol_abs_1.csv"

STYLES = {
    "chembl": dict(color="#bdbdbd", s=8, alpha=0.25, z=1, label="ChEMBL"),
    "chromophore": dict(color="#2ca02c", s=10, alpha=0.35, z=2, label="Chromophores"),
}


def embed(X: np.ndarray, method: str, seed: int, pca_dim: int = 50) -> tuple[np.ndarray, str]:
    n = X.shape[0]
    if method == "pca":
        pca = PCA(n_components=2, random_state=seed)
        Z = pca.fit_transform(X)
        ev = pca.explained_variance_ratio_
        return Z, f"PCA (EVR {100*ev[0]:.1f}% / {100*ev[1]:.1f}%)"

    Xp = X
    if X.shape[1] > pca_dim and n > pca_dim + 1:
        Xp = PCA(n_components=min(pca_dim, n - 1), random_state=seed).fit_transform(X)

    if method == "umap":
        if not HAVE_UMAP:
            raise RuntimeError("umap-learn required")
        reducer = UMAP(
            n_components=2,
            n_neighbors=min(30, max(5, n // 20)),
            min_dist=0.15,
            metric="euclidean",
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = reducer.fit_transform(Xp)
        return Z, f"UMAP (PCA→{Xp.shape[1]})"

    if method == "tsne":
        perp = min(40, max(5, (n - 1) // 4))
        reducer = TSNE(
            n_components=2,
            perplexity=perp,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = reducer.fit_transform(Xp)
        return Z, f"t-SNE (perp={perp}, PCA prep)"

    raise ValueError(method)


def plot_panel(
    ax,
    Z: np.ndarray,
    labels: np.ndarray,
    rl_label: str,
    rl_scores: np.ndarray,
    title: str,
) -> None:
    for name in ("chembl", "chromophore"):
        m = labels == name
        if not m.any():
            continue
        st = STYLES[name]
        ax.scatter(
            Z[m, 0],
            Z[m, 1],
            c=st["color"],
            s=st["s"],
            alpha=st["alpha"],
            linewidths=0,
            label=f"{st['label']} (n={m.sum()})",
            zorder=st["z"],
        )
    m = labels == rl_label
    sc = ax.scatter(
        Z[m, 0],
        Z[m, 1],
        c=rl_scores,
        cmap="plasma",
        s=18,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.15,
        zorder=5,
        label=f"{rl_label} (n={m.sum()})",
    )
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    return sc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", type=Path, default=SWEEP_CSV)
    p.add_argument("--unimol", type=Path, default=UNIMOL_CSV)
    p.add_argument("--chembl", type=Path, default=DEFAULT_CHEMBL)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--max-ref", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--pca-prep", type=int, default=50)
    args = p.parse_args()

    out = ensure_dir(args.out)

    print("[INFO] Loading references…")
    chembl = read_smiles(args.chembl, max_n=None, seed=args.seed)
    chromo = read_smiles(args.train, max_n=args.max_ref, seed=args.seed)
    sweep_smi, sweep_sc = load_rl_smiles(args.sweep, max_n=None, seed=args.seed)
    uni_smi, uni_sc = load_rl_smiles(args.unimol, max_n=None, seed=args.seed)
    print(
        f"  chembl={len(chembl)}, chromo={len(chromo)}, "
        f"sweep={len(sweep_smi)}, unimol={len(uni_smi)}"
    )

    print("[INFO] Fingerprints…")
    blocks, labels, smiles_all = [], [], []
    score_maps = {"rl_sweep": dict(zip(sweep_smi, sweep_sc)), "rl_unimol": dict(zip(uni_smi, uni_sc))}

    for name, smis in (
        ("chembl", chembl),
        ("chromophore", chromo),
        ("rl_sweep", sweep_smi),
        ("rl_unimol", uni_smi),
    ):
        valid, fps, n_inv = smiles_to_fps(smis, radius=args.radius, n_bits=args.n_bits, unique=True)
        print(f"  {name}: {len(valid)} (invalid≈{n_inv})")
        if not fps:
            continue
        blocks.append(fps_to_numpy(fps))
        labels.extend([name] * len(fps))
        smiles_all.extend(valid)

    X = np.vstack(blocks)
    labels_a = np.asarray(labels)
    print(f"  matrix {X.shape}")

    for method in ("pca", "umap", "tsne"):
        print(f"[INFO] {method.upper()}…")
        try:
            Z, tag = embed(X, method, args.seed, pca_dim=args.pca_prep)
        except Exception as exc:
            print(f"[WARN] skip {method}: {exc}")
            continue

        pd.DataFrame(
            {"smiles": smiles_all, "set": labels_a, "dim1": Z[:, 0], "dim2": Z[:, 1]}
        ).to_csv(out / f"coordinates_{method}.csv", index=False)

        # scores aligned to kept RL smiles
        sweep_scores = np.array(
            [score_maps["rl_sweep"].get(s, np.nan) for s, lb in zip(smiles_all, labels_a) if lb == "rl_sweep"]
        )
        uni_scores = np.array(
            [score_maps["rl_unimol"].get(s, np.nan) for s, lb in zip(smiles_all, labels_a) if lb == "rl_unimol"]
        )

        # individual plots
        for rl_label, scores, fname, nice in (
            ("rl_sweep", sweep_scores, f"{method}_01_rl_sweep.png", "RL TL sweep"),
            ("rl_unimol", uni_scores, f"{method}_02_rl_unimol.png", "RL UniMol abs"),
        ):
            fig, ax = plt.subplots(figsize=(8.5, 7))
            sc = plot_panel(ax, Z, labels_a, rl_label, scores, f"{nice}\n{tag}")
            fig.colorbar(sc, ax=ax, label="Score")
            fig.tight_layout()
            fig.savefig(out / fname, dpi=160)
            plt.close(fig)
            print(f"[OK] {fname}")

        # side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        sc1 = plot_panel(axes[0], Z, labels_a, "rl_sweep", sweep_scores, f"RL TL sweep\n{tag}")
        sc2 = plot_panel(axes[1], Z, labels_a, "rl_unimol", uni_scores, f"RL UniMol abs\n{tag}")
        fig.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.04, label="Score")
        fig.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.04, label="Score")
        fig.suptitle("ChEMBL + chromophores + RL (joint embedding)", y=1.02)
        fig.tight_layout()
        fig.savefig(out / f"{method}_00_side_by_side.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {method}_00_side_by_side.png")

    (out / "README.md").write_text(
        "\n".join(
            [
                "# RL chemspace compare: TL sweep vs UniMol abs",
                "",
                "Joint PCA / UMAP / t-SNE on ChEMBL + chromophores + both RL sets.",
                "For each method: two plots (sweep / unimol) with identical reference coordinates,",
                "plus a side-by-side panel.",
                "",
                f"- Sweep: `{args.sweep.name}`",
                f"- UniMol: `{args.unimol.name}`",
                f"- ChEMBL n={int((labels_a=='chembl').sum())}, chromophore n={int((labels_a=='chromophore').sum())}",
                f"- RL sweep n={int((labels_a=='rl_sweep').sum())}, RL unimol n={int((labels_a=='rl_unimol').sum())}",
                "",
                "Files: `{pca,umap,tsne}_01_rl_sweep.png`, `{pca,umap,tsne}_02_rl_unimol.png`,",
                "`{method}_00_side_by_side.png`, `coordinates_*.csv`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] → {out}")


if __name__ == "__main__":
    main()
