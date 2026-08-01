"""
Compare chemical space of RL-generated molecules vs chromophore training set vs ChEMBL.

Embeddings (Morgan fingerprints, joint fit):
  - PCA
  - UMAP (PCA→50 prep)
  - t-SNE (PCA→50 prep)

Outputs under results_rl_unimol_abs/chemspace_compare/:
  01_pca_overview.png
  02_umap_overview.png
  03_tsne_overview.png
  04_methods_side_by_side.png
  05_umap_rl_colored_by_score.png
  06_pca_variance.png
  coordinates_*.csv

Usage (from repo root):
  .\\.venv_eval\\Scripts\\python.exe results_rl_unimol_abs\\viz_chemspace_compare.py
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
from utils import (  # noqa: E402
    DEFAULT_CHEMBL,
    DEFAULT_TRAIN,
    ensure_dir,
    fps_to_numpy,
    read_smiles,
    smiles_to_fps,
)

OUT_DIR = Path(__file__).resolve().parent / "chemspace_compare"
DEFAULT_RL_CSV = ROOT / "rl_unimol_abs_1.csv"

try:
    from umap import UMAP

    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

STYLES = {
    "chembl": dict(color="#9e9e9e", label="ChEMBL", s=10, alpha=0.28, z=1, marker="."),
    "chromophore": dict(color="#2ca02c", label="Chromophores (train)", s=14, alpha=0.40, z=2, marker="."),
    "rl": dict(color="#d62728", label="RL generated", s=22, alpha=0.55, z=3, marker="o"),
}


def load_rl_smiles(path: Path, max_n: int | None, seed: int) -> tuple[list[str], np.ndarray]:
    """Valid unique SMILES + matching Score array (aligned after unique)."""
    df = pd.read_csv(path)
    if "SMILES" not in df.columns:
        raise ValueError(f"No SMILES column in {path}")
    state = df["SMILES_state"] if "SMILES_state" in df.columns else 1
    score = df["Score"] if "Score" in df.columns else np.nan
    sub = df.loc[state == 1, ["SMILES", "Score"] if "Score" in df.columns else ["SMILES"]].copy()
    if "Score" not in sub.columns:
        sub["Score"] = np.nan

    smis = sub["SMILES"].astype(str).tolist()
    scores = sub["Score"].to_numpy(dtype=float)

    # unique by canonicalization happens in smiles_to_fps; keep score of first occurrence
    seen: set[str] = set()
    uniq_smi: list[str] = []
    uniq_score: list[float] = []
    from rdkit import Chem

    for smi, sc in zip(smis, scores):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol, canonical=True)
        if canon in seen:
            continue
        seen.add(canon)
        uniq_smi.append(canon)
        uniq_score.append(float(sc))

    if max_n is not None and len(uniq_smi) > max_n:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(uniq_smi), size=max_n, replace=False)
        idx = np.sort(idx)
        uniq_smi = [uniq_smi[i] for i in idx]
        uniq_score = [uniq_score[i] for i in idx]

    return uniq_smi, np.asarray(uniq_score, dtype=float)


def build_matrix(
    chembl: list[str],
    chromo: list[str],
    rl: list[str],
    radius: int,
    n_bits: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    blocks: list[np.ndarray] = []
    labels: list[str] = []
    kept_smiles: list[str] = []

    for name, smis in (("chembl", chembl), ("chromophore", chromo), ("rl", rl)):
        valid, fps, n_inv = smiles_to_fps(smis, radius=radius, n_bits=n_bits, unique=True)
        print(f"  {name}: {len(valid)} valid unique FP (invalid skipped≈{n_inv})")
        if not fps:
            continue
        blocks.append(fps_to_numpy(fps))
        labels.extend([name] * len(fps))
        kept_smiles.extend(valid)

    X = np.vstack(blocks)
    return X, np.asarray(labels), kept_smiles


def embed_umap(X: np.ndarray, seed: int, pca_dim: int = 50) -> tuple[np.ndarray, str]:
    if not HAVE_UMAP:
        raise RuntimeError("umap-learn is not installed")
    n = X.shape[0]
    Xp = X
    if X.shape[1] > pca_dim and n > pca_dim + 1:
        Xp = PCA(n_components=min(pca_dim, n - 1), random_state=seed).fit_transform(X)
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
    return Z, f"UMAP (PCA→{Xp.shape[1]} prep)"


def embed_tsne(X: np.ndarray, seed: int, pca_dim: int = 50) -> tuple[np.ndarray, str]:
    n = X.shape[0]
    Xp = X
    if X.shape[1] > pca_dim and n > pca_dim + 1:
        Xp = PCA(n_components=min(pca_dim, n - 1), random_state=seed).fit_transform(X)
    perplexity = min(40, max(5, (n - 1) // 4))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = reducer.fit_transform(Xp)
    return Z, f"t-SNE (perplexity={perplexity}, PCA prep)"


def plot_overview(Z: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    order = ("chembl", "chromophore", "rl")
    for name in order:
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
            marker=st["marker"],
            linewidths=0.2 if name == "rl" else 0,
            edgecolors="k" if name == "rl" else "none",
            label=f"{st['label']} (n={m.sum()})",
            zorder=st["z"],
        )
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def plot_side_by_side(
    panels: list[tuple[np.ndarray, np.ndarray, str]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.2))
    if len(panels) == 1:
        axes = [axes]
    order = ("chembl", "chromophore", "rl")
    for ax, (Z, labels, title) in zip(axes, panels):
        for name in order:
            m = labels == name
            if not m.any():
                continue
            st = STYLES[name]
            ax.scatter(
                Z[m, 0],
                Z[m, 1],
                c=st["color"],
                s=st["s"] * 0.85,
                alpha=st["alpha"],
                marker=st["marker"],
                linewidths=0.15 if name == "rl" else 0,
                edgecolors="k" if name == "rl" else "none",
                label=st["label"] if ax is axes[0] else None,
                zorder=st["z"],
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.grid(True, alpha=0.25)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=STYLES[n]["color"],
            markersize=8,
            label=STYLES[n]["label"],
        )
        for n in order
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=True)
    fig.suptitle("Chemical space: ChEMBL vs chromophores vs RL generated", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def plot_rl_by_score(
    Z: np.ndarray,
    labels: np.ndarray,
    rl_scores: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Background refs + RL points colored by Score."""
    fig, ax = plt.subplots(figsize=(9, 7))
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
            alpha=0.22,
            marker=".",
            linewidths=0,
            label=st["label"],
            zorder=st["z"],
        )
    m = labels == "rl"
    sc = ax.scatter(
        Z[m, 0],
        Z[m, 1],
        c=rl_scores,
        cmap="plasma",
        s=28,
        alpha=0.75,
        edgecolors="k",
        linewidths=0.2,
        label="RL (color = Score)",
        zorder=3,
    )
    fig.colorbar(sc, ax=ax, label="Score")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def save_coords(Z: np.ndarray, labels: np.ndarray, smiles: list[str], path: Path) -> None:
    pd.DataFrame(
        {
            "smiles": smiles,
            "set": labels,
            "dim1": Z[:, 0],
            "dim2": Z[:, 1],
        }
    ).to_csv(path, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Chemspace: RL vs chromophores vs ChEMBL")
    p.add_argument("--rl-csv", type=Path, default=DEFAULT_RL_CSV)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--chembl", type=Path, default=DEFAULT_CHEMBL)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--max-ref", type=int, default=4000, help="Max chromophore SMILES (subsample)")
    p.add_argument("--max-chembl", type=int, default=None, help="Max ChEMBL SMILES")
    p.add_argument("--max-gen", type=int, default=None, help="Max RL SMILES")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pca-prep", type=int, default=50)
    args = p.parse_args()

    out = ensure_dir(args.out)
    print("[INFO] Loading SMILES…")
    chembl = read_smiles(args.chembl, max_n=args.max_chembl, seed=args.seed)
    chromo = read_smiles(args.train, max_n=args.max_ref, seed=args.seed)
    rl_smi, rl_scores_all = load_rl_smiles(args.rl_csv, max_n=args.max_gen, seed=args.seed)
    print(f"  loaded chembl={len(chembl)}, chromophore={len(chromo)}, rl={len(rl_smi)}")

    print("[INFO] Fingerprints…")
    X, labels, smiles = build_matrix(chembl, chromo, rl_smi, args.radius, args.n_bits)
    print(f"  matrix {X.shape}")

    # Align RL scores to kept RL smiles order in matrix
    rl_canon_to_score = dict(zip(rl_smi, rl_scores_all))
    rl_scores = np.array(
        [rl_canon_to_score.get(s, np.nan) for s, lb in zip(smiles, labels) if lb == "rl"]
    )

    panels: list[tuple[np.ndarray, np.ndarray, str]] = []

    print("[INFO] PCA…")
    # fuller PCA for scree + 2D projection
    n_comp = min(30, X.shape[0] - 1, X.shape[1])
    pca_full = PCA(n_components=n_comp, random_state=args.seed)
    Z_full = pca_full.fit_transform(X)
    Z_pca = Z_full[:, :2]
    ev = pca_full.explained_variance_ratio_
    tag_pca = f"PCA (EVR PC1={100*ev[0]:.1f}%, PC2={100*ev[1]:.1f}%)"
    plot_overview(Z_pca, labels, f"Chemical space — {tag_pca}", out / "01_pca_overview.png")
    save_coords(Z_pca, labels, smiles, out / "coordinates_pca.csv")
    plot_rl_by_score(
        Z_pca,
        labels,
        rl_scores,
        f"PCA — RL colored by Score\n({tag_pca})",
        out / "05a_pca_rl_colored_by_score.png",
    )
    panels.append((Z_pca, labels, tag_pca))

    # scree
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(1, len(ev) + 1), 100 * ev, color="#4c72b0")
    ax.plot(np.arange(1, len(ev) + 1), 100 * np.cumsum(ev), "o-", color="#c44e52", label="cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title(f"PCA scree (top {n_comp} PCs; cum2={100*(ev[0]+ev[1]):.1f}%)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "06_pca_variance.png", dpi=140)
    plt.close(fig)
    print("[OK] 06_pca_variance.png")

    if HAVE_UMAP:
        print("[INFO] UMAP…")
        Z_umap, tag_umap = embed_umap(X, args.seed, pca_dim=args.pca_prep)
        plot_overview(Z_umap, labels, f"Chemical space — {tag_umap}", out / "02_umap_overview.png")
        save_coords(Z_umap, labels, smiles, out / "coordinates_umap.csv")
        panels.append((Z_umap, labels, tag_umap))
        plot_rl_by_score(
            Z_umap,
            labels,
            rl_scores,
            f"UMAP — RL colored by Score\n({tag_umap})",
            out / "05b_umap_rl_colored_by_score.png",
        )
    else:
        print("[WARN] umap-learn missing — skip UMAP plots")

    print("[INFO] t-SNE…")
    Z_tsne, tag_tsne = embed_tsne(X, args.seed, pca_dim=args.pca_prep)
    plot_overview(Z_tsne, labels, f"Chemical space — {tag_tsne}", out / "03_tsne_overview.png")
    save_coords(Z_tsne, labels, smiles, out / "coordinates_tsne.csv")
    plot_rl_by_score(
        Z_tsne,
        labels,
        rl_scores,
        f"t-SNE — RL colored by Score\n({tag_tsne})",
        out / "05c_tsne_rl_colored_by_score.png",
    )
    panels.append((Z_tsne, labels, tag_tsne))

    plot_side_by_side(panels, out / "04_methods_side_by_side.png")

    # short summary
    summary_lines = [
        "# Chemical space comparison",
        "",
        f"- ChEMBL: {int((labels == 'chembl').sum())}",
        f"- Chromophores (train, subsampled to max {args.max_ref}): {int((labels == 'chromophore').sum())}",
        f"- RL generated (valid unique): {int((labels == 'rl').sum())}",
        f"- Fingerprints: Morgan r={args.radius}, {args.n_bits} bits",
        f"- PCA PC1+PC2 explained variance: {100*(ev[0]+ev[1]):.1f}%",
        "",
        "Grey = ChEMBL, green = chromophore train, red = RL molecules.",
        "UMAP/t-SNE use a PCA→50 preprocessing step for speed/stability.",
    ]
    (out / "README.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"[DONE] → {out}")


if __name__ == "__main__":
    main()
