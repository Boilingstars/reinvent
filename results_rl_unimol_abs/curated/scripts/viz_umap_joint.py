"""
Single joint UMAP so RL points share the same coordinates across plots.

Includes:
  - molecules.csv (scaffold_id densities)
  - chembl_scaffold_classes_top5_rl.csv (top-5 densities)
  - ChEMBL drugs
  - chromophore train
  - RL from rl_unimol_abs_1.csv (points by Score)

Writes comparable figures to:
  results_rl_unimol_abs/chemspace_joint/
  and refreshes key plots in chemspace_scaffold_classes/ and chemspace_molecules/

Usage:
  .\\.venv_eval\\Scripts\\python.exe results_rl_unimol_abs\\viz_umap_joint.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.decomposition import PCA

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
from viz_chemspace_compare import DEFAULT_RL_CSV, load_rl_smiles  # noqa: E402
from viz_umap_scaffold_classes import (  # noqa: E402
    CHEMBL_COLOR,
    CHROMO_COLOR,
    kde_field,
    resolve_class_colors,
    tinted_cmap,
)

try:
    from umap import UMAP

    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

OUT = Path(__file__).resolve().parent / "chemspace_joint"
MOL_CSV = ROOT / "molecules.csv"
TOP5_CSV = ROOT / "chembl_scaffold_classes_top5_rl.csv"
TOP5 = ["stilbene", "benzothiazole", "chalcone", "azobenzene", "thiazolidinedione"]

MOL_CONTRAST = [
    "#FF0000",
    "#0033FF",
    "#00C853",
    "#AA00FF",
    "#FF6D00",
    "#00B8D4",
    "#FFD600",
    "#6D4C41",
    "#C51162",
    "#212121",
    "#1A237E",
    "#FF1744",
    "#76FF03",
    "#F50057",
    "#00E5FF",
    "#FFAB00",
    "#6200EA",
    "#64DD17",
    "#D50000",
    "#304FFE",
    "#AEEA00",
    "#D500F9",
    "#00BFA5",
    "#FF6F00",
    "#2979FF",
    "#C6FF00",
    "#AA00FF",
    "#00E676",
    "#FF3D00",
]


def fit_umap(X: np.ndarray, seed: int, pca_dim: int = 50) -> tuple[np.ndarray, str]:
    n = X.shape[0]
    Xp = X
    if X.shape[1] > pca_dim and n > pca_dim + 1:
        Xp = PCA(n_components=min(pca_dim, n - 1), random_state=seed).fit_transform(X)
    reducer = UMAP(
        n_components=2,
        n_neighbors=min(40, max(10, n // 40)),
        min_dist=0.12,
        metric="euclidean",
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = reducer.fit_transform(Xp)
    return Z, f"UMAP joint (PCA→{Xp.shape[1]}, n={n})"


def add_group(
    blocks: list,
    labels: list,
    smiles_out: list,
    sources: list,
    smis: list[str],
    label: str,
    source: str,
    radius: int,
    n_bits: int,
) -> None:
    valid, fps, n_inv = smiles_to_fps(smis, radius=radius, n_bits=n_bits, unique=True)
    print(f"  {source}/{label}: {len(valid)} FP (invalid≈{n_inv})")
    if not fps:
        return
    blocks.append(fps_to_numpy(fps))
    labels.extend([label] * len(fps))
    smiles_out.extend(valid)
    sources.extend([source] * len(fps))


def plot_top5_style(
    Z: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    rl_scores: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Densities: top5 + ChEMBL + chromophore; RL points by Score."""
    colors = resolve_class_colors(TOP5)
    fig, ax = plt.subplots(figsize=(12, 9))
    patches = []

    for cls in TOP5:
        m = (labels == cls) & (sources == "top5")
        if not m.any():
            continue
        grid = kde_field(Z[m])
        if grid is None:
            continue
        xx, yy, zz = grid
        ax.contourf(xx, yy, zz, levels=6, cmap=tinted_cmap(colors[cls], cls), alpha=0.5)
        ax.contour(xx, yy, zz, levels=6, colors=[colors[cls]], linewidths=1.1, alpha=0.85)
        patches.append(Patch(facecolor=colors[cls], edgecolor=colors[cls], alpha=0.55, label=cls))

    for name, color, nice, src in (
        ("chembl_drugs", CHEMBL_COLOR, "ChEMBL drugs", "chembl"),
        ("chromophore", CHROMO_COLOR, "Chromophores (train)", "chromophore"),
    ):
        m = (labels == name) & (sources == src)
        if not m.any():
            continue
        grid = kde_field(Z[m])
        if grid is None:
            continue
        xx, yy, zz = grid
        ax.contourf(xx, yy, zz, levels=6, cmap=tinted_cmap(color, name), alpha=0.35)
        ax.contour(xx, yy, zz, levels=6, colors=[color], linewidths=1.2, alpha=0.8)
        patches.append(Patch(facecolor=color, edgecolor=color, alpha=0.45, label=nice))

    rl_m = labels == "rl"
    sc = ax.scatter(
        Z[rl_m, 0],
        Z[rl_m, 1],
        c=rl_scores,
        cmap="plasma",
        s=22,
        alpha=0.88,
        edgecolors="k",
        linewidths=0.25,
        zorder=10,
    )
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label="RL Score")
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.12, 1.0), fontsize=8, title="Density regions")
    ax.set_title(title)
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def plot_molecules_style(
    Z: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    name_map: dict[str, str],
    rl_scores: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Densities: molecules.csv scaffold_id; RL points by Score (same Z)."""
    groups = sorted({lb for lb, src in zip(labels, sources) if src == "molecules"})
    colors = {g: MOL_CONTRAST[i % len(MOL_CONTRAST)] for i, g in enumerate(groups)}
    fig, ax = plt.subplots(figsize=(13, 9))
    patches = []
    for g in groups:
        m = (labels == g) & (sources == "molecules")
        if m.sum() < 12:
            ax.scatter(Z[m, 0], Z[m, 1], c=colors[g], s=16, alpha=0.35, linewidths=0, zorder=2)
        else:
            grid = kde_field(Z[m], grid_n=90)
            if grid is None:
                continue
            xx, yy, zz = grid
            ax.contourf(xx, yy, zz, levels=5, cmap=tinted_cmap(colors[g], g), alpha=0.45)
            ax.contour(xx, yy, zz, levels=5, colors=[colors[g]], linewidths=1.1, alpha=0.85)
        nice = name_map.get(g, g)
        patches.append(Patch(facecolor=colors[g], edgecolor=colors[g], alpha=0.55, label=f"{g} ({nice})"))

    rl_m = labels == "rl"
    sc = ax.scatter(
        Z[rl_m, 0],
        Z[rl_m, 1],
        c=rl_scores,
        cmap="plasma",
        s=22,
        alpha=0.88,
        edgecolors="k",
        linewidths=0.25,
        zorder=10,
    )
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label="RL Score")
    ax.legend(
        handles=patches,
        loc="upper left",
        bbox_to_anchor=(1.12, 1.0),
        fontsize=7,
        title="molecules.csv scaffolds",
    )
    ax.set_title(title)
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def plot_side_by_side_rl_only(
    Z: np.ndarray,
    labels: np.ndarray,
    rl_scores: np.ndarray,
    out_path: Path,
) -> None:
    """Sanity: only RL on shared axes (same positions)."""
    rl_m = labels == "rl"
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        Z[rl_m, 0],
        Z[rl_m, 1],
        c=rl_scores,
        cmap="plasma",
        s=22,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.2,
    )
    fig.colorbar(sc, ax=ax, label="RL Score")
    ax.set_title("RL only (joint UMAP coordinates)")
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--molecules", type=Path, default=MOL_CSV)
    p.add_argument("--top5", type=Path, default=TOP5_CSV)
    p.add_argument("--rl-csv", type=Path, default=DEFAULT_RL_CSV)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--chembl", type=Path, default=DEFAULT_CHEMBL)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--max-per-top5", type=int, default=800)
    p.add_argument("--max-ref", type=int, default=3500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--pca-prep", type=int, default=50)
    args = p.parse_args()

    if not HAVE_UMAP:
        raise SystemExit("umap-learn required")

    out = ensure_dir(args.out)
    blocks: list[np.ndarray] = []
    labels: list[str] = []
    smiles_out: list[str] = []
    sources: list[str] = []

    print("[INFO] molecules.csv…")
    mol_df = pd.read_csv(args.molecules)
    name_map = (
        mol_df.groupby("scaffold_id")["scaffold_name"]
        .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else s.name)
        .to_dict()
        if "scaffold_name" in mol_df.columns
        else {}
    )
    for g, sub in mol_df.groupby("scaffold_id"):
        add_group(
            blocks,
            labels,
            smiles_out,
            sources,
            sub["smiles"].astype(str).tolist(),
            str(g),
            "molecules",
            args.radius,
            args.n_bits,
        )

    print("[INFO] top5 scaffold classes…")
    top5_df = pd.read_csv(args.top5, usecols=["smiles", "scaffold_class"])
    rng = np.random.RandomState(args.seed)
    for cls in TOP5:
        sub = top5_df[top5_df["scaffold_class"] == cls]
        smis = sub["smiles"].astype(str).tolist()
        if len(smis) > args.max_per_top5:
            idx = rng.choice(len(smis), size=args.max_per_top5, replace=False)
            smis = [smis[i] for i in idx]
        add_group(blocks, labels, smiles_out, sources, smis, cls, "top5", args.radius, args.n_bits)

    print("[INFO] ChEMBL + chromophores + RL…")
    chembl = read_smiles(args.chembl, max_n=None, seed=args.seed)
    chromo = read_smiles(args.train, max_n=args.max_ref, seed=args.seed)
    add_group(blocks, labels, smiles_out, sources, chembl, "chembl_drugs", "chembl", args.radius, args.n_bits)
    add_group(blocks, labels, smiles_out, sources, chromo, "chromophore", "chromophore", args.radius, args.n_bits)

    rl_smi, rl_scores_all = load_rl_smiles(args.rl_csv, max_n=None, seed=args.seed)
    score_map = dict(zip(rl_smi, rl_scores_all))
    valid_rl, fps_rl, n_inv = smiles_to_fps(rl_smi, radius=args.radius, n_bits=args.n_bits, unique=True)
    print(f"  rl: {len(valid_rl)} FP (invalid≈{n_inv})")
    blocks.append(fps_to_numpy(fps_rl))
    labels.extend(["rl"] * len(valid_rl))
    smiles_out.extend(valid_rl)
    sources.extend(["rl"] * len(valid_rl))
    rl_scores = np.array([score_map.get(s, np.nan) for s in valid_rl], dtype=float)

    X = np.vstack(blocks)
    labels_a = np.asarray(labels)
    sources_a = np.asarray(sources)
    print(f"[INFO] matrix {X.shape}")
    print("[INFO] Fitting joint UMAP…")
    Z, tag = fit_umap(X, args.seed, pca_dim=args.pca_prep)

    coords = pd.DataFrame(
        {
            "smiles": smiles_out,
            "label": labels_a,
            "source": sources_a,
            "umap1": Z[:, 0],
            "umap2": Z[:, 1],
        }
    )
    coords.loc[coords["label"] == "rl", "score"] = rl_scores
    coords.to_csv(out / "coordinates_umap_joint.csv", index=False)

    # Joint folder plots
    plot_top5_style(
        Z,
        labels_a,
        sources_a,
        rl_scores,
        f"Top-5 + refs densities; RL by Score\n{tag}",
        out / "01_joint_top5_densities_rl_score.png",
    )
    plot_molecules_style(
        Z,
        labels_a,
        sources_a,
        {str(k): str(v) for k, v in name_map.items()},
        rl_scores,
        f"molecules.csv densities; RL by Score (same UMAP)\n{tag}",
        out / "02_joint_molecules_densities_rl_score.png",
    )
    plot_side_by_side_rl_only(Z, labels_a, rl_scores, out / "00_rl_only_joint_coords.png")

    # Refresh the two folders the user compares
    sc_dir = ensure_dir(Path(__file__).resolve().parent / "chemspace_scaffold_classes")
    mol_dir = ensure_dir(Path(__file__).resolve().parent / "chemspace_molecules")
    plot_top5_style(
        Z,
        labels_a,
        sources_a,
        rl_scores,
        f"Scaffold-class densities + refs; RL points by Score\n{tag}",
        sc_dir / "01_umap_density_classes_rl_score.png",
    )
    plot_molecules_style(
        Z,
        labels_a,
        sources_a,
        {str(k): str(v) for k, v in name_map.items()},
        rl_scores,
        f"molecules.csv scaffold densities + RL by Score\n{tag}",
        mol_dir / "01_umap_scaffold_densities_rl_score.png",
    )
    # also copy joint coords into both folders for reference
    coords.to_csv(sc_dir / "coordinates_umap_joint.csv", index=False)
    coords.to_csv(mol_dir / "coordinates_umap_joint.csv", index=False)

    (out / "README.md").write_text(
        "\n".join(
            [
                "# Joint UMAP (comparable RL positions)",
                "",
                "One embedding fitted on: molecules.csv + top5 classes + ChEMBL + chromophores + RL.",
                "Plots `01` and `02` use **identical** RL coordinates.",
                "",
                f"- {tag}",
                "- `00_rl_only_joint_coords.png` — RL alone",
                "- `01_joint_top5_densities_rl_score.png`",
                "- `02_joint_molecules_densities_rl_score.png`",
                "",
                "Also refreshed:",
                "- `chemspace_scaffold_classes/01_umap_density_classes_rl_score.png`",
                "- `chemspace_molecules/01_umap_scaffold_densities_rl_score.png`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] → {out}")


if __name__ == "__main__":
    main()
