"""
UMAP chemical space for molecules.csv (scaffold density regions)
+ RL molecules from rl_unimol_abs_1.csv as Score-colored points.

Outputs → results_rl_unimol_abs/chemspace_molecules/

Usage:
  .\\.venv_eval\\Scripts\\python.exe results_rl_unimol_abs\\viz_umap_molecules_csv.py
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
from utils import ensure_dir, fps_to_numpy, smiles_to_fps  # noqa: E402
from viz_chemspace_compare import DEFAULT_RL_CSV, load_rl_smiles  # noqa: E402
from viz_umap_scaffold_classes import kde_field, tinted_cmap  # noqa: E402

try:
    from umap import UMAP

    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

OUT_DIR = Path(__file__).resolve().parent / "chemspace_molecules"
DEFAULT_MOL = ROOT / "molecules.csv"

# High-contrast cycling palette (29 scaffolds)
CONTRAST = [
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
    "#F50057",
    "#00E676",
    "#FF3D00",
]


def colors_for(names: list[str]) -> dict[str, str]:
    return {n: CONTRAST[i % len(CONTRAST)] for i, n in enumerate(sorted(names))}


def fit_umap(X: np.ndarray, seed: int, pca_dim: int = 50) -> tuple[np.ndarray, str]:
    if not HAVE_UMAP:
        raise RuntimeError("umap-learn required")
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
    return Z, f"UMAP (PCA→{Xp.shape[1]}, n={n})"


def plot_density_rl(
    Z: np.ndarray,
    labels: np.ndarray,
    group_names: list[str],
    name_map: dict[str, str],
    rl_scores: np.ndarray,
    title: str,
    out_path: Path,
    colors: dict[str, str],
) -> None:
    fig, ax = plt.subplots(figsize=(13, 9))
    pad = 0.04
    xmin, xmax = Z[:, 0].min(), Z[:, 0].max()
    ymin, ymax = Z[:, 1].min(), Z[:, 1].max()
    dx, dy = xmax - xmin, ymax - ymin

    legend_patches: list[Patch] = []
    for g in group_names:
        m = labels == g
        if m.sum() < 12:
            # too few for KDE — scatter faint
            ax.scatter(
                Z[m, 0],
                Z[m, 1],
                c=colors[g],
                s=18,
                alpha=0.35,
                linewidths=0,
                zorder=2,
            )
        else:
            grid = kde_field(Z[m], grid_n=90)
            if grid is None:
                continue
            xx, yy, zz = grid
            col = colors[g]
            ax.contourf(xx, yy, zz, levels=5, cmap=tinted_cmap(col, f"c_{g}"), alpha=0.5)
            ax.contour(xx, yy, zz, levels=5, colors=[col], linewidths=1.2, alpha=0.9)
        nice = name_map.get(g, g)
        legend_patches.append(
            Patch(facecolor=colors[g], edgecolor=colors[g], alpha=0.55, label=f"{g} ({nice})")
        )

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
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("RL Score")

    ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
    ax.set_ylim(ymin - pad * dy, ymax + pad * dy)
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(1.14, 1.0),
        fontsize=7,
        frameon=True,
        title="molecules.csv scaffolds",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def plot_by_scaffold_class(
    Z: np.ndarray,
    labels_fine: np.ndarray,
    class_labels: np.ndarray,
    rl_scores: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Broader scaffold_class densities (5 groups) + RL points."""
    class_colors = {
        "chromophore": "#FF0000",
        "heterocycle": "#0033FF",
        "drug_class": "#00C853",
        "scaffold": "#AA00FF",
        "natural_product": "#FF6D00",
    }
    fig, ax = plt.subplots(figsize=(11, 8))
    patches = []
    for g, col in class_colors.items():
        m = class_labels == g
        if not m.any():
            continue
        grid = kde_field(Z[m], grid_n=100)
        if grid is None:
            continue
        xx, yy, zz = grid
        ax.contourf(xx, yy, zz, levels=6, cmap=tinted_cmap(col, f"cls_{g}"), alpha=0.45)
        ax.contour(xx, yy, zz, levels=6, colors=[col], linewidths=1.4, alpha=0.9)
        patches.append(Patch(facecolor=col, edgecolor=col, alpha=0.5, label=g))

    rl_m = labels_fine == "rl"
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
    fig.colorbar(sc, ax=ax, label="RL Score")
    ax.legend(handles=patches, loc="best", title="scaffold_class")
    ax.set_title(title)
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--molecules", type=Path, default=DEFAULT_MOL)
    p.add_argument("--rl-csv", type=Path, default=DEFAULT_RL_CSV)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--group-col", type=str, default="scaffold_id", help="Column for density groups")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pca-prep", type=int, default=50)
    args = p.parse_args()

    if not HAVE_UMAP:
        raise SystemExit("Install umap-learn")

    out = ensure_dir(args.out)
    print("[INFO] Loading molecules.csv…")
    mol_df = pd.read_csv(args.molecules)
    if "smiles" not in mol_df.columns:
        raise SystemExit("molecules.csv needs smiles column")
    group_col = args.group_col
    if group_col not in mol_df.columns:
        raise SystemExit(f"Missing column {group_col}")

    name_map = {}
    if "scaffold_name" in mol_df.columns:
        name_map = (
            mol_df.groupby(group_col)["scaffold_name"]
            .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else s.name)
            .to_dict()
        )

    # fingerprints per group
    blocks = []
    labels = []
    smiles_out = []
    class_labels_list = []  # parallel for scaffold_class of molecules only

    groups = sorted(mol_df[group_col].dropna().unique().tolist())
    print(f"  groups ({group_col}): {len(groups)}")
    for g in groups:
        sub = mol_df[mol_df[group_col] == g]
        smis = sub["smiles"].astype(str).tolist()
        valid, fps, n_inv = smiles_to_fps(smis, radius=args.radius, n_bits=args.n_bits, unique=True)
        print(f"  {g}: {len(valid)} FP (invalid≈{n_inv})")
        if not fps:
            continue
        blocks.append(fps_to_numpy(fps))
        labels.extend([str(g)] * len(fps))
        smiles_out.extend(valid)
        # map class for valid smiles order — approximate via first match
        if "scaffold_class" in sub.columns:
            # rebuild class by canonical match
            class_by_smi = {}
            from rdkit import Chem

            for smi, cls in zip(sub["smiles"], sub["scaffold_class"]):
                mol = Chem.MolFromSmiles(str(smi))
                if mol is None:
                    continue
                class_by_smi[Chem.MolToSmiles(mol, canonical=True)] = str(cls)
            for s in valid:
                class_labels_list.append(class_by_smi.get(s, "unknown"))
        else:
            class_labels_list.extend(["unknown"] * len(valid))

    print("[INFO] Loading RL…")
    rl_smi, rl_scores_all = load_rl_smiles(args.rl_csv, max_n=None, seed=args.seed)
    valid_rl, fps_rl, n_inv = smiles_to_fps(rl_smi, radius=args.radius, n_bits=args.n_bits, unique=True)
    print(f"  rl: {len(valid_rl)} FP (invalid≈{n_inv})")
    score_map = dict(zip(rl_smi, rl_scores_all))
    rl_scores = np.array([score_map.get(s, np.nan) for s in valid_rl], dtype=float)
    blocks.append(fps_to_numpy(fps_rl))
    labels.extend(["rl"] * len(fps_rl))
    smiles_out.extend(valid_rl)
    class_labels_list.extend(["rl"] * len(fps_rl))

    X = np.vstack(blocks)
    labels_arr = np.asarray(labels)
    print(f"  matrix {X.shape}")

    print("[INFO] UMAP…")
    Z, tag = fit_umap(X, args.seed, pca_dim=args.pca_prep)

    coords = pd.DataFrame(
        {
            "smiles": smiles_out,
            "label": labels_arr,
            "umap1": Z[:, 0],
            "umap2": Z[:, 1],
            "scaffold_class": class_labels_list,
        }
    )
    coords.loc[coords["label"] == "rl", "score"] = rl_scores
    coords.to_csv(out / "coordinates_umap.csv", index=False)

    group_names = [g for g in groups if (labels_arr == str(g)).any()]
    colors = colors_for([str(g) for g in group_names])

    plot_density_rl(
        Z,
        labels_arr,
        [str(g) for g in group_names],
        {str(k): str(v) for k, v in name_map.items()},
        rl_scores,
        f"molecules.csv scaffold densities + RL by Score\n{tag}",
        out / "01_umap_scaffold_densities_rl_score.png",
        colors,
    )

    # broader class view (same embedding)
    class_arr = np.asarray(class_labels_list)
    # for class density, only non-rl molecule points; Z same
    plot_by_scaffold_class(
        Z,
        labels_arr,
        class_arr,
        rl_scores,
        f"molecules.csv scaffold_class densities + RL by Score\n{tag}",
        out / "02_umap_scaffold_class_densities_rl_score.png",
    )

    counts = coords["label"].value_counts().rename_axis("label").reset_index(name="n")
    counts.to_csv(out / "set_counts.csv", index=False)

    (out / "README.md").write_text(
        "\n".join(
            [
                "# UMAP: molecules.csv + RL",
                "",
                f"- Source molecules: `{args.molecules.name}` grouped by `{group_col}`",
                f"- RL: `{args.rl_csv.name}` as points colored by Score",
                f"- Embedding: {tag}",
                "",
                "- `01_…` — density per scaffold_id (contrasting colors) + RL",
                "- `02_…` — density per scaffold_class (5 broad groups) + RL",
                "- `coordinates_umap.csv`, `set_counts.csv`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] → {out}")


if __name__ == "__main__":
    main()
