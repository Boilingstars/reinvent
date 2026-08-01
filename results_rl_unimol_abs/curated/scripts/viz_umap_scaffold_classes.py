"""
Joint UMAP chemical space:
  - scaffold classes from chembl_scaffold_classes.csv (density regions)
  - ChEMBL drugs (density)
  - chromophore training set (density)
  - RL-generated molecules (points colored by Score)

Outputs → results_rl_unimol_abs/chemspace_scaffold_classes/

Usage (from repo root):
  .\\.venv_eval\\Scripts\\python.exe results_rl_unimol_abs\\viz_umap_scaffold_classes.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

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

try:
    from umap import UMAP

    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

OUT_DIR = Path(__file__).resolve().parent / "chemspace_scaffold_classes"
DEFAULT_CLASSES = ROOT / "chembl_scaffold_classes_top5_rl.csv"

# High-contrast palette (extended for set1 + set2 + extras)
CONTRAST_PALETTE = [
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
]

# Named overrides for known classes (set1 + set2)
CLASS_COLORS: dict[str, str] = {
    # set1
    "anthraquinone": "#FF0000",
    "carbazole": "#0033FF",
    "chalcone": "#00C853",
    "coumarin": "#AA00FF",
    "flavone": "#FF6D00",
    "indole": "#00B8D4",
    "phenothiazine": "#FFD600",
    "purine": "#6D4C41",
    "quinoline": "#C51162",
    "steroid": "#212121",
    # set2
    "adamantane": "#FF0000",
    "azobenzene": "#0033FF",
    "benzimidazole": "#00C853",
    "benzothiazole": "#AA00FF",
    "hydantoin": "#FF6D00",
    "isoquinoline": "#00B8D4",
    "naphthalimide": "#FFD600",
    "quinazoline": "#6D4C41",
    "stilbene": "#C51162",
    "thiazolidinedione": "#212121",
}
CHEMBL_COLOR = "#90A4AE"  # blue-grey
CHROMO_COLOR = "#1B5E20"  # dark green
REF_LABELS = {"chembl_drugs", "chromophore", "rl"}


def resolve_class_colors(class_names: list[str] | set[str]) -> dict[str, str]:
    """Map class names to contrasting colors (named overrides first, then palette)."""
    names = sorted(class_names)
    used = set()
    out: dict[str, str] = {}
    for name in names:
        if name in CLASS_COLORS and CLASS_COLORS[name] not in used:
            out[name] = CLASS_COLORS[name]
            used.add(CLASS_COLORS[name])
        else:
            for col in CONTRAST_PALETTE:
                if col not in used:
                    out[name] = col
                    used.add(col)
                    break
            else:
                out[name] = f"C{len(out) % 10}"
    return out


def scaffold_class_names(labels: np.ndarray) -> list[str]:
    return sorted({lb for lb in np.unique(labels) if lb not in REF_LABELS})


def kde_field(xy: np.ndarray, grid_n: int = 100, pad: float = 0.08):
    if len(xy) < 12:
        return None
    try:
        kde = gaussian_kde(xy.T, bw_method="scott")
    except Exception:
        return None
    xmin, xmax = float(xy[:, 0].min()), float(xy[:, 0].max())
    ymin, ymax = float(xy[:, 1].min()), float(xy[:, 1].max())
    dx = xmax - xmin + 1e-9
    dy = ymax - ymin + 1e-9
    xs = np.linspace(xmin - pad * dx, xmax + pad * dx, grid_n)
    ys = np.linspace(ymin - pad * dy, ymax + pad * dy, grid_n)
    xx, yy = np.meshgrid(xs, ys)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, zz


def tinted_cmap(hex_color: str, name: str, n: int = 256):
    """Colormap from transparent-ish white to solid class color."""
    from matplotlib.colors import LinearSegmentedColormap

    r, g, b = to_rgb(hex_color)
    colors = [(1.0, 1.0, 1.0, 0.0), (r, g, b, 1.0)]
    return LinearSegmentedColormap.from_list(name, colors, N=n)


def load_scaffold_classes(
    path: Path,
    max_per_class: int | None,
    seed: int,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Return smiles, labels (scaffold_class), and class→ru name map."""
    df = pd.read_csv(path, usecols=["smiles", "scaffold_class", "core_name_ru"])
    df = df.dropna(subset=["smiles", "scaffold_class"])
    ru_map = (
        df.groupby("scaffold_class")["core_name_ru"]
        .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else s.name)
        .to_dict()
    )
    rng = np.random.RandomState(seed)
    parts = []
    for cls, g in df.groupby("scaffold_class"):
        if max_per_class is not None and len(g) > max_per_class:
            g = g.sample(n=max_per_class, random_state=rng)
        parts.append(g)
    sub = pd.concat(parts, ignore_index=True)
    return sub["smiles"].astype(str).tolist(), sub["scaffold_class"].astype(str).tolist(), ru_map


def build_joint_matrix(
    class_smiles: list[str],
    class_labels: list[str],
    chembl: list[str],
    chromo: list[str],
    rl: list[str],
    radius: int,
    n_bits: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """labels: class name / 'chembl_drugs' / 'chromophore' / 'rl'."""
    blocks: list[np.ndarray] = []
    labels: list[str] = []
    smiles_out: list[str] = []

    # scaffold classes — keep class label even after unique FP (drop dups within class)
    by_class: dict[str, list[str]] = {}
    for smi, lb in zip(class_smiles, class_labels):
        by_class.setdefault(lb, []).append(smi)

    for cls in sorted(by_class):
        valid, fps, n_inv = smiles_to_fps(by_class[cls], radius=radius, n_bits=n_bits, unique=True)
        print(f"  class {cls}: {len(valid)} FP (invalid≈{n_inv})")
        if not fps:
            continue
        blocks.append(fps_to_numpy(fps))
        labels.extend([cls] * len(fps))
        smiles_out.extend(valid)

    for name, smis in (("chembl_drugs", chembl), ("chromophore", chromo), ("rl", rl)):
        valid, fps, n_inv = smiles_to_fps(smis, radius=radius, n_bits=n_bits, unique=True)
        print(f"  {name}: {len(valid)} FP (invalid≈{n_inv})")
        if not fps:
            continue
        blocks.append(fps_to_numpy(fps))
        labels.extend([name] * len(fps))
        smiles_out.extend(valid)

    return np.vstack(blocks), np.asarray(labels), smiles_out


def fit_umap(X: np.ndarray, seed: int, pca_dim: int = 50) -> tuple[np.ndarray, str]:
    if not HAVE_UMAP:
        raise RuntimeError("umap-learn is required")
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


def plot_density_umap(
    Z: np.ndarray,
    labels: np.ndarray,
    rl_scores: np.ndarray,
    ru_map: dict[str, str],
    title: str,
    out_path: Path,
    levels: int = 6,
    class_colors: dict[str, str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 9))

    # Global limits
    pad = 0.04
    xmin, xmax = Z[:, 0].min(), Z[:, 0].max()
    ymin, ymax = Z[:, 1].min(), Z[:, 1].max()
    dx, dy = xmax - xmin, ymax - ymin

    class_names = scaffold_class_names(labels)
    colors = class_colors or resolve_class_colors(class_names)
    legend_patches: list[Patch] = []

    for cls in class_names:
        m = labels == cls
        grid = kde_field(Z[m])
        if grid is None:
            continue
        xx, yy, zz = grid
        col = colors.get(cls, "#333333")
        cmap = tinted_cmap(col, f"cmap_{cls}")
        ax.contourf(xx, yy, zz, levels=levels, cmap=cmap, alpha=0.55)
        ax.contour(xx, yy, zz, levels=levels, colors=[col], linewidths=1.1, alpha=0.85)
        ru = ru_map.get(cls, cls)
        legend_patches.append(
            Patch(facecolor=col, edgecolor=col, alpha=0.55, label=f"{cls} ({ru})")
        )

    for name, color, nice in (
        ("chembl_drugs", CHEMBL_COLOR, "ChEMBL drugs"),
        ("chromophore", CHROMO_COLOR, "Chromophores (train)"),
    ):
        m = labels == name
        if not m.any():
            continue
        grid = kde_field(Z[m])
        if grid is None:
            continue
        xx, yy, zz = grid
        cmap = tinted_cmap(color, f"cmap_{name}")
        ax.contourf(xx, yy, zz, levels=levels, cmap=cmap, alpha=0.40)
        ax.contour(xx, yy, zz, levels=levels, colors=[color], linewidths=1.0, alpha=0.7)
        legend_patches.append(Patch(facecolor=color, edgecolor=color, alpha=0.5, label=nice))

    # RL points
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
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("RL Score")

    ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
    ax.set_ylim(ymin - pad * dy, ymax + pad * dy)
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    # Legend outside
    ax.legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(1.12, 1.0),
        fontsize=8,
        frameon=True,
        title="Density regions",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def plot_overview_scatter(
    Z: np.ndarray,
    labels: np.ndarray,
    rl_scores: np.ndarray,
    ru_map: dict[str, str],
    title: str,
    out_path: Path,
    class_colors: dict[str, str] | None = None,
) -> None:
    """Point overview (no KDE) for sanity check."""
    fig, ax = plt.subplots(figsize=(11, 8))
    class_names = scaffold_class_names(labels)
    colors = class_colors or resolve_class_colors(class_names)
    for cls in class_names:
        m = labels == cls
        ax.scatter(
            Z[m, 0],
            Z[m, 1],
            c=colors.get(cls, "#333"),
            s=6,
            alpha=0.25,
            linewidths=0,
            label=cls,
            zorder=1,
        )
    for name, color, nice, s, a, z in (
        ("chembl_drugs", CHEMBL_COLOR, "ChEMBL drugs", 8, 0.3, 2),
        ("chromophore", CHROMO_COLOR, "Chromophores", 10, 0.35, 3),
    ):
        m = labels == name
        if m.any():
            ax.scatter(Z[m, 0], Z[m, 1], c=color, s=s, alpha=a, linewidths=0, label=nice, zorder=z)

    rl_m = labels == "rl"
    sc = ax.scatter(
        Z[rl_m, 0],
        Z[rl_m, 1],
        c=rl_scores,
        cmap="plasma",
        s=18,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.2,
        zorder=5,
    )
    fig.colorbar(sc, ax=ax, label="RL Score")
    ax.set_title(title)
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.legend(fontsize=7, loc="best", markerscale=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def plot_classes_only_density(
    Z: np.ndarray,
    labels: np.ndarray,
    ru_map: dict[str, str],
    title: str,
    out_path: Path,
    class_colors: dict[str, str] | None = None,
) -> None:
    """Density of scaffold classes only (no RL) + chromophore/chembl outlines."""
    fig, ax = plt.subplots(figsize=(12, 9))
    class_names = scaffold_class_names(labels)
    colors = class_colors or resolve_class_colors(class_names)
    patches = []
    for cls in class_names:
        m = labels == cls
        grid = kde_field(Z[m])
        if grid is None:
            continue
        xx, yy, zz = grid
        col = colors.get(cls, "#333")
        cmap = tinted_cmap(col, f"c_{cls}")
        ax.contourf(xx, yy, zz, levels=6, cmap=cmap, alpha=0.5)
        ax.contour(xx, yy, zz, levels=6, colors=[col], linewidths=0.8, alpha=0.6)
        patches.append(
            Patch(
                facecolor=col,
                edgecolor=col,
                alpha=0.55,
                label=f"{cls} ({ru_map.get(cls, cls)})",
            )
        )
    for name, color, nice in (
        ("chembl_drugs", CHEMBL_COLOR, "ChEMBL drugs"),
        ("chromophore", CHROMO_COLOR, "Chromophores (train)"),
    ):
        m = labels == name
        if not m.any():
            continue
        grid = kde_field(Z[m])
        if grid is None:
            continue
        xx, yy, zz = grid
        ax.contour(xx, yy, zz, levels=5, colors=[color], linewidths=1.6, alpha=0.9)
        patches.append(Patch(facecolor="none", edgecolor=color, linewidth=2, label=nice + " (contour)"))

    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def main() -> None:
    p = argparse.ArgumentParser(description="UMAP: scaffold classes + ChEMBL + chromophores + RL")
    p.add_argument("--classes-csv", type=Path, default=DEFAULT_CLASSES)
    p.add_argument("--rl-csv", type=Path, default=DEFAULT_RL_CSV)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--chembl", type=Path, default=DEFAULT_CHEMBL)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--max-per-class", type=int, default=800, help="Subsample per scaffold class")
    p.add_argument("--max-ref", type=int, default=3500, help="Max chromophore SMILES")
    p.add_argument("--max-chembl", type=int, default=None)
    p.add_argument("--max-gen", type=int, default=None)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pca-prep", type=int, default=50)
    p.add_argument(
        "--replot-only",
        action="store_true",
        help="Redraw plots from existing coordinates_umap.csv (no UMAP recompute)",
    )
    args = p.parse_args()

    out = ensure_dir(args.out)

    # Russian names for legend
    ru_map: dict[str, str] = {}
    if args.classes_csv.is_file():
        tmp = pd.read_csv(args.classes_csv, usecols=["scaffold_class", "core_name_ru"])
        ru_map = (
            tmp.groupby("scaffold_class")["core_name_ru"]
            .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else s.name)
            .to_dict()
        )

    if args.replot_only:
        coords_path = out / "coordinates_umap.csv"
        if not coords_path.is_file():
            raise SystemExit(f"Missing {coords_path}; run without --replot-only first")
        coords = pd.read_csv(coords_path)
        Z = coords[["umap1", "umap2"]].to_numpy(dtype=float)
        labels = coords["label"].to_numpy()
        rl_scores = coords.loc[coords["label"] == "rl", "score"].to_numpy(dtype=float)
        tag = f"UMAP (replot, n={len(coords)})"
        print(f"[INFO] Replotting from {coords_path} ({len(coords)} points)")
    else:
        if not HAVE_UMAP:
            raise SystemExit("Install umap-learn first")

        print("[INFO] Loading data…")
        class_smi, class_lb, ru_map = load_scaffold_classes(
            args.classes_csv, max_per_class=args.max_per_class, seed=args.seed
        )
        chembl = read_smiles(args.chembl, max_n=args.max_chembl, seed=args.seed)
        chromo = read_smiles(args.train, max_n=args.max_ref, seed=args.seed)
        rl_smi, rl_scores_all = load_rl_smiles(args.rl_csv, max_n=args.max_gen, seed=args.seed)
        print(
            f"  classes={len(class_smi)}, chembl={len(chembl)}, "
            f"chromo={len(chromo)}, rl={len(rl_smi)}"
        )

        print("[INFO] Fingerprints…")
        X, labels, smiles = build_joint_matrix(
            class_smi, class_lb, chembl, chromo, rl_smi, args.radius, args.n_bits
        )
        print(f"  matrix {X.shape}")

        print("[INFO] UMAP…")
        Z, tag = fit_umap(X, args.seed, pca_dim=args.pca_prep)

        score_map = dict(zip(rl_smi, rl_scores_all))
        rl_scores = np.array(
            [score_map.get(s, np.nan) for s, lb in zip(smiles, labels) if lb == "rl"]
        )

        coords = pd.DataFrame(
            {"smiles": smiles, "label": labels, "umap1": Z[:, 0], "umap2": Z[:, 1]}
        )
        coords.loc[coords["label"] == "rl", "score"] = rl_scores
        coords.to_csv(out / "coordinates_umap.csv", index=False)

        counts = coords["label"].value_counts().rename_axis("label").reset_index(name="n")
        counts.to_csv(out / "set_counts.csv", index=False)

    plot_density_umap(
        Z,
        labels,
        rl_scores,
        ru_map,
        f"Scaffold-class densities + refs; RL points by Score\n{tag}",
        out / "01_umap_density_classes_rl_score.png",
    )
    plot_overview_scatter(
        Z,
        labels,
        rl_scores,
        ru_map,
        f"UMAP overview (points)\n{tag}",
        out / "02_umap_scatter_overview.png",
    )
    plot_classes_only_density(
        Z,
        labels,
        ru_map,
        f"Scaffold-class densities (+ ChEMBL/chromophore contours)\n{tag}",
        out / "03_umap_density_classes_only.png",
    )

    summary = [
        "# UMAP: scaffold classes + ChEMBL + chromophores + RL",
        "",
        f"- Embedding: {tag}",
        f"- Scaffold classes: " + ", ".join(scaffold_class_names(labels)),
        f"- ChEMBL drugs: {int((labels == 'chembl_drugs').sum())}",
        f"- Chromophores: {int((labels == 'chromophore').sum())}",
        f"- RL generated: {int((labels == 'rl').sum())}",
        "",
        "Density regions = KDE of each scaffold class / ChEMBL / chromophores.",
        "RL molecules shown as points colored by Score (colorbar).",
        "",
        "Files:",
        "- `01_umap_density_classes_rl_score.png` — main figure",
        "- `02_umap_scatter_overview.png` — all sets as points",
        "- `03_umap_density_classes_only.png` — densities without RL points",
        "- `coordinates_umap.csv`, `set_counts.csv`",
        "- `class_proximity/` — Tanimoto / UMAP proximity to scaffold classes",
    ]
    (out / "README.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"[DONE] → {out}")


if __name__ == "__main__":
    main()
