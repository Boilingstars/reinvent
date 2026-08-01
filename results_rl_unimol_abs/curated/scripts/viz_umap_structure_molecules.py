"""
UMAP for molecules.csv scaffolds — same style as previous chemspace plots.

Uses joint UMAP coordinates (chemspace_joint/) so RL positions match other figures.
Saves into structure_interpretation_molecules/.

Usage:
  .\\.venv_eval\\Scripts\\python.exe results_rl_unimol_abs\\viz_umap_structure_molecules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "results_rl_unimol_abs"))
from viz_umap_scaffold_classes import kde_field, tinted_cmap  # noqa: E402

JOINT = Path(__file__).resolve().parent / "chemspace_joint" / "coordinates_umap_joint.csv"
ASSIGN = (
    Path(__file__).resolve().parent
    / "structure_interpretation_molecules"
    / "rl_nearest_molecules_scaffold.csv"
)
MOL_CSV = ROOT / "molecules.csv"
OUT = Path(__file__).resolve().parent / "structure_interpretation_molecules"

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

CLASS_COLORS = {
    "chromophore": "#C51162",
    "heterocycle": "#0033FF",
    "natural_product": "#FF6D00",
    "scaffold": "#AA00FF",
    "drug_class": "#00C853",
}


def set_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#fafafa",
            "figure.facecolor": "white",
        }
    )


def plot_scaffold_densities_rl_score(coords: pd.DataFrame, name_map: dict, out_path: Path) -> None:
    mol = coords[coords["source"] == "molecules"].copy()
    rl = coords[coords["label"] == "rl"].copy()
    groups = sorted(mol["label"].unique())
    colors = {g: MOL_CONTRAST[i % len(MOL_CONTRAST)] for i, g in enumerate(groups)}

    fig, ax = plt.subplots(figsize=(13, 9))
    patches = []
    for g in groups:
        sub = mol[mol["label"] == g]
        xy = sub[["umap1", "umap2"]].to_numpy()
        if len(xy) < 12:
            ax.scatter(xy[:, 0], xy[:, 1], c=colors[g], s=16, alpha=0.35, linewidths=0, zorder=2)
        else:
            grid = kde_field(xy, grid_n=90)
            if grid is None:
                continue
            xx, yy, zz = grid
            ax.contourf(xx, yy, zz, levels=5, cmap=tinted_cmap(colors[g], g), alpha=0.45)
            ax.contour(xx, yy, zz, levels=5, colors=[colors[g]], linewidths=1.1, alpha=0.85)
        nice = name_map.get(g, g)
        patches.append(Patch(facecolor=colors[g], edgecolor=colors[g], alpha=0.55, label=f"{g} ({nice})"))

    # faint ChEMBL / chromophore context
    for src, col, lab in (
        ("chembl", "#90a4ae", "ChEMBL"),
        ("chromophore", "#66bb6a", "Chromophores"),
    ):
        bg = coords[coords["source"] == src]
        if len(bg) < 30:
            continue
        ax.scatter(bg["umap1"], bg["umap2"], c=col, s=4, alpha=0.12, linewidths=0, zorder=1, label="_nolegend_")

    sc = ax.scatter(
        rl["umap1"],
        rl["umap2"],
        c=rl["score"].fillna(0),
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
    ax.set_title("molecules.csv scaffold densities + RL by Score\n(joint UMAP coordinates)")
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def plot_scaffold_class_densities(coords: pd.DataFrame, mol_df: pd.DataFrame, out_path: Path) -> None:
    mol = coords[coords["source"] == "molecules"].copy()
    cls_map = (
        mol_df.groupby("scaffold_id")["scaffold_class"]
        .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else "unknown")
        .to_dict()
    )
    mol["scaffold_class"] = mol["label"].map(cls_map)
    rl = coords[coords["label"] == "rl"].copy()

    fig, ax = plt.subplots(figsize=(11, 8.5))
    patches = []
    for cls, col in CLASS_COLORS.items():
        sub = mol[mol["scaffold_class"] == cls]
        if len(sub) < 20:
            continue
        xy = sub[["umap1", "umap2"]].to_numpy()
        grid = kde_field(xy, grid_n=90)
        if grid is None:
            continue
        xx, yy, zz = grid
        ax.contourf(xx, yy, zz, levels=5, cmap=tinted_cmap(col, cls), alpha=0.45)
        ax.contour(xx, yy, zz, levels=5, colors=[col], linewidths=1.2, alpha=0.85)
        patches.append(Patch(facecolor=col, edgecolor=col, alpha=0.55, label=cls))

    sc = ax.scatter(
        rl["umap1"],
        rl["umap2"],
        c=rl["score"].fillna(0),
        cmap="plasma",
        s=22,
        alpha=0.88,
        edgecolors="k",
        linewidths=0.25,
        zorder=10,
    )
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label="RL Score")
    ax.legend(handles=patches, loc="best", title="scaffold_class")
    ax.set_title("molecules.csv scaffold_class densities + RL by Score\n(joint UMAP)")
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def plot_rl_by_nearest(coords: pd.DataFrame, assign: pd.DataFrame, name_map: dict, out_path: Path) -> None:
    mol = coords[coords["source"] == "molecules"].copy()
    rl = coords[coords["label"] == "rl"].copy()
    near = dict(zip(assign["smiles"], assign["nearest_scaffold"]))
    rl["nearest_scaffold"] = rl["smiles"].map(near)
    rl = rl.dropna(subset=["nearest_scaffold"])

    # top scaffolds by assignment count
    top = assign["nearest_scaffold"].value_counts().head(10).index.tolist()
    colors = {g: MOL_CONTRAST[i % len(MOL_CONTRAST)] for i, g in enumerate(sorted(mol["label"].unique()))}

    fig, ax = plt.subplots(figsize=(12, 9))

    # faint densities for top assigned families
    for g in top[:8]:
        sub = mol[mol["label"] == g]
        if len(sub) < 15:
            continue
        xy = sub[["umap1", "umap2"]].to_numpy()
        grid = kde_field(xy, grid_n=80)
        if grid is None:
            continue
        xx, yy, zz = grid
        ax.contourf(xx, yy, zz, levels=4, cmap=tinted_cmap(colors[g], f"n_{g}"), alpha=0.28)
        ax.contour(xx, yy, zz, levels=4, colors=[colors[g]], linewidths=0.9, alpha=0.55)

    other = ~rl["nearest_scaffold"].isin(top)
    if other.any():
        ax.scatter(
            rl.loc[other, "umap1"],
            rl.loc[other, "umap2"],
            c="#9e9e9e",
            s=12,
            alpha=0.35,
            linewidths=0,
            zorder=3,
            label=f"other ({other.sum()})",
        )
    for g in top:
        m = rl["nearest_scaffold"] == g
        if not m.any():
            continue
        ax.scatter(
            rl.loc[m, "umap1"],
            rl.loc[m, "umap2"],
            c=colors[g],
            s=20,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.3,
            zorder=5,
            label=f"{g} ({m.sum()}) — {str(name_map.get(g, ''))[:22]}",
        )

    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Nearest scaffold_id", framealpha=0.95)
    ax.set_title("UMAP: RL colored by nearest molecules.csv scaffold\n(joint coordinates + max Tanimoto assignment)")
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def main() -> None:
    set_style()
    out = OUT
    out.mkdir(parents=True, exist_ok=True)

    if not JOINT.is_file():
        raise SystemExit(f"Missing joint coords: {JOINT}\nRun viz_umap_joint.py first.")
    if not ASSIGN.is_file():
        raise SystemExit(f"Missing assignment CSV: {ASSIGN}\nRun viz_structure_vs_molecules.py first.")

    print("[INFO] Loading joint UMAP + assignment…")
    coords = pd.read_csv(JOINT)
    assign = pd.read_csv(ASSIGN)
    mol_df = pd.read_csv(MOL_CSV)
    name_map = (
        mol_df.groupby("scaffold_id")["scaffold_name"]
        .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else s.name)
        .to_dict()
    )

    plot_scaffold_densities_rl_score(coords, name_map, out / "04_umap_molecules_densities_rl_score.png")
    plot_scaffold_class_densities(coords, mol_df, out / "05_umap_scaffold_class_densities_rl_score.png")
    plot_rl_by_nearest(coords, assign, name_map, out / "06_umap_rl_by_nearest_scaffold.png")

    # also refresh chemspace_molecules copy for consistency
    mol_dir = Path(__file__).resolve().parent / "chemspace_molecules"
    mol_dir.mkdir(parents=True, exist_ok=True)
    plot_scaffold_densities_rl_score(coords, name_map, mol_dir / "01_umap_scaffold_densities_rl_score.png")
    plot_scaffold_class_densities(coords, mol_df, mol_dir / "02_umap_scaffold_class_densities_rl_score.png")

    print(f"[DONE] → {out}")


if __name__ == "__main__":
    main()
