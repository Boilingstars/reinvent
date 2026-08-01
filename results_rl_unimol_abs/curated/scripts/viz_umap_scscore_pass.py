"""
UMAP: top-5 scaffold class densities + 25 SCScore-pass molecules highlighted.

Uses joint UMAP coordinates so positions match other curated figures.

Usage:
  .\\.venv_eval\\Scripts\\python.exe results_rl_unimol_abs\\curated\\scripts\\viz_umap_scscore_pass.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

CURATED = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
ROOT = CURATED.parents[1]
sys.path.insert(0, str(ROOT / "epoch_eval"))
sys.path.insert(0, str(SCRIPTS))

from utils import canonicalize  # noqa: E402
from viz_umap_scaffold_classes import kde_field, tinted_cmap  # noqa: E402

JOINT = CURATED / "tables" / "coordinates_umap_joint.csv"
PASS = CURATED / "generator_check" / "synthesizability" / "tables" / "scscore_pass_sorted.csv"
CLASS_CSV = CURATED / "tables" / "rl_tanimoto_to_classes.csv"
OUT = CURATED / "generator_check" / "synthesizability"

TOP5 = ["stilbene", "benzothiazole", "chalcone", "azobenzene", "thiazolidinedione"]
COLORS = {
    "stilbene": "#C51162",
    "benzothiazole": "#6200EA",
    "chalcone": "#00C853",
    "azobenzene": "#2962FF",
    "thiazolidinedione": "#212121",
}


def main() -> None:
    coords = pd.read_csv(JOINT)
    pass_df = pd.read_csv(PASS)
    pass_canons = {canonicalize(str(s)) for s in pass_df["SMILES"]}
    pass_canons.discard(None)
    score_map = {canonicalize(str(r.SMILES)): float(r.score) for _, r in pass_df.iterrows() if canonicalize(str(r.SMILES))}
    sc_map = {canonicalize(str(r.SMILES)): float(r.scscore) for _, r in pass_df.iterrows() if canonicalize(str(r.SMILES))}

    near = {}
    if CLASS_CSV.is_file():
        cls = pd.read_csv(CLASS_CSV)
        smi_col = "smiles" if "smiles" in cls.columns else "SMILES"
        for _, r in cls.iterrows():
            c = canonicalize(str(r[smi_col]))
            if c:
                near[c] = r["nearest_class"]

    # top5 densities
    top5 = coords[coords["source"] == "top5"].copy()
    rl = coords[coords["label"] == "rl"].copy()
    rl["canon"] = rl["smiles"].map(lambda s: canonicalize(str(s)))
    rl["is_pass"] = rl["canon"].isin(pass_canons)
    n_matched = int(rl["is_pass"].sum())
    print(f"[INFO] SCScore-pass matched in joint UMAP: {n_matched}/{len(pass_canons)}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)

    def draw_densities(ax) -> list:
        patches = []
        for cls in TOP5:
            sub = top5[top5["label"] == cls]
            xy = sub[["umap1", "umap2"]].to_numpy()
            if len(xy) < 15:
                continue
            grid = kde_field(xy, grid_n=90)
            if grid is None:
                continue
            xx, yy, zz = grid
            ax.contourf(xx, yy, zz, levels=5, cmap=tinted_cmap(COLORS[cls], cls), alpha=0.42)
            ax.contour(xx, yy, zz, levels=5, colors=[COLORS[cls]], linewidths=1.0, alpha=0.8)
            patches.append(Patch(facecolor=COLORS[cls], edgecolor=COLORS[cls], alpha=0.55, label=cls))
        return patches

    # Panel 1: all RL faint + pass highlighted by Score
    ax = axes[0]
    patches = draw_densities(ax)
    other = rl[~rl["is_pass"]]
    ax.scatter(
        other["umap1"],
        other["umap2"],
        c="#bdbdbd",
        s=10,
        alpha=0.25,
        linewidths=0,
        zorder=2,
        label=f"other RL (n={len(other)})",
    )
    pas = rl[rl["is_pass"]].copy()
    pas["score_plot"] = pas["canon"].map(score_map)
    sc = ax.scatter(
        pas["umap1"],
        pas["umap2"],
        c=pas["score_plot"],
        cmap="plasma",
        s=70,
        alpha=0.95,
        edgecolors="k",
        linewidths=0.7,
        zorder=8,
        vmin=0.9,
        vmax=1.0,
    )
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="Score (SCScore-pass)")
    ax.legend(handles=patches + ax.get_legend_handles_labels()[0][:1], loc="best", fontsize=8)
    # rebuild legend properly
    h1 = patches + [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#bdbdbd", markersize=6, label=f"other RL (n={len(other)})"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#fde725", markeredgecolor="k", markersize=8, label=f"SCScore-pass (n={n_matched})"),
    ]
    ax.legend(handles=h1, loc="best", fontsize=8, framealpha=0.95)
    ax.set_title("Top-5 densities + SCScore-pass (color=Score)")
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.grid(True, alpha=0.2)

    # Panel 2: pass colored by nearest top-5 class
    ax = axes[1]
    patches = draw_densities(ax)
    ax.scatter(
        other["umap1"],
        other["umap2"],
        c="#bdbdbd",
        s=8,
        alpha=0.2,
        linewidths=0,
        zorder=2,
    )
    pas = pas.copy()
    pas["nearest_class"] = pas["canon"].map(near)
    for cls in TOP5:
        m = pas["nearest_class"] == cls
        if not m.any():
            continue
        ax.scatter(
            pas.loc[m, "umap1"],
            pas.loc[m, "umap2"],
            c=COLORS[cls],
            s=80,
            alpha=0.95,
            edgecolors="white",
            linewidths=0.9,
            zorder=8,
            label=f"pass → {cls} ({int(m.sum())})",
        )
    unk = pas["nearest_class"].isna() | ~pas["nearest_class"].isin(TOP5)
    if unk.any():
        ax.scatter(
            pas.loc[unk, "umap1"],
            pas.loc[unk, "umap2"],
            c="#ff6d00",
            s=80,
            alpha=0.95,
            edgecolors="k",
            linewidths=0.6,
            zorder=8,
            label=f"pass other ({int(unk.sum())})",
        )
    ax.legend(loc="best", fontsize=7.5, framealpha=0.95, title="SCScore-pass nearest class")
    ax.set_title("Same 25 molecules by nearest scaffold class")
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Joint UMAP: top-5 chromophore families + molecules with SCScore ≤ 3\n"
        "(baseline: λ∈[450,480], Score≥0.8, clean)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out_path = OUT / "05_umap_top5_scscore_pass.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")

    # save matched coords
    export = pas.copy()
    export["scscore"] = export["canon"].map(sc_map)
    export.to_csv(OUT / "tables" / "scscore_pass_umap_coords.csv", index=False)
    print("[OK] tables/scscore_pass_umap_coords.csv")


if __name__ == "__main__":
    main()
