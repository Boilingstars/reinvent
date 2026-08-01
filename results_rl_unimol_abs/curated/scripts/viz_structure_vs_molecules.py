"""
Structural interpretation of RL vs molecules.csv scaffold families (29 groups).

Outputs → results_rl_unimol_abs/structure_interpretation_molecules/

Usage:
  .\\.venv_eval\\Scripts\\python.exe results_rl_unimol_abs\\viz_structure_vs_molecules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "epoch_eval"))
sys.path.insert(0, str(ROOT / "results_rl_unimol_abs"))
from utils import smiles_to_fps  # noqa: E402
from viz_chemspace_compare import DEFAULT_RL_CSV, load_rl_smiles  # noqa: E402

OUT = Path(__file__).resolve().parent / "structure_interpretation_molecules"
MOL_CSV = ROOT / "molecules.csv"
JOINT = Path(__file__).resolve().parent / "chemspace_joint" / "coordinates_umap_joint.csv"
COMPARE_UMAP = Path(__file__).resolve().parent / "chemspace_rl_compare" / "coordinates_umap.csv"

CONTRAST = [
    "#C51162",
    "#0033FF",
    "#00C853",
    "#AA00FF",
    "#FF6D00",
    "#00B8D4",
    "#FFD600",
    "#6D4C41",
    "#D50000",
    "#212121",
    "#1A237E",
    "#FF1744",
    "#76FF03",
    "#F50057",
    "#00E5FF",
    "#FFAB00",
    "#6200EA",
    "#64DD17",
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
    "#455A64",
]


def set_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#fafafa",
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#e0e0e0",
            "grid.linewidth": 0.6,
        }
    )


def max_tc_to_fps(query_fps: list, ref_fps: list) -> np.ndarray:
    out = np.zeros(len(query_fps), dtype=float)
    for i, q in enumerate(query_fps):
        out[i] = max(DataStructs.BulkTanimotoSimilarity(q, ref_fps)) if ref_fps else np.nan
    return out


def kde_contour(ax, xy, color, levels=5, alpha_fill=0.10):
    if len(xy) < 15:
        return
    try:
        kde = gaussian_kde(xy.T)
    except Exception:
        return
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    xs = np.linspace(xmin - 0.08 * (xmax - xmin + 1e-9), xmax + 0.08 * (xmax - xmin + 1e-9), 70)
    ys = np.linspace(ymin - 0.08 * (ymax - ymin + 1e-9), ymax + 0.08 * (ymax - ymin + 1e-9), 70)
    xx, yy = np.meshgrid(xs, ys)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=levels, colors=[color], alpha=alpha_fill)
    ax.contour(xx, yy, zz, levels=levels, colors=[color], linewidths=0.9, alpha=0.5)


def main() -> None:
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    set_style()

    print("[INFO] Loading molecules.csv + RL…")
    mol_df = pd.read_csv(MOL_CSV)
    name_map = (
        mol_df.groupby("scaffold_id")["scaffold_name"]
        .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else s.name)
        .to_dict()
    )
    class_map = (
        mol_df.groupby("scaffold_id")["scaffold_class"]
        .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else "unknown")
        .to_dict()
        if "scaffold_class" in mol_df.columns
        else {}
    )

    rl_smi, rl_scores = load_rl_smiles(DEFAULT_RL_CSV, max_n=None, seed=42)
    rl_valid, rl_fps, _ = smiles_to_fps(rl_smi, unique=True)
    score_map = dict(zip(rl_smi, rl_scores))
    rl_sc = np.array([score_map.get(s, np.nan) for s in rl_valid], dtype=float)

    scaffolds = sorted(mol_df["scaffold_id"].dropna().unique().tolist())
    colors = {s: CONTRAST[i % len(CONTRAST)] for i, s in enumerate(scaffolds)}

    print("[INFO] Fingerprints per scaffold_id…")
    ref_fps: dict[str, list] = {}
    for sid in scaffolds:
        smis = mol_df.loc[mol_df["scaffold_id"] == sid, "smiles"].astype(str).tolist()
        _, fps, _ = smiles_to_fps(smis, unique=True)
        ref_fps[sid] = fps
        print(f"  {sid}: {len(fps)}")

    print("[INFO] Max Tanimoto RL → each scaffold…")
    max_tc = {}
    for sid, fps in ref_fps.items():
        print(f"  → {sid}")
        max_tc[sid] = max_tc_to_fps(rl_fps, fps)

    stack = np.column_stack([max_tc[s] for s in scaffolds])
    nearest_idx = np.argmax(stack, axis=1)
    nearest_tc = np.max(stack, axis=1)
    nearest = [scaffolds[i] for i in nearest_idx]

    assign = pd.DataFrame(
        {
            "smiles": rl_valid,
            "score": rl_sc,
            "nearest_scaffold": nearest,
            "nearest_tc": nearest_tc,
            "scaffold_name": [name_map.get(s, s) for s in nearest],
            "scaffold_class": [class_map.get(s, "unknown") for s in nearest],
        }
    )
    for sid in scaffolds:
        assign[f"max_tc_{sid}"] = max_tc[sid]
    assign.to_csv(out / "rl_nearest_molecules_scaffold.csv", index=False)

    # summary
    rows = []
    for sid in scaffolds:
        m = assign["nearest_scaffold"] == sid
        hi = m & (assign["score"] >= 0.8)
        rows.append(
            {
                "scaffold_id": sid,
                "scaffold_name": name_map.get(sid, sid),
                "scaffold_class": class_map.get(sid, ""),
                "n": int(m.sum()),
                "frac": float(m.mean()),
                "n_highscore": int(hi.sum()),
                "frac_highscore": float(hi.sum() / max((assign["score"] >= 0.8).sum(), 1)),
                "mean_score": float(assign.loc[m, "score"].mean()) if m.any() else np.nan,
                "mean_max_tc": float(np.mean(max_tc[sid])),
                "mean_nearest_tc": float(assign.loc[m, "nearest_tc"].mean()) if m.any() else np.nan,
            }
        )
    summary = pd.DataFrame(rows).sort_values("frac", ascending=False)
    summary.to_csv(out / "scaffold_assignment_summary.csv", index=False)

    # Top scaffolds for plotting (those with enough RL)
    top_show = summary.head(10)["scaffold_id"].tolist()

    # ── Dashboard ──────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28, left=0.06, right=0.98, top=0.90, bottom=0.07)

    # UMAP by nearest scaffold (top families only colored; rest grey)
    ax0 = fig.add_subplot(gs[0, 0])
    if JOINT.is_file() or COMPARE_UMAP.is_file():
        if JOINT.is_file():
            coords = pd.read_csv(JOINT)
            rl_xy = coords[coords["label"] == "rl"].copy()
            if "source" in coords.columns:
                chembl = coords[coords["source"] == "chembl"].copy()
                chromo = coords[coords["source"] == "chromophore"].copy()
                mol_pts = coords[coords["source"] == "molecules"].copy()
            else:
                chembl = coords[coords["label"] == "chembl_drugs"].copy()
                chromo = coords[coords["label"] == "chromophore"].copy()
                mol_pts = pd.DataFrame()
        else:
            coords = pd.read_csv(COMPARE_UMAP)
            rl_xy = coords[coords["set"] == "rl_unimol"].rename(
                columns={"dim1": "umap1", "dim2": "umap2", "set": "label"}
            )
            chembl = coords[coords["set"] == "chembl"].rename(columns={"dim1": "umap1", "dim2": "umap2"})
            chromo = coords[coords["set"] == "chromophore"].rename(columns={"dim1": "umap1", "dim2": "umap2"})
            mol_pts = pd.DataFrame()

        near_map = dict(zip(assign["smiles"], assign["nearest_scaffold"]))
        rl_xy["nearest_scaffold"] = rl_xy["smiles"].map(near_map)
        rl_xy = rl_xy.dropna(subset=["nearest_scaffold"])

        if len(chembl):
            ax0.scatter(chembl["umap1"], chembl["umap2"], c="#cfd8dc", s=3, alpha=0.18, linewidths=0, zorder=1)
        if len(chromo):
            kde_contour(ax0, chromo[["umap1", "umap2"]].to_numpy(), "#1B5E20", alpha_fill=0.08)

        # faint densities for top molecule scaffolds if joint coords have them
        if len(mol_pts) and "label" in mol_pts.columns:
            for sid in top_show[:6]:
                sub = mol_pts[mol_pts["label"] == sid]
                if len(sub) >= 15:
                    kde_contour(ax0, sub[["umap1", "umap2"]].to_numpy(), colors[sid], alpha_fill=0.07)

        other = ~rl_xy["nearest_scaffold"].isin(top_show[:8])
        if other.any():
            ax0.scatter(
                rl_xy.loc[other, "umap1"],
                rl_xy.loc[other, "umap2"],
                c="#9e9e9e",
                s=10,
                alpha=0.35,
                linewidths=0,
                zorder=3,
                label="other",
            )
        for sid in top_show[:8]:
            m = rl_xy["nearest_scaffold"] == sid
            if not m.any():
                continue
            ax0.scatter(
                rl_xy.loc[m, "umap1"],
                rl_xy.loc[m, "umap2"],
                c=colors[sid],
                s=18,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.3,
                zorder=5,
                label=f"{sid} ({m.sum()})",
            )
        ax0.legend(fontsize=6.5, loc="best", title="Nearest molecules.csv scaffold", framealpha=0.95)
        ax0.set_title("UMAP: RL colored by nearest molecules.csv scaffold", fontweight="medium")
        ax0.set_xlabel("UMAP Dim 1")
        ax0.set_ylabel("UMAP Dim 2")
    else:
        ax0.text(0.5, 0.5, "UMAP coordinates not found", ha="center")
        ax0.axis("off")

    # composition top scaffolds
    ax1 = fig.add_subplot(gs[0, 1])
    show = summary.head(10)
    x = np.arange(len(show))
    w = 0.38
    ax1.bar(
        x - w / 2,
        show["frac"].to_numpy(),
        w,
        color=[colors[s] for s in show["scaffold_id"]],
        alpha=0.45,
        edgecolor="k",
        lw=0.35,
        label="All RL",
    )
    ax1.bar(
        x + w / 2,
        show["frac_highscore"].to_numpy(),
        w,
        color=[colors[s] for s in show["scaffold_id"]],
        alpha=0.95,
        edgecolor="k",
        lw=0.35,
        label="Score ≥ 0.8",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"{r.scaffold_id}\n{str(r.scaffold_name)[:18]}" for _, r in show.iterrows()],
        fontsize=6.5,
        rotation=25,
        ha="right",
    )
    ax1.set_ylabel("Share")
    ax1.set_title("Top-10 nearest scaffolds from molecules.csv", fontweight="medium")
    ax1.legend(fontsize=8)

    # by scaffold_class (broader)
    ax2 = fig.add_subplot(gs[1, 0])
    by_cls = (
        assign.groupby("scaffold_class")
        .agg(n=("smiles", "size"), mean_score=("score", "mean"))
        .sort_values("n", ascending=False)
    )
    hi = assign[assign["score"] >= 0.8]
    hi_cls = hi["scaffold_class"].value_counts(normalize=True)
    all_cls = assign["scaffold_class"].value_counts(normalize=True)
    classes = list(all_cls.index)
    cls_colors = {
        "chromophore": "#C51162",
        "heterocycle": "#0033FF",
        "drug_class": "#00C853",
        "scaffold": "#AA00FF",
        "natural_product": "#FF6D00",
        "unknown": "#9e9e9e",
    }
    x = np.arange(len(classes))
    ax2.bar(x - 0.2, [all_cls.get(c, 0) for c in classes], 0.38, color=[cls_colors.get(c, "#777") for c in classes], alpha=0.45, edgecolor="k", lw=0.3, label="All RL")
    ax2.bar(x + 0.2, [hi_cls.get(c, 0) for c in classes], 0.38, color=[cls_colors.get(c, "#777") for c in classes], alpha=0.95, edgecolor="k", lw=0.3, label="Score ≥ 0.8")
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=20, ha="right")
    ax2.set_ylabel("Share")
    ax2.set_title("Broader scaffold_class of nearest molecules.csv hit", fontweight="medium")
    ax2.legend(fontsize=8)

    # mean max-Tc bar for all 29 (sorted)
    ax3 = fig.add_subplot(gs[1, 1])
    s2 = summary.sort_values("mean_max_tc", ascending=True)
    ax3.barh(
        s2["scaffold_id"],
        s2["mean_max_tc"],
        color=[colors[s] for s in s2["scaffold_id"]],
        edgecolor="k",
        lw=0.25,
        height=0.75,
    )
    ax3.set_xlabel("Mean max Tanimoto (RL → scaffold family)")
    ax3.set_title("Structural proximity to each molecules.csv family", fontweight="medium")
    ax3.tick_params(axis="y", labelsize=7)

    fig.suptitle(
        "RL vs molecules.csv scaffold families\n"
        "Nearest family = argmax Morgan Tanimoto to that scaffold_id group",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(out / "01_molecules_scaffold_interpretation.png", dpi=180)
    plt.close(fig)
    print("[OK] 01_molecules_scaffold_interpretation.png")

    # donut high-score top scaffolds
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    hi_counts = hi["nearest_scaffold"].value_counts()
    top_hi = hi_counts.head(8)
    other_n = int(hi_counts.iloc[8:].sum()) if len(hi_counts) > 8 else 0
    labels_pie = list(top_hi.index) + (["other"] if other_n else [])
    sizes = list(top_hi.to_numpy()) + ([other_n] if other_n else [])
    pie_colors = [colors.get(s, "#9e9e9e") for s in labels_pie]
    axes[0].pie(
        sizes,
        labels=None,
        colors=pie_colors,
        autopct=lambda p: f"{p:.0f}%" if p >= 5 else "",
        startangle=90,
        pctdistance=0.72,
        wedgeprops=dict(width=0.42, edgecolor="white", lw=1.4),
    )
    axes[0].legend(
        [mpatches.Patch(color=pie_colors[i], label=labels_pie[i]) for i in range(len(labels_pie))],
        [f"{labels_pie[i]} ({name_map.get(labels_pie[i], '')[:22]})" for i in range(len(labels_pie))],
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        fontsize=7,
        frameon=False,
    )
    axes[0].set_title(f"High-Score (S≥0.8) nearest scaffold\nn={len(hi)}", fontweight="medium")

    # mean score by top scaffolds
    top_mean = summary.head(10)
    axes[1].barh(
        top_mean["scaffold_id"][::-1],
        top_mean["mean_score"][::-1],
        color=[colors[s] for s in top_mean["scaffold_id"][::-1]],
        edgecolor="k",
        lw=0.35,
    )
    axes[1].set_xlabel("Mean RL Score")
    axes[1].set_title("Mean Score of RL assigned to each family", fontweight="medium")
    fig.tight_layout()
    fig.savefig(out / "02_highscore_molecules_scaffolds.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 02_highscore_molecules_scaffolds.png")

    # representative molecules for top 5 assigned scaffolds
    top5_assigned = summary.head(5)["scaffold_id"].tolist()
    mols, legs = [], []
    for sid in top5_assigned:
        sub = assign[assign["nearest_scaffold"] == sid].copy()
        sub["rank"] = 0.55 * sub["score"].fillna(0) + 0.45 * sub["nearest_tc"]
        for _, r in sub.sort_values("rank", ascending=False).head(2).iterrows():
            m = Chem.MolFromSmiles(r["smiles"])
            if m is None:
                continue
            mols.append(m)
            legs.append(f"{sid}\nS={r['score']:.2f} Tc={r['nearest_tc']:.2f}")
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(260, 210), legends=legs)
        img.save(out / "03_representative_rl_by_molecules_scaffold.png")
        print("[OK] 03_representative_rl_by_molecules_scaffold.png")

    # interpretation text
    top3 = summary.head(3)
    lines = [
        "# RL vs molecules.csv scaffolds",
        "",
        "Each RL molecule is assigned to the nearest `scaffold_id` group in `molecules.csv`",
        "by maximum Morgan Tanimoto.",
        "",
        f"- RL molecules: **{len(assign)}**",
        f"- High-Score (S≥0.8): **{(assign['score']>=0.8).sum()}**",
        "",
        "## Top nearest scaffolds (all RL)",
    ]
    for _, r in top3.iterrows():
        lines.append(
            f"- **{r['scaffold_id']}** ({r['scaffold_name']}): {100*r['frac']:.1f}% of RL, "
            f"mean Score={r['mean_score']:.2f}, high-Score share={100*r['frac_highscore']:.1f}%"
        )
    lines += [
        "",
        "## Broader scaffold_class among nearest hits",
    ]
    for c, frac in all_cls.items():
        lines.append(f"- **{c}**: {100*frac:.1f}% all RL; high-Score {100*hi_cls.get(c, 0):.1f}%")
    lines += [
        "",
        "## Why use molecules.csv in addition to top-5 ChEMBL classes?",
        "- `molecules.csv` has **finer** families (fluorescein, cyanine, BODIPY, rhodamine, …)",
        "  that are more dye/chromophore-specific than stilbene/chalcone alone.",
        "- Good for asking: *does RL rediscover classic dye scaffolds or invent stilbene-like generics?*",
        "- Top-5 ChEMBL classes answer *broad chemotype neighborhood*;",
        "  molecules.csv answers *which named dye/drug scaffolds are closest*.",
        "",
        "Compare with `structure_interpretation/` (top-5 ChEMBL families).",
    ]
    (out / "INTERPRETATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] → {out}")
    print(summary[["scaffold_id", "frac", "frac_highscore", "mean_score"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
