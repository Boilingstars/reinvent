"""
Structure-aware interpretation of RL generation vs scaffold classes.

Beautiful multi-panel figure + molecule grids + summary tables.

Outputs → results_rl_unimol_abs/structure_interpretation/

Usage:
  .\\.venv_eval\\Scripts\\python.exe results_rl_unimol_abs\\viz_structure_interpretation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from rdkit import Chem
from rdkit.Chem import Draw
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "epoch_eval"))
sys.path.insert(0, str(ROOT / "results_rl_unimol_abs"))

OUT = Path(__file__).resolve().parent / "structure_interpretation"
RL_TC = (
    Path(__file__).resolve().parent
    / "chemspace_scaffold_classes"
    / "class_proximity"
    / "tables"
    / "rl_tanimoto_to_classes.csv"
)
CHROMO_TC = (
    Path(__file__).resolve().parent
    / "chemspace_scaffold_classes"
    / "chromophore_proximity"
    / "tables"
    / "chromophore_tanimoto_to_classes.csv"
)
COMPARE = (
    Path(__file__).resolve().parent
    / "chemspace_scaffold_classes"
    / "rl_vs_chromophore_proximity.csv"
)
# prefer joint coords if present, else rl_compare umap
JOINT = Path(__file__).resolve().parent / "chemspace_joint" / "coordinates_umap_joint.csv"
COMPARE_UMAP = Path(__file__).resolve().parent / "chemspace_rl_compare" / "coordinates_umap.csv"

TOP5 = ["stilbene", "benzothiazole", "chalcone", "azobenzene", "thiazolidinedione"]
COLORS = {
    "stilbene": "#C51162",
    "benzothiazole": "#AA00FF",
    "chalcone": "#00C853",
    "azobenzene": "#0033FF",
    "thiazolidinedione": "#212121",
}
RU = {
    "stilbene": "стильбен",
    "benzothiazole": "бензотиазол",
    "chalcone": "халкон",
    "azobenzene": "азобензол",
    "thiazolidinedione": "тиазолидиндион",
}


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
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )


def kde_contour(ax, xy, color, levels=5, alpha_fill=0.12):
    if len(xy) < 15:
        return
    try:
        kde = gaussian_kde(xy.T)
    except Exception:
        return
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    pad_x = 0.08 * (xmax - xmin + 1e-9)
    pad_y = 0.08 * (ymax - ymin + 1e-9)
    xs = np.linspace(xmin - pad_x, xmax + pad_x, 80)
    ys = np.linspace(ymin - pad_y, ymax + pad_y, 80)
    xx, yy = np.meshgrid(xs, ys)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=levels, colors=[color], alpha=alpha_fill)
    ax.contour(xx, yy, zz, levels=levels, colors=[color], linewidths=1.0, alpha=0.55)


def load_umap_with_nearest(rl_tc: pd.DataFrame) -> pd.DataFrame | None:
    nearest = dict(zip(rl_tc["smiles"], rl_tc["nearest_class"]))
    score = dict(zip(rl_tc["smiles"], rl_tc["score"]))
    nearest_tc = dict(zip(rl_tc["smiles"], rl_tc["nearest_tc"]))

    if JOINT.is_file():
        coords = pd.read_csv(JOINT)
        # joint has label rl + source
        rl = coords[coords["label"] == "rl"].copy()
        bg_chembl = coords[(coords["label"] == "chembl_drugs") | (coords["source"] == "chembl")].copy()
        bg_chromo = coords[(coords["label"] == "chromophore") | (coords["source"] == "chromophore")].copy()
        # top5 for faint hulls
        top5_pts = coords[coords["source"] == "top5"].copy() if "source" in coords.columns else coords[coords["label"].isin(TOP5)].copy()
    elif COMPARE_UMAP.is_file():
        coords = pd.read_csv(COMPARE_UMAP)
        rl = coords[coords["set"] == "rl_unimol"].copy()
        rl = rl.rename(columns={"set": "label", "dim1": "umap1", "dim2": "umap2"})
        bg_chembl = coords[coords["set"] == "chembl"].rename(columns={"dim1": "umap1", "dim2": "umap2"})
        bg_chromo = coords[coords["set"] == "chromophore"].rename(columns={"dim1": "umap1", "dim2": "umap2"})
        top5_pts = pd.DataFrame()
    else:
        return None

    rl["nearest_class"] = rl["smiles"].map(nearest)
    rl["score"] = rl["smiles"].map(score)
    rl["nearest_tc"] = rl["smiles"].map(nearest_tc)
    rl = rl.dropna(subset=["nearest_class"])
    return {"rl": rl, "chembl": bg_chembl, "chromo": bg_chromo, "top5": top5_pts}


def panel_umap_by_class(ax, data):
    chembl = data["chembl"]
    chromo = data["chromo"]
    rl = data["rl"]
    top5 = data["top5"]

    if len(chembl):
        ax.scatter(chembl["umap1"], chembl["umap2"], c="#cfd8dc", s=4, alpha=0.2, linewidths=0, zorder=1)
    if len(chromo):
        xy = chromo[["umap1", "umap2"]].to_numpy()
        kde_contour(ax, xy, "#1B5E20", alpha_fill=0.10)
        ax.scatter(chromo["umap1"], chromo["umap2"], c="#66bb6a", s=5, alpha=0.15, linewidths=0, zorder=2)

    # faint class clouds from top5 refs if available
    if len(top5) and "label" in top5.columns:
        for cls in TOP5:
            sub = top5[top5["label"] == cls]
            if len(sub) < 20:
                continue
            kde_contour(ax, sub[["umap1", "umap2"]].to_numpy(), COLORS[cls], alpha_fill=0.06)

    for cls in TOP5:
        m = rl["nearest_class"] == cls
        if not m.any():
            continue
        ax.scatter(
            rl.loc[m, "umap1"],
            rl.loc[m, "umap2"],
            c=COLORS[cls],
            s=22,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.35,
            zorder=5,
            label=f"{cls} ({m.sum()})",
        )

    ax.set_title("UMAP: RL by nearest scaffold class", fontweight="medium")
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.legend(fontsize=7, framealpha=0.95, loc="best", title="Nearest class")


def panel_composition(ax, rl_tc: pd.DataFrame, chromo_tc: pd.DataFrame | None):
    all_c = rl_tc["nearest_class"].value_counts(normalize=True).reindex(TOP5).fillna(0)
    hi = rl_tc[rl_tc["score"] >= 0.8]["nearest_class"].value_counts(normalize=True).reindex(TOP5).fillna(0)
    x = np.arange(len(TOP5))
    w = 0.36
    ax.bar(x - w / 2, all_c.to_numpy(), w, color=[COLORS[c] for c in TOP5], alpha=0.45, edgecolor="k", lw=0.4, label="All RL")
    ax.bar(x + w / 2, hi.to_numpy(), w, color=[COLORS[c] for c in TOP5], alpha=0.95, edgecolor="k", lw=0.4, label="Score ≥ 0.8")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n({RU[c]})" for c in TOP5], fontsize=7.5)
    ax.set_ylabel("Share of molecules")
    ax.set_title("Scaffold preference: all vs high-Score", fontweight="medium")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(0.45, float(max(all_c.max(), hi.max())) * 1.25))


def panel_delta(ax, compare: pd.DataFrame):
    df = compare.copy()
    df["delta"] = df["rl_mean_max_tc"] - df["chromo_mean_max_tc"]
    df = df.set_index("class").reindex(TOP5).reset_index()
    colors = [COLORS[c] for c in df["class"]]
    ax.barh(df["class"], df["delta"], color=colors, edgecolor="k", lw=0.4, height=0.65)
    ax.axvline(0, color="#424242", lw=1)
    ax.set_xlabel("Δ mean max-Tc (RL − train chromophores)")
    ax.set_title("Where generation shifted vs training set", fontweight="medium")
    for y, v in enumerate(df["delta"]):
        ax.text(v + (0.002 if v >= 0 else -0.002), y, f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)


def panel_score_vs_tc(ax, rl_tc: pd.DataFrame):
    for cls in TOP5:
        m = rl_tc["nearest_class"] == cls
        ax.scatter(
            rl_tc.loc[m, f"max_tc_{cls}"],
            rl_tc.loc[m, "score"],
            c=COLORS[cls],
            s=14,
            alpha=0.45,
            linewidths=0,
            label=cls,
        )
    ax.set_xlabel("Max Tanimoto to assigned class")
    ax.set_ylabel("RL Score")
    ax.set_title("Optical Score vs structural class match", fontweight="medium")
    ax.legend(fontsize=7, markerscale=1.4, framealpha=0.95)


def make_dashboard(rl_tc, chromo_tc, compare, umap_data, out: Path):
    set_style()
    fig = plt.figure(figsize=(14.5, 10.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.28, left=0.06, right=0.98, top=0.90, bottom=0.07)

    ax0 = fig.add_subplot(gs[0, 0])
    if umap_data is not None:
        panel_umap_by_class(ax0, umap_data)
    else:
        ax0.text(0.5, 0.5, "UMAP coords missing", ha="center")
        ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    panel_composition(ax1, rl_tc, chromo_tc)

    ax2 = fig.add_subplot(gs[1, 0])
    panel_delta(ax2, compare)

    ax3 = fig.add_subplot(gs[1, 1])
    panel_score_vs_tc(ax3, rl_tc)

    fig.suptitle(
        "Structural interpretation of RL generation\n"
        "Nearest scaffold class = argmax Morgan Tanimoto to top-5 chromophore-related families",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(out / "01_structure_interpretation_dashboard.png", dpi=180)
    plt.close(fig)
    print("[OK] 01_structure_interpretation_dashboard.png")


def molecule_grids(rl_tc: pd.DataFrame, out: Path):
    """High-Score representatives per nearest class."""
    for cls in TOP5:
        sub = rl_tc[(rl_tc["nearest_class"] == cls) & (rl_tc["score"] >= 0.7)].copy()
        if sub.empty:
            sub = rl_tc[rl_tc["nearest_class"] == cls].copy()
        col = f"max_tc_{cls}"
        sub["rank"] = 0.55 * sub["score"] + 0.45 * sub[col]
        sub = sub.sort_values("rank", ascending=False).head(6)
        mols, legs = [], []
        for _, r in sub.iterrows():
            m = Chem.MolFromSmiles(r["smiles"])
            if m is None:
                continue
            mols.append(m)
            legs.append(f"S={r['score']:.2f}  Tc={r[col]:.2f}")
        if not mols:
            continue
        img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(280, 220), legends=legs)
        img.save(out / f"02_reps_{cls}.png")
        print(f"[OK] 02_reps_{cls}.png")

    # combined high-score strip: 2 per class
    mols, legs = [], []
    for cls in TOP5:
        sub = rl_tc[rl_tc["nearest_class"] == cls].copy()
        col = f"max_tc_{cls}"
        sub["rank"] = 0.6 * sub["score"].fillna(0) + 0.4 * sub[col]
        for _, r in sub.sort_values("rank", ascending=False).head(2).iterrows():
            m = Chem.MolFromSmiles(r["smiles"])
            if m is None:
                continue
            mols.append(m)
            legs.append(f"{cls}\nS={r['score']:.2f}")
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(260, 210), legends=legs)
        img.save(out / "02_reps_all_classes.png")
        print("[OK] 02_reps_all_classes.png")


def donut_high_score(rl_tc: pd.DataFrame, out: Path):
    set_style()
    hi = rl_tc[rl_tc["score"] >= 0.8]
    counts = hi["nearest_class"].value_counts().reindex(TOP5).fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    wedges, texts, autotexts = axes[0].pie(
        counts.to_numpy(),
        labels=None,
        colors=[COLORS[c] for c in TOP5],
        autopct=lambda p: f"{p:.0f}%" if p >= 4 else "",
        startangle=90,
        pctdistance=0.72,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.5),
    )
    axes[0].legend(
        [mpatches.Patch(color=COLORS[c], label=f"{c} ({RU[c]})") for c in TOP5],
        [f"{c} ({RU[c]})" for c in TOP5],
        loc="center left",
        bbox_to_anchor=(0.95, 0.5),
        fontsize=8,
        frameon=False,
    )
    axes[0].set_title(f"High-Score hits (S≥0.8), n={len(hi)}\nnearest scaffold family", fontweight="medium")

    # mean Score by class
    means = rl_tc.groupby("nearest_class")["score"].mean().reindex(TOP5)
    axes[1].barh(TOP5, means.to_numpy(), color=[COLORS[c] for c in TOP5], edgecolor="k", lw=0.4, height=0.65)
    axes[1].set_xlabel("Mean RL Score")
    axes[1].set_title("Mean Score within each nearest-class cohort", fontweight="medium")
    for i, v in enumerate(means.to_numpy()):
        axes[1].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "03_highscore_class_profile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 03_highscore_class_profile.png")


def write_summary(rl_tc: pd.DataFrame, compare: pd.DataFrame, out: Path):
    hi = rl_tc[rl_tc["score"] >= 0.8]
    lines = [
        "# Structural interpretation of RL UniMol generation",
        "",
        "## Method",
        "Each RL molecule is assigned to the **nearest scaffold family** among top-5",
        "(stilbene, benzothiazole, chalcone, azobenzene, thiazolidinedione)",
        "by **maximum Morgan Tanimoto** to molecules of that family.",
        "",
        "This is a soft structural label — not a hard SMARTS/substructure match.",
        "",
        "## Key numbers",
        f"- RL molecules classified: **{len(rl_tc)}**",
        f"- High-Score (S≥0.8): **{len(hi)}** ({100*len(hi)/len(rl_tc):.1f}%)",
        "",
        "### Nearest-class share (all RL)",
    ]
    for cls, n in rl_tc["nearest_class"].value_counts().reindex(TOP5).fillna(0).items():
        lines.append(f"- **{cls}** ({RU[cls]}): {100*n/len(rl_tc):.1f}%")
    lines += ["", "### Nearest-class share (Score ≥ 0.8)"]
    for cls, n in hi["nearest_class"].value_counts().reindex(TOP5).fillna(0).items():
        lines.append(f"- **{cls}** ({RU[cls]}): {100*n/max(len(hi),1):.1f}%")
    lines += ["", "### Shift vs train chromophores (Δ mean max-Tc)"]
    for _, r in compare.sort_values("rl_mean_max_tc", ascending=False).iterrows():
        d = r["rl_mean_max_tc"] - r["chromo_mean_max_tc"]
        lines.append(f"- **{r['class']}**: Δ={d:+.3f} (RL {r['rl_mean_max_tc']:.3f} vs train {r['chromo_mean_max_tc']:.3f})")
    lines += [
        "",
        "## How to read the figures",
        "- `01_…dashboard`: UMAP by class + composition + Δ vs train + Score–Tc scatter",
        "- `02_reps_*`: example structures for each family (high Score preferred)",
        "- `03_…profile`: donut of high-Score families + mean Score by class",
        "",
        "## Chemical takeaway",
        "Generation concentrates on **stilbene-like** and **benzothiazole-like** neighborhoods;",
        "high-Score molecules amplify this bias. Chalcone/thiazolidinedione remain secondary.",
        "Relative to the training chromophore set, RL **over-weights stilbene/benzothiazole**",
        "and under-weights some other families — a signature of the λ_abs reward landscape.",
    ]
    (out / "INTERPRETATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    rl_tc = pd.read_csv(RL_TC)
    chromo_tc = pd.read_csv(CHROMO_TC) if CHROMO_TC.is_file() else None
    compare = pd.read_csv(COMPARE)
    umap_data = load_umap_with_nearest(rl_tc)

    make_dashboard(rl_tc, chromo_tc, compare, umap_data, out)
    molecule_grids(rl_tc, out)
    donut_high_score(rl_tc, out)
    write_summary(rl_tc, compare, out)

    # export table for canvas / reporting
    summary = (
        rl_tc.groupby("nearest_class")
        .agg(n=("smiles", "size"), mean_score=("score", "mean"), mean_nearest_tc=("nearest_tc", "mean"))
        .reindex(TOP5)
        .reset_index()
    )
    hi = rl_tc[rl_tc["score"] >= 0.8]
    summary["n_highscore"] = summary["nearest_class"].map(hi["nearest_class"].value_counts()).fillna(0).astype(int)
    summary["frac_all"] = summary["n"] / summary["n"].sum()
    summary["frac_highscore"] = summary["n_highscore"] / max(len(hi), 1)
    summary = summary.merge(compare.rename(columns={"class": "nearest_class"}), on="nearest_class", how="left")
    summary.to_csv(out / "class_structure_summary.csv", index=False)
    print(f"[DONE] → {out}")


if __name__ == "__main__":
    main()
