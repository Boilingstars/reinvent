"""
visualize_rl_unimol_abs.py
=========================
Visualize REINVENT RL + Uni-Mol absorption results from rl_unimol_abs_1.csv.

Creates plots and summary tables under results_rl_unimol_abs/.

Usage (from repo root):
    python results_rl_unimol_abs/visualize_rl_unimol_abs.py
    python results_rl_unimol_abs/visualize_rl_unimol_abs.py --csv rl_unimol_abs_1.csv
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CSV = ROOT / "rl_unimol_abs_1.csv"

# Column aliases in the RL CSV
COL = {
    "score": "Score",
    "smiles": "SMILES",
    "state": "SMILES_state",
    "lam_t": "lambda_abs",
    "lam": "lambda_abs (raw)",
    "qed_t": "QED",
    "qed": "QED (raw)",
    "sa_t": "SA score",
    "sa": "SA score (raw)",
    "uw_t": "Unwanted SMARTS",
    "uw": "Unwanted SMARTS (raw)",
    "patterns": "matchting_patterns (Unwanted SMARTS)",
    "step": "step",
    "agent": "Agent",
    "prior": "Prior",
    "target": "Target",
    "scaffold": "Scaffold",
}


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # convenience flags
    df["valid"] = df[COL["state"]] == 1
    df["has_lambda"] = df[COL["lam"]] > 0
    df["clean"] = (df[COL["uw"]] >= 0.999) & df["valid"]  # no unwanted SMARTS (component score ~1)
    df["unique_smiles"] = ~df[COL["smiles"]].duplicated(keep="first")
    return df


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overview_hists(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    panels = [
        (axes[0, 0], COL["score"], "Score (aggregate)", None),
        (axes[0, 1], COL["lam"], "λ_abs raw (nm)", df["has_lambda"]),
        (axes[0, 2], COL["lam_t"], "λ_abs transformed", None),
        (axes[1, 0], COL["qed"], "QED raw", df[COL["qed"]] > 0),
        (axes[1, 1], COL["sa"], "SA score raw (↓ easier)", df[COL["sa"]] > 0),
        (axes[1, 2], COL["step"], "RL step", None),
    ]
    for ax, col, title, mask in panels:
        s = df.loc[mask, col] if mask is not None else df[col]
        ax.hist(s.dropna(), bins=40, color="#4c72b0", edgecolor="white", alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25, axis="y")
    fig.suptitle("RL Uni-Mol absorption — property distributions", y=1.01)
    savefig(fig, out / "01_property_histograms.png")


def plot_score_vs_lambda(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    m = df["valid"] & df["has_lambda"]
    sc = axes[0].scatter(
        df.loc[m, COL["lam"]],
        df.loc[m, COL["score"]],
        c=df.loc[m, COL["step"]],
        cmap="viridis",
        s=18,
        alpha=0.75,
        edgecolors="none",
    )
    fig.colorbar(sc, ax=axes[0], label="step")
    axes[0].set_xlabel("λ_abs (nm)")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Score vs λ_abs (valid, colored by step)")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(
        df.loc[m, COL["qed"]],
        df.loc[m, COL["lam"]],
        c=df.loc[m, COL["score"]],
        cmap="magma",
        s=18,
        alpha=0.75,
        edgecolors="none",
    )
    cb = fig.colorbar(axes[1].collections[0], ax=axes[1], label="Score")
    axes[1].set_xlabel("QED")
    axes[1].set_ylabel("λ_abs (nm)")
    axes[1].set_title("λ_abs vs QED (colored by Score)")
    axes[1].grid(True, alpha=0.3)
    savefig(fig, out / "02_score_lambda_qed_scatter.png")


def plot_step_trajectories(df: pd.DataFrame, out: Path) -> None:
    g = (
        df.groupby(COL["step"], as_index=False)
        .agg(
            n=(COL["smiles"], "size"),
            mean_score=(COL["score"], "mean"),
            median_score=(COL["score"], "median"),
            mean_lam=(COL["lam"], "mean"),
            median_lam=(COL["lam"], "median"),
            frac_valid=("valid", "mean"),
            frac_has_lam=("has_lambda", "mean"),
            frac_clean=("clean", "mean"),
            mean_qed=(COL["qed"], "mean"),
            mean_sa=(COL["sa"], "mean"),
            n_unique=(COL["smiles"], "nunique"),
        )
    )
    g["uniqueness"] = g["n_unique"] / g["n"]
    g.to_csv(out / "tables" / "metrics_by_step.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    step = g[COL["step"]]
    axes[0, 0].plot(step, g["mean_score"], "o-", label="mean")
    axes[0, 0].plot(step, g["median_score"], "s--", label="median")
    axes[0, 0].set_title("Score vs step")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(step, g["mean_lam"], "o-", color="#c44e52")
    axes[0, 1].plot(step, g["median_lam"], "s--", color="#c44e52", alpha=0.7)
    axes[0, 1].set_title("λ_abs (nm) vs step")

    axes[0, 2].plot(step, g["frac_valid"], "o-", label="valid")
    axes[0, 2].plot(step, g["frac_has_lam"], "s-", label="λ>0")
    axes[0, 2].plot(step, g["frac_clean"], "^-", label="clean SMARTS")
    axes[0, 2].set_ylim(0, 1.05)
    axes[0, 2].set_title("Validity / quality fractions")
    axes[0, 2].legend(fontsize=8)

    axes[1, 0].plot(step, g["mean_qed"], "o-", color="#55a868")
    axes[1, 0].set_title("Mean QED vs step")

    axes[1, 1].plot(step, g["mean_sa"], "o-", color="#8172b3")
    axes[1, 1].set_title("Mean SA (raw) vs step")

    axes[1, 2].plot(step, g["uniqueness"], "o-", color="#ccb974")
    axes[1, 2].set_ylim(0, 1.05)
    axes[1, 2].set_title("Uniqueness (#unique/#batch)")

    for ax in axes.ravel():
        ax.set_xlabel("RL step")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Evolution of RL batch metrics over steps", y=1.01)
    savefig(fig, out / "03_metrics_vs_step.png")


def plot_lambda_by_step_box(df: pd.DataFrame, out: Path) -> None:
    m = df["has_lambda"] & df["valid"]
    sub = df.loc[m, [COL["step"], COL["lam"], COL["score"]]].copy()
    # aggregate every few steps for readability if many steps
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    steps = sorted(sub[COL["step"]].unique())
    data = [sub.loc[sub[COL["step"]] == s, COL["lam"]].values for s in steps]
    axes[0].boxplot(data, tick_labels=steps, showfliers=False)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("λ_abs (nm)")
    axes[0].set_title("λ_abs distribution by RL step (valid, λ>0)")
    axes[0].tick_params(axis="x", rotation=90, labelsize=7)
    axes[0].grid(True, alpha=0.3, axis="y")

    data_s = [sub.loc[sub[COL["step"]] == s, COL["score"]].values for s in steps]
    axes[1].boxplot(data_s, tick_labels=steps, showfliers=False)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Score distribution by RL step")
    axes[1].tick_params(axis="x", rotation=90, labelsize=7)
    axes[1].grid(True, alpha=0.3, axis="y")
    savefig(fig, out / "04_lambda_score_boxplot_by_step.png")


def plot_pairwise_properties(df: pd.DataFrame, out: Path) -> None:
    m = df["valid"] & df["has_lambda"] & (df[COL["qed"]] > 0) & (df[COL["sa"]] > 0)
    cols = [COL["lam"], COL["qed"], COL["sa"], COL["score"], COL["lam_t"]]
    sub = df.loc[m, cols].rename(
        columns={
            COL["lam"]: "lambda_nm",
            COL["qed"]: "QED",
            COL["sa"]: "SA",
            COL["score"]: "Score",
            COL["lam_t"]: "lambda_t",
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = sns.pairplot(sub.sample(min(800, len(sub)), random_state=42), corner=True, diag_kind="hist", plot_kws={"s": 12, "alpha": 0.5})
        g.fig.suptitle("Pairwise property relationships (valid subset)", y=1.02)
        g.fig.savefig(out / "05_pairwise_properties.png", dpi=140, bbox_inches="tight")
        plt.close(g.fig)

    corr = sub.corr(numeric_only=True)
    corr.to_csv(out / "tables" / "property_correlation.csv")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Correlation matrix")
    savefig(fig, out / "06_correlation_heatmap.png")


def plot_nll_panel(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].scatter(df[COL["prior"]], df[COL["agent"]], c=df[COL["score"]], cmap="viridis", s=12, alpha=0.6)
    axes[0].set_xlabel("Prior NLL")
    axes[0].set_ylabel("Agent NLL")
    axes[0].set_title("Agent vs Prior NLL")
    axes[0].grid(True, alpha=0.3)

    m = df["valid"]
    axes[1].scatter(df.loc[m, COL["agent"]], df.loc[m, COL["lam"]], c=df.loc[m, COL["step"]], cmap="plasma", s=12, alpha=0.65)
    axes[1].set_xlabel("Agent NLL")
    axes[1].set_ylabel("λ_abs (nm)")
    axes[1].set_title("λ_abs vs Agent NLL")
    axes[1].grid(True, alpha=0.3)

    by = df.groupby(COL["step"])[[COL["agent"], COL["prior"]]].mean()
    axes[2].plot(by.index, by[COL["agent"]], "o-", label="Agent")
    axes[2].plot(by.index, by[COL["prior"]], "s-", label="Prior")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("mean NLL")
    axes[2].set_title("Mean NLL over RL steps")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    savefig(fig, out / "07_nll_panels.png")


def plot_top_molecules(df: pd.DataFrame, out: Path, top_n: int = 24) -> None:
    """Grid of top-scoring valid molecules (2D depiction)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except ImportError:
        print("[WARN] RDKit missing — skip molecule grid")
        return

    cand = (
        df[df["valid"] & df["has_lambda"] & df["clean"]]
        .sort_values(COL["score"], ascending=False)
        .drop_duplicates(COL["smiles"])
        .head(top_n)
    )
    if cand.empty:
        cand = (
            df[df["valid"] & df["has_lambda"]]
            .sort_values(COL["score"], ascending=False)
            .drop_duplicates(COL["smiles"])
            .head(top_n)
        )

    mols, legends = [], []
    for _, r in cand.iterrows():
        mol = Chem.MolFromSmiles(r[COL["smiles"]])
        if mol is None:
            continue
        mols.append(mol)
        legends.append(
            f"S={r[COL['score']]:.2f}\nλ={r[COL['lam']]:.0f} nm\nQED={r[COL['qed']]:.2f}"
        )

    if not mols:
        return
    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(280, 220), legends=legends)
    img.save(out / "08_top_molecules_grid.png")

    cand_out = cand[
        [COL["smiles"], COL["score"], COL["lam"], COL["qed"], COL["sa"], COL["step"], COL["scaffold"]]
    ].copy()
    cand_out.to_csv(out / "tables" / "top_molecules.csv", index=False)


def plot_chemspace_umap(df: pd.DataFrame, out: Path) -> None:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        from sklearn.decomposition import PCA
    except ImportError as exc:
        print(f"[WARN] chemspace skip: {exc}")
        return

    try:
        from umap import UMAP

        use_umap = True
    except Exception:
        use_umap = False

    sub = (
        df[df["valid"] & df["has_lambda"]]
        .drop_duplicates(COL["smiles"])
        .copy()
    )
    if len(sub) < 20:
        return

    fps = []
    keep_idx = []
    for i, smi in enumerate(sub[COL["smiles"]].tolist()):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((2048,), dtype=np.float64)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        keep_idx.append(i)
    if len(fps) < 20:
        return

    X = np.vstack(fps)
    sub = sub.iloc[keep_idx].reset_index(drop=True)
    Xp = PCA(n_components=min(50, X.shape[0] - 1), random_state=42).fit_transform(X)
    if use_umap:
        Z = UMAP(n_neighbors=20, min_dist=0.2, random_state=42).fit_transform(Xp)
        tag = "UMAP"
    else:
        from sklearn.manifold import TSNE

        Z = TSNE(n_components=2, perplexity=min(30, len(Xp) // 4), random_state=42).fit_transform(Xp)
        tag = "t-SNE"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc0 = axes[0].scatter(Z[:, 0], Z[:, 1], c=sub[COL["lam"]], cmap="plasma", s=22, alpha=0.85, edgecolors="none")
    fig.colorbar(sc0, ax=axes[0], label="λ_abs (nm)")
    axes[0].set_title(f"{tag}: colored by λ_abs")
    axes[0].set_xlabel("Dim 1")
    axes[0].set_ylabel("Dim 2")

    sc1 = axes[1].scatter(Z[:, 0], Z[:, 1], c=sub[COL["score"]], cmap="viridis", s=22, alpha=0.85, edgecolors="none")
    fig.colorbar(sc1, ax=axes[1], label="Score")
    axes[1].set_title(f"{tag}: colored by Score")
    axes[1].set_xlabel("Dim 1")
    axes[1].set_ylabel("Dim 2")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Chemical space of valid generated molecules", y=1.02)
    savefig(fig, out / "09_chemspace_lambda_score.png")

    coords = sub[[COL["smiles"], COL["lam"], COL["score"], COL["step"]]].copy()
    coords["dim1"] = Z[:, 0]
    coords["dim2"] = Z[:, 1]
    coords.to_csv(out / "tables" / "chemspace_coordinates.csv", index=False)


def plot_lambda_targets(df: pd.DataFrame, out: Path) -> None:
    """Highlight molecules in useful optical windows."""
    m = df["valid"] & df["has_lambda"] & df["clean"]
    sub = df.loc[m].drop_duplicates(COL["smiles"])
    windows = [
        ("UV < 350 nm", sub[COL["lam"]] < 350),
        ("near-UV / violet 350–420", (sub[COL["lam"]] >= 350) & (sub[COL["lam"]] < 420)),
        ("blue-cyan 420–500", (sub[COL["lam"]] >= 420) & (sub[COL["lam"]] < 500)),
        ("green-yellow 500–580", (sub[COL["lam"]] >= 500) & (sub[COL["lam"]] < 580)),
        ("red / NIR ≥ 580", sub[COL["lam"]] >= 580),
    ]
    rows = []
    for name, mask in windows:
        part = sub.loc[mask]
        rows.append(
            {
                "window": name,
                "n": int(mask.sum()),
                "mean_score": float(part[COL["score"]].mean()) if len(part) else np.nan,
                "mean_qed": float(part[COL["qed"]].mean()) if len(part) else np.nan,
                "mean_lambda": float(part[COL["lam"]].mean()) if len(part) else np.nan,
            }
        )
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "tables" / "lambda_windows.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].barh(tab["window"], tab["n"], color="#4c72b0")
    axes[0].set_xlabel("# unique clean molecules")
    axes[0].set_title("Count by λ_abs window")
    axes[0].grid(True, alpha=0.3, axis="x")

    axes[1].hist(sub[COL["lam"]], bins=40, color="#dd8452", edgecolor="white")
    for x, label in [(350, "350"), (420, "420"), (500, "500"), (580, "580")]:
        axes[1].axvline(x, color="k", ls="--", lw=1, alpha=0.5)
        axes[1].text(x, axes[1].get_ylim()[1] * 0.9, label, rotation=90, va="top", fontsize=8)
    axes[1].set_xlabel("λ_abs (nm)")
    axes[1].set_ylabel("count")
    axes[1].set_title("λ_abs histogram (valid, clean, unique)")
    axes[1].grid(True, alpha=0.3, axis="y")
    savefig(fig, out / "10_lambda_windows.png")


def write_summary(df: pd.DataFrame, out: Path) -> None:
    summary = {
        "n_rows": int(len(df)),
        "n_unique_smiles": int(df[COL["smiles"]].nunique()),
        "frac_valid": float(df["valid"].mean()),
        "frac_has_lambda": float(df["has_lambda"].mean()),
        "frac_clean_unwanted": float(df["clean"].mean()),
        "steps": [int(df[COL["step"]].min()), int(df[COL["step"]].max())],
        "score_mean": float(df[COL["score"]].mean()),
        "score_median": float(df[COL["score"]].median()),
        "lambda_nm_mean_positive": float(df.loc[df["has_lambda"], COL["lam"]].mean()),
        "lambda_nm_median_positive": float(df.loc[df["has_lambda"], COL["lam"]].median()),
        "top_score": float(df[COL["score"]].max()),
        "best_smiles": str(df.loc[df[COL["score"]].idxmax(), COL["smiles"]]),
        "best_lambda_nm": float(df.loc[df[COL["score"]].idxmax(), COL["lam"]]),
    }
    (out / "tables" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(out / "tables" / "summary.csv", index=False)

    # human-readable markdown
    lines = [
        "# RL Uni-Mol absorption — summary",
        "",
        f"- Rows: **{summary['n_rows']}** | unique SMILES: **{summary['n_unique_smiles']}**",
        f"- Valid (SMILES_state=1): **{100*summary['frac_valid']:.1f}%**",
        f"- With λ_abs > 0: **{100*summary['frac_has_lambda']:.1f}%**",
        f"- Clean (no unwanted SMARTS): **{100*summary['frac_clean_unwanted']:.1f}%**",
        f"- Steps: {summary['steps'][0]}–{summary['steps'][1]}",
        f"- Score mean/median: {summary['score_mean']:.3f} / {summary['score_median']:.3f}",
        f"- λ_abs (nm) mean/median among positives: "
        f"{summary['lambda_nm_mean_positive']:.1f} / {summary['lambda_nm_median_positive']:.1f}",
        f"- Best score molecule: λ={summary['best_lambda_nm']:.1f} nm, score={summary['top_score']:.3f}",
        "",
        "Plots are numbered `01_…`–`10_…` in this folder; numeric tables in `tables/`.",
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize RL Uni-Mol absorption CSV")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out-dir", type=Path, default=HERE)
    args = p.parse_args()

    out = args.out_dir
    (out / "tables").mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    print(f"[INFO] Loading {args.csv}")
    df = load_df(args.csv)
    print(f"[INFO] {len(df)} rows, steps {df[COL['step']].min()}–{df[COL['step']].max()}")

    write_summary(df, out)
    plot_overview_hists(df, out)
    plot_score_vs_lambda(df, out)
    plot_step_trajectories(df, out)
    plot_lambda_by_step_box(df, out)
    plot_pairwise_properties(df, out)
    plot_nll_panel(df, out)
    plot_top_molecules(df, out)
    plot_chemspace_umap(df, out)
    plot_lambda_targets(df, out)

    print(f"[OK] Results → {out}")
    print(f"     See {out / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
