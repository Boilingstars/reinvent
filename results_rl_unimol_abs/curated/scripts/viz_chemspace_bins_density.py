"""
Score-bin distance analysis + density chemspace views for PCA / UMAP / t-SNE.

Uses embeddings already saved by viz_chemspace_compare.py.

Outputs → results_rl_unimol_abs/chemspace_compare/bins_density/
  tables/distance_by_score_bin.csv
  01_distance_to_refs_by_bin.png
  02_centroid_distance_by_bin.png
  03_{pca,umap,tsne}_bins_in_space.png
  04_{pca,umap,tsne}_density_refs_rl_points.png
  05_density_side_by_side.png

Usage (from repo root):
  .\\.venv_eval\\Scripts\\python.exe results_rl_unimol_abs\\viz_chemspace_bins_density.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "results_rl_unimol_abs"))
from viz_chemspace_compare import DEFAULT_RL_CSV, OUT_DIR, load_rl_smiles  # noqa: E402

METHODS = (
    ("pca", "coordinates_pca.csv", "PCA"),
    ("umap", "coordinates_umap.csv", "UMAP"),
    ("tsne", "coordinates_tsne.csv", "t-SNE"),
)

# Fixed Score bins for comparable tables across methods
SCORE_EDGES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0001])
BIN_LABELS = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]
BIN_COLORS = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]  # plasma-like


def assign_bins(scores: np.ndarray) -> np.ndarray:
    idx = np.digitize(scores, SCORE_EDGES, right=False) - 1
    idx = np.clip(idx, 0, len(BIN_LABELS) - 1)
    return idx


def kde_field(xy: np.ndarray, grid_n: int = 120):
    if len(xy) < 8:
        return None
    try:
        kde = gaussian_kde(xy.T)
    except Exception:
        return None
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    pad_x = 0.10 * (xmax - xmin + 1e-9)
    pad_y = 0.10 * (ymax - ymin + 1e-9)
    xs = np.linspace(xmin - pad_x, xmax + pad_x, grid_n)
    ys = np.linspace(ymin - pad_y, ymax + pad_y, grid_n)
    xx, yy = np.meshgrid(xs, ys)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, zz


def load_embedding(path: Path, score_map: dict[str, float]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["score"] = [
        score_map.get(s, np.nan) if set_name == "rl" else np.nan
        for s, set_name in zip(df["smiles"], df["set"])
    ]
    return df


def analyze_distances(df: pd.DataFrame, method: str) -> pd.DataFrame:
    Z = df[["dim1", "dim2"]].to_numpy(dtype=float)
    sets = df["set"].to_numpy()

    chromo = Z[sets == "chromophore"]
    chembl = Z[sets == "chembl"]
    rl_m = sets == "rl"
    rl_Z = Z[rl_m]
    rl_scores = df.loc[rl_m, "score"].to_numpy(dtype=float)

    if len(rl_Z) == 0:
        return pd.DataFrame()

    bins = assign_bins(rl_scores)
    c_chromo = chromo.mean(axis=0) if len(chromo) else np.array([np.nan, np.nan])
    c_chembl = chembl.mean(axis=0) if len(chembl) else np.array([np.nan, np.nan])

    # nearest-neighbor distances (to ref clouds)
    d_nn_chromo = (
        cdist(rl_Z, chromo).min(axis=1) if len(chromo) else np.full(len(rl_Z), np.nan)
    )
    d_nn_chembl = (
        cdist(rl_Z, chembl).min(axis=1) if len(chembl) else np.full(len(rl_Z), np.nan)
    )
    d_cent_chromo = np.linalg.norm(rl_Z - c_chromo, axis=1)
    d_cent_chembl = np.linalg.norm(rl_Z - c_chembl, axis=1)

    rows = []
    for b, label in enumerate(BIN_LABELS):
        m = bins == b
        n = int(m.sum())
        if n == 0:
            rows.append(
                {
                    "method": method,
                    "score_bin": label,
                    "bin_mid": 0.5 * (SCORE_EDGES[b] + min(SCORE_EDGES[b + 1], 1.0)),
                    "n": 0,
                    "mean_score": np.nan,
                    "mean_nn_dist_to_chromophore": np.nan,
                    "median_nn_dist_to_chromophore": np.nan,
                    "mean_nn_dist_to_chembl": np.nan,
                    "median_nn_dist_to_chembl": np.nan,
                    "mean_dist_to_chromophore_centroid": np.nan,
                    "mean_dist_to_chembl_centroid": np.nan,
                    "bin_centroid_dist_to_chromophore_centroid": np.nan,
                    "bin_centroid_dist_to_chembl_centroid": np.nan,
                    "centroid_dim1": np.nan,
                    "centroid_dim2": np.nan,
                }
            )
            continue

        cent = rl_Z[m].mean(axis=0)
        rows.append(
            {
                "method": method,
                "score_bin": label,
                "bin_mid": 0.5 * (SCORE_EDGES[b] + min(SCORE_EDGES[b + 1], 1.0)),
                "n": n,
                "mean_score": float(np.nanmean(rl_scores[m])),
                "mean_nn_dist_to_chromophore": float(np.nanmean(d_nn_chromo[m])),
                "median_nn_dist_to_chromophore": float(np.nanmedian(d_nn_chromo[m])),
                "mean_nn_dist_to_chembl": float(np.nanmean(d_nn_chembl[m])),
                "median_nn_dist_to_chembl": float(np.nanmedian(d_nn_chembl[m])),
                "mean_dist_to_chromophore_centroid": float(np.nanmean(d_cent_chromo[m])),
                "mean_dist_to_chembl_centroid": float(np.nanmean(d_cent_chembl[m])),
                "bin_centroid_dist_to_chromophore_centroid": float(
                    np.linalg.norm(cent - c_chromo)
                ),
                "bin_centroid_dist_to_chembl_centroid": float(
                    np.linalg.norm(cent - c_chembl)
                ),
                "centroid_dim1": float(cent[0]),
                "centroid_dim2": float(cent[1]),
            }
        )
    return pd.DataFrame(rows)


def plot_distance_bars(table: pd.DataFrame, out_path: Path) -> None:
    methods = [m for m, _, _ in METHODS if m in set(table["method"])]
    fig, axes = plt.subplots(1, len(methods), figsize=(5.0 * len(methods), 4.6), sharey=False)
    if len(methods) == 1:
        axes = [axes]

    x = np.arange(len(BIN_LABELS))
    width = 0.36
    for ax, method in zip(axes, methods):
        sub = table[table["method"] == method].set_index("score_bin").reindex(BIN_LABELS)
        ax.bar(
            x - width / 2,
            sub["mean_nn_dist_to_chromophore"],
            width,
            color="#2ca02c",
            alpha=0.85,
            label="mean NN → chromophore",
        )
        ax.bar(
            x + width / 2,
            sub["mean_nn_dist_to_chembl"],
            width,
            color="#9e9e9e",
            alpha=0.85,
            label="mean NN → ChEMBL",
        )
        for i, n in enumerate(sub["n"].fillna(0).astype(int)):
            ax.text(i, 0.02, f"n={n}", ha="center", va="bottom", fontsize=7, transform=ax.get_xaxis_transform())
        title = next(t for m, _, t in METHODS if m == method)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Mean nearest-neighbor distance (embedding units)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Distance of RL Score-bins to reference clouds", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def plot_centroid_distances(table: pd.DataFrame, out_path: Path) -> None:
    methods = [m for m, _, _ in METHODS if m in set(table["method"])]
    fig, axes = plt.subplots(1, len(methods), figsize=(5.0 * len(methods), 4.4), sharey=False)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        sub = table[table["method"] == method].set_index("score_bin").reindex(BIN_LABELS)
        ax.plot(
            BIN_LABELS,
            sub["bin_centroid_dist_to_chromophore_centroid"],
            "o-",
            color="#2ca02c",
            lw=2,
            label="bin centroid → chromophore centroid",
        )
        ax.plot(
            BIN_LABELS,
            sub["bin_centroid_dist_to_chembl_centroid"],
            "s--",
            color="#616161",
            lw=2,
            label="bin centroid → ChEMBL centroid",
        )
        title = next(t for m, _, t in METHODS if m == method)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("Centroid–centroid distance")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("Score-bin centroids vs reference centroids", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def plot_bins_in_space(df: pd.DataFrame, method_title: str, out_path: Path) -> None:
    Z = df[["dim1", "dim2"]].to_numpy(dtype=float)
    sets = df["set"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 7))

    for name, color, alpha, s, z in (
        ("chembl", "#bdbdbd", 0.20, 8, 1),
        ("chromophore", "#a5d6a7", 0.25, 10, 2),
    ):
        m = sets == name
        ax.scatter(Z[m, 0], Z[m, 1], c=color, s=s, alpha=alpha, linewidths=0, zorder=z, label=name)

    rl_m = sets == "rl"
    rl_Z = Z[rl_m]
    scores = df.loc[rl_m, "score"].to_numpy(dtype=float)
    bins = assign_bins(scores)

    for b, label in enumerate(BIN_LABELS):
        m = bins == b
        if not m.any():
            continue
        ax.scatter(
            rl_Z[m, 0],
            rl_Z[m, 1],
            c=[BIN_COLORS[b]],
            s=28,
            alpha=0.8,
            edgecolors="k",
            linewidths=0.2,
            label=f"RL Score {label}",
            zorder=4,
        )
        cent = rl_Z[m].mean(axis=0)
        ax.scatter(
            [cent[0]],
            [cent[1]],
            c=[BIN_COLORS[b]],
            s=160,
            marker="X",
            edgecolors="k",
            linewidths=0.8,
            zorder=6,
        )

    ax.set_title(f"{method_title}: RL Score bins (X = bin centroids)")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def plot_density_with_rl(
    df: pd.DataFrame,
    method_title: str,
    out_path: Path,
    color_rl_by_score: bool = True,
) -> None:
    Z = df[["dim1", "dim2"]].to_numpy(dtype=float)
    sets = df["set"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 7))

    # Shared axis limits from all points
    pad = 0.05
    xmin, xmax = Z[:, 0].min(), Z[:, 0].max()
    ymin, ymax = Z[:, 1].min(), Z[:, 1].max()
    dx, dy = xmax - xmin, ymax - ymin

    for name, cmap, levels in (
        ("chembl", "Greys", 8),
        ("chromophore", "Greens", 8),
    ):
        m = sets == name
        grid = kde_field(Z[m])
        if grid is None:
            continue
        xx, yy, zz = grid
        # filled density region
        ax.contourf(xx, yy, zz, levels=levels, cmap=cmap, alpha=0.35)
        ax.contour(xx, yy, zz, levels=levels, colors="k", linewidths=0.35, alpha=0.25)

    rl_m = sets == "rl"
    rl_Z = Z[rl_m]
    if color_rl_by_score:
        scores = df.loc[rl_m, "score"].to_numpy(dtype=float)
        sc = ax.scatter(
            rl_Z[:, 0],
            rl_Z[:, 1],
            c=scores,
            cmap="plasma",
            s=26,
            alpha=0.85,
            edgecolors="k",
            linewidths=0.25,
            zorder=5,
        )
        fig.colorbar(sc, ax=ax, label="RL Score")
    else:
        ax.scatter(
            rl_Z[:, 0],
            rl_Z[:, 1],
            c="#d62728",
            s=26,
            alpha=0.8,
            edgecolors="k",
            linewidths=0.25,
            zorder=5,
            label="RL generated",
        )

    legend_extra = [
        Patch(facecolor="#9e9e9e", edgecolor="k", alpha=0.45, label="ChEMBL density"),
        Patch(facecolor="#2ca02c", edgecolor="k", alpha=0.45, label="Chromophore density"),
    ]
    ax.legend(handles=legend_extra, loc="best", fontsize=9)
    ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
    ax.set_ylim(ymin - pad * dy, ymax + pad * dy)
    ax.set_title(f"{method_title}: reference densities + RL points")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def plot_density_side_by_side(dfs: dict[str, pd.DataFrame], out_path: Path) -> None:
    keys = [m for m, _, _ in METHODS if m in dfs]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.4 * len(keys), 5.0))
    if len(keys) == 1:
        axes = [axes]

    for ax, method in zip(axes, keys):
        df = dfs[method]
        title = next(t for m, _, t in METHODS if m == method)
        Z = df[["dim1", "dim2"]].to_numpy(dtype=float)
        sets = df["set"].to_numpy()
        for name, cmap, levels in (
            ("chembl", "Greys", 7),
            ("chromophore", "Greens", 7),
        ):
            grid = kde_field(Z[sets == name], grid_n=100)
            if grid is None:
                continue
            xx, yy, zz = grid
            ax.contourf(xx, yy, zz, levels=levels, cmap=cmap, alpha=0.35)
            ax.contour(xx, yy, zz, levels=levels, colors="k", linewidths=0.3, alpha=0.2)

        rl_m = sets == "rl"
        scores = df.loc[rl_m, "score"].to_numpy(dtype=float)
        sc = ax.scatter(
            Z[rl_m, 0],
            Z[rl_m, 1],
            c=scores,
            cmap="plasma",
            s=14,
            alpha=0.8,
            edgecolors="none",
            zorder=5,
        )
        ax.set_title(title)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.grid(True, alpha=0.2)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Score")

    legend_extra = [
        Patch(facecolor="#9e9e9e", edgecolor="k", alpha=0.45, label="ChEMBL density"),
        Patch(facecolor="#2ca02c", edgecolor="k", alpha=0.45, label="Chromophore density"),
    ]
    fig.legend(handles=legend_extra, loc="upper center", ncol=2, frameon=True)
    fig.suptitle("Reference density fields + RL generated points", y=1.04, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--coords-dir", type=Path, default=OUT_DIR)
    p.add_argument("--rl-csv", type=Path, default=DEFAULT_RL_CSV)
    p.add_argument("--out", type=Path, default=OUT_DIR / "bins_density")
    args = p.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)

    print("[INFO] Loading RL scores…")
    rl_smi, rl_scores = load_rl_smiles(args.rl_csv, max_n=None, seed=42)
    score_map = dict(zip(rl_smi, rl_scores))

    tables = []
    dfs: dict[str, pd.DataFrame] = {}

    for method, csv_name, title in METHODS:
        path = args.coords_dir / csv_name
        if not path.is_file():
            print(f"[WARN] missing {path} — skip {method}")
            continue
        print(f"[INFO] {title}: distances + density…")
        df = load_embedding(path, score_map)
        dfs[method] = df
        tab = analyze_distances(df, method)
        tables.append(tab)
        plot_bins_in_space(df, title, out / f"03_{method}_bins_in_space.png")
        plot_density_with_rl(df, title, out / f"04_{method}_density_refs_rl_points.png")

    if not tables:
        raise SystemExit("No embeddings found. Run viz_chemspace_compare.py first.")

    table = pd.concat(tables, ignore_index=True)
    table_path = out / "tables" / "distance_by_score_bin.csv"
    table.to_csv(table_path, index=False)
    print(f"[OK] {table_path}")

    # Wide pivot for quick reading
    for metric in (
        "mean_nn_dist_to_chromophore",
        "mean_nn_dist_to_chembl",
        "bin_centroid_dist_to_chromophore_centroid",
        "bin_centroid_dist_to_chembl_centroid",
        "n",
    ):
        wide = table.pivot(index="score_bin", columns="method", values=metric)
        wide = wide.reindex(BIN_LABELS)
        wide.to_csv(out / "tables" / f"pivot_{metric}.csv")

    plot_distance_bars(table, out / "01_distance_to_refs_by_bin.png")
    plot_centroid_distances(table, out / "02_centroid_distance_by_bin.png")
    plot_density_side_by_side(dfs, out / "05_density_side_by_side.png")

    # Markdown summary table (NN distances)
    lines = [
        "# Score-bin distances in chemical space",
        "",
        "Bins: " + ", ".join(BIN_LABELS),
        "",
        "Mean nearest-neighbor distance of RL molecules in each Score bin "
        "to the nearest chromophore / ChEMBL molecule in the embedding.",
        "",
    ]
    for method, _, title in METHODS:
        sub = table[table["method"] == method]
        if sub.empty:
            continue
        lines.append(f"## {title}")
        lines.append("")
        lines.append(
            "| Score bin | n | mean Score | mean NN to chromo | mean NN to ChEMBL | "
            "centroid to chromo | centroid to ChEMBL |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['score_bin']} | {int(r['n'])} | {r['mean_score']:.3f} | "
                f"{r['mean_nn_dist_to_chromophore']:.3f} | {r['mean_nn_dist_to_chembl']:.3f} | "
                f"{r['bin_centroid_dist_to_chromophore_centroid']:.3f} | "
                f"{r['bin_centroid_dist_to_chembl_centroid']:.3f} |"
            )
        lines.append("")
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] → {out}")


if __name__ == "__main__":
    main()
