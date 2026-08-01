"""
Proximity of query molecules (RL and/or chromophore train) to scaffold classes.

Outputs under --out (default chemspace_scaffold_classes/):
  class_proximity/          — RL vs classes
  chromophore_proximity/    — train chromophores vs classes
  08_rl_vs_chromophore_class_proximity.png
  rl_vs_chromophore_proximity.csv

Usage:
  .\\.venv_eval\\Scripts\\python.exe results_rl_unimol_abs\\viz_class_proximity.py \\
      --classes-csv chembl_scaffold_classes_top10_rl.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "epoch_eval"))
sys.path.insert(0, str(ROOT / "results_rl_unimol_abs"))
from utils import DEFAULT_TRAIN, read_smiles, smiles_to_fps  # noqa: E402
from viz_chemspace_compare import DEFAULT_RL_CSV, load_rl_smiles  # noqa: E402
from viz_umap_scaffold_classes import (  # noqa: E402
    OUT_DIR,
    load_scaffold_classes,
    resolve_class_colors,
)

SCORE_EDGES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0001])
BIN_LABELS = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]
DEFAULT_CLASSES = ROOT / "chembl_scaffold_classes_top5_rl.csv"


def max_tanimoto_to_fps(query_fps: list, ref_fps: list, batch: int = 256) -> np.ndarray:
    if not query_fps or not ref_fps:
        return np.full(len(query_fps), np.nan)
    out = np.zeros(len(query_fps), dtype=np.float64)
    for i in range(0, len(query_fps), batch):
        chunk = query_fps[i : i + batch]
        for j, q in enumerate(chunk):
            out[i + j] = max(DataStructs.BulkTanimotoSimilarity(q, ref_fps))
    return out


def mean_topk_tanimoto(query_fps: list, ref_fps: list, k: int = 5) -> np.ndarray:
    if not query_fps or not ref_fps:
        return np.full(len(query_fps), np.nan)
    k = min(k, len(ref_fps))
    out = np.zeros(len(query_fps), dtype=np.float64)
    for i, q in enumerate(query_fps):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(q, ref_fps), dtype=np.float64)
        out[i] = np.mean(np.partition(sims, -k)[-k:])
    return out


def assign_bins(scores: np.ndarray) -> np.ndarray:
    idx = np.digitize(scores, SCORE_EDGES, right=False) - 1
    return np.clip(idx, 0, len(BIN_LABELS) - 1)


def plot_mean_max_tc(
    summary: pd.DataFrame, out_path: Path, colors: dict[str, str], title: str, ylabel: str
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    classes = summary["class"].tolist()
    x = np.arange(len(classes))
    ax.bar(
        x,
        summary["mean_max_tc"],
        color=[colors.get(c, "#333") for c in classes],
        edgecolor="k",
        linewidth=0.4,
        alpha=0.9,
    )
    ax.errorbar(
        x,
        summary["mean_max_tc"],
        yerr=summary["std_max_tc"],
        fmt="none",
        ecolor="k",
        capsize=3,
        lw=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def plot_tc_boxplots(
    tc_long: pd.DataFrame, out_path: Path, colors: dict[str, str], title: str
) -> None:
    classes = sorted(tc_long["class"].unique())
    data = [tc_long.loc[tc_long["class"] == c, "max_tc"].to_numpy() for c in classes]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    bp = ax.boxplot(
        data,
        tick_labels=classes,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="k", lw=1.2),
    )
    for patch, c in zip(bp["boxes"], classes):
        patch.set_facecolor(colors.get(c, "#888"))
        patch.set_alpha(0.8)
    ax.set_ylabel("Max Tanimoto to class")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def plot_nearest_class_bar(
    assign: pd.DataFrame, out_path: Path, colors: dict[str, str], title: str, ylabel: str
) -> None:
    counts = assign["nearest_class"].value_counts()
    classes = [c for c in colors if c in counts.index] + [
        c for c in counts.index if c not in colors
    ]
    vals = [int(counts.get(c, 0)) for c in classes]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        np.arange(len(classes)),
        vals,
        color=[colors.get(c, "#333") for c in classes],
        edgecolor="k",
        linewidth=0.4,
    )
    total = sum(vals) or 1
    for i, v in enumerate(vals):
        ax.text(i, v, f"{100 * v / total:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def plot_score_class_heatmap(mat: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_xlabel("Scaffold class")
    ax.set_ylabel("RL Score bin")
    ax.set_title("Mean max Tanimoto by Score bin x class")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="w")
    fig.colorbar(im, ax=ax, label="Mean max Tc")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def plot_umap_nn_distances(
    summary: pd.DataFrame, out_path: Path, colors: dict[str, str], title: str
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    classes = summary["class"].tolist()
    x = np.arange(len(classes))
    w = 0.38
    ax.bar(
        x - w / 2,
        summary["mean_umap_nn_dist"],
        w,
        color=[colors.get(c, "#333") for c in classes],
        edgecolor="k",
        linewidth=0.3,
        label="mean NN UMAP dist",
        alpha=0.9,
    )
    ax.bar(
        x + w / 2,
        summary["umap_centroid_dist"],
        w,
        color=[colors.get(c, "#333") for c in classes],
        edgecolor="k",
        linewidth=0.3,
        alpha=0.45,
        hatch="//",
        label="centroid-centroid UMAP dist",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Distance in UMAP space")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def plot_umap_colored_by_nearest(
    coords: pd.DataFrame,
    assign: pd.DataFrame,
    out_path: Path,
    colors: dict[str, str],
    query_label: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    bg = coords[coords["label"] != query_label]
    ax.scatter(bg["umap1"], bg["umap2"], c="#dddddd", s=4, alpha=0.25, linewidths=0, zorder=1)

    merged = coords[coords["label"] == query_label].merge(
        assign[["smiles", "nearest_class", "nearest_tc"]], on="smiles", how="left"
    )
    for cls in sorted(colors):
        m = merged["nearest_class"] == cls
        if not m.any():
            continue
        ax.scatter(
            merged.loc[m, "umap1"],
            merged.loc[m, "umap2"],
            c=colors[cls],
            s=18 if query_label == "chromophore" else 22,
            alpha=0.75 if query_label == "chromophore" else 0.85,
            edgecolors="k",
            linewidths=0.15,
            label=f"{cls} (n={m.sum()})",
            zorder=3,
        )
    ax.legend(fontsize=8, loc="best", title="Nearest class (max Tc)")
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def plot_topk_vs_max(summary: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    classes = summary["class"].tolist()
    x = np.arange(len(classes))
    w = 0.38
    ax.bar(x - w / 2, summary["mean_max_tc"], w, label="mean max Tc", color="#1565C0", edgecolor="k", lw=0.3)
    ax.bar(x + w / 2, summary["mean_topk5_tc"], w, label="mean top-5 Tc", color="#FF8F00", edgecolor="k", lw=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Tanimoto")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def plot_rl_vs_chromophore_compare(rl_sum: pd.DataFrame, chromo_sum: pd.DataFrame, out_path: Path) -> None:
    merged = (
        rl_sum[["class", "mean_max_tc"]]
        .rename(columns={"mean_max_tc": "rl"})
        .merge(
            chromo_sum[["class", "mean_max_tc"]].rename(columns={"mean_max_tc": "chromophore"}),
            on="class",
        )
        .sort_values("rl", ascending=False)
    )
    classes = merged["class"].tolist()
    x = np.arange(len(classes))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x - w / 2, merged["rl"], w, label="RL generated", color="#C51162", edgecolor="k", lw=0.3)
    ax.bar(
        x + w / 2,
        merged["chromophore"],
        w,
        label="Chromophore train",
        color="#1B5E20",
        edgecolor="k",
        lw=0.3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Mean max Tanimoto to class")
    ax.set_title("RL vs training chromophores: proximity to scaffold classes")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def analyze_query(
    *,
    query_name: str,
    query_smiles: list[str],
    query_scores: np.ndarray | None,
    class_fps: dict[str, list],
    ru_map: dict[str, str],
    coords: pd.DataFrame | None,
    coords_label: str,
    out: Path,
    radius: int,
    n_bits: int,
) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)

    print(f"[INFO] Fingerprints for query={query_name}…")
    q_valid, q_fps, _ = smiles_to_fps(query_smiles, radius=radius, n_bits=n_bits, unique=True)

    if query_scores is None:
        q_scores = np.full(len(q_valid), np.nan)
    else:
        score_map: dict[str, float] = {}
        for smi, sc in zip(query_smiles, query_scores):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                continue
            canon = Chem.MolToSmiles(mol, canonical=True)
            score_map.setdefault(canon, float(sc))
        q_scores = np.array([score_map.get(s, np.nan) for s in q_valid], dtype=float)

    print(f"[INFO] Tanimoto {query_name} -> classes…")
    max_tc: dict[str, np.ndarray] = {}
    topk_tc: dict[str, np.ndarray] = {}
    long_rows = []
    for cls, fps in class_fps.items():
        print(f"  -> {cls}")
        mx = max_tanimoto_to_fps(q_fps, fps)
        tk = mean_topk_tanimoto(q_fps, fps, k=5)
        max_tc[cls] = mx
        topk_tc[cls] = tk
        for smi, sc, v, t5 in zip(q_valid, q_scores, mx, tk):
            long_rows.append({"smiles": smi, "score": sc, "class": cls, "max_tc": v, "topk5_tc": t5})

    tc_long = pd.DataFrame(long_rows)
    classes = sorted(class_fps.keys())
    stack = np.column_stack([max_tc[c] for c in classes])
    nearest_idx = np.argmax(stack, axis=1)
    nearest_tc = np.max(stack, axis=1)
    assign = pd.DataFrame(
        {
            "smiles": q_valid,
            "score": q_scores,
            "nearest_class": [classes[i] for i in nearest_idx],
            "nearest_tc": nearest_tc,
        }
    )
    for cls in classes:
        assign[f"max_tc_{cls}"] = max_tc[cls]
        assign[f"topk5_tc_{cls}"] = topk_tc[cls]

    assign.to_csv(out / "tables" / f"{query_name}_tanimoto_to_classes.csv", index=False)
    tc_long.to_csv(out / "tables" / f"{query_name}_tanimoto_long.csv", index=False)

    rows = []
    for cls in classes:
        rows.append(
            {
                "class": cls,
                "core_name_ru": ru_map.get(cls, ""),
                "n_ref": len(class_fps[cls]),
                "mean_max_tc": float(np.mean(max_tc[cls])),
                "median_max_tc": float(np.median(max_tc[cls])),
                "std_max_tc": float(np.std(max_tc[cls])),
                "mean_topk5_tc": float(np.mean(topk_tc[cls])),
                "frac_nearest": float((assign["nearest_class"] == cls).mean()),
                "n_nearest": int((assign["nearest_class"] == cls).sum()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("mean_max_tc", ascending=False)

    if coords is not None and (coords["label"] == coords_label).any():
        print(f"[INFO] UMAP distances for {query_name}…")
        Z_q = coords.loc[coords["label"] == coords_label, ["umap1", "umap2", "smiles"]].copy()
        Z_q = Z_q.set_index("smiles").reindex(q_valid)
        q_xy = Z_q[["umap1", "umap2"]].to_numpy(dtype=float)
        umap_nn, umap_cent = {}, {}
        q_cent = np.nanmean(q_xy, axis=0)
        for cls in classes:
            xy = coords.loc[coords["label"] == cls, ["umap1", "umap2"]].to_numpy(dtype=float)
            if len(xy) == 0 or np.isnan(q_xy).all():
                umap_nn[cls] = np.nan
                umap_cent[cls] = np.nan
                continue
            d = cdist(q_xy, xy)
            umap_nn[cls] = float(np.nanmean(np.nanmin(d, axis=1)))
            umap_cent[cls] = float(np.linalg.norm(q_cent - xy.mean(axis=0)))
        summary["mean_umap_nn_dist"] = summary["class"].map(umap_nn)
        summary["umap_centroid_dist"] = summary["class"].map(umap_cent)

    summary.to_csv(out / "tables" / "class_proximity_summary.csv", index=False)

    colors = resolve_class_colors(classes)
    nice = "RL generated" if query_name == "rl" else "Chromophore train"

    plot_mean_max_tc(
        summary,
        out / "01_mean_max_tanimoto_by_class.png",
        colors,
        title=f"{nice}: mean max Tanimoto to scaffold classes",
        ylabel=f"Mean max Tanimoto ({query_name} -> class)",
    )
    plot_tc_boxplots(
        tc_long,
        out / "02_max_tanimoto_boxplots.png",
        colors,
        title=f"{nice}: max Tanimoto distributions",
    )
    plot_nearest_class_bar(
        assign,
        out / "03_nearest_class_assignment.png",
        colors,
        title=f"{nice}: nearest scaffold class (max Tc)",
        ylabel=f"# {query_name} molecules",
    )
    plot_topk_vs_max(summary, out / "04_max_vs_topk5_tanimoto.png", title=f"{nice}: max vs top-5 Tc")

    if np.isfinite(q_scores).any():
        bins = assign_bins(q_scores)
        heat = []
        for b, blab in enumerate(BIN_LABELS):
            m = bins == b
            row = {"score_bin": blab}
            for cls in classes:
                row[cls] = float(np.mean(max_tc[cls][m])) if m.any() else np.nan
            heat.append(row)
        heat_df = pd.DataFrame(heat).set_index("score_bin")[classes]
        heat_df.to_csv(out / "tables" / "mean_max_tc_by_score_bin.csv")
        plot_score_class_heatmap(heat_df, out / "05_score_bin_x_class_heatmap.png")

    if "mean_umap_nn_dist" in summary.columns:
        plot_umap_nn_distances(
            summary,
            out / "06_umap_distance_to_classes.png",
            colors,
            title=f"{nice}: UMAP distance to class clouds",
        )
    if coords is not None:
        plot_umap_colored_by_nearest(
            coords,
            assign,
            out / f"07_umap_{query_name}_by_nearest_class.png",
            colors,
            query_label=coords_label,
            title=f"{nice} colored by nearest scaffold class (Tanimoto)",
        )

    top3 = summary.head(3)
    lines = [f"# Class proximity — {query_name}", "", f"Query molecules: **{len(q_valid)}**", "", "## Top classes", ""]
    for _, r in top3.iterrows():
        lines.append(
            f"- **{r['class']}** ({r['core_name_ru']}): mean max-Tc={r['mean_max_tc']:.3f}, "
            f"nearest-share={100 * r['frac_nearest']:.1f}%"
        )
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] {query_name} -> {out}")
    print(summary[["class", "mean_max_tc", "frac_nearest"]].to_string(index=False))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--classes-csv", type=Path, default=DEFAULT_CLASSES)
    p.add_argument("--rl-csv", type=Path, default=DEFAULT_RL_CSV)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--coords", type=Path, default=OUT_DIR / "coordinates_umap.csv")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--max-per-class", type=int, default=800)
    p.add_argument("--max-chromo", type=int, default=3500)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("[INFO] Loading scaffold classes…")
    class_smi, class_lb, ru_map = load_scaffold_classes(
        args.classes_csv, max_per_class=args.max_per_class, seed=args.seed
    )
    class_fps: dict[str, list] = {}
    for cls in sorted(set(class_lb)):
        smis = [s for s, lb in zip(class_smi, class_lb) if lb == cls]
        _, fps, _ = smiles_to_fps(smis, radius=args.radius, n_bits=args.n_bits, unique=True)
        class_fps[cls] = fps
        print(f"  {cls}: {len(fps)} FP")

    coords = pd.read_csv(args.coords) if args.coords.is_file() else None

    rl_smi, rl_scores = load_rl_smiles(args.rl_csv, max_n=None, seed=args.seed)
    if coords is not None and (coords["label"] == "rl").any():
        rl_list = coords.loc[coords["label"] == "rl", "smiles"].tolist()
        score_map = dict(zip(rl_smi, rl_scores))
        rl_sc = np.array([score_map.get(s, np.nan) for s in rl_list], dtype=float)
    else:
        rl_list, rl_sc = rl_smi, rl_scores

    rl_sum = analyze_query(
        query_name="rl",
        query_smiles=rl_list,
        query_scores=rl_sc,
        class_fps=class_fps,
        ru_map=ru_map,
        coords=coords,
        coords_label="rl",
        out=args.out / "class_proximity",
        radius=args.radius,
        n_bits=args.n_bits,
    )

    chromo = read_smiles(args.train, max_n=args.max_chromo, seed=args.seed)
    if coords is not None and (coords["label"] == "chromophore").any():
        chromo_list = coords.loc[coords["label"] == "chromophore", "smiles"].tolist()
    else:
        chromo_list = chromo

    chromo_sum = analyze_query(
        query_name="chromophore",
        query_smiles=chromo_list,
        query_scores=None,
        class_fps=class_fps,
        ru_map=ru_map,
        coords=coords,
        coords_label="chromophore",
        out=args.out / "chromophore_proximity",
        radius=args.radius,
        n_bits=args.n_bits,
    )

    plot_rl_vs_chromophore_compare(
        rl_sum, chromo_sum, args.out / "08_rl_vs_chromophore_class_proximity.png"
    )
    compare = rl_sum[["class", "mean_max_tc", "frac_nearest"]].rename(
        columns={"mean_max_tc": "rl_mean_max_tc", "frac_nearest": "rl_frac_nearest"}
    ).merge(
        chromo_sum[["class", "mean_max_tc", "frac_nearest"]].rename(
            columns={"mean_max_tc": "chromo_mean_max_tc", "frac_nearest": "chromo_frac_nearest"}
        ),
        on="class",
    )
    compare.to_csv(args.out / "rl_vs_chromophore_proximity.csv", index=False)
    print(f"[DONE] all proximity -> {args.out}")


if __name__ == "__main__":
    main()
