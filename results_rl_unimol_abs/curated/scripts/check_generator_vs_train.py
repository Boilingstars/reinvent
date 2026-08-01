"""
Проверка генератора REINVENT относительно тренировочного набора
при заданных условиях (λ-окно, Score, validity, clean).

Сравнивает сгенерированные молекулы (RL CSV) с data/train.smi:
  - novelty (exact + max Tanimoto to train)
  - попадание в условия
  - химпространство (PCA / UMAP): train + RL, цвет = Score

Outputs → results_rl_unimol_abs/curated/generator_check/

Usage (from repo root):
  .\\.venv_eval\\Scripts\\python.exe results_rl_unimol_abs\\curated\\scripts\\check_generator_vs_train.py

  # свои условия:
  .\\.venv_eval\\Scripts\\python.exe results_rl_unimol_abs\\curated\\scripts\\check_generator_vs_train.py \\
      --lambda-min 450 --lambda-max 480 --min-score 0.8 --require-clean
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from rdkit import DataStructs
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "epoch_eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (  # noqa: E402
    canonicalize,
    fps_to_numpy,
    read_smiles,
    smiles_to_fps,
)

try:
    from umap import UMAP

    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

CURATED = Path(__file__).resolve().parents[1]
DEFAULT_RL = CURATED / "data" / "rl_unimol_abs_1.csv"
DEFAULT_TRAIN = CURATED / "data" / "refs" / "train.smi"
if not DEFAULT_TRAIN.is_file():
    DEFAULT_TRAIN = ROOT / "data" / "train.smi"
DEFAULT_OUT = CURATED / "generator_check"


def set_style() -> None:
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


def load_rl(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["valid"] = df["SMILES_state"] == 1 if "SMILES_state" in df.columns else True
    lam = df["lambda_abs (raw)"] if "lambda_abs (raw)" in df.columns else df.get("lambda_abs", 0)
    df["lambda_nm"] = pd.to_numeric(lam, errors="coerce").fillna(0.0)
    df["score"] = pd.to_numeric(df["Score"], errors="coerce").fillna(0.0)
    qed = df["QED (raw)"] if "QED (raw)" in df.columns else df.get("QED", np.nan)
    sa = df["SA score (raw)"] if "SA score (raw)" in df.columns else df.get("SA score", np.nan)
    df["qed"] = pd.to_numeric(qed, errors="coerce")
    df["sa"] = pd.to_numeric(sa, errors="coerce")
    if "Unwanted SMARTS (raw)" in df.columns:
        df["clean"] = (pd.to_numeric(df["Unwanted SMARTS (raw)"], errors="coerce") >= 0.999) & df["valid"]
    elif "Unwanted SMARTS" in df.columns:
        df["clean"] = (pd.to_numeric(df["Unwanted SMARTS"], errors="coerce") >= 0.999) & df["valid"]
    else:
        df["clean"] = df["valid"]
    df["canon"] = df["SMILES"].map(lambda s: canonicalize(str(s)))
    return df


def apply_conditions(
    df: pd.DataFrame,
    *,
    lambda_min: float | None,
    lambda_max: float | None,
    min_score: float | None,
    max_score: float | None,
    require_valid: bool,
    require_clean: bool,
    require_lambda: bool,
    max_sa: float | None = None,
    min_qed: float | None = None,
) -> pd.Series:
    m = pd.Series(True, index=df.index)
    if require_valid:
        m &= df["valid"]
    if require_clean:
        m &= df["clean"]
    if require_lambda:
        m &= df["lambda_nm"] > 0
    if lambda_min is not None:
        m &= df["lambda_nm"] >= lambda_min
    if lambda_max is not None:
        m &= df["lambda_nm"] <= lambda_max
    if min_score is not None:
        m &= df["score"] >= min_score
    if max_score is not None:
        m &= df["score"] <= max_score
    if max_sa is not None:
        m &= df["sa"].notna() & (df["sa"] <= max_sa)
    if min_qed is not None:
        m &= df["qed"].notna() & (df["qed"] >= min_qed)
    return m


def max_tc_to_refs(query_fps: list, ref_fps: list, chunk: int = 256) -> np.ndarray:
    """Max Tanimoto of each query FP against reference set."""
    out = np.zeros(len(query_fps), dtype=float)
    if not ref_fps or not query_fps:
        return out
    for i in range(0, len(query_fps), chunk):
        qs = query_fps[i : i + chunk]
        # BulkTanimotoSimilarity(query, refs) for each query
        for j, q in enumerate(qs):
            sims = DataStructs.BulkTanimotoSimilarity(q, ref_fps)
            out[i + j] = float(max(sims)) if sims else 0.0
    return out


def fit_pca_umap(X: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray | None, str]:
    pca = PCA(n_components=2, random_state=seed)
    Z_pca = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_
    tag = f"PCA EVR {100 * ev[0]:.1f}% / {100 * ev[1]:.1f}%"
    Z_umap = None
    if HAVE_UMAP:
        n = X.shape[0]
        Xp = X
        if X.shape[1] > 50 and n > 51:
            Xp = PCA(n_components=min(50, n - 1), random_state=seed).fit_transform(X)
        reducer = UMAP(
            n_components=2,
            n_neighbors=min(30, max(5, n // 25)),
            min_dist=0.15,
            metric="euclidean",
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z_umap = reducer.fit_transform(Xp)
    return Z_pca, Z_umap, tag


def plot_chemspace(
    Z: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    title: str,
    out_path: Path,
    cond_mask: np.ndarray | None = None,
) -> None:
    """Train = gray; RL colored by Score. Passers shown on a separate panel (no overlays)."""
    train_m = labels == "train"
    rl_m = labels == "rl"
    has_pass = cond_mask is not None and cond_mask.any()
    n_panels = 2 if has_pass else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(11 if n_panels == 1 else 14, 6.5), sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]

    def _panel(ax, show_mask: np.ndarray | None, panel_title: str) -> None:
        ax.scatter(
            Z[train_m, 0],
            Z[train_m, 1],
            c="#cfd8dc",
            s=8,
            alpha=0.28,
            linewidths=0,
            zorder=1,
            label=f"Train (n={int(train_m.sum())})",
        )
        if show_mask is None:
            m = rl_m
            lab = "RL (all, color=Score)"
        else:
            m = show_mask
            lab = f"Pass conditions (n={int(m.sum())})"
        sc = ax.scatter(
            Z[m, 0],
            Z[m, 1],
            c=scores[m],
            cmap="plasma",
            s=26,
            alpha=0.9,
            edgecolors="k",
            linewidths=0.2,
            norm=Normalize(vmin=0.0, vmax=1.0),
            zorder=3,
            label=lab,
        )
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="Score")
        ax.set_title(panel_title, fontsize=11)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.legend(loc="best", fontsize=8, framealpha=0.95)

    _panel(axes[0], None, "All RL vs train")
    if has_pass:
        _panel(axes[1], cond_mask, f"Only pass conditions (n={int(cond_mask.sum())})")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def plot_score_colored_props(df: pd.DataFrame, cond: pd.Series, out: Path, cond_label: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Score hist
    ax = axes[0, 0]
    ax.hist(df.loc[df["valid"], "score"], bins=40, color="#90a4ae", alpha=0.7, label="valid all")
    if cond.any():
        ax.hist(df.loc[cond, "score"], bins=40, color="#c51162", alpha=0.75, label="pass conditions")
    ax.axvline(0.8, color="#212121", ls="--", lw=1, label="Score=0.8")
    ax.set_xlabel("Score")
    ax.set_ylabel("count")
    ax.set_title("Score distribution")
    ax.legend(fontsize=8)

    # λ vs Score: fail = gray, pass = plasma
    ax = axes[0, 1]
    m = df["valid"] & (df["lambda_nm"] > 0)
    fail = m & ~cond
    ok = m & cond
    if fail.any():
        ax.scatter(
            df.loc[fail, "lambda_nm"],
            df.loc[fail, "score"],
            c="#bdbdbd",
            s=12,
            alpha=0.35,
            edgecolors="none",
            label="other valid",
            zorder=1,
        )
    if ok.any():
        sc = ax.scatter(
            df.loc[ok, "lambda_nm"],
            df.loc[ok, "score"],
            c=df.loc[ok, "score"],
            cmap="plasma",
            s=22,
            alpha=0.9,
            edgecolors="k",
            linewidths=0.2,
            norm=Normalize(0, 1),
            label="pass conditions",
            zorder=3,
        )
        fig.colorbar(sc, ax=ax, label="Score (pass)")
    ax.set_xlabel("λ_abs (nm)")
    ax.set_ylabel("Score")
    ax.set_title("Score vs λ_abs (pass highlighted)")
    ax.legend(fontsize=8)

    # Novelty vs Score: same scheme
    ax = axes[1, 0]
    if "max_tc_train" in df.columns:
        m2 = df["valid"] & df["max_tc_train"].notna()
        fail2 = m2 & ~cond
        ok2 = m2 & cond
        if fail2.any():
            ax.scatter(
                df.loc[fail2, "max_tc_train"],
                df.loc[fail2, "score"],
                c="#bdbdbd",
                s=12,
                alpha=0.35,
                edgecolors="none",
                label="other valid",
                zorder=1,
            )
        if ok2.any():
            sc2 = ax.scatter(
                df.loc[ok2, "max_tc_train"],
                df.loc[ok2, "score"],
                c=df.loc[ok2, "score"],
                cmap="plasma",
                s=22,
                alpha=0.9,
                edgecolors="k",
                linewidths=0.2,
                norm=Normalize(0, 1),
                label="pass conditions",
                zorder=3,
            )
            fig.colorbar(sc2, ax=ax, label="Score (pass)")
        ax.set_xlabel("Max Tanimoto to train")
        ax.set_ylabel("Score")
        ax.set_title("Novelty vs Score (pass highlighted)")
        ax.legend(fontsize=8)

    # Funnel / counts
    ax = axes[1, 1]
    stages = [
        ("All rows", len(df)),
        ("Valid", int(df["valid"].sum())),
        ("Clean", int(df["clean"].sum())),
        ("λ > 0", int((df["lambda_nm"] > 0).sum())),
        ("Pass conditions", int(cond.sum())),
    ]
    if "in_train" in df.columns:
        stages.append(("Pass & novel", int((cond & ~df["in_train"]).sum())))
    y = np.arange(len(stages))[::-1]
    vals = [s[1] for s in stages]
    colors = ["#90a4ae", "#78909c", "#5c6bc0", "#26a69a", "#c51162", "#ff6d00"][: len(stages)]
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in stages], fontsize=9)
    ax.set_xlabel("count")
    ax.set_title(f"Filter funnel\n{cond_label}", fontsize=10)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.01, yi, str(v), va="center", fontsize=8)

    fig.suptitle("Generator check vs training set — score-colored views", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "01_score_colored_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 01_score_colored_overview.png")


def plot_novelty_summary(df: pd.DataFrame, cond: pd.Series, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # max Tc hist
    ax = axes[0]
    if "max_tc_train" in df.columns:
        ax.hist(df.loc[df["valid"], "max_tc_train"].dropna(), bins=40, color="#90a4ae", alpha=0.75, label="all valid")
        if cond.any():
            ax.hist(df.loc[cond, "max_tc_train"].dropna(), bins=40, color="#c51162", alpha=0.75, label="pass conditions")
        ax.axvline(0.4, color="#212121", ls="--", lw=1, label="Tc=0.4")
        ax.set_xlabel("Max Tanimoto to train")
        ax.set_ylabel("count")
        ax.set_title("Structural novelty vs train")
        ax.legend(fontsize=8)

    # in_train share by score bins
    ax = axes[1]
    if "in_train" in df.columns:
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
        labels = ["[0–0.2)", "[0.2–0.4)", "[0.4–0.6)", "[0.6–0.8)", "[0.8–1]"]
        d = df[df["valid"]].copy()
        d["sbin"] = pd.cut(d["score"], bins=bins, labels=labels, right=False)
        g = d.groupby("sbin", observed=False).agg(
            n=("SMILES", "size"),
            frac_in_train=("in_train", "mean"),
            mean_tc=("max_tc_train", "mean"),
        )
        x = np.arange(len(g))
        ax.bar(x, g["frac_in_train"], color="#5c6bc0", edgecolor="k", lw=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(g.index.astype(str), rotation=20, ha="right")
        ax.set_ylabel("Fraction exact match to train")
        ax.set_title("Train-set copies by Score bin")
        ax.set_ylim(0, max(0.05, float(g["frac_in_train"].max()) * 1.3 + 0.01))

    fig.tight_layout()
    fig.savefig(out / "02_novelty_vs_train.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 02_novelty_vs_train.png")


def plot_condition_pass_rate(df: pd.DataFrame, cond: pd.Series, out: Path, cond_label: str) -> None:
    """Pass-rate by RL step, points colored by mean Score."""
    if "step" not in df.columns:
        return
    d = df.copy()
    d["pass"] = cond.astype(bool)
    g = d.groupby("step", as_index=False).agg(
        n=("SMILES", "size"),
        n_pass=("pass", "sum"),
        mean_score=("score", "mean"),
    )
    g["pass_rate"] = g["n_pass"] / g["n"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sc = ax.scatter(
        g["step"],
        g["pass_rate"],
        c=g["mean_score"],
        cmap="plasma",
        s=55,
        edgecolors="k",
        linewidths=0.4,
        norm=Normalize(0, 1),
        zorder=3,
    )
    ax.plot(g["step"], g["pass_rate"], color="#616161", lw=1, alpha=0.5, zorder=2)
    fig.colorbar(sc, ax=ax, label="Mean Score (all in step)")
    ax.set_xlabel("RL step")
    ax.set_ylabel("Fraction passing conditions")
    ax.set_title(f"Condition pass-rate over training\n{cond_label}")
    ax.set_ylim(-0.02, min(1.05, float(g["pass_rate"].max()) * 1.15 + 0.05))
    fig.tight_layout()
    fig.savefig(out / "03_pass_rate_by_step.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 03_pass_rate_by_step.png")


def main() -> None:
    p = argparse.ArgumentParser(description="Check REINVENT generator vs train set under conditions")
    p.add_argument("--rl-csv", type=Path, default=DEFAULT_RL)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--lambda-min", type=float, default=450.0, help="Min λ_abs (nm); None to disable")
    p.add_argument("--lambda-max", type=float, default=480.0, help="Max λ_abs (nm); None to disable")
    p.add_argument("--min-score", type=float, default=0.8)
    p.add_argument("--max-score", type=float, default=None)
    p.add_argument("--require-valid", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--require-clean", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--require-lambda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-sa", type=float, default=None, help="Synthesizability: keep SA ≤ this (↓ easier)")
    p.add_argument("--min-qed", type=float, default=None, help="Optional QED lower bound")
    p.add_argument("--no-lambda-window", action="store_true", help="Disable λ min/max filter")
    p.add_argument("--max-train", type=int, default=4000)
    p.add_argument("--max-rl", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-umap", action="store_true")
    args = p.parse_args()

    if args.no_lambda_window:
        args.lambda_min = None
        args.lambda_max = None

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    set_style()

    cond_parts = []
    if args.require_valid:
        cond_parts.append("valid")
    if args.require_clean:
        cond_parts.append("clean")
    if args.require_lambda:
        cond_parts.append("λ>0")
    if args.lambda_min is not None or args.lambda_max is not None:
        cond_parts.append(f"λ∈[{args.lambda_min},{args.lambda_max}]")
    if args.min_score is not None:
        cond_parts.append(f"Score≥{args.min_score}")
    if args.max_score is not None:
        cond_parts.append(f"Score≤{args.max_score}")
    if args.max_sa is not None:
        cond_parts.append(f"SA≤{args.max_sa}")
    if args.min_qed is not None:
        cond_parts.append(f"QED≥{args.min_qed}")
    cond_label = " & ".join(cond_parts) if cond_parts else "no filters"

    print(f"[INFO] Conditions: {cond_label}")
    print(f"[INFO] Loading RL: {args.rl_csv}")
    df = load_rl(args.rl_csv)
    if args.max_rl is not None and len(df) > args.max_rl:
        df = df.sample(args.max_rl, random_state=args.seed).sort_index()

    print(f"[INFO] Loading train: {args.train}")
    train_smi = read_smiles(args.train, max_n=args.max_train, seed=args.seed)
    train_canon = {canonicalize(s) for s in train_smi}
    train_canon.discard(None)

    df["in_train"] = df["canon"].isin(train_canon)

    print("[INFO] Fingerprints + max Tanimoto to train…")
    train_smi_fp, train_fps, _ = smiles_to_fps(train_smi, unique=True)
    # RL unique valid for FP
    rl_sub = df[df["valid"] & df["canon"].notna()].drop_duplicates("canon")
    rl_smi_fp, rl_fps, _ = smiles_to_fps(rl_sub["canon"].tolist(), unique=False)
    max_tc = max_tc_to_refs(rl_fps, train_fps)
    tc_map = dict(zip(rl_smi_fp, max_tc))
    df["max_tc_train"] = df["canon"].map(tc_map)

    cond = apply_conditions(
        df,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        min_score=args.min_score,
        max_score=args.max_score,
        require_valid=args.require_valid,
        require_clean=args.require_clean,
        require_lambda=args.require_lambda,
        max_sa=args.max_sa,
        min_qed=args.min_qed,
    )
    df["pass_conditions"] = cond

    # Summary metrics
    summary = {
        "conditions": cond_label,
        "n_rl_rows": int(len(df)),
        "n_valid": int(df["valid"].sum()),
        "n_clean": int(df["clean"].sum()),
        "n_pass_conditions": int(cond.sum()),
        "pass_rate": float(cond.mean()),
        "n_pass_novel": int((cond & ~df["in_train"]).sum()),
        "frac_pass_in_train": float(df.loc[cond, "in_train"].mean()) if cond.any() else None,
        "mean_score_all": float(df.loc[df["valid"], "score"].mean()),
        "mean_score_pass": float(df.loc[cond, "score"].mean()) if cond.any() else None,
        "mean_max_tc_all": float(df.loc[df["valid"], "max_tc_train"].mean()),
        "mean_max_tc_pass": float(df.loc[cond, "max_tc_train"].mean()) if cond.any() else None,
        "median_max_tc_pass": float(df.loc[cond, "max_tc_train"].median()) if cond.any() else None,
        "n_train_fps": len(train_fps),
    }
    (out / "tables" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    df.to_csv(out / "tables" / "rl_with_novelty.csv", index=False)
    print(json.dumps(summary, indent=2))

    plot_score_colored_props(df, cond, out, cond_label)
    plot_novelty_summary(df, cond, out)
    plot_condition_pass_rate(df, cond, out, cond_label)

    # Chemspace: subsample for speed
    print("[INFO] Chemspace PCA/UMAP…")
    rl_plot = df[df["valid"] & df["canon"].notna()].drop_duplicates("canon")
    if len(rl_plot) > 2000:
        rl_plot = rl_plot.sample(2000, random_state=args.seed)

    all_smi = list(train_smi_fp) + rl_plot["canon"].tolist()
    labels_list = ["train"] * len(train_smi_fp) + ["rl"] * len(rl_plot)
    _, fps_all, _ = smiles_to_fps(all_smi, unique=False)
    X = fps_to_numpy(fps_all)
    Z_pca, Z_umap, tag = fit_pca_umap(X, args.seed)

    labels_a = np.array(labels_list)
    scores_a = np.array(
        [np.nan] * len(train_smi_fp) + rl_plot["score"].tolist(),
        dtype=float,
    )
    # condition mask on embedding rows
    pass_canons = set(df.loc[cond, "canon"].dropna())
    cond_rows = np.array([(lb == "rl" and smi in pass_canons) for lb, smi in zip(labels_list, all_smi)])

    plot_chemspace(
        Z_pca,
        labels_a,
        scores_a,
        f"PCA: train + RL (color=Score)\nConditions: {cond_label}\n{tag}",
        out / "04_pca_train_rl_by_score.png",
        cond_mask=cond_rows,
    )
    if Z_umap is not None and not args.skip_umap:
        plot_chemspace(
            Z_umap,
            labels_a,
            scores_a,
            f"UMAP: train + RL (color=Score)\nConditions: {cond_label}",
            out / "05_umap_train_rl_by_score.png",
            cond_mask=cond_rows,
        )

    # Save coords
    pd.DataFrame(
        {
            "smiles": all_smi,
            "label": labels_list,
            "score": scores_a,
            "pass_conditions": cond_rows,
            "pca1": Z_pca[:, 0],
            "pca2": Z_pca[:, 1],
            **(
                {"umap1": Z_umap[:, 0], "umap2": Z_umap[:, 1]}
                if Z_umap is not None
                else {}
            ),
        }
    ).to_csv(out / "tables" / "coordinates.csv", index=False)

    # Markdown report
    lines = [
        "# Generator check vs training set",
        "",
        f"**Conditions:** `{cond_label}`",
        "",
        "## Summary",
        f"- RL rows: **{summary['n_rl_rows']}**",
        f"- Pass conditions: **{summary['n_pass_conditions']}** ({100 * summary['pass_rate']:.1f}%)",
        f"- Pass & novel (not exact train): **{summary['n_pass_novel']}**",
        f"- Mean max-Tc to train (pass): **{summary['mean_max_tc_pass']:.3f}**" if summary["mean_max_tc_pass"] is not None else "",
        f"- Mean Score (pass): **{summary['mean_score_pass']:.3f}**" if summary["mean_score_pass"] is not None else "",
        "",
        "## Figures",
        "- `01_score_colored_overview.png` — Score / λ / novelty / funnel",
        "- `02_novelty_vs_train.png` — Tanimoto to train",
        "- `03_pass_rate_by_step.png` — условия по шагам RL",
        "- `04_pca_train_rl_by_score.png` — PCA: все RL | только pass (цвет = Score)",
        "- `05_umap_train_rl_by_score.png` — UMAP то же",
        "",
        "Train set = REINVENT transfer-learning SMILES (`train.smi`).",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] → {out}")


if __name__ == "__main__":
    main()
