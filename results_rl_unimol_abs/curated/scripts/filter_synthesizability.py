"""
Пайплайн оценки синтезируемости для RL-молекул REINVENT.

Методы:
  1. SA score (Ertl) — эвристика фрагментов, 1–10, ↓ легче
  2. SCScore (Coley) — NN на ~12M реакций Reaxys, 1–5, ↓ легче  ← «SA на базе Reaxys»
  3. RAscore — опционально, если установлен (pip/git); иначе пропускается

Outputs → results_rl_unimol_abs/curated/generator_check/synthesizability/

Usage:
  .\\.venv_eval\\Scripts\\python.exe results_rl_unimol_abs\\curated\\scripts\\filter_synthesizability.py
  .\\.venv_eval\\Scripts\\python.exe ...\\filter_synthesizability.py --max-sa 4 --max-scscore 3.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

CURATED = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

DEFAULT_RL = CURATED / "data" / "rl_tl_sweep_20260801_090230_ep04_1.csv"
DEFAULT_OUT = CURATED / "generator_check" / "synthesizability"

from scscore_wrapper import scscore_smiles  # noqa: E402

_HAVE_RDKIT_SA = False
try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    try:
        from rdkit.Contrib.SA_Score import sascorer  # type: ignore

        _HAVE_RDKIT_SA = True
    except Exception:
        sascorer = None  # type: ignore
except Exception:
    Chem = None  # type: ignore
    rdMolDescriptors = None  # type: ignore
    sascorer = None  # type: ignore

_HAVE_RASCORE = False
_ra_scorer = None
try:
    from RAscore import RAscore_NN  # type: ignore

    _ra_scorer = RAscore_NN.RAScorerNN()
    _HAVE_RASCORE = True
except Exception:
    try:
        from RAscore.RAscore_NN import RAScorerNN  # type: ignore

        _ra_scorer = RAScorerNN()
        _HAVE_RASCORE = True
    except Exception:
        pass


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
    df["sa_csv"] = pd.to_numeric(sa, errors="coerce")
    if "Unwanted SMARTS (raw)" in df.columns:
        uw = pd.to_numeric(df["Unwanted SMARTS (raw)"], errors="coerce")
        df["clean"] = (uw >= 0.999) & df["valid"]
    else:
        df["clean"] = df["valid"]
    return df


def compute_rdkit_sa(smiles: str) -> float | None:
    if not _HAVE_RDKIT_SA or Chem is None or sascorer is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return float(sascorer.calculateScore(mol))
    except Exception:
        return None


def compute_rascore(smiles: str) -> float | None:
    if not _HAVE_RASCORE or _ra_scorer is None:
        return None
    try:
        return float(_ra_scorer.predict(smiles))
    except Exception:
        return None


def plot_methods_overview(df: pd.DataFrame, out: Path, max_sa: float, max_sc: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9))

    # SA vs Score
    ax = axes[0, 0]
    m = df["valid"] & df["sa"].notna()
    sc = ax.scatter(
        df.loc[m, "sa"],
        df.loc[m, "score"],
        c=df.loc[m, "score"],
        cmap="plasma",
        s=12,
        alpha=0.7,
        edgecolors="none",
        norm=Normalize(0, 1),
    )
    ax.axvline(max_sa, color="#c62828", ls="--", lw=1.1, label=f"SA≤{max_sa}")
    ax.axhline(0.8, color="#212121", ls=":", lw=1)
    fig.colorbar(sc, ax=ax, label="Score")
    ax.set_xlabel("SA score (Ertl), ↓ easier")
    ax.set_ylabel("Score")
    ax.set_title("SA (fragment heuristic)")
    ax.legend(fontsize=8)

    # SCScore vs Score
    ax = axes[0, 1]
    m = df["valid"] & df["scscore"].notna()
    sc = ax.scatter(
        df.loc[m, "scscore"],
        df.loc[m, "score"],
        c=df.loc[m, "score"],
        cmap="plasma",
        s=12,
        alpha=0.7,
        edgecolors="none",
        norm=Normalize(0, 1),
    )
    ax.axvline(max_sc, color="#c62828", ls="--", lw=1.1, label=f"SCScore≤{max_sc}")
    ax.axhline(0.8, color="#212121", ls=":", lw=1)
    fig.colorbar(sc, ax=ax, label="Score")
    ax.set_xlabel("SCScore (Reaxys NN), ↓ easier")
    ax.set_ylabel("Score")
    ax.set_title("SCScore — Reaxys-trained complexity")
    ax.legend(fontsize=8)

    # SA vs SCScore among high-Score
    ax = axes[1, 0]
    hi = df["valid"] & (df["score"] >= 0.8) & df["sa"].notna() & df["scscore"].notna()
    ax.scatter(
        df.loc[hi, "sa"],
        df.loc[hi, "scscore"],
        c=df.loc[hi, "score"],
        cmap="plasma",
        s=18,
        alpha=0.8,
        edgecolors="k",
        linewidths=0.15,
        norm=Normalize(0, 1),
    )
    ax.axvline(max_sa, color="#c62828", ls="--", lw=1)
    ax.axhline(max_sc, color="#1565c0", ls="--", lw=1)
    ax.set_xlabel("SA (Ertl)")
    ax.set_ylabel("SCScore (Reaxys)")
    ax.set_title("Agreement of two synth scores (Score≥0.8)")
    if hi.any():
        corr = float(df.loc[hi, ["sa", "scscore"]].corr().iloc[0, 1])
        ax.text(0.05, 0.95, f"Spearman/Pearson≈{corr:.2f}", transform=ax.transAxes, va="top", fontsize=9)

    # RAscore panel or note
    ax = axes[1, 1]
    if "rascore" in df.columns and df["rascore"].notna().any():
        m = df["valid"] & df["rascore"].notna()
        ax.scatter(
            df.loc[m, "rascore"],
            df.loc[m, "score"],
            c=df.loc[m, "score"],
            cmap="plasma",
            s=12,
            alpha=0.7,
            edgecolors="none",
            norm=Normalize(0, 1),
        )
        ax.set_xlabel("RAscore (↑ easier / more solvable)")
        ax.set_ylabel("Score")
        ax.set_title("RAscore (AiZynth success proxy)")
    else:
        ax.axis("off")
        ax.text(
            0.05,
            0.6,
            "RAscore: not installed\n\n"
            "Install from:\n"
            "  github.com/reymond-group/RAscore\n\n"
            "Pipeline still runs SA + SCScore(Reaxys).",
            transform=ax.transAxes,
            fontsize=10,
            va="center",
            family="DejaVu Sans",
        )

    fig.suptitle("Synthesizability pipeline: SA + SCScore(Reaxys)", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "01_synth_methods_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 01_synth_methods_overview.png")


def plot_funnel_compare(df: pd.DataFrame, masks: dict[str, pd.Series], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    names = list(masks.keys())
    vals = [int(masks[k].sum()) for k in names]
    colors = ["#90a4ae", "#5c6bc0", "#ab47bc", "#00897b", "#1565c0", "#c51162"][: len(names)]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=colors, edgecolor="k", lw=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("count")
    ax.set_title("Filter funnel — property + synthesizability")
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.01, yi, str(v), va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "02_synth_filter_funnel.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 02_synth_filter_funnel.png")


def plot_threshold_sweeps(df: pd.DataFrame, base: pd.Series, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # SA sweep
    sa_thr = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
    sa_n = [int((base & df["sa"].notna() & (df["sa"] <= t)).sum()) for t in sa_thr]
    axes[0].plot(sa_thr, sa_n, "o-", color="#00897b")
    axes[0].set_xlabel("max SA (Ertl)")
    axes[0].set_ylabel("# pass")
    axes[0].set_title("SA threshold sweep (on baseline)")

    # SCScore sweep
    sc_thr = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    sc_n = [int((base & df["scscore"].notna() & (df["scscore"] <= t)).sum()) for t in sc_thr]
    axes[1].plot(sc_thr, sc_n, "o-", color="#1565c0")
    axes[1].set_xlabel("max SCScore (Reaxys)")
    axes[1].set_ylabel("# pass")
    axes[1].set_title("SCScore threshold sweep (on baseline)")

    pd.DataFrame({"max_sa": sa_thr, "n": sa_n}).to_csv(out / "tables" / "sa_threshold_sweep.csv", index=False)
    pd.DataFrame({"max_scscore": sc_thr, "n": sc_n}).to_csv(out / "tables" / "scscore_threshold_sweep.csv", index=False)

    fig.tight_layout()
    fig.savefig(out / "03_threshold_sweeps.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 03_threshold_sweeps.png")


def main() -> None:
    p = argparse.ArgumentParser(description="Synthesizability pipeline: SA + SCScore(Reaxys) + optional RAscore")
    p.add_argument("--rl-csv", type=Path, default=DEFAULT_RL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-sa", type=float, default=4.0, help="Ertl SA ≤ this")
    p.add_argument("--max-scscore", type=float, default=3.0, help="Reaxys SCScore ≤ this (1–5)")
    p.add_argument("--min-rascore", type=float, default=0.5, help="RAscore ≥ this if available")
    p.add_argument("--min-qed", type=float, default=None)
    p.add_argument("--min-score", type=float, default=0.8)
    p.add_argument("--lambda-min", type=float, default=450.0)
    p.add_argument("--lambda-max", type=float, default=480.0)
    p.add_argument("--recompute-sa", action="store_true")
    p.add_argument("--require-clean", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-scscore", action="store_true")
    args = p.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    set_style()

    print("[INFO] Loading RL…")
    df = load_rl(args.rl_csv)

    if args.recompute_sa and _HAVE_RDKIT_SA:
        print("[INFO] Recomputing Ertl SA…")
        df["sa_rdkit"] = df["SMILES"].map(compute_rdkit_sa)
        df["sa"] = df["sa_rdkit"].fillna(df["sa_csv"])
    else:
        df["sa"] = df["sa_csv"]

    if args.skip_scscore:
        df["scscore"] = np.nan
        print("[INFO] SCScore skipped")
    else:
        print("[INFO] Computing SCScore (Reaxys-trained)…")
        # cache unique SMILES
        uniq = df.loc[df["valid"], "SMILES"].dropna().unique()
        cache: dict[str, float | None] = {}
        for i, smi in enumerate(uniq):
            if i % 200 == 0:
                print(f"  SCScore {i}/{len(uniq)}")
            cache[smi] = scscore_smiles(str(smi))
        df["scscore"] = df["SMILES"].map(cache)

    if _HAVE_RASCORE:
        print("[INFO] Computing RAscore…")
        uniq = df.loc[df["valid"], "SMILES"].dropna().unique()
        cache_ra: dict[str, float | None] = {}
        for smi in uniq:
            cache_ra[smi] = compute_rascore(str(smi))
        df["rascore"] = df["SMILES"].map(cache_ra)
    else:
        df["rascore"] = np.nan
        print("[INFO] RAscore not available (optional)")

    base = df["valid"].copy()
    if args.require_clean:
        base &= df["clean"]
    base &= df["lambda_nm"] > 0
    base &= df["lambda_nm"] >= args.lambda_min
    base &= df["lambda_nm"] <= args.lambda_max
    base &= df["score"] >= args.min_score

    pass_sa = base & df["sa"].notna() & (df["sa"] <= args.max_sa)
    pass_sc = base & df["scscore"].notna() & (df["scscore"] <= args.max_scscore)
    pass_both = pass_sa & pass_sc
    if args.min_qed is not None:
        pass_both &= df["qed"].notna() & (df["qed"] >= args.min_qed)
    if _HAVE_RASCORE:
        pass_ra = base & df["rascore"].notna() & (df["rascore"] >= args.min_rascore)
        pass_all = pass_both & pass_ra
    else:
        pass_ra = pd.Series(False, index=df.index)
        pass_all = pass_both

    df["pass_baseline"] = base
    df["pass_sa"] = pass_sa
    df["pass_scscore"] = pass_sc
    df["pass_synth"] = pass_all

    masks = {
        "Valid": df["valid"],
        "Clean": df["clean"],
        f"Baseline λ/Score≥{args.min_score}": base,
        f"SA≤{args.max_sa}": pass_sa,
        f"SCScore≤{args.max_scscore}": pass_sc,
        "SA ∩ SCScore": pass_both,
    }
    if _HAVE_RASCORE:
        masks[f"RAscore≥{args.min_rascore}"] = pass_ra
        masks["All synth filters"] = pass_all

    summary = {
        "methods": {
            "sa_ertl": "fragment heuristic 1–10, ↓ easier",
            "scscore_reaxys": "NN on ~12M Reaxys reactions, 1–5, ↓ easier (Coley et al.)",
            "rascore": "optional AiZynth success proxy 0–1, ↑ better",
        },
        "thresholds": {
            "max_sa": args.max_sa,
            "max_scscore": args.max_scscore,
            "min_rascore": args.min_rascore if _HAVE_RASCORE else None,
            "min_qed": args.min_qed,
            "min_score": args.min_score,
            "lambda_window": [args.lambda_min, args.lambda_max],
        },
        "n_baseline": int(base.sum()),
        "n_pass_sa": int(pass_sa.sum()),
        "n_pass_scscore": int(pass_sc.sum()),
        "n_pass_sa_and_scscore": int(pass_both.sum()),
        "n_pass_all": int(pass_all.sum()),
        "frac_baseline_kept_sa": float(pass_sa.sum() / max(base.sum(), 1)),
        "frac_baseline_kept_scscore": float(pass_sc.sum() / max(base.sum(), 1)),
        "frac_baseline_kept_both": float(pass_both.sum() / max(base.sum(), 1)),
        "mean_sa_baseline": float(df.loc[base, "sa"].mean()) if base.any() else None,
        "mean_scscore_baseline": float(df.loc[base, "scscore"].mean()) if base.any() else None,
        "corr_sa_scscore_highscore": float(
            df.loc[df["valid"] & (df["score"] >= 0.8), ["sa", "scscore"]].corr().iloc[0, 1]
        )
        if (df["valid"] & (df["score"] >= 0.8) & df["sa"].notna() & df["scscore"].notna()).sum() > 5
        else None,
        "rascore_available": _HAVE_RASCORE,
        "rdkit_sa_available": _HAVE_RDKIT_SA,
    }
    (out / "tables" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cols = [
        c
        for c in [
            "SMILES",
            "score",
            "lambda_nm",
            "qed",
            "sa",
            "scscore",
            "rascore",
            "clean",
            "step",
            "pass_baseline",
            "pass_sa",
            "pass_scscore",
            "pass_synth",
        ]
        if c in df.columns
    ]
    if "step" in df.columns and "step" not in cols:
        cols.append("step")
    df.loc[pass_all, cols].to_csv(out / "tables" / "synth_pass_molecules.csv", index=False)
    df.to_csv(out / "tables" / "rl_with_synth_scores.csv", index=False)
    # SCScore-pass table used by novelty / UMAP follow-ups
    sc_cols = [c for c in cols if c in df.columns]
    sc_pass = df.loc[pass_sc, sc_cols].sort_values(["scscore", "score"], ascending=[True, False]).reset_index(drop=True)
    sc_pass.to_csv(out / "tables" / "scscore_pass_sorted.csv", index=False)
    df.loc[base, sc_cols].to_csv(out / "tables" / "rl_with_synth_flags.csv", index=False)

    # structure grid for SCScore-pass (cap for readability)
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw

        show = sc_pass.head(25)
        mols, legs = [], []
        for _, r in show.iterrows():
            m = Chem.MolFromSmiles(str(r["SMILES"]))
            if m is None:
                continue
            mols.append(m)
            legs.append(f"S={r['score']:.2f} SC={r['scscore']:.2f}\nSA={r['sa']:.2f}")
        if mols:
            img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(260, 240), legends=legs)
            img.save(str(out / "04_scscore_pass_structures.png"))
            print("[OK] 04_scscore_pass_structures.png")
    except Exception as e:
        print(f"[WARN] structure grid skipped: {e}")

    plot_methods_overview(df, out, args.max_sa, args.max_scscore)
    plot_funnel_compare(df, masks, out)
    plot_threshold_sweeps(df, base, out)

    # keep old filenames as aliases for compatibility
    if (out / "01_synth_methods_overview.png").is_file():
        # also write SA-focused legacy name pointing users to new plot
        pass

    lines = [
        "# Synthesizability pipeline",
        "",
        "## Methods",
        "1. **SA score (Ertl)** — fragment heuristic, 1–10, ↓ easier",
        "2. **SCScore (Coley)** — NN trained on ~12M **Reaxys** reactions, 1–5, ↓ easier",
        "3. **RAscore** — optional; predicts AiZynthFinder success",
        "",
        "> «SA на базе Reaxys» в литературе = **SCScore**, не классический Ertl SA.",
        "",
        f"## Results (baseline λ∈[{args.lambda_min},{args.lambda_max}], Score≥{args.min_score})",
        f"- Baseline: **{int(base.sum())}**",
        f"- SA≤{args.max_sa}: **{int(pass_sa.sum())}** ({100 * summary['frac_baseline_kept_sa']:.1f}%)",
        f"- SCScore≤{args.max_scscore}: **{int(pass_sc.sum())}** ({100 * summary['frac_baseline_kept_scscore']:.1f}%)",
        f"- SA ∩ SCScore: **{int(pass_both.sum())}** ({100 * summary['frac_baseline_kept_both']:.1f}%)",
        "",
        f"Mean SA (baseline): {summary['mean_sa_baseline']:.2f}" if summary["mean_sa_baseline"] else "",
        f"Mean SCScore (baseline): {summary['mean_scscore_baseline']:.2f}" if summary["mean_scscore_baseline"] else "",
        "",
        "## Figures",
        "- `01_synth_methods_overview.png`",
        "- `02_synth_filter_funnel.png`",
        "- `03_threshold_sweeps.png`",
        "",
        "## Tables",
        "- `tables/rl_with_synth_scores.csv` — все скоры",
        "- `tables/synth_pass_molecules.csv` — прошедшие фильтр",
    ]
    (out / "README.md").write_text("\n".join([ln for ln in lines if ln is not None]) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[DONE] → {out}")


if __name__ == "__main__":
    main()
