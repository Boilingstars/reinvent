"""
REINVENT challenge suite — five failure-mode exams.

1) Coverage / Recall of target chemical space (molecules.csv)
2) Murcko scaffold collapse / dominance (class-agnostic)
3) Memorization vs extrapolation (vs train)
4) Novelty–Precision frontier
5) Prior drift + reward / surrogate hacking

One primary figure per task → generator_check/challenge/

Usage:
  .venv_eval\\Scripts\\python.exe results_rl_unimol_abs\\curated\\scripts\\run_challenge.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

CURATED = Path(__file__).resolve().parents[1]
OUT = CURATED / "generator_check" / "challenge"
RL_CSV = CURATED / "data" / "rl_unimol_abs_1.csv"
NOVELTY_CSV = CURATED / "generator_check" / "tables" / "rl_with_novelty.csv"
MOLECULES_CSV = CURATED / "data" / "molecules.csv"
TRAIN_SMI = CURATED / "data" / "refs" / "train.smi"

LAM_MIN, LAM_MAX = 450.0, 480.0
SCORE_THR = 0.8
TC_COVER = 0.50  # neighbor threshold for coverage


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#fafafa",
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#e8e8e8",
            "grid.linewidth": 0.55,
        }
    )


def canon(s: str) -> str | None:
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m, canonical=True) if m else None


def murcko_generic(smi: str) -> str | None:
    m = Chem.MolFromSmiles(str(smi))
    if not m:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(m)
        if core is None or core.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(core), canonical=True)
    except Exception:
        return None


def fp(smi: str):
    m = Chem.MolFromSmiles(str(smi))
    if not m:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def max_tc_to_refs(query_fps: list, ref_fps: list) -> np.ndarray:
    out = np.zeros(len(query_fps), dtype=float)
    if not ref_fps:
        return out
    for i, q in enumerate(query_fps):
        if q is None:
            out[i] = np.nan
            continue
        out[i] = float(max(DataStructs.BulkTanimotoSimilarity(q, ref_fps)))
    return out


def any_neighbor_above(query_fps: list, ref_fps: list, thr: float) -> np.ndarray:
    """For each REF, whether any QUERY has Tc >= thr."""
    hits = np.zeros(len(ref_fps), dtype=bool)
    if not query_fps:
        return hits
    q_ok = [q for q in query_fps if q is not None]
    if not q_ok:
        return hits
    for j, r in enumerate(ref_fps):
        if r is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(r, q_ok)
        hits[j] = float(max(sims)) >= thr
    return hits


def scaffold_stats(labels: pd.Series) -> dict:
    labs = labels.dropna()
    labs = labs[labs.astype(str).str.len() > 0]
    n = len(labs)
    if n == 0:
        return {"n": 0, "n_unique": 0, "dominance": 0.0, "entropy_norm": 0.0, "n_eff": 0.0, "collapse_index": 0.0}
    vc = labs.value_counts()
    p = (vc / n).astype(float)
    d = float(p.iloc[0])
    H = float(-(p * np.log(p.clip(1e-15))).sum())
    k = int(len(vc))
    Hn = float(H / np.log(k)) if k > 1 else 0.0
    return {
        "n": int(n),
        "n_unique": k,
        "dominance": d,
        "entropy_norm": Hn,
        "n_eff": float(np.exp(H)),
        "collapse_index": float(d * (1.0 - Hn)),
        "top_scaffold": str(vc.index[0]),
    }


def load_rl() -> pd.DataFrame:
    df = pd.read_csv(RL_CSV)
    df["valid"] = df["SMILES_state"] == 1 if "SMILES_state" in df.columns else True
    df["score"] = pd.to_numeric(df["Score"], errors="coerce").fillna(0.0)
    df["prior"] = pd.to_numeric(df["Prior"], errors="coerce")
    df["agent"] = pd.to_numeric(df["Agent"], errors="coerce")
    lam = df["lambda_abs (raw)"] if "lambda_abs (raw)" in df.columns else df["lambda_abs"]
    df["lambda_nm"] = pd.to_numeric(lam, errors="coerce").fillna(0.0)
    qed = df["QED (raw)"] if "QED (raw)" in df.columns else df.get("QED")
    df["qed"] = pd.to_numeric(qed, errors="coerce")
    if "Unwanted SMARTS (raw)" in df.columns:
        uw = pd.to_numeric(df["Unwanted SMARTS (raw)"], errors="coerce")
        df["clean"] = (uw >= 0.999) & df["valid"]
    else:
        df["clean"] = df["valid"]
    df["canon"] = df["SMILES"].map(canon)
    df["on_target"] = (
        df["valid"]
        & df["clean"]
        & (df["lambda_nm"] >= LAM_MIN)
        & (df["lambda_nm"] <= LAM_MAX)
        & (df["score"] >= SCORE_THR)
    )
    if NOVELTY_CSV.is_file():
        nov = pd.read_csv(NOVELTY_CSV)
        if "SMILES" in nov.columns and "max_tc_train" in nov.columns:
            df["max_tc_train"] = df["SMILES"].map(dict(zip(nov["SMILES"], nov["max_tc_train"])))
    if "max_tc_train" not in df.columns:
        df["max_tc_train"] = np.nan
    print("[INFO] Computing generic Murcko…")
    df["murcko_generic"] = df["SMILES"].map(murcko_generic)
    return df


# ─── 1. Coverage / Recall ─────────────────────────────────────────────────────


def task_coverage_recall(df: pd.DataFrame, out: Path) -> dict:
    """
    Target chemical space := molecules.csv (literature chromophore set).
    Coverage@τ  = fraction of unique Murcko-generic scaffolds in molecules.csv
                  that have ≥1 RL neighbor with Tc≥τ
    Recall@τ    = fraction of molecules.csv compounds with ≥1 RL neighbor Tc≥τ
    """
    mol = pd.read_csv(MOLECULES_CSV)
    smi_col = "smiles" if "smiles" in mol.columns else "SMILES"
    mol["canon"] = mol[smi_col].map(canon)
    mol = mol.dropna(subset=["canon"]).drop_duplicates("canon")
    mol["murcko_generic"] = mol["canon"].map(murcko_generic)

    rl = df[df["valid"] & df["canon"].notna()].drop_duplicates("canon")
    print(f"[INFO] Coverage: RL={len(rl)}  ref molecules={len(mol)}")

    rl_fps = [fp(s) for s in rl["canon"]]
    ref_fps = [fp(s) for s in mol["canon"]]

    thresholds = np.round(np.arange(0.30, 0.85, 0.05), 2)
    recall = []
    for t in thresholds:
        hits = any_neighbor_above(rl_fps, ref_fps, float(t))
        recall.append(float(hits.mean()))

    # scaffold coverage at TC_COVER
    sc_refs = mol.dropna(subset=["murcko_generic"]).drop_duplicates("murcko_generic")
    sc_fps = [fp(s) for s in sc_refs["canon"]]  # use a representative mol per scaffold
    # better: for each unique scaffold, take first mol's fp already; check neighbor of that mol
    # Coverage of scaffolds: scaffold covered if ANY mol with that scaffold is hit
    hits_mol = any_neighbor_above(rl_fps, ref_fps, TC_COVER)
    mol = mol.reset_index(drop=True)
    mol["_hit"] = hits_mol
    sc_cover = (
        mol.dropna(subset=["murcko_generic"])
        .groupby("murcko_generic")["_hit"]
        .any()
    )
    coverage = float(sc_cover.mean()) if len(sc_cover) else 0.0

    # also on-target RL only recall for contrast
    rl_ot = df[df["on_target"] & df["canon"].notna()].drop_duplicates("canon")
    ot_fps = [fp(s) for s in rl_ot["canon"]]
    recall_ot = []
    for t in thresholds:
        hits = any_neighbor_above(ot_fps, ref_fps, float(t))
        recall_ot.append(float(hits.mean()) if len(ot_fps) else 0.0)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(thresholds, recall, "o-", color="#1565c0", lw=2.2, ms=7, label="all valid RL")
    ax.plot(thresholds, recall_ot, "s--", color="#c62828", lw=2.0, ms=6, label=f"on-target RL (λ∈[{LAM_MIN:.0f},{LAM_MAX:.0f}], S≥{SCORE_THR})")
    ax.axvline(TC_COVER, color="#666", ls=":", lw=1.2)
    ax.axhline(coverage, color="#2e7d32", ls="--", lw=1.3, label=f"scaffold Coverage@{TC_COVER:.2f} = {coverage:.1%}")
    # annotate Coverage point on all-RL curve
    i = int(np.argmin(np.abs(thresholds - TC_COVER)))
    ax.scatter([thresholds[i]], [recall[i]], s=90, zorder=5, color="#1565c0", edgecolors="k", lw=0.6)
    ax.annotate(
        f"Recall@{TC_COVER:.2f} = {recall[i]:.1%}",
        xy=(thresholds[i], recall[i]),
        xytext=(thresholds[i] + 0.06, recall[i] + 0.08),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#333", lw=0.8),
    )
    ax.set_xlabel("Tanimoto threshold τ to molecules.csv")
    ax.set_ylabel("Recall (fraction of ref molecules covered)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(thresholds.min() - 0.02, thresholds.max() + 0.02)
    ax.set_title("1 · Coverage / Recall of target chemical space\n(target = molecules.csv chromophores)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "01_coverage_recall.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 01_coverage_recall.png")

    return {
        "n_ref_molecules": int(len(mol)),
        "n_ref_scaffolds": int(len(sc_cover)),
        "coverage_scaffold_at_tc": coverage,
        "tc_cover": TC_COVER,
        "recall_curve": {str(t): float(r) for t, r in zip(thresholds, recall)},
        "recall_on_target_curve": {str(t): float(r) for t, r in zip(thresholds, recall_ot)},
        "recall_at_tc_cover": float(recall[i]),
        "n_rl_valid": int(len(rl)),
        "n_rl_on_target": int(len(rl_ot)),
    }


# ─── 2. Scaffold collapse ─────────────────────────────────────────────────────


def task_scaffold_collapse(df: pd.DataFrame, out: Path) -> dict:
    valid = df["valid"] & df["murcko_generic"].notna()
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    labels = ["0–0.5", "0.5–0.6", "0.6–0.7", "0.7–0.8", "0.8–0.9", "0.9–1.0"]
    sub = df.loc[valid].copy()
    sub["score_bin"] = pd.cut(sub["score"], bins=bins, labels=labels, right=False)

    rows = []
    for lab in labels:
        g = sub[sub["score_bin"] == lab]
        st = scaffold_stats(g["murcko_generic"])
        st["score_bin"] = lab
        rows.append(st)
    by = pd.DataFrame(rows)
    st_hi = scaffold_stats(df.loc[valid & (df["score"] >= SCORE_THR), "murcko_generic"])

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(by))
    ax.bar(x, by["dominance"], color="#c62828", alpha=0.85, edgecolor="k", lw=0.3, label="top-1 share (dominance)")
    ax.plot(x, by["entropy_norm"], "o-", color="#1565c0", lw=2.2, ms=7, label="H_norm")
    ax.set_xticks(x)
    ax.set_xticklabels(by["score_bin"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Score bin")
    ax.set_ylabel("value")
    ax.set_title(
        f"2 · Murcko scaffold collapse (generic, class-agnostic)\n"
        f"high-Score: dominance={st_hi['dominance']:.3f}  H_norm={st_hi['entropy_norm']:.3f}  "
        f"N_eff={st_hi['n_eff']:.0f}/{st_hi['n_unique']}"
    )
    ax.legend(fontsize=8, loc="center right")
    # note
    ax.text(
        0.02,
        0.98,
        "Collapse if dominance↑ and H_norm↓ as Score↑",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color="#555",
    )
    fig.tight_layout()
    fig.savefig(out / "02_scaffold_collapse.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 02_scaffold_collapse.png")

    by.to_csv(out / "tables" / "murcko_collapse_by_score.csv", index=False)
    return {
        "by_score_bin": by.to_dict(orient="records"),
        "high_score": st_hi,
        "collapse_worsens_with_score": bool(by.iloc[-1]["dominance"] > by.iloc[0]["dominance"]),
    }


# ─── 3. Memorization vs extrapolation ─────────────────────────────────────────


def task_memorization(df: pd.DataFrame, out: Path) -> dict:
    sub = df[df["valid"] & df["max_tc_train"].notna()].copy()
    x = sub["max_tc_train"].to_numpy()
    y = sub["score"].to_numpy()
    on = sub["on_target"].to_numpy()

    # quadrants
    tc_mem, tc_ext = 0.70, 0.40
    mem = (x >= tc_mem) & (y >= SCORE_THR)
    ext = (x < tc_ext) & (y >= SCORE_THR)
    mid = (x >= tc_ext) & (x < tc_mem) & (y >= SCORE_THR)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    # background regions (data coordinates)
    ax.add_patch(Rectangle((-0.01, SCORE_THR), tc_ext + 0.01, 1.05 - SCORE_THR, facecolor="#e8f5e9", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((tc_mem, SCORE_THR), 1.02 - tc_mem, 1.05 - SCORE_THR, facecolor="#ffebee", edgecolor="none", zorder=0))
    ax.scatter(x[~on], y[~on], s=10, c="#bdbdbd", alpha=0.45, linewidths=0, label="other valid", zorder=2)
    ax.scatter(x[on], y[on], s=18, c="#1565c0", alpha=0.75, linewidths=0, label="on-target (λ+Score)", zorder=3)
    ax.axhline(SCORE_THR, color="#333", ls="--", lw=1, zorder=1)
    ax.axvline(tc_ext, color="#2e7d32", ls=":", lw=1.2, zorder=1)
    ax.axvline(tc_mem, color="#c62828", ls=":", lw=1.2, zorder=1)
    ax.text(tc_ext / 2, 0.93, "extrapolation\n(novel + high Score)", ha="center", va="center", fontsize=8, color="#1b5e20", zorder=4)
    ax.text((tc_mem + 1) / 2, 0.93, "memorization\n(near-train)", ha="center", va="center", fontsize=8, color="#b71c1c", zorder=4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("max Tanimoto to train  →  more memorization")
    ax.set_ylabel("Score")
    ax.set_title(
        f"3 · Memorization vs extrapolation\n"
        f"high-Score: extrapolate (Tc<{tc_ext})={ext.sum()}  mid={mid.sum()}  memorize (Tc≥{tc_mem})={mem.sum()}"
    )
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "03_memorization_vs_extrapolation.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 03_memorization_vs_extrapolation.png")

    n_hi = int((y >= SCORE_THR).sum())
    return {
        "n_highscore": n_hi,
        "n_extrapolate": int(ext.sum()),
        "n_mid": int(mid.sum()),
        "n_memorize": int(mem.sum()),
        "frac_extrapolate_of_highscore": float(ext.sum() / max(n_hi, 1)),
        "frac_memorize_of_highscore": float(mem.sum() / max(n_hi, 1)),
        "mean_max_tc_highscore": float(x[y >= SCORE_THR].mean()) if n_hi else None,
        "tc_extrapolate": tc_ext,
        "tc_memorize": tc_mem,
    }


# ─── 4. Novelty–Precision frontier ────────────────────────────────────────────


def task_novelty_precision(df: pd.DataFrame, out: Path) -> dict:
    """
    novelty   = 1 - max_tc_train
    precision = Score  (surrogate for on-target quality; λ window as marker)
    Success = high novelty AND high precision (upper-right).
    """
    sub = df[df["valid"] & df["max_tc_train"].notna()].copy()
    sub["novelty"] = 1.0 - sub["max_tc_train"]
    sub["precision"] = sub["score"]

    # Pareto frontier (maximize novelty and precision)
    pts = sub[["novelty", "precision"]].to_numpy()
    # sort by novelty desc
    order = np.argsort(-pts[:, 0])
    pts_s = pts[order]
    pareto = []
    best_p = -1.0
    for n, p in pts_s:
        if p > best_p:
            pareto.append((n, p))
            best_p = p
    pareto = np.array(pareto) if pareto else np.zeros((0, 2))

    # "true success": novelty ≥ 0.6 (Tc≤0.4) AND precision ≥ SCORE_THR AND on_target
    success = sub["on_target"] & (sub["novelty"] >= 0.6)
    n_success = int(success.sum())

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.scatter(
        sub.loc[~sub["on_target"], "novelty"],
        sub.loc[~sub["on_target"], "precision"],
        s=9,
        c="#bdbdbd",
        alpha=0.4,
        linewidths=0,
        label="other valid",
    )
    ax.scatter(
        sub.loc[sub["on_target"] & ~success, "novelty"],
        sub.loc[sub["on_target"] & ~success, "precision"],
        s=16,
        c="#1565c0",
        alpha=0.7,
        linewidths=0,
        label="on-target but not novel",
    )
    ax.scatter(
        sub.loc[success, "novelty"],
        sub.loc[success, "precision"],
        s=36,
        c="#00c853",
        edgecolors="k",
        linewidths=0.4,
        zorder=5,
        label=f"novel ∩ on-target (n={n_success})",
    )
    if len(pareto):
        ax.plot(pareto[:, 0], pareto[:, 1], "-", color="#e65100", lw=2.0, label="Pareto frontier")
        ax.scatter(pareto[:, 0], pareto[:, 1], s=22, c="#e65100", zorder=4)
    # success box
    ax.add_patch(
        Rectangle(
            (0.6, SCORE_THR),
            0.4,
            1.0 - SCORE_THR,
            fill=False,
            edgecolor="#00c853",
            lw=1.5,
            ls="--",
        )
    )
    ax.text(0.80, 0.92, "success\nquadrant", ha="center", va="center", fontsize=8, color="#1b5e20")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Novelty = 1 − max Tc(train)")
    ax.set_ylabel("Precision ≈ Score (on-target quality)")
    ax.set_title(
        f"4 · Novelty–Precision frontier\n"
        f"true successes (novel∧λ∧S≥{SCORE_THR}): {n_success} / {int(sub['on_target'].sum())} on-target "
        f"({n_success / max(int(sub['on_target'].sum()), 1):.1%})"
    )
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "04_novelty_precision_frontier.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 04_novelty_precision_frontier.png")

    return {
        "n_on_target": int(sub["on_target"].sum()),
        "n_true_success": n_success,
        "frac_true_success_of_on_target": float(n_success / max(int(sub["on_target"].sum()), 1)),
        "n_pareto": int(len(pareto)),
        "mean_novelty_on_target": float(sub.loc[sub["on_target"], "novelty"].mean()) if sub["on_target"].any() else None,
        "mean_novelty_all_highscore": float(sub.loc[sub["score"] >= SCORE_THR, "novelty"].mean()),
    }


# ─── 5. Prior drift + reward / surrogate hacking ───────────────────────────────


def task_prior_hacking(df: pd.DataFrame, out: Path) -> dict:
    sub = df[df["valid"] & df["prior"].notna()].copy()
    hi = sub["score"] >= SCORE_THR
    lo = sub["score"] < 0.3

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # left: Prior NLL vs Score (drift)
    ax = axes[0]
    sc = ax.scatter(
        sub["prior"],
        sub["score"],
        c=sub["lambda_nm"].clip(300, 600),
        cmap="viridis",
        s=12,
        alpha=0.65,
        linewidths=0,
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("λ_abs (nm)")
    ax.axhline(SCORE_THR, color="#c62828", ls="--", lw=1.1)
    # mean prior markers
    if hi.any():
        ax.axvline(sub.loc[hi, "prior"].median(), color="#c62828", ls=":", lw=1.3)
    if lo.any():
        ax.axvline(sub.loc[lo, "prior"].median(), color="#1565c0", ls=":", lw=1.3)
    ax.set_xlabel("Prior NLL  →  less likely under prior (drift)")
    ax.set_ylabel("Score")
    ax.set_title(
        f"Prior drift\n"
        f"median Prior: high-S={sub.loc[hi,'prior'].median():.1f}  low-S={sub.loc[lo,'prior'].median():.1f}"
    )

    # right: QED vs Score among on-λ molecules (reward hacking / cheap winners)
    ax = axes[1]
    in_lam = sub[(sub["lambda_nm"] >= LAM_MIN) & (sub["lambda_nm"] <= LAM_MAX) & sub["qed"].notna()]
    ax.scatter(in_lam["qed"], in_lam["score"], s=14, c="#6a1b9a", alpha=0.55, linewidths=0)
    ax.axhline(SCORE_THR, color="#c62828", ls="--", lw=1.1)
    ax.axvline(0.2, color="#ef6c00", ls=":", lw=1.2)
    cheap = in_lam[(in_lam["score"] >= SCORE_THR) & (in_lam["qed"] < 0.2)]
    good = in_lam[(in_lam["score"] >= SCORE_THR) & (in_lam["qed"] >= 0.35)]
    ax.scatter(cheap["qed"], cheap["score"], s=22, c="#ef6c00", edgecolors="k", lw=0.3, label=f"cheap winners QED<0.2 (n={len(cheap)})")
    ax.scatter(good["qed"], good["score"], s=22, c="#00c853", edgecolors="k", lw=0.3, label=f"QED≥0.35 (n={len(good)})")
    ax.set_xlabel("QED")
    ax.set_ylabel("Score")
    ax.set_xlim(-0.02, 1.02)
    ax.set_title(
        f"Reward hacking (λ∈[{LAM_MIN:.0f},{LAM_MAX:.0f}])\n"
        f"among high-S: cheap={len(cheap)} / {(in_lam['score']>=SCORE_THR).sum()} "
        f"({len(cheap)/max(int((in_lam['score']>=SCORE_THR).sum()),1):.0%})"
    )
    ax.legend(fontsize=7.5, loc="lower right")

    fig.suptitle("5 · Prior drift & reward / surrogate hacking", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "05_prior_drift_reward_hacking.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 05_prior_drift_reward_hacking.png")

    # surrogate: λ concentration among high-score
    hi_all = sub[hi]
    in_window = ((hi_all["lambda_nm"] >= LAM_MIN) & (hi_all["lambda_nm"] <= LAM_MAX)).mean() if len(hi_all) else 0.0

    return {
        "median_prior_highscore": float(sub.loc[hi, "prior"].median()) if hi.any() else None,
        "median_prior_lowscore": float(sub.loc[lo, "prior"].median()) if lo.any() else None,
        "prior_drift_delta": float(sub.loc[hi, "prior"].median() - sub.loc[lo, "prior"].median())
        if hi.any() and lo.any()
        else None,
        "frac_highscore_in_lambda_window": float(in_window),
        "cheap_qed_rate_in_lambda_highscore": float(len(cheap) / max(int((in_lam["score"] >= SCORE_THR).sum()), 1)),
        "n_cheap": int(len(cheap)),
        "n_good_qed": int(len(good)),
    }


def main() -> None:
    set_style()
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)

    df = load_rl()
    print(f"[INFO] valid={df['valid'].sum()}  on_target={df['on_target'].sum()}")

    summary = {
        "definitions": {
            "target_property": f"λ∈[{LAM_MIN},{LAM_MAX}] nm and Score≥{SCORE_THR}, clean+valid",
            "target_chemical_space": "molecules.csv (literature chromophores)",
            "coverage": f"fraction of Murcko-generic scaffolds in molecules.csv with ≥1 RL neighbor Tc≥{TC_COVER}",
            "recall": "fraction of molecules.csv compounds with ≥1 RL neighbor Tc≥τ",
            "scaffold_collapse": "generic Bemis–Murcko dominance / H_norm vs Score (no named motifs)",
            "memorization": "high Score with max_tc_train≥0.7",
            "extrapolation": "high Score with max_tc_train<0.4",
            "novelty_precision": "novelty=1-max_tc_train, precision=Score; success=novelty≥0.6 ∧ on-target",
            "prior_drift": "higher Prior NLL among high-Score vs low-Score",
            "reward_hacking": "high Score with very low QED inside λ window",
        },
        "coverage_recall": task_coverage_recall(df, out),
        "scaffold_collapse": task_scaffold_collapse(df, out),
        "memorization_vs_extrapolation": task_memorization(df, out),
        "novelty_precision_frontier": task_novelty_precision(df, out),
        "prior_drift_reward_hacking": task_prior_hacking(df, out),
    }

    (out / "tables" / "challenge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # annotated export
    cols = [
        c
        for c in [
            "SMILES",
            "canon",
            "score",
            "prior",
            "agent",
            "lambda_nm",
            "qed",
            "max_tc_train",
            "murcko_generic",
            "on_target",
            "valid",
            "clean",
            "step",
        ]
        if c in df.columns
    ]
    df.loc[df["valid"], cols].to_csv(out / "tables" / "rl_challenge_annotated.csv", index=False)

    cr = summary["coverage_recall"]
    sc = summary["scaffold_collapse"]["high_score"]
    me = summary["memorization_vs_extrapolation"]
    npf = summary["novelty_precision_frontier"]
    ph = summary["prior_drift_reward_hacking"]

    report = f"""# Challenge report

Five exams of the RL generator. One figure per exam.

## Is this set enough?
Yes for a first challenge battery, with Coverage defined against **molecules.csv** as the
target chemical space (not just the λ window). Together they probe: reach of target chemistry,
mode collapse, generalization, joint novelty×quality, and reward/prior pathologies.

## Results (short)

| # | Exam | Key number | Figure |
|---|---|---|---|
| 1 | Coverage / Recall | Coverage@{TC_COVER:.2f}={cr['coverage_scaffold_at_tc']:.1%}; Recall@{TC_COVER:.2f}={cr['recall_at_tc_cover']:.1%} | `01_coverage_recall.png` |
| 2 | Scaffold collapse | dominance={sc['dominance']:.3f}, H_norm={sc['entropy_norm']:.3f}, N_eff={sc['n_eff']:.0f} | `02_scaffold_collapse.png` |
| 3 | Memorization vs extrapolation | extrapolate={me['frac_extrapolate_of_highscore']:.1%}, memorize={me['frac_memorize_of_highscore']:.1%} of high-Score | `03_memorization_vs_extrapolation.png` |
| 4 | Novelty–Precision frontier | true successes={npf['n_true_success']} ({npf['frac_true_success_of_on_target']:.1%} of on-target) | `04_novelty_precision_frontier.png` |
| 5 | Prior drift / hacking | ΔPrior(high−low)={ph['prior_drift_delta']:.1f}; cheap QED rate={ph['cheap_qed_rate_in_lambda_highscore']:.1%} | `05_prior_drift_reward_hacking.png` |

## Reading guide
- **Low Coverage/Recall** → generator does not populate literature chromophore space.
- **Dominance↑ / H_norm↓ with Score** → scaffold collapse under RL.
- **Few extrapolation points** → no generalization beyond train neighborhood.
- **Empty success quadrant** on frontier → cannot be novel *and* on-target at once.
- **Prior↑ with Score + cheap QED** → drifted from prior and/or reward hacking the surrogate.
"""
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"[DONE] → {out}")


if __name__ == "__main__":
    main()
