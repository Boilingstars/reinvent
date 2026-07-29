"""
eval_summary_table.py — merge numerical metrics from sibling scripts into one
table and recommend underfit / optimal / overfit epochs.

Expects CSVs produced by:
  eval_tanimoto, eval_fcd, eval_mahalanobis, eval_hitrate, eval_novelty_diversity

Heuristic (set-level, not nearest-neighbour):
  * underfit  — early epochs: low similarity / high FCD to chromophores,
                high similarity to ChEMBL
  * overfit   — late epochs: rising max-Tanimoto / falling novelty to train,
                FCD to chromophores may keep falling while diversity drops
  * optimal   — best trade-off: low FFD/FCD to chromophores, still high novelty
                & diversity, chromophore-like hit-rate not collapsing uniqueness

Usage:
  # first run the metric scripts, then:
  python epoch_eval/eval_summary_table.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DEFAULT_OUT, ensure_dir


METRIC_FILES = {
    "tanimoto": "tanimoto/tanimoto_by_epoch.csv",
    "fcd": "fcd/fcd_by_epoch.csv",
    "mahalanobis": "mahalanobis/mahalanobis_by_epoch.csv",
    "hitrate": "hitrate/hitrate_by_epoch.csv",
    "novelty": "novelty_diversity/novelty_diversity_by_epoch.csv",
}


def _load(eval_root: Path, rel: str) -> pd.DataFrame | None:
    path = eval_root / rel
    if not path.is_file():
        print(f"[WARN] missing {path}")
        return None
    return pd.read_csv(path)


def merge_tables(eval_root: Path) -> pd.DataFrame:
    frames = []
    for key, rel in METRIC_FILES.items():
        df = _load(eval_root, rel)
        if df is None:
            continue
        # keep epoch + metric columns; prefix non-epoch cols with source
        cols = {}
        for c in df.columns:
            if c == "epoch":
                continue
            cols[c] = f"{key}__{c}"
        frames.append(df.rename(columns=cols))

    if not frames:
        raise SystemExit(
            f"No metric CSVs under {eval_root}. Run the individual eval_*.py scripts first."
        )

    out = frames[0]
    for fr in frames[1:]:
        out = out.merge(fr, on="epoch", how="outer")
    return out.sort_values("epoch").reset_index(drop=True)


def recommend_epochs(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime labels and a composite score."""
    eps = df["epoch"].to_numpy()

    # Prefer lower distance to chromophores, higher novelty & diversity
    def col(*names: str) -> np.ndarray | None:
        for n in names:
            if n in df.columns and df[n].notna().any():
                return df[n].to_numpy(dtype=float)
        return None

    fcd_c = col("fcd__fcd_vs_chromophores", "fcd__ffd_vs_chromophores")
    fcd_d = col("fcd__fcd_vs_chembl", "fcd__ffd_vs_chembl")
    tc_mean = col("tanimoto__mean_max_tanimoto_vs_chromophores")
    tc_cross = col("tanimoto__cross_mean_vs_chromophores")
    nov = col("novelty__novelty_vs_chromophores")
    div = col("novelty__internal_diversity")
    hit = col("hitrate__hitrate_chromophore_like")
    mah_c = col("mahalanobis__mean_mahal_vs_chromophores")

    n = len(df)
    score = np.zeros(n)

    def _norm_good_high(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        lo, hi = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(lo) or hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    def _norm_good_low(x: np.ndarray) -> np.ndarray:
        return 1.0 - _norm_good_high(x)

    if fcd_c is not None:
        score += 1.5 * _norm_good_low(fcd_c)
    if mah_c is not None:
        score += 1.0 * _norm_good_low(mah_c)
    if tc_cross is not None:
        # want moderate similarity to chromophores — penalize extremes via distance to mid-high target
        target = 0.15  # cross-mean Tc is typically small
        score += 0.5 * _norm_good_low(np.abs(tc_cross - target))
    if tc_mean is not None:
        # mean-max to train: sweet spot ~0.35–0.45; rising further → memorisation
        score += 1.0 * _norm_good_low(np.abs(tc_mean - 0.40))
    if nov is not None:
        score += 1.2 * _norm_good_high(nov)
    if div is not None:
        score += 1.0 * _norm_good_high(div)
    if hit is not None:
        score += 1.0 * _norm_good_high(hit)
    if fcd_d is not None:
        # drifting away from drug space is expected; mild preference for not collapsing back
        score += 0.3 * _norm_good_high(fcd_d)

    best_idx = int(np.nanargmax(score))
    best_ep = int(eps[best_idx])

    # regime by position relative to best and Tanimoto trend
    regimes = []
    for i, ep in enumerate(eps):
        if i < best_idx:
            regimes.append("underfit / early")
        elif i == best_idx:
            regimes.append("recommended")
        else:
            # if novelty drops or mean-max to train rises sharply → overfit
            over = False
            if nov is not None and tc_mean is not None:
                if nov[i] < nov[best_idx] - 0.02 and tc_mean[i] > tc_mean[best_idx] + 0.03:
                    over = True
            if fcd_c is not None and div is not None:
                if fcd_c[i] <= fcd_c[best_idx] and div[i] < div[best_idx] - 0.01:
                    over = True
            regimes.append("overfit / late" if over else "late (monitor)")

    out = df.copy()
    out["composite_score"] = score
    out["regime"] = regimes
    out["recommended_epoch"] = best_ep
    return out


def write_interpretation(df: pd.DataFrame, path: Path) -> None:
    best = int(df["recommended_epoch"].iloc[0])
    lines = [
        "# Epoch recommendation (set-level metrics)",
        "",
        f"Recommended epoch: **{best}**",
        "",
        "How to read regimes:",
        "- underfit / early — still close to drug-like prior; chromophore match weak",
        "- recommended — best composite of chromophore proximity, novelty, diversity, hit-rate",
        "- overfit / late — memorising train set (↑ max-Tc / ↓ novelty / ↓ diversity)",
        "",
        "Note: comparisons use full set statistics (cross-mean / distributional distances),",
        "not single nearest-neighbour pairs.",
        "",
        df[
            [
                c
                for c in (
                    "epoch",
                    "regime",
                    "composite_score",
                    "tanimoto__mean_max_tanimoto_vs_chromophores",
                    "tanimoto__cross_mean_vs_chromophores",
                    "fcd__ffd_vs_chromophores",
                    "novelty__novelty_vs_chromophores",
                    "novelty__internal_diversity",
                    "hitrate__hitrate_chromophore_like",
                )
                if c in df.columns
            ]
        ].to_string(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Merge metric tables and recommend stop epoch")
    p.add_argument("--eval-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    out = ensure_dir(args.out_dir or args.eval_root / "summary")

    merged = merge_tables(args.eval_root)
    scored = recommend_epochs(merged)
    csv_path = out / "metrics_summary_by_epoch.csv"
    scored.to_csv(csv_path, index=False)
    write_interpretation(scored, out / "epoch_recommendation.md")
    print(scored[["epoch", "regime", "composite_score", "recommended_epoch"]].to_string(index=False))
    print(f"[OK] {csv_path}")
    print(f"[OK] {out / 'epoch_recommendation.md'}")


if __name__ == "__main__":
    main()
