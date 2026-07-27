#!/usr/bin/env python3
"""
Tanimoto evaluation for REINVENT4 sampling output.

The script compares generated molecules, e.g. finetuned.csv, against a reference
set, e.g. train.smi / val.smi / test.smi, using Morgan fingerprints and
Tanimoto similarity.

Main outputs:
  tanimoto_eval/summary_metrics.csv
  tanimoto_eval/generated_with_tanimoto.csv
  tanimoto_eval/tanimoto_distribution.csv
  tanimoto_eval/tanimoto_histogram.png
  tanimoto_eval/top_most_similar.csv
  tanimoto_eval/top_most_novel.csv

Examples:
  python evaluate_tanimoto_reinvent.py \
    --generated finetuned.csv \
    --reference train.smi \
    --out_dir tanimoto_train_eval

  python evaluate_tanimoto_reinvent.py \
    --generated finetuned.csv \
    --reference train.smi \
    --reference val.smi \
    --out_dir tanimoto_train_val_eval

  python evaluate_tanimoto_reinvent.py \
    --generated samples/epoch_010.csv \
    --reference train.smi \
    --smiles_col SMILES

Dependencies:
  conda install -c conda-forge rdkit pandas numpy tqdm matplotlib seaborn
or
  pip install rdkit pandas numpy tqdm matplotlib seaborn
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")


# Common column names in REINVENT/RDKit workflows. The script can also guess.
DEFAULT_SMILES_COLUMNS = [
    "SMILES",
    "smiles",
    "Smiles",
    "canonical_smiles",
    "sampled_smiles",
    "generated_smiles",
    "input_smiles",
    "randomized_smiles",
    "mol",
    "molecule",
]


def read_smiles(path: Path, smiles_col: str | None = None) -> pd.DataFrame:
    """
    Read SMILES from .csv/.tsv/.smi/.smiles/.txt.

    Returns a DataFrame with columns:
      source_file, row_id, smiles_raw
    """
    suffix = path.suffix.lower()

    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)

        col = smiles_col
        if col is None:
            for c in DEFAULT_SMILES_COLUMNS:
                if c in df.columns:
                    col = c
                    break

        # If no known column found, try the first object/string column that looks like SMILES.
        if col is None:
            candidate_cols = list(df.columns)
            for c in candidate_cols:
                values = df[c].dropna().astype(str).head(50).tolist()
                if not values:
                    continue
                n_valid = sum(Chem.MolFromSmiles(v) is not None for v in values)
                if n_valid >= max(3, int(0.5 * len(values))):
                    col = c
                    break

        if col is None or col not in df.columns:
            raise ValueError(
                f"Cannot detect SMILES column in {path}. Columns are: {list(df.columns)}. "
                f"Use --smiles_col COLUMN_NAME."
            )

        out = pd.DataFrame(
            {
                "source_file": str(path),
                "row_id": np.arange(len(df)),
                "smiles_raw": df[col].astype(str),
            }
        )
        # Preserve original columns with prefix, useful for REINVENT scores/NLL/etc.
        for c in df.columns:
            if c != col:
                out[f"input__{c}"] = df[c]
        return out

    if suffix in {".smi", ".smiles", ".txt"}:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Common .smi format: first token is SMILES, rest can be molecule name.
                parts = line.split()
                rows.append(
                    {
                        "source_file": str(path),
                        "row_id": i,
                        "smiles_raw": parts[0],
                        "input__name": " ".join(parts[1:]) if len(parts) > 1 else "",
                    }
                )
        return pd.DataFrame(rows)

    raise ValueError(f"Unsupported file extension for {path}. Use .csv, .tsv, .smi, .smiles or .txt")


def canonicalize_one(smiles: str) -> tuple[str | None, str | None]:
    """Return canonical isomeric SMILES and error string if invalid."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None, "MolFromSmiles returned None"
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), None
    except Exception as e:
        return None, str(e)


def canonicalize_df(df: pd.DataFrame) -> pd.DataFrame:
    can = []
    errors = []
    for smi in df["smiles_raw"].tolist():
        c, err = canonicalize_one(smi)
        can.append(c)
        errors.append(err)
    out = df.copy()
    out["canonical_smiles"] = can
    out["is_valid"] = out["canonical_smiles"].notna()
    out["error"] = errors
    return out


def build_fps(smiles: Iterable[str], radius: int, n_bits: int):
    """Build Morgan fingerprints for canonical SMILES."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    kept_smiles = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fps.append(generator.GetFingerprint(mol))
        kept_smiles.append(smi)
    return kept_smiles, fps


def nearest_neighbor_tanimoto(query_fps, ref_fps, ref_smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    For each query molecule: maximum Tanimoto to reference set and closest reference SMILES.
    """
    max_sims = []
    nearest_smiles = []

    if len(ref_fps) == 0:
        return np.full(len(query_fps), np.nan), [""] * len(query_fps)

    for fp in tqdm(query_fps, desc="Nearest-neighbor Tanimoto to reference"):
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        if not sims:
            max_sims.append(np.nan)
            nearest_smiles.append("")
        else:
            idx = int(np.argmax(sims))
            max_sims.append(float(sims[idx]))
            nearest_smiles.append(ref_smiles[idx])

    return np.asarray(max_sims, dtype=float), nearest_smiles


def sampled_pairwise_tanimoto(fps, n_pairs: int, seed: int = 1) -> np.ndarray:
    """Sample random pairwise Tanimoto values within generated molecules."""
    n = len(fps)
    if n < 2 or n_pairs <= 0:
        return np.asarray([], dtype=float)

    max_possible = n * (n - 1) // 2
    m = min(n_pairs, max_possible)
    rng = np.random.default_rng(seed)

    values = []
    seen = set()
    attempts = 0
    max_attempts = m * 20
    while len(values) < m and attempts < max_attempts:
        i, j = rng.choice(n, size=2, replace=False)
        if i > j:
            i, j = j, i
        if (i, j) in seen:
            attempts += 1
            continue
        seen.add((i, j))
        values.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
        attempts += 1

    return np.asarray(values, dtype=float)


def distribution_summary(values: np.ndarray, prefix: str) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_p05": np.nan,
            f"{prefix}_p10": np.nan,
            f"{prefix}_p25": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p75": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_p95": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_n": int(len(values)),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_p05": float(np.percentile(values, 5)),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p25": float(np.percentile(values, 25)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p75": float(np.percentile(values, 75)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }


def fraction_thresholds(values: np.ndarray, prefix: str) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {}

    out = {}
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
        key_thr = str(thr).replace(".", "_")
        out[f"{prefix}_fraction_ge_{key_thr}"] = float(np.mean(values >= thr))
        out[f"{prefix}_fraction_lt_{key_thr}"] = float(np.mean(values < thr))
    return out


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_histogram(values: np.ndarray, out_png: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        values = values[~np.isnan(values)]
        plt.figure(figsize=(8, 5))
        sns.histplot(values, bins=40, kde=True)
        plt.xlim(0, 1)
        plt.xlabel("Maximum Tanimoto similarity to reference")
        plt.ylabel("Number of generated molecules")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
    except Exception as e:
        print(f"Plotting skipped: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate REINVENT-generated molecules by Tanimoto similarity."
    )
    parser.add_argument(
        "--generated",
        required=True,
        help="Generated molecules file, e.g. finetuned.csv from REINVENT sampling.",
    )
    parser.add_argument(
        "--reference",
        action="append",
        required=True,
        help=(
            "Reference SMILES file. Can be used multiple times, e.g. "
            "--reference train.smi --reference val.smi"
        ),
    )
    parser.add_argument(
        "--smiles_col",
        default=None,
        help="SMILES column name for CSV/TSV files. If omitted, the script tries to detect it.",
    )
    parser.add_argument(
        "--out_dir",
        default="tanimoto_eval",
        help="Output directory.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius. radius=2 corresponds to ECFP4-like fingerprints.",
    )
    parser.add_argument(
        "--n_bits",
        type=int,
        default=2048,
        help="Morgan fingerprint bit length.",
    )
    parser.add_argument(
        "--pair_sample",
        type=int,
        default=20000,
        help="Number of random generated-generated pairs for internal diversity estimate.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=100,
        help="Number of most similar / most novel molecules to save.",
    )
    args = parser.parse_args()

    generated_path = Path(args.generated)
    reference_paths = [Path(p) for p in args.reference]
    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)

    # ---------- Read and canonicalize generated molecules ----------
    gen_df = read_smiles(generated_path, smiles_col=args.smiles_col)
    gen_df = canonicalize_df(gen_df)

    raw_n = len(gen_df)
    valid_df = gen_df[gen_df["is_valid"]].copy()
    valid_n = len(valid_df)
    invalid_n = raw_n - valid_n

    # Mark duplicates among valid generated molecules.
    valid_df["is_duplicate_generated"] = valid_df.duplicated("canonical_smiles", keep="first")
    unique_valid_df = valid_df.drop_duplicates("canonical_smiles", keep="first").copy()
    unique_valid_n = len(unique_valid_df)

    print(f"Generated raw molecules:       {raw_n}")
    print(f"Generated valid molecules:     {valid_n}")
    print(f"Generated invalid molecules:   {invalid_n}")
    print(f"Generated unique valid mols:   {unique_valid_n}")

    # ---------- Read and canonicalize reference molecules ----------
    ref_parts = []
    for p in reference_paths:
        rdf = read_smiles(p, smiles_col=args.smiles_col)
        rdf["reference_label"] = p.stem
        ref_parts.append(rdf)
    ref_df = pd.concat(ref_parts, ignore_index=True)
    ref_df = canonicalize_df(ref_df)
    ref_valid_df = ref_df[ref_df["is_valid"]].copy()
    ref_unique_df = ref_valid_df.drop_duplicates("canonical_smiles", keep="first").copy()

    print(f"Reference raw molecules:       {len(ref_df)}")
    print(f"Reference valid molecules:     {len(ref_valid_df)}")
    print(f"Reference unique valid mols:   {len(ref_unique_df)}")

    # ---------- Fingerprints ----------
    gen_smiles, gen_fps = build_fps(
        unique_valid_df["canonical_smiles"].tolist(),
        radius=args.radius,
        n_bits=args.n_bits,
    )
    ref_smiles, ref_fps = build_fps(
        ref_unique_df["canonical_smiles"].tolist(),
        radius=args.radius,
        n_bits=args.n_bits,
    )

    # Keep only molecules that were successfully fingerprinted.
    fp_gen_df = unique_valid_df[unique_valid_df["canonical_smiles"].isin(gen_smiles)].copy()
    fp_gen_df = fp_gen_df.set_index("canonical_smiles").loc[gen_smiles].reset_index()

    # ---------- Generated -> reference nearest-neighbor Tanimoto ----------
    max_tani, nearest_ref = nearest_neighbor_tanimoto(gen_fps, ref_fps, ref_smiles)
    fp_gen_df["max_tanimoto_to_reference"] = max_tani
    fp_gen_df["nearest_reference_smiles"] = nearest_ref
    fp_gen_df["is_exact_in_reference"] = fp_gen_df["max_tanimoto_to_reference"] >= 0.999999

    # ---------- Internal diversity among generated molecules ----------
    pairwise_values = sampled_pairwise_tanimoto(gen_fps, n_pairs=args.pair_sample, seed=1)
    mean_pairwise = float(np.mean(pairwise_values)) if len(pairwise_values) else np.nan
    internal_diversity = 1.0 - mean_pairwise if not math.isnan(mean_pairwise) else np.nan

    # ---------- Summary metrics ----------
    summary = {
        "generated_file": str(generated_path),
        "reference_files": ";".join(str(p) for p in reference_paths),
        "fingerprint": f"Morgan_radius_{args.radius}_nBits_{args.n_bits}",
        "generated_raw_n": raw_n,
        "generated_valid_n": valid_n,
        "generated_invalid_n": invalid_n,
        "generated_validity": valid_n / raw_n if raw_n else np.nan,
        "generated_unique_valid_n": unique_valid_n,
        "generated_uniqueness_among_valid": unique_valid_n / valid_n if valid_n else np.nan,
        "reference_raw_n": len(ref_df),
        "reference_valid_n": len(ref_valid_df),
        "reference_unique_valid_n": len(ref_unique_df),
        "exact_generated_in_reference_n": int(fp_gen_df["is_exact_in_reference"].sum()),
        "exact_generated_in_reference_fraction_unique_valid": float(fp_gen_df["is_exact_in_reference"].mean()) if len(fp_gen_df) else np.nan,
        "sampled_pairwise_tanimoto_generated_mean": mean_pairwise,
        "internal_diversity_1_minus_pairwise_mean": internal_diversity,
        "sampled_pairwise_pairs_n": int(len(pairwise_values)),
    }
    summary.update(distribution_summary(max_tani, "max_tanimoto_to_reference"))
    summary.update(fraction_thresholds(max_tani, "max_tanimoto_to_reference"))

    summary_df = pd.DataFrame([summary])

    # ---------- Distribution table ----------
    dist_df = pd.DataFrame({"max_tanimoto_to_reference": max_tani})
    bins = np.linspace(0, 1, 21)  # 0.05 bins
    dist_df["bin"] = pd.cut(dist_df["max_tanimoto_to_reference"], bins=bins, include_lowest=True)
    hist_df = (
        dist_df.groupby("bin", observed=False)
        .size()
        .reset_index(name="count")
    )
    hist_df["fraction"] = hist_df["count"] / hist_df["count"].sum() if hist_df["count"].sum() else np.nan

    # ---------- Save outputs ----------
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    fp_gen_df.to_csv(out_dir / "generated_with_tanimoto.csv", index=False)
    hist_df.to_csv(out_dir / "tanimoto_distribution.csv", index=False)

    fp_gen_df.sort_values("max_tanimoto_to_reference", ascending=False).head(args.top_n).to_csv(
        out_dir / "top_most_similar.csv", index=False
    )
    fp_gen_df.sort_values("max_tanimoto_to_reference", ascending=True).head(args.top_n).to_csv(
        out_dir / "top_most_novel.csv", index=False
    )

    if len(pairwise_values):
        pd.DataFrame({"sampled_pairwise_tanimoto_generated": pairwise_values}).to_csv(
            out_dir / "sampled_pairwise_tanimoto_generated.csv", index=False
        )

    plot_histogram(
        max_tani,
        out_dir / "tanimoto_histogram.png",
        title="Generated molecules: max Tanimoto to reference",
    )

    print("\nSaved files:")
    for name in [
        "summary_metrics.csv",
        "generated_with_tanimoto.csv",
        "tanimoto_distribution.csv",
        "tanimoto_histogram.png",
        "top_most_similar.csv",
        "top_most_novel.csv",
    ]:
        print(f"  {out_dir / name}")

    print("\nMain summary:")
    cols_to_show = [
        "generated_validity",
        "generated_uniqueness_among_valid",
        "exact_generated_in_reference_fraction_unique_valid",
        "max_tanimoto_to_reference_mean",
        "max_tanimoto_to_reference_median",
        "max_tanimoto_to_reference_p90",
        "max_tanimoto_to_reference_fraction_ge_0_7",
        "max_tanimoto_to_reference_fraction_ge_0_85",
        "internal_diversity_1_minus_pairwise_mean",
    ]
    for c in cols_to_show:
        if c in summary:
            print(f"  {c}: {summary[c]}")


if __name__ == "__main__":
    main()
