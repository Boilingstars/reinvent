#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust visualisation of REINVENT sampling files by epoch.

This version avoids the pandas duplicate-SMILES indexing bug that caused:
  ValueError: Length of values (...) does not match length of index (...)

Run example:
  python vizual_fixed.py --epochs_dir samples_by_epoch --reference train.smi --out_dir epoch_visualization_fixed

If SMILES column is not detected automatically:
  python vizual_fixed.py --epochs_dir samples_by_epoch --reference train.smi --smiles_col SMILES --out_dir epoch_visualization_fixed
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

SMILES_COLUMNS = [
    "SMILES", "smiles", "Smiles", "canonical_smiles", "sampled_smiles",
    "generated_smiles", "input_smiles", "randomized_smiles", "mol", "molecule"
]


def epoch_from_filename(path: Path) -> str:
    m = re.search(r"(?:epoch|ep|checkpoint|ckpt)[_\- ]*(\d+)", path.stem, flags=re.I)
    if m:
        return str(int(m.group(1)))
    m = re.search(r"(\d+)", path.stem)
    if m:
        return str(int(m.group(1)))
    return path.stem


def epoch_sort_key(x):
    s = str(x)
    return (0, int(s)) if s.isdigit() else (1, s)


def read_smiles_file(path: Path, smiles_col: str | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix in [".csv", ".tsv"]:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)

        col = smiles_col
        if col is None:
            for c in SMILES_COLUMNS:
                if c in df.columns:
                    col = c
                    break

        if col is None:
            # Try to guess a SMILES-like column.
            for c in df.columns:
                vals = df[c].dropna().astype(str).head(50).tolist()
                if not vals:
                    continue
                n_valid = sum(Chem.MolFromSmiles(v) is not None for v in vals)
                if n_valid >= max(3, int(0.5 * len(vals))):
                    col = c
                    break

        if col is None or col not in df.columns:
            raise ValueError(
                f"Cannot find SMILES column in {path}. Columns: {list(df.columns)}. "
                f"Run with --smiles_col COLUMN_NAME"
            )

        return pd.DataFrame({"smiles_raw": df[col].astype(str), "source_file": str(path)})

    if suffix in [".smi", ".smiles", ".txt"]:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rows.append({"smiles_raw": line.split()[0], "source_file": str(path)})
        return pd.DataFrame(rows)

    raise ValueError(f"Unsupported file format: {path}")


def canonicalize(smi: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def fingerprint_dataframe(df: pd.DataFrame, radius: int, n_bits: int):
    """
    Returns df2, fps, X. df2 has exactly the same number/order as fps and X.
    This is the important anti-bug part: no set_index(...).loc[...] by SMILES.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    df = df.reset_index(drop=True).copy()

    keep_rows = []
    fps = []
    arrs = []

    for i, smi in enumerate(df["canonical_smiles"].tolist()):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        keep_rows.append(i)
        fps.append(fp)
        arrs.append(arr)

    df2 = df.iloc[keep_rows].reset_index(drop=True)
    X = np.vstack(arrs) if arrs else np.zeros((0, n_bits), dtype=np.int8)
    return df2, fps, X


def max_tanimoto_to_ref(query_fps, ref_fps, ref_smiles):
    vals = []
    nearest = []
    if not ref_fps:
        return np.full(len(query_fps), np.nan), [""] * len(query_fps)

    for fp in tqdm(query_fps, desc="Tanimoto to reference"):
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        idx = int(np.argmax(sims))
        vals.append(float(sims[idx]))
        nearest.append(ref_smiles[idx])
    return np.array(vals, dtype=float), nearest


def internal_diversity(fps, n_pairs=20000, seed=1):
    if len(fps) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(fps)
    m = min(n_pairs, n * (n - 1) // 2)
    vals = []
    seen = set()
    attempts = 0
    while len(vals) < m and attempts < m * 20:
        i, j = rng.choice(n, 2, replace=False)
        if i > j:
            i, j = j, i
        if (i, j) not in seen:
            seen.add((i, j))
            vals.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
        attempts += 1
    mean_pair = float(np.mean(vals)) if vals else np.nan
    div = 1.0 - mean_pair if not np.isnan(mean_pair) else np.nan
    return mean_pair, div


def make_plots(mol_df, cloud_df, X, summary_df, out_dir: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA

    # Histograms and violin plots if Tanimoto exists.
    if "max_tanimoto_to_reference" in mol_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=mol_df,
            x="max_tanimoto_to_reference",
            hue="epoch",
            bins=30,
            stat="density",
            common_norm=False,
            element="step",
            fill=False,
        )
        plt.xlim(0, 1)
        plt.xlabel("Max Tanimoto to reference/train")
        plt.ylabel("Density")
        plt.title("Tanimoto distribution by epoch")
        plt.tight_layout()
        plt.savefig(out_dir / "tanimoto_hist_by_epoch.png", dpi=220)
        plt.close()

        order = sorted(mol_df["epoch"].astype(str).unique(), key=epoch_sort_key)
        plt.figure(figsize=(max(9, 0.8 * len(order)), 6))
        sns.violinplot(data=mol_df, x="epoch", y="max_tanimoto_to_reference", order=order, inner="quartile", cut=0)
        sns.stripplot(data=mol_df, x="epoch", y="max_tanimoto_to_reference", order=order, color="black", alpha=0.25, size=2)
        plt.ylim(0, 1)
        plt.xlabel("Epoch")
        plt.ylabel("Max Tanimoto to reference/train")
        plt.title("Tanimoto by epoch")
        plt.tight_layout()
        plt.savefig(out_dir / "tanimoto_violin_by_epoch.png", dpi=220)
        plt.close()

    # Metrics by epoch.
    metric_cols = [
        "validity", "uniqueness", "mean_max_tanimoto_to_reference",
        "median_max_tanimoto_to_reference", "fraction_ge_0_7", "fraction_ge_0_85",
        "internal_diversity"
    ]
    metric_cols = [c for c in metric_cols if c in summary_df.columns]
    if metric_cols:
        plot_df = summary_df.copy()
        plot_df["epoch_num"] = pd.to_numeric(plot_df["epoch"], errors="coerce")
        if plot_df["epoch_num"].notna().all():
            plot_df = plot_df.sort_values("epoch_num")
            x = plot_df["epoch_num"]
            xlabel = "Epoch"
        else:
            plot_df = plot_df.sort_values("epoch", key=lambda s: s.astype(str).map(epoch_sort_key))
            x = np.arange(len(plot_df))
            xlabel = "Epoch index"

        fig, axes = plt.subplots(len(metric_cols), 1, figsize=(9, max(4, 2.5 * len(metric_cols))), sharex=True)
        if len(metric_cols) == 1:
            axes = [axes]
        for ax, c in zip(axes, metric_cols):
            ax.plot(x, plot_df[c], marker="o")
            ax.set_ylabel(c)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel(xlabel)
        plt.tight_layout()
        plt.savefig(out_dir / "metrics_by_epoch.png", dpi=220)
        plt.close()

    # PCA chemical-space cloud.
    print(f"PCA input: table rows={len(cloud_df)}, fingerprint rows={X.shape[0]}")
    if len(cloud_df) != X.shape[0]:
        # This should never happen in this fixed script.
        n = min(len(cloud_df), X.shape[0])
        print(f"WARNING: length mismatch; cutting both to {n}")
        cloud_df = cloud_df.iloc[:n].reset_index(drop=True)
        X = X[:n]

    if X.shape[0] >= 2:
        print("Computing PCA embedding...")
        xy = PCA(n_components=2, random_state=1).fit_transform(X.astype(np.float32))
        pca_df = cloud_df.reset_index(drop=True).copy()
        pca_df["PCA1"] = xy[:, 0]
        pca_df["PCA2"] = xy[:, 1]
        pca_df.to_csv(out_dir / "embedding_pca.csv", index=False)

        plt.figure(figsize=(9, 7))
        sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="epoch", s=16, alpha=0.65, linewidth=0)
        plt.title("Chemical space cloud: Morgan fingerprints + PCA")
        plt.tight_layout()
        plt.savefig(out_dir / "chemical_space_pca.png", dpi=220)
        plt.close()

    # Optional UMAP.
    try:
        import umap
        if X.shape[0] >= 5:
            print("Computing UMAP embedding...")
            reducer = umap.UMAP(n_components=2, metric="jaccard", n_neighbors=25, min_dist=0.15, random_state=1)
            xy = reducer.fit_transform(X.astype(bool))
            umap_df = cloud_df.reset_index(drop=True).copy()
            umap_df["UMAP1"] = xy[:, 0]
            umap_df["UMAP2"] = xy[:, 1]
            umap_df.to_csv(out_dir / "embedding_umap.csv", index=False)

            plt.figure(figsize=(9, 7))
            sns.scatterplot(data=umap_df, x="UMAP1", y="UMAP2", hue="epoch", s=16, alpha=0.65, linewidth=0)
            plt.title("Chemical space cloud: Morgan fingerprints + UMAP")
            plt.tight_layout()
            plt.savefig(out_dir / "chemical_space_umap.png", dpi=220)
            plt.close()
    except Exception as e:
        print(f"UMAP skipped: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_dir", required=True)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--smiles_col", default=None)
    parser.add_argument("--out_dir", default="epoch_visualization_fixed")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n_bits", type=int, default=2048)
    parser.add_argument("--max_molecules_per_epoch", type=int, default=5000)
    parser.add_argument("--max_reference_molecules", type=int, default=5000)
    parser.add_argument("--pair_sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for ext in ["*.csv", "*.tsv", "*.smi", "*.smiles", "*.txt"]:
        files.extend(Path(args.epochs_dir).glob(ext))
    files = sorted(set(files), key=lambda p: epoch_sort_key(epoch_from_filename(p)))
    if not files:
        raise FileNotFoundError(f"No files found in {args.epochs_dir}")

    all_epoch_dfs = []
    summary_rows = []

    for path in files:
        epoch = epoch_from_filename(path)
        print(f"Reading epoch {epoch}: {path}")
        raw = read_smiles_file(path, smiles_col=args.smiles_col)
        raw_n = len(raw)
        raw["canonical_smiles"] = [canonicalize(s) for s in raw["smiles_raw"].tolist()]
        valid = raw[raw["canonical_smiles"].notna()].copy()
        valid_n = len(valid)

        # Drop duplicates only within this epoch, not across epochs.
        valid_unique = valid.drop_duplicates("canonical_smiles", keep="first").copy()
        unique_n = len(valid_unique)

        if args.max_molecules_per_epoch and len(valid_unique) > args.max_molecules_per_epoch:
            valid_unique = valid_unique.sample(args.max_molecules_per_epoch, random_state=args.seed).reset_index(drop=True)

        valid_unique["epoch"] = epoch
        valid_unique["set"] = "generated"
        fp_df, fps, X = fingerprint_dataframe(valid_unique, args.radius, args.n_bits)
        all_epoch_dfs.append(fp_df)

        mean_pair, div = internal_diversity(fps, n_pairs=args.pair_sample, seed=args.seed)
        summary_rows.append({
            "epoch": epoch,
            "file": str(path),
            "raw_n": raw_n,
            "valid_n": valid_n,
            "invalid_n": raw_n - valid_n,
            "validity": valid_n / raw_n if raw_n else np.nan,
            "unique_valid_n": unique_n,
            "uniqueness": unique_n / valid_n if valid_n else np.nan,
            "used_for_visualization_n": len(fp_df),
            "mean_pairwise_tanimoto": mean_pair,
            "internal_diversity": div,
        })

    mol_df = pd.concat(all_epoch_dfs, ignore_index=True)

    ref_vis_df = None
    if args.reference:
        print(f"Reading reference: {args.reference}")
        ref = read_smiles_file(Path(args.reference), smiles_col=args.smiles_col)
        ref["canonical_smiles"] = [canonicalize(s) for s in ref["smiles_raw"].tolist()]
        ref = ref[ref["canonical_smiles"].notna()].drop_duplicates("canonical_smiles", keep="first").reset_index(drop=True)
        ref_fp_df, ref_fps, ref_X = fingerprint_dataframe(ref, args.radius, args.n_bits)
        ref_smiles = ref_fp_df["canonical_smiles"].tolist()

        gen_fp_df, gen_fps, gen_X = fingerprint_dataframe(mol_df, args.radius, args.n_bits)
        max_tani, nearest = max_tanimoto_to_ref(gen_fps, ref_fps, ref_smiles)
        mol_df = gen_fp_df.copy()
        mol_df["max_tanimoto_to_reference"] = max_tani
        mol_df["nearest_reference_smiles"] = nearest
        mol_df["exact_in_reference"] = mol_df["max_tanimoto_to_reference"] >= 0.999999

        # Summary Tanimoto by epoch.
        summary_df = pd.DataFrame(summary_rows)
        tani = mol_df.groupby("epoch", observed=False)["max_tanimoto_to_reference"].agg(
            mean_max_tanimoto_to_reference="mean",
            median_max_tanimoto_to_reference="median",
            p90_max_tanimoto_to_reference=lambda s: float(np.percentile(s.dropna(), 90)) if len(s.dropna()) else np.nan,
            exact_fraction=lambda s: float(np.mean(s >= 0.999999)) if len(s) else np.nan,
            fraction_ge_0_7=lambda s: float(np.mean(s >= 0.7)) if len(s) else np.nan,
            fraction_ge_0_85=lambda s: float(np.mean(s >= 0.85)) if len(s) else np.nan,
        ).reset_index()
        summary_df = summary_df.merge(tani, on="epoch", how="left")

        ref_vis_df = ref_fp_df.copy()
        if args.max_reference_molecules and len(ref_vis_df) > args.max_reference_molecules:
            ref_vis_df = ref_vis_df.sample(args.max_reference_molecules, random_state=args.seed).reset_index(drop=True)
        ref_vis_df["epoch"] = "reference"
        ref_vis_df["set"] = "reference"
    else:
        summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values("epoch", key=lambda s: s.astype(str).map(epoch_sort_key))
    mol_df.to_csv(out_dir / "per_molecule_epoch_data.csv", index=False)
    summary_df.to_csv(out_dir / "per_epoch_summary.csv", index=False)

    # Distribution table.
    if "max_tanimoto_to_reference" in mol_df.columns:
        bins = np.linspace(0, 1, 21)
        dist = mol_df[["epoch", "max_tanimoto_to_reference"]].copy()
        dist["bin"] = pd.cut(dist["max_tanimoto_to_reference"], bins=bins, include_lowest=True)
        dist = dist.groupby(["epoch", "bin"], observed=False).size().reset_index(name="count")
        dist["fraction"] = dist.groupby("epoch")["count"].transform(lambda x: x / x.sum() if x.sum() else np.nan)
        dist.to_csv(out_dir / "tanimoto_distribution_by_epoch.csv", index=False)

    # Cloud table and fingerprints.
    cloud_parts = [mol_df[["canonical_smiles", "epoch", "set"]].copy()]
    if ref_vis_df is not None:
        cloud_parts.append(ref_vis_df[["canonical_smiles", "epoch", "set"]].copy())
    cloud_df = pd.concat(cloud_parts, ignore_index=True)
    cloud_df, cloud_fps, cloud_X = fingerprint_dataframe(cloud_df, args.radius, args.n_bits)

    make_plots(mol_df, cloud_df, cloud_X, summary_df, out_dir)

    print("\nDONE. Open this folder:")
    print(out_dir.resolve())
    print("\nMain files:")
    for f in [
        "tanimoto_hist_by_epoch.png", "tanimoto_violin_by_epoch.png", "metrics_by_epoch.png",
        "chemical_space_pca.png", "chemical_space_umap.png", "per_epoch_summary.csv"
    ]:
        if (out_dir / f).exists():
            print("  ", out_dir / f)


if __name__ == "__main__":
    main()
