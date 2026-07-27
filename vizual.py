"""
Visualize REINVENT-generated molecule distributions across epochs.

What it does:
  1) Reads generated SMILES files, one file per epoch/checkpoint.
  2) Optionally reads reference/train SMILES.
  3) Computes Morgan fingerprints.
  4) Computes max Tanimoto to reference for every generated molecule.
  5) Builds visualizations:
       - chemical_space_pca.png
       - chemical_space_umap.png, if umap-learn is installed
       - tanimoto_hist_by_epoch.png, if reference is provided
       - tanimoto_violin_by_epoch.png, if reference is provided
       - metrics_by_epoch.png
       - per_epoch_summary.csv
       - per_molecule_epoch_data.csv
       - embedding_pca.csv
       - embedding_umap.csv, if UMAP is available

Input convention:
  --epochs_dir should contain files such as:
      samples_epoch_002.csv
      samples_epoch_004.csv
      samples_epoch_006.csv
      samples_epoch_008.csv
      samples_epoch_010.csv

Supported file formats:
  .csv, .tsv, .smi, .smiles, .txt

Examples:
  python visualize_epochs_reinvent.py \
    --epochs_dir samples_by_epoch \
    --reference train.smi \
    --out_dir epoch_visualization

  python visualize_epochs_reinvent.py \
    --epochs_dir samples_by_epoch \
    --reference train.smi \
    --smiles_col SMILES \
    --max_molecules_per_epoch 5000 \
    --max_reference_molecules 5000 \
    --out_dir epoch_visualization

Dependencies:
  conda install -c conda-forge rdkit pandas numpy tqdm matplotlib seaborn scikit-learn
Optional for UMAP:
  pip install umap-learn
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")


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

        if col is None:
            # Guess a SMILES-like column.
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
                f"Cannot detect SMILES column in {path}. Columns: {list(df.columns)}. "
                f"Use --smiles_col COLUMN_NAME."
            )

        return pd.DataFrame(
            {
                "source_file": str(path),
                "row_id": np.arange(len(df)),
                "smiles_raw": df[col].astype(str),
            }
        )

    if suffix in {".smi", ".smiles", ".txt"}:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rows.append(
                    {
                        "source_file": str(path),
                        "row_id": i,
                        "smiles_raw": line.split()[0],
                    }
                )
        return pd.DataFrame(rows)

    raise ValueError(f"Unsupported extension: {path}. Use .csv/.tsv/.smi/.smiles/.txt")


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonicalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["canonical_smiles"] = [canonicalize_smiles(s) for s in out["smiles_raw"].tolist()]
    out["is_valid"] = out["canonical_smiles"].notna()
    return out


def epoch_from_filename(path: Path) -> str:
    """Extract epoch label from filename. Works with epoch_002, ep-10, checkpoint_20, etc."""
    m = re.search(r"(?:epoch|ep|checkpoint|ckpt)[_\- ]*(\d+)", path.stem, flags=re.IGNORECASE)
    if m:
        return str(int(m.group(1)))
    m = re.search(r"(\d+)", path.stem)
    if m:
        return str(int(m.group(1)))
    return path.stem


def sort_epoch_labels(labels: Iterable[str]) -> list[str]:
    def key(x: str):
        return (0, int(x)) if str(x).isdigit() else (1, str(x))

    return sorted(labels, key=key)


def collect_epoch_files(epochs_dir: Path) -> list[Path]:
    files = []
    for ext in ["*.csv", "*.tsv", "*.smi", "*.smiles", "*.txt"]:
        files.extend(epochs_dir.glob(ext))
    return sorted(set(files))


def build_fingerprints(smiles: list[str], radius: int, n_bits: int):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    kept_smiles = []
    fps = []
    arrays = []

    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        kept_smiles.append(smi)
        fps.append(fp)
        arrays.append(arr)

    if arrays:
        X = np.vstack(arrays)
    else:
        X = np.zeros((0, n_bits), dtype=np.int8)
    return kept_smiles, fps, X


def max_tanimoto_to_reference(query_fps, ref_fps, ref_smiles: list[str]):
    max_values = []
    nearest_smiles = []

    if len(ref_fps) == 0:
        return np.full(len(query_fps), np.nan), [""] * len(query_fps)

    for fp in tqdm(query_fps, desc="Tanimoto to reference"):
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        idx = int(np.argmax(sims))
        max_values.append(float(sims[idx]))
        nearest_smiles.append(ref_smiles[idx])

    return np.asarray(max_values, dtype=float), nearest_smiles


def sampled_pairwise_tanimoto(fps, n_pairs: int, seed: int = 1) -> np.ndarray:
    n = len(fps)
    if n < 2 or n_pairs <= 0:
        return np.asarray([], dtype=float)

    rng = np.random.default_rng(seed)
    max_possible = n * (n - 1) // 2
    m = min(n_pairs, max_possible)
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


def downsample_df(df: pd.DataFrame, max_n: int | None, seed: int) -> pd.DataFrame:
    if max_n is None or max_n <= 0 or len(df) <= max_n:
        return df
    return df.sample(n=max_n, random_state=seed).reset_index(drop=True)


def make_pca_embedding(X: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=seed)
    return pca.fit_transform(X.astype(np.float32))


def make_umap_embedding(X: np.ndarray, seed: int) -> np.ndarray | None:
    try:
        import umap  # type: ignore
    except Exception:
        return None

    reducer = umap.UMAP(
        n_components=2,
        metric="jaccard",
        n_neighbors=25,
        min_dist=0.15,
        random_state=seed,
    )
    # UMAP with jaccard works naturally with boolean/binary vectors.
    return reducer.fit_transform(X.astype(bool))


def save_scatter(df: pd.DataFrame, x: str, y: str, hue: str, out_png: Path, title: str, alpha: float = 0.75):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        s=16,
        alpha=alpha,
        linewidth=0,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def save_tanimoto_hist(df: pd.DataFrame, out_png: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df,
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
    plt.title("Distribution of max Tanimoto to reference by epoch")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def save_tanimoto_violin(df: pd.DataFrame, out_png: Path, epoch_order: list[str]):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(max(9, 0.7 * len(epoch_order)), 6))
    sns.violinplot(
        data=df,
        x="epoch",
        y="max_tanimoto_to_reference",
        order=epoch_order,
        inner="quartile",
        cut=0,
    )
    sns.stripplot(
        data=df,
        x="epoch",
        y="max_tanimoto_to_reference",
        order=epoch_order,
        color="black",
        alpha=0.25,
        size=2,
        jitter=True,
    )
    plt.ylim(0, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Max Tanimoto to reference/train")
    plt.title("Max Tanimoto to reference by epoch")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def save_metrics_by_epoch(summary_df: pd.DataFrame, out_png: Path):
    import matplotlib.pyplot as plt

    plot_df = summary_df.copy()
    plot_df["epoch_num"] = pd.to_numeric(plot_df["epoch"], errors="coerce")
    if plot_df["epoch_num"].notna().all():
        plot_df = plot_df.sort_values("epoch_num")
        x = plot_df["epoch_num"]
        xlabel = "Epoch"
    else:
        x = np.arange(len(plot_df))
        xlabel = "Epoch index"

    metrics = [
        "validity",
        "uniqueness",
        "mean_max_tanimoto_to_reference",
        "median_max_tanimoto_to_reference",
        "fraction_ge_0_7",
        "fraction_ge_0_85",
        "internal_diversity",
    ]
    metrics = [m for m in metrics if m in plot_df.columns]

    n = len(metrics)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(9, max(3 * n, 5)), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ax.plot(x, plot_df[metric], marker="o")
        ax.set_ylabel(metric)
        if metric not in {"mean_max_tanimoto_to_reference", "median_max_tanimoto_to_reference"}:
            ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel(xlabel)
    fig.suptitle("Metrics by epoch", y=1.0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def save_tanimoto_distribution_table(df: pd.DataFrame, out_csv: Path):
    bins = np.linspace(0, 1, 21)
    tmp = df[["epoch", "max_tanimoto_to_reference"]].copy()
    tmp["bin"] = pd.cut(tmp["max_tanimoto_to_reference"], bins=bins, include_lowest=True)
    dist = tmp.groupby(["epoch", "bin"], observed=False).size().reset_index(name="count")
    dist["fraction"] = dist.groupby("epoch")["count"].transform(lambda s: s / s.sum() if s.sum() else np.nan)
    dist.to_csv(out_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize generated molecule clouds and Tanimoto histograms by epoch.")
    parser.add_argument("--epochs_dir", required=True, help="Directory with one generated SMILES/CSV file per epoch.")
    parser.add_argument("--reference", default=None, help="Optional reference/train SMILES file for Tanimoto-to-train histograms.")
    parser.add_argument("--smiles_col", default=None, help="SMILES column name for CSV/TSV files. If omitted, script tries to detect it.")
    parser.add_argument("--out_dir", default="epoch_visualization", help="Output directory.")
    parser.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius. 2 = ECFP4-like.")
    parser.add_argument("--n_bits", type=int, default=2048, help="Morgan fingerprint bit length.")
    parser.add_argument("--max_molecules_per_epoch", type=int, default=5000, help="Downsample each epoch for visualization; 0 means no limit.")
    parser.add_argument("--max_reference_molecules", type=int, default=5000, help="Downsample reference for cloud plot; 0 means no limit.")
    parser.add_argument("--pair_sample", type=int, default=20000, help="Number of random pairs for internal diversity per epoch.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    args = parser.parse_args()

    epochs_dir = Path(args.epochs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect_epoch_files(epochs_dir)
    if not files:
        raise FileNotFoundError(f"No .csv/.tsv/.smi/.smiles/.txt files found in {epochs_dir}")

    all_rows = []
    summary_rows = []
    epoch_fps = {}

    for path in files:
        epoch = epoch_from_filename(path)
        print(f"Reading epoch {epoch}: {path}")
        df = read_smiles(path, smiles_col=args.smiles_col)
        raw_n = len(df)
        df = canonicalize_df(df)
        valid_df = df[df["is_valid"]].copy()
        valid_n = len(valid_df)
        invalid_n = raw_n - valid_n

        valid_df = valid_df.drop_duplicates("canonical_smiles", keep="first").copy()
        unique_valid_n = len(valid_df)
        valid_df = downsample_df(valid_df, args.max_molecules_per_epoch, args.seed)

        smiles, fps, X = build_fingerprints(valid_df["canonical_smiles"].tolist(), args.radius, args.n_bits)
        valid_df = valid_df[valid_df["canonical_smiles"].isin(smiles)].copy()
        valid_df = valid_df.set_index("canonical_smiles").loc[smiles].reset_index()
        valid_df["epoch"] = epoch
        valid_df["set"] = "generated"
        all_rows.append(valid_df)
        epoch_fps[epoch] = fps

        pair_vals = sampled_pairwise_tanimoto(fps, n_pairs=args.pair_sample, seed=args.seed)
        mean_pair = float(np.mean(pair_vals)) if len(pair_vals) else np.nan
        internal_div = 1.0 - mean_pair if not np.isnan(mean_pair) else np.nan

        summary_rows.append(
            {
                "epoch": epoch,
                "file": str(path),
                "raw_n": raw_n,
                "valid_n": valid_n,
                "invalid_n": invalid_n,
                "validity": valid_n / raw_n if raw_n else np.nan,
                "unique_valid_n": unique_valid_n,
                "uniqueness": unique_valid_n / valid_n if valid_n else np.nan,
                "used_for_visualization_n": len(valid_df),
                "mean_pairwise_tanimoto": mean_pair,
                "internal_diversity": internal_div,
            }
        )

    mol_df = pd.concat(all_rows, ignore_index=True)
    epoch_order = sort_epoch_labels(mol_df["epoch"].unique())
    mol_df["epoch"] = pd.Categorical(mol_df["epoch"], categories=epoch_order, ordered=True)

    # Reference data, optional.
    ref_fps = []
    ref_smiles = []
    ref_df_vis = None

    if args.reference:
        print(f"Reading reference: {args.reference}")
        ref_df = read_smiles(Path(args.reference), smiles_col=args.smiles_col)
        ref_df = canonicalize_df(ref_df)
        ref_df = ref_df[ref_df["is_valid"]].drop_duplicates("canonical_smiles", keep="first").copy()

        ref_smiles, ref_fps, ref_X = build_fingerprints(ref_df["canonical_smiles"].tolist(), args.radius, args.n_bits)

        ref_df_vis = ref_df[ref_df["canonical_smiles"].isin(ref_smiles)].copy()
        ref_df_vis = ref_df_vis.set_index("canonical_smiles").loc[ref_smiles].reset_index()
        ref_df_vis["epoch"] = "reference"
        ref_df_vis["set"] = "reference"
        ref_df_vis = downsample_df(ref_df_vis, args.max_reference_molecules, args.seed)

        # Compute max Tanimoto to reference for all generated molecules.
        gen_smiles, gen_fps, _ = build_fingerprints(mol_df["canonical_smiles"].tolist(), args.radius, args.n_bits)
        max_tani, nearest = max_tanimoto_to_reference(gen_fps, ref_fps, ref_smiles)

        # mol_df order should match gen_smiles after filtering; enforce it.
        mol_df = mol_df[mol_df["canonical_smiles"].isin(gen_smiles)].copy()
        mol_df = mol_df.set_index("canonical_smiles").loc[gen_smiles].reset_index()
        mol_df["max_tanimoto_to_reference"] = max_tani
        mol_df["nearest_reference_smiles"] = nearest
        mol_df["exact_in_reference"] = mol_df["max_tanimoto_to_reference"] >= 0.999999

        # Add Tanimoto summary metrics per epoch.
        summary_df_tmp = pd.DataFrame(summary_rows)
        tani_summary = (
            mol_df.groupby("epoch", observed=False)["max_tanimoto_to_reference"]
            .agg(
                mean_max_tanimoto_to_reference="mean",
                median_max_tanimoto_to_reference="median",
                p90_max_tanimoto_to_reference=lambda s: float(np.percentile(s.dropna(), 90)) if len(s.dropna()) else np.nan,
                exact_fraction=lambda s: float(np.mean(s >= 0.999999)) if len(s) else np.nan,
                fraction_ge_0_7=lambda s: float(np.mean(s >= 0.7)) if len(s) else np.nan,
                fraction_ge_0_85=lambda s: float(np.mean(s >= 0.85)) if len(s) else np.nan,
            )
            .reset_index()
        )
        summary_df = summary_df_tmp.merge(tani_summary, on="epoch", how="left")
    else:
        summary_df = pd.DataFrame(summary_rows)

    # Sort summary.
    summary_df["epoch_sort"] = summary_df["epoch"].apply(lambda x: int(x) if str(x).isdigit() else np.nan)
    if summary_df["epoch_sort"].notna().all():
        summary_df = summary_df.sort_values("epoch_sort")
    else:
        summary_df = summary_df.sort_values("epoch")
    summary_df = summary_df.drop(columns=["epoch_sort"])

    # Save molecule-level and summary data.
    mol_df.to_csv(out_dir / "per_molecule_epoch_data.csv", index=False)
    summary_df.to_csv(out_dir / "per_epoch_summary.csv", index=False)

    # Histograms/violin by epoch, if reference provided.
    if args.reference and "max_tanimoto_to_reference" in mol_df.columns:
        save_tanimoto_hist(mol_df, out_dir / "tanimoto_hist_by_epoch.png")
        save_tanimoto_violin(mol_df, out_dir / "tanimoto_violin_by_epoch.png", epoch_order=epoch_order)
        save_tanimoto_distribution_table(mol_df, out_dir / "tanimoto_distribution_by_epoch.csv")

    # Metrics line plot.
    save_metrics_by_epoch(summary_df, out_dir / "metrics_by_epoch.png")

    # Build combined data for chemical-space cloud.
    cloud_df = mol_df[["canonical_smiles", "epoch", "set"]].copy()
    if ref_df_vis is not None:
        cloud_df = pd.concat(
            [cloud_df, ref_df_vis[["canonical_smiles", "epoch", "set"]]],
            ignore_index=True,
        )

    cloud_smiles, cloud_fps, cloud_X = build_fingerprints(cloud_df["canonical_smiles"].tolist(), args.radius, args.n_bits)
    cloud_df = cloud_df[cloud_df["canonical_smiles"].isin(cloud_smiles)].copy()
    cloud_df = cloud_df.set_index("canonical_smiles").loc[cloud_smiles].reset_index()

    # PCA cloud.
    print("Computing PCA embedding...")
    pca_xy = make_pca_embedding(cloud_X, seed=args.seed)
    pca_df = cloud_df.copy()
    pca_df["PCA1"] = pca_xy[:, 0]
    pca_df["PCA2"] = pca_xy[:, 1]
    pca_df.to_csv(out_dir / "embedding_pca.csv", index=False)
    save_scatter(
        pca_df,
        x="PCA1",
        y="PCA2",
        hue="epoch",
        out_png=out_dir / "chemical_space_pca.png",
        title="Chemical space cloud by epoch, Morgan fingerprints + PCA",
        alpha=0.65,
    )

    # UMAP cloud, optional.
    print("Computing UMAP embedding, if umap-learn is installed...")
    umap_xy = make_umap_embedding(cloud_X, seed=args.seed)
    if umap_xy is not None:
        umap_df = cloud_df.copy()
        umap_df["UMAP1"] = umap_xy[:, 0]
        umap_df["UMAP2"] = umap_xy[:, 1]
        umap_df.to_csv(out_dir / "embedding_umap.csv", index=False)
        save_scatter(
            umap_df,
            x="UMAP1",
            y="UMAP2",
            hue="epoch",
            out_png=out_dir / "chemical_space_umap.png",
            title="Chemical space cloud by epoch, Morgan fingerprints + UMAP",
            alpha=0.65,
        )
    else:
        print("UMAP skipped. Install it with: pip install umap-learn")

    print("\nDone. Saved outputs to:", out_dir.resolve())
    print("\nMost important files:")
    print("  per_epoch_summary.csv")
    print("  per_molecule_epoch_data.csv")
    print("  chemical_space_pca.png")
    if (out_dir / "chemical_space_umap.png").exists():
        print("  chemical_space_umap.png")
    if args.reference:
        print("  tanimoto_hist_by_epoch.png")
        print("  tanimoto_violin_by_epoch.png")
        print("  metrics_by_epoch.png")


if __name__ == "__main__":
    main()