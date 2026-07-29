"""
eval_fcd.py — Fréchet ChemNet Distance (FCD) across TL epochs.

Computes FCD of each epoch sample set vs:
  * chromophore fine-tuning set
  * ChEMBL drug-like set

Uses the official `fcd` package when available (`pip install fcd`).
Falls back to Fréchet distance on PCA-reduced Morgan fingerprints (FFD)
if ChemNet weights cannot be loaded.

Writes CSV + FCD curves.

Usage:
  python epoch_eval/eval_fcd.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# fcd 1.2.x still calls np.row_stack, removed in NumPy 2
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack  # type: ignore[attr-defined]

# fcd 1.2.x: linalg.sqrtm(..., disp=False) -> (matrix, info)
import scipy.linalg as _sla

_orig_sqrtm = _sla.sqrtm


def _sqrtm_compat(A, *args, **kwargs):
    want_tuple = "disp" in kwargs
    kwargs.pop("disp", None)
    result = _orig_sqrtm(A, *args, **kwargs)
    if want_tuple:
        return result, True
    return result


_sla.sqrtm = _sqrtm_compat  # type: ignore[assignment]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    DEFAULT_CHEMBL,
    DEFAULT_OUT,
    DEFAULT_SAMPLES,
    DEFAULT_TRAIN,
    ensure_dir,
    fps_to_numpy,
    frechet_distance,
    gaussian_stats,
    load_epoch_smiles,
    read_smiles,
    smiles_to_fps,
)

_CHEMNET_MODEL = None


def _load_chemnet():
    global _CHEMNET_MODEL
    if _CHEMNET_MODEL is not None:
        return _CHEMNET_MODEL
    from fcd import load_ref_model  # type: ignore

    _CHEMNET_MODEL = load_ref_model()
    return _CHEMNET_MODEL


def try_chemnet_fcd(smiles_a: list[str], smiles_b: list[str]) -> float | None:
    """ChemNet FCD via embeddings + our Fréchet (avoids fcd/scipy API mismatch)."""
    try:
        from fcd import get_predictions  # type: ignore
    except Exception as exc:
        print(f"[WARN] fcd not importable: {exc}")
        return None
    try:
        model = _load_chemnet()
        act1 = np.asarray(get_predictions(model, smiles_a), dtype=np.float64)
        act2 = np.asarray(get_predictions(model, smiles_b), dtype=np.float64)
        if len(act1) < 2 or len(act2) < 2:
            return float("nan")
        mu1, s1 = gaussian_stats(act1)
        mu2, s2 = gaussian_stats(act2)
        return frechet_distance(mu1, s1, mu2, s2)
    except Exception as exc:
        print(f"[WARN] ChemNet FCD failed: {exc}")
        return None


def fingerprint_frechet(
    smiles_a: list[str],
    smiles_b: list[str],
    radius: int,
    n_bits: int,
    pca_dim: int,
    seed: int,
) -> float:
    """Fréchet distance on PCA of Morgan FP (fallback / always available)."""
    from sklearn.decomposition import PCA

    _, fps_a, _ = smiles_to_fps(smiles_a, radius, n_bits)
    _, fps_b, _ = smiles_to_fps(smiles_b, radius, n_bits)
    Xa = fps_to_numpy(fps_a)
    Xb = fps_to_numpy(fps_b)
    if len(Xa) < 2 or len(Xb) < 2:
        return float("nan")

    X = np.vstack([Xa, Xb])
    dim = min(pca_dim, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=dim, random_state=seed)
    Z = pca.fit_transform(X)
    Za, Zb = Z[: len(Xa)], Z[len(Xa) :]
    mu1, s1 = gaussian_stats(Za)
    mu2, s2 = gaussian_stats(Zb)
    return frechet_distance(mu1, s1, mu2, s2)


def compute_fcd_table(
    epochs: dict[int, list[str]],
    train: list[str],
    chembl: list[str],
    radius: int,
    n_bits: int,
    pca_dim: int,
    seed: int,
    prefer_chemnet: bool,
) -> pd.DataFrame:
    rows = []
    chemnet_ok = prefer_chemnet
    for ep in sorted(epochs):
        gen = epochs[ep]
        # canonicalize via fps helper
        gen_valid, _, _ = smiles_to_fps(gen, radius, n_bits, unique=True)

        fcd_chrom = try_chemnet_fcd(gen_valid, train) if chemnet_ok else None
        fcd_drug = try_chemnet_fcd(gen_valid, chembl) if chemnet_ok else None
        if chemnet_ok and fcd_chrom is None:
            print("[INFO] ChemNet FCD unavailable — using fingerprint Fréchet (FFD).")
            chemnet_ok = False

        ffd_chrom = fingerprint_frechet(gen_valid, train, radius, n_bits, pca_dim, seed)
        ffd_drug = fingerprint_frechet(gen_valid, chembl, radius, n_bits, pca_dim, seed)

        row = {
            "epoch": ep,
            "n_molecules": len(gen_valid),
            "fcd_vs_chromophores": fcd_chrom if fcd_chrom is not None else np.nan,
            "fcd_vs_chembl": fcd_drug if fcd_drug is not None else np.nan,
            "ffd_vs_chromophores": ffd_chrom,
            "ffd_vs_chembl": ffd_drug,
            "backend": "chemnet+ffd" if fcd_chrom is not None else "ffd_only",
        }
        rows.append(row)
        print(
            f"  epoch {ep:>3}: FCD_chrom={row['fcd_vs_chromophores']}  "
            f"FFD_chrom={ffd_chrom:.3f}  FFD_chembl={ffd_drug:.3f}"
        )
    return pd.DataFrame(rows)


def plot_fcd_curves(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    ep = df["epoch"]

    has_fcd = df["fcd_vs_chromophores"].notna().any()
    if has_fcd:
        axes[0].plot(ep, df["fcd_vs_chromophores"], "o-", label="FCD → chromophores")
        axes[0].plot(ep, df["fcd_vs_chembl"], "s-", label="FCD → ChEMBL")
        axes[0].set_title("Fréchet ChemNet Distance")
    else:
        axes[0].text(0.5, 0.5, "ChemNet FCD not available\n(install: pip install fcd)",
                     ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title("Fréchet ChemNet Distance")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Distance (↓ better match)")
    axes[0].grid(True, alpha=0.3)
    if has_fcd:
        axes[0].legend(fontsize=8)

    axes[1].plot(ep, df["ffd_vs_chromophores"], "o-", label="FFD → chromophores")
    axes[1].plot(ep, df["ffd_vs_chembl"], "s-", label="FFD → ChEMBL")
    axes[1].set_title("Fréchet Fingerprint Distance (PCA-Morgan)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Distance (↓ better match)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="FCD / FFD evaluation across TL epochs")
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--chembl", type=Path, default=DEFAULT_CHEMBL)
    p.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT / "fcd")
    p.add_argument("--max-ref", type=int, default=2000)
    p.add_argument("--max-per-epoch", type=int, default=None)
    p.add_argument("--pca-dim", type=int, default=64)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-chemnet", action="store_true", help="Skip official FCD, only FFD")
    args = p.parse_args()

    out = ensure_dir(args.out_dir)
    train = read_smiles(args.train, max_n=args.max_ref, seed=args.seed)
    chembl = read_smiles(args.chembl, max_n=args.max_ref, seed=args.seed)
    # canonicalize refs once
    train, _, _ = smiles_to_fps(train, args.radius, args.n_bits)
    chembl, _, _ = smiles_to_fps(chembl, args.radius, args.n_bits)

    epochs = load_epoch_smiles(args.samples_dir, max_per_epoch=args.max_per_epoch, seed=args.seed)
    if not epochs:
        raise SystemExit(f"No epoch CSVs in {args.samples_dir}")

    print(f"[INFO] Epochs: {list(epochs)}")
    df = compute_fcd_table(
        epochs,
        train,
        chembl,
        args.radius,
        args.n_bits,
        args.pca_dim,
        args.seed,
        prefer_chemnet=not args.no_chemnet,
    )
    csv_path = out / "fcd_by_epoch.csv"
    df.to_csv(csv_path, index=False)
    plot_fcd_curves(df, out / "fcd_curves.png")
    print(f"[OK] {csv_path}")
    print(f"[OK] {out / 'fcd_curves.png'}")


if __name__ == "__main__":
    main()
