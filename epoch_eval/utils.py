"""Shared helpers for epoch-wise REINVENT transfer-learning evaluation."""

from __future__ import annotations

import glob
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs

    RDLogger.DisableLog("rdApp.*")
except ImportError as exc:  # pragma: no cover
    raise SystemExit("RDKit is required: pip install rdkit") from exc


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = ROOT / "data" / "train.smi"
DEFAULT_CHEMBL = ROOT / "data" / "chembl_drugs.smi"
DEFAULT_SAMPLES = ROOT / "samples_by_epoch"
DEFAULT_OUT = ROOT / "epochs" / "eval"


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_smiles(path: Path | str, max_n: Optional[int] = None, seed: int = 42) -> List[str]:
    """Load SMILES from .smi / .csv (column smiles/SMILES or first column)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    smiles: List[str] = []
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        col = next((c for c in ("SMILES", "smiles") if c in df.columns), df.columns[0])
        smiles = df[col].dropna().astype(str).tolist()
    else:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                s = line.strip().split()[0] if line.strip() and not line.startswith("#") else ""
                if s:
                    smiles.append(s)

    if max_n is not None and len(smiles) > max_n:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(smiles), size=max_n, replace=False)
        smiles = [smiles[i] for i in sorted(idx)]
    return smiles


def canonicalize(smi: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def smiles_to_fps(
    smiles: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
    unique: bool = True,
) -> Tuple[List[str], list, int]:
    """Return (canonical SMILES, RDKit bitvects, n_invalid)."""
    valid: List[str] = []
    fps = []
    invalid = 0
    seen = set()
    for smi in smiles:
        canon = canonicalize(smi)
        if canon is None:
            invalid += 1
            continue
        if unique and canon in seen:
            continue
        if unique:
            seen.add(canon)
        mol = Chem.MolFromSmiles(canon)
        if mol is None:
            invalid += 1
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        valid.append(canon)
        fps.append(fp)
    return valid, fps, invalid


def fps_to_numpy(fps: Sequence) -> np.ndarray:
    """BitVect list -> dense float matrix (N, n_bits)."""
    if not fps:
        return np.zeros((0, 0), dtype=np.float64)
    n_bits = fps[0].GetNumBits()
    arr = np.zeros((len(fps), n_bits), dtype=np.float64)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


def discover_epoch_files(results_dir: Path | str) -> Dict[int, Path]:
    """Map epoch -> first matching samples_epoch_XXX.csv."""
    results_dir = Path(results_dir)
    patterns = [
        str(results_dir / "samples_epoch_*.csv"),
        str(results_dir / "*epoch*.csv"),
    ]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))

    epoch_to_file: Dict[int, Path] = {}
    for f in files:
        m = re.search(r"epoch[_\-]?(\d+)", Path(f).name, flags=re.I)
        if not m:
            digits = "".join(ch for ch in Path(f).name if ch.isdigit())
            if not digits:
                continue
            epoch = int(digits)
        else:
            epoch = int(m.group(1))
        if epoch not in epoch_to_file:
            epoch_to_file[epoch] = Path(f)
    return dict(sorted(epoch_to_file.items()))


def load_epoch_smiles(
    results_dir: Path | str,
    max_per_epoch: Optional[int] = None,
    seed: int = 42,
) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for epoch, path in discover_epoch_files(results_dir).items():
        smi = read_smiles(path, max_n=max_per_epoch, seed=seed)
        out[epoch] = smi
    return out


def tanimoto_matrix(fps_a: Sequence, fps_b: Sequence) -> np.ndarray:
    n_a, n_b = len(fps_a), len(fps_b)
    if n_a == 0 or n_b == 0:
        return np.zeros((n_a, n_b), dtype=np.float64)
    mat = np.zeros((n_a, n_b), dtype=np.float64)
    for i, fp in enumerate(fps_a):
        mat[i, :] = DataStructs.BulkTanimotoSimilarity(fp, list(fps_b))
    return mat


def mean_pairwise(fps_a: Sequence, fps_b: Sequence) -> float:
    """Set-level mean Tanimoto over ALL pairs (not nearest-neighbour)."""
    mat = tanimoto_matrix(fps_a, fps_b)
    return float(mat.mean()) if mat.size else float("nan")


def mean_intra(fps: Sequence) -> float:
    n = len(fps)
    if n < 2:
        return float("nan")
    total, count = 0.0, 0
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], list(fps[i + 1 :]))
        total += float(np.sum(sims))
        count += len(sims)
    return total / count if count else float("nan")


def max_sims_to_ref(fps_query: Sequence, fps_ref: Sequence) -> np.ndarray:
    mat = tanimoto_matrix(fps_query, fps_ref)
    if mat.size == 0:
        return np.array([], dtype=np.float64)
    return mat.max(axis=1)


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """Squared Fréchet distance between two Gaussians."""
    from scipy.linalg import sqrtm

    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def gaussian_stats(X: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    if X.shape[0] < 2:
        cov = np.eye(X.shape[1]) * eps
    else:
        cov = np.cov(X, rowvar=False)
        cov = cov + np.eye(cov.shape[0]) * eps
    return mu, cov
