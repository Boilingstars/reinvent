"""
Python3-compatible wrapper around Coley SCScore (trained on Reaxys).

Weights: vendor/scscore_reaxys/models/model_1024bool.json.gz
Source: https://github.com/connorcoley/scscore
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

WEIGHTS = (
    Path(__file__).resolve().parent
    / "vendor"
    / "scscore_reaxys"
    / "models"
    / "model_1024bool.json.gz"
)

FP_LEN = 1024
FP_RAD = 2
SCORE_SCALE = 5.0


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class SCScorerReaxys:
    """Synthetic complexity score 1–5 (↓ easier), learned from ~12M Reaxys reactions."""

    def __init__(self, weight_path: Path | None = None):
        path = Path(weight_path) if weight_path else WEIGHTS
        if not path.is_file():
            raise FileNotFoundError(f"SCScore weights not found: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            raw = json.load(fh)
        self.vars = [np.asarray(x, dtype=np.float64) for x in raw]
        self.fp_len = FP_LEN
        self.fp_rad = FP_RAD

    def mol_to_fp(self, mol) -> np.ndarray:
        if mol is None:
            return np.zeros((self.fp_len,), dtype=np.float64)
        bv = AllChem.GetMorganFingerprintAsBitVect(
            mol, self.fp_rad, nBits=self.fp_len, useChirality=True
        )
        return np.asarray(bv, dtype=np.float64)

    def apply(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        for i in range(0, len(self.vars), 2):
            last = i == len(self.vars) - 2
            W, b = self.vars[i], self.vars[i + 1]
            x = x @ W + b
            if not last:
                x = np.maximum(x, 0.0)  # ReLU
        # output is shape (1, 1)
        logit = float(np.asarray(x).ravel()[0])
        return 1.0 + (SCORE_SCALE - 1.0) * _sigmoid(logit)

    def score_smiles(self, smi: str) -> float | None:
        if not smi:
            return None
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        fp = self.mol_to_fp(mol)
        if float(fp.sum()) == 0.0:
            return None
        return float(self.apply(fp))


_SCORER: SCScorerReaxys | None = None


def get_scscorer() -> SCScorerReaxys:
    global _SCORER
    if _SCORER is None:
        _SCORER = SCScorerReaxys()
    return _SCORER


def scscore_smiles(smi: str) -> float | None:
    try:
        return get_scscorer().score_smiles(smi)
    except Exception:
        return None


if __name__ == "__main__":
    for s in ("CCCOCCC", "CCCNc1ccccc1", "c1ccccc1"):
        print(s, scscore_smiles(s))
