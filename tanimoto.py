"""
tanimoto.py — вычисление похожести молекул по индексу Танимото (Tanimoto)
для разных эпох предобучения модели REINVENT.

Сравнивает сгенерированные молекулы между эпохами и относительно обучающей выборки.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, RDLogger
    from rdkit import __version__ as rdkit_version
    RDLogger.DisableLog("rdApp.*")  # глушим rdkit warnings
except ImportError:
    raise SystemExit(
        "RDKit не установлен. Установите: pip install rdkit-pypi"
    )


# ---------------------------------------------------------------------------
# Утилиты для SMILES
# ---------------------------------------------------------------------------

def read_smiles_file(path: str) -> List[str]:
    """Читает SMILES из файла. Поддерживает .smi (по одному SMILES на строку)
    и .csv (с колонкой 'smiles' или первой колонкой)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    ext = os.path.splitext(path)[1].lower()
    smiles_list = []

    if ext == ".csv":
        df = pd.read_csv(path)
        col = "smiles" if "smiles" in df.columns else df.columns[0]
        smiles_list = df[col].dropna().astype(str).tolist()
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    smiles_list.append(s)
    return smiles_list


def canonicalize(smi: str) -> str:
    """Канонизирует SMILES, возвращает None если молекула невалидна."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def mol_to_morgan_fp(mol, radius: int = 2, n_bits: int = 2048):
    """Молекула -> Morgan fingerprint (bit vector)."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def smiles_to_fingerprints(smiles_list: List[str], radius: int = 2, n_bits: int = 2048):
    """Превращает список SMILES в (валидные канонические SMILES, fingerprints).
    Невалидные молекулы отбрасываются, но их количество логируется."""
    valid_smiles = []
    fps = []
    invalid = 0
    for smi in smiles_list:
        canon = canonicalize(smi)
        if canon is None:
            invalid += 1
            continue
        mol = Chem.MolFromSmiles(canon)
        if mol is None:
            invalid += 1
            continue
        try:
            fp = mol_to_morgan_fp(mol, radius=radius, n_bits=n_bits)
            valid_smiles.append(canon)
            fps.append(fp)
        except Exception:
            invalid += 1
    return valid_smiles, fps, invalid


# ---------------------------------------------------------------------------
# Сбор сгенерированных молекул по эпохам
# ---------------------------------------------------------------------------

def discover_epoch_files(results_dir: str) -> Dict[int, List[str]]:
    """Ищет CSV-файлы, соответствующие разным эпохам.

    Ожидаемые шаблоны имён:
        samples_epoch_2.csv
        samples_epoch_4.csv
        samples_epoch_6.csv
        ...
    или альтернативно:
        TL_reinvent_epoch_2.samples.csv
    """
    pattern_candidates = [
        os.path.join(results_dir, "samples_epoch_*.csv"),
        os.path.join(results_dir, "*epoch*.csv"),
        os.path.join(results_dir, "epoch_*.csv"),
    ]

    files = []
    for pat in pattern_candidates:
        files.extend(glob.glob(pat))
    files = sorted(set(files))

    epoch_to_files: Dict[int, List[str]] = defaultdict(list)
    for f in files:
        name = os.path.basename(f)
        digits = "".join(ch for ch in name if ch.isdigit())
        if not digits:
            continue
        # берём последнее "разумное" число в имени как номер эпохи
        epoch = int(digits)
        if 0 < epoch < 10000:
            epoch_to_files[epoch].append(f)

    return dict(sorted(epoch_to_files.items()))


def load_smiles_from_csv(path: str) -> List[str]:
    """Загружает SMILES из CSV, сгенерированного REINVENT.
    REINVENT кладёт результат в колонку 'smiles'."""
    df = pd.read_csv(path)
    col = "smiles" if "smiles" in df.columns else df.columns[0]
    return df[col].dropna().astype(str).tolist()


# ---------------------------------------------------------------------------
# Подсчёт Tanimoto
# ---------------------------------------------------------------------------

def tanimoto_matrix(fps_a, fps_b) -> np.ndarray:
    """Считает попарную матрицу Танимото между двумя наборами Morgan FP.
    Возвращает матрицу shape (len(a), len(b))."""
    n_a, n_b = len(fps_a), len(fps_b)
    if n_a == 0 or n_b == 0:
        return np.zeros((n_a, n_b), dtype=np.float32)
    # BulkTanimotoSimilarity работает быстрее, чем цикл
    mat = np.zeros((n_a, n_b), dtype=np.float32)
    for i, fp in enumerate(fps_a):
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps_b)
        mat[i, :] = sims
    return mat


def average_intra_similarity(fps) -> float:
    """Средний Танимото ВНУТРИ набора (все пары)."""
    n = len(fps)
    if n < 2:
        return float("nan")
    total, count = 0.0, 0
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        total += float(np.sum(sims))
        count += len(sims)
    return total / count if count else float("nan")


def average_to_reference(fps_query, fps_ref) -> Tuple[float, np.ndarray]:
    """Средний Танимото каждой молекулы из query к ближайшему соседу в ref.
    Возвращает (mean_max_sim, массив max_sim по каждой query молекуле)."""
    mat = tanimoto_matrix(fps_query, fps_ref)
    if mat.size == 0:
        return float("nan"), np.array([])
    max_sim = mat.max(axis=1)
    return float(max_sim.mean()), max_sim


# ---------------------------------------------------------------------------
# Главный сценарий
# ---------------------------------------------------------------------------

def compare_epochs(
    train_smi: str,
    results_dir: str,
    out_csv: str = "tanimoto_summary.csv",
    out_pairs_csv: str = "tanimoto_pairs.csv",
    radius: int = 2,
    n_bits: int = 2048,
    max_per_epoch: int = 5000,
):
    """Сравнивает молекулы разных эпох предобучения.

    Метрики для каждой эпохи:
        - valid_molecules       : сколько молекул прошло валидацию
        - unique_molecules      : сколько уникальных
        - mean_intra_tanimoto   : средний Танимото внутри эпохи
        - mean_max_sim_to_train : средний максимальный Танимото к обучающей выборке
        - mean_max_sim_to_prior : то же, но к prior.prior (если задан)

    Дополнительно сохраняет попарные сравнения между соседними эпохами:
        epoch_a, epoch_b, mean_tanimoto
    """
    print(f"[INFO] Загружаю обучающую выборку: {train_smi}")
    train_raw = read_smiles_file(train_smi)
    _, train_fps, train_invalid = smiles_to_fingerprints(train_raw, radius, n_bits)
    print(f"  валидных молекул в train: {len(train_fps)} (невалидных: {train_invalid})")

    epoch_files = discover_epoch_files(results_dir)
    if not epoch_files:
        print(f"[WARN] В директории {results_dir} не найдено CSV-файлов с эпохами.")
        print("       Ожидаются файлы вида samples_epoch_<N>.csv")
        return

    print(f"[INFO] Найдено эпох: {len(epoch_files)} — {list(epoch_files.keys())}")

    # Словари для агрегатов
    epoch_summaries = []
    epoch_fps: Dict[int, list] = {}
    epoch_smiles: Dict[int, list] = {}

    for epoch, files in epoch_files.items():
        all_smiles = []
        for f in files:
            all_smiles.extend(load_smiles_from_csv(f))

        if max_per_epoch and len(all_smiles) > max_per_epoch:
            all_smiles = list(np.random.RandomState(42).choice(
                all_smiles, size=max_per_epoch, replace=False))

        valid_smi, fps, invalid = smiles_to_fingerprints(all_smiles, radius, n_bits)
        unique_smi = list(dict.fromkeys(valid_smi))  # сохраняем порядок

        # пересчитаем fps на уникальных
        _, unique_fps, _ = smiles_to_fingerprints(unique_smi, radius, n_bits)

        epoch_smiles[epoch] = unique_smi
        epoch_fps[epoch] = unique_fps

        mean_intra = average_intra_similarity(unique_fps) if unique_fps else float("nan")
        mean_to_train, _ = average_to_reference(unique_fps, train_fps) if unique_fps else (float("nan"), np.array([]))

        epoch_summaries.append({
            "epoch": epoch,
            "n_generated": len(all_smiles),
            "n_valid": len(valid_smi),
            "n_unique": len(unique_smi),
            "mean_intra_tanimoto": mean_intra,
            "mean_max_sim_to_train": mean_to_train,
        })
        print(f"  epoch {epoch:>3}: "
              f"valid={len(valid_smi):>5} unique={len(unique_smi):>5} "
              f"intra={mean_intra:.3f} → train={mean_to_train:.3f}")

    summary_df = pd.DataFrame(epoch_summaries)
    summary_df.to_csv(out_csv, index=False)
    print(f"[INFO] Сводная таблица сохранена: {out_csv}")

    # Попарные сравнения соседних эпох
    epochs_sorted = sorted(epoch_fps.keys())
    pair_rows = []
    for i in range(len(epochs_sorted) - 1):
        a, b = epochs_sorted[i], epochs_sorted[i + 1]
        mat = tanimoto_matrix(epoch_fps[a], epoch_fps[b])
        if mat.size:
            mean_ab = float(mat.mean())
            # средний максимальный Танимото a->b и b->a
            mean_a_to_b = float(mat.max(axis=1).mean())
            mean_b_to_a = float(mat.max(axis=0).mean())
        else:
            mean_ab = mean_a_to_b = mean_b_to_a = float("nan")
        pair_rows.append({
            "epoch_a": a, "epoch_b": b,
            "mean_tanimoto": mean_ab,
            f"mean_{a}_to_{b}": mean_a_to_b,
            f"mean_{b}_to_{a}": mean_b_to_a,
        })

    pairs_df = pd.DataFrame(pair_rows)
    pairs_df.to_csv(out_pairs_csv, index=False)
    print(f"[INFO] Попарные сравнения сохранены: {out_pairs_csv}")
    print()
    print("=" * 60)
    print("СВОДКА ПО ЭПОХАМ")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    return summary_df, pairs_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Сравнение молекул разных эпох предобучения REINVENT по Танимото."
    )
    parser.add_argument("--train", required=True,
                        help="Путь к обучающему .smi файлу (например train.smi).")
    parser.add_argument("--results_dir", required=True,
                        help="Папка с CSV-файлами samples_epoch_<N>.csv.")
    parser.add_argument("--out_csv", default="tanimoto_summary.csv",
                        help="Куда сохранить сводную таблицу.")
    parser.add_argument("--out_pairs", default="tanimoto_pairs.csv",
                        help="Куда сохранить попарные сравнения эпох.")
    parser.add_argument("--radius", type=int, default=2,
                        help="Радиус Morgan fingerprint (по умолчанию 2).")
    parser.add_argument("--n_bits", type=int, default=2048,
                        help="Длина Morgan fingerprint (по умолчанию 2048).")
    parser.add_argument("--max_per_epoch", type=int, default=5000,
                        help="Максимум молекул с эпохи (для скорости).")
    args = parser.parse_args()

    compare_epochs(
        train_smi=args.train,
        results_dir=args.results_dir,
        out_csv=args.out_csv,
        out_pairs_csv=args.out_pairs,
        radius=args.radius,
        n_bits=args.n_bits,
        max_per_epoch=args.max_per_epoch,
    )


if __name__ == "__main__":
    main()
