"""
vizualization_hist.py — гистограммы и boxplot'ы для сравнения эпох
предобучения REINVENT по индексу Танимото.
"""

import argparse
import csv
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    raise SystemExit("Нужен RDKit: pip install rdkit-pypi")


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _canon(smi: str):
    m = Chem.MolFromSmiles(smi)
    return None if m is None else Chem.MolToSmiles(m)


def _fp(smi: str, radius: int = 2, n_bits: int = 2048):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)


def load_train_fps(path: str, radius: int, n_bits: int,
                   encoding: str = 'utf-8', sep: str = ','):
    if path.endswith(".csv"):
        for enc in (encoding, 'utf-8-sig', 'utf-8', 'cp1251', 'latin-1'):
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                break
            except Exception:
                continue
        else:
            raise ValueError(f"Не удалось прочитать {path} ни в одной кодировке")
        smiles_col = None
        for col in df.columns:
            if 'smiles' in col.lower():
                smiles_col = col
                break
        if smiles_col is None:
            smiles_col = df.columns[0]
        smis = df[smiles_col].dropna().astype(str).tolist()
    else:
        with open(path, encoding=encoding) as f:
            smis = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    fps = []
    for s in smis:
        c = _canon(s)
        if c is None:
            continue
        fp = _fp(c, radius, n_bits)
        if fp is not None:
            fps.append(fp)
    return fps


_CSV_ENCODINGS = ('utf-8-sig', 'utf-8', 'cp1251', 'cp1252', 'latin-1')
_HEADER_TOKENS = frozenset({
    'SMILES', 'SMILES_STATE', 'NLL', 'INPUT_SMILES', 'TANIMOTO',
})
_SMILES_PATTERN = re.compile(r'^[A-Za-z0-9@+\-\[\]()=#$%\\/\.:]+$')


def _encoding_candidates(preferred: str = 'utf-8'):
    seen = set()
    for enc in (preferred, *_CSV_ENCODINGS):
        if enc and enc not in seen:
            seen.add(enc)
            yield enc


def _is_probably_text(raw: bytes) -> bool:
    if not raw:
        return False
    if raw[:2] == b'PK' or raw[:4] in (b'\x89PNG', b'\x80\x04\x8a\x00'):
        return False
    sample = raw[:8192]
    ctrl = sum(1 for b in sample if b == 0 or (b < 32 and b not in (9, 10, 13)))
    return ctrl / len(sample) < 0.02


def _looks_like_smiles(value: str) -> bool:
    s = value.strip()
    if len(s) < 2:
        return False
    if s.upper() in _HEADER_TOKENS:
        return False
    if not _SMILES_PATTERN.match(s):
        return False
    return bool(re.search(r'[A-Za-z]', s))


def _decode_text(raw: bytes, preferred: str = 'utf-8'):
    for enc in _encoding_candidates(preferred):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode('latin-1'), 'latin-1'


def _extract_smiles_from_line(line: str, sep: str = ',') -> str | None:
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    for sep_try in dict.fromkeys((sep, ',', '\t', ';')):
        try:
            fields = next(csv.reader([line], delimiter=sep_try))
        except csv.Error:
            continue
        if not fields:
            continue
        candidate = fields[0].strip()
        if _looks_like_smiles(candidate):
            return candidate
    if _looks_like_smiles(line):
        return line
    return None


def _smiles_from_dataframe(df: pd.DataFrame) -> list[str]:
    for col in df.columns:
        if 'smiles' in str(col).lower():
            return df[col].dropna().astype(str).str.strip().tolist()
    if df.shape[1] > 1:
        return df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    parts = df.iloc[:, 0].astype(str).str.split(',', n=1, expand=True)
    if parts.shape[1] >= 1:
        return parts[0].str.strip().tolist()
    return df.iloc[:, 0].dropna().astype(str).str.strip().tolist()


def _filter_smiles(candidates: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in candidates:
        if not _looks_like_smiles(item):
            continue
        smi = item.strip()
        if smi not in seen:
            seen.add(smi)
            out.append(smi)
    return out


def load_smiles_from_epoch_file(filepath: str, encoding: str = 'utf-8',
                                sep: str = ',') -> list[str]:
    """Читает REINVENT CSV и возвращает SMILES из первой колонки.

    Поддерживает:
    - стандартный REINVENT CSV с заголовком SMILES,...
    - строки вида ``SMILES,state,NLL`` без заголовка
    - utf-8, utf-8-sig, cp1251, cp1252, latin-1
    """
    with open(filepath, 'rb') as fh:
        raw = fh.read()

    if not _is_probably_text(raw):
        raise ValueError(f"{filepath}: файл не похож на текстовый CSV")

    text, used_enc = _decode_text(raw, encoding)
    line_smiles = []
    for line in text.splitlines():
        smi = _extract_smiles_from_line(line, sep=sep)
        if smi:
            line_smiles.append(smi)

    if line_smiles:
        print(f"    Прочитано {len(line_smiles)} SMILES из {os.path.basename(filepath)} "
              f"(encoding={used_enc})")
        print(f"    Пример: {line_smiles[0][:80]}")
        return line_smiles

    for enc in _encoding_candidates(encoding):
        for sep_try in dict.fromkeys((sep, ',', '\t', ';')):
            for header in ('infer', None):
                try:
                    df = pd.read_csv(
                        filepath,
                        encoding=enc,
                        sep=sep_try,
                        header=0 if header == 'infer' else None,
                    )
                    if df.empty:
                        continue
                    valid = _filter_smiles(_smiles_from_dataframe(df))
                    if valid:
                        print(f"    Прочитано {len(valid)} SMILES из {os.path.basename(filepath)} "
                              f"(pandas, encoding={enc}, sep={repr(sep_try)})")
                        print(f"    Пример: {valid[0][:80]}")
                        return valid
                except Exception:
                    continue

    raise ValueError(
        f"В {filepath} не найдено валидных SMILES "
        f"(проверены кодировки: {', '.join(_encoding_candidates(encoding))})"
    )


def collect_epoch_distributions(results_dir: str, train_fps, radius: int, n_bits: int,
                                max_per_epoch: int = 5000,
                                encoding: str = 'utf-8', sep: str = ','):
    import glob
    from collections import defaultdict

    files = sorted(glob.glob(os.path.join(results_dir, "samples_epoch_*.csv")))
    if not files:
        print("[WARN] Не найдено ни одного файла samples_epoch_*.csv")
        return {}

    print(f"[INFO] Найдено файлов: {len(files)}")
    for f in files:
        print(f"  {os.path.basename(f)}")

    epoch_to_files = defaultdict(list)
    for f in files:
        name = os.path.basename(f)
        digits = "".join(ch for ch in name if ch.isdigit())
        if not digits:
            continue
        epoch = int(digits)
        if 0 <= epoch < 10000:   # включаем эпоху 0
            epoch_to_files[epoch].append(f)

    distributions = {}
    for ep in sorted(epoch_to_files):
        smiles = []
        for f in epoch_to_files[ep]:
            try:
                smiles.extend(load_smiles_from_epoch_file(f, encoding=encoding, sep=sep))
            except Exception as e:
                print(f"[ERROR] {f} не прочитан: {e}")
                continue

        if not smiles:
            print(f"  epoch {ep:>3}: нет молекул, пропускаем")
            continue

        if max_per_epoch and len(smiles) > max_per_epoch:
            smiles = list(np.random.RandomState(42).choice(
                smiles, size=max_per_epoch, replace=False))

        max_sims = []
        for s in smiles:
            c = _canon(s)
            if c is None:
                continue
            fp = _fp(c, radius, n_bits)
            if fp is None:
                continue
            sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
            max_sims.append(float(max(sims)))
        if not max_sims:
            print(f"  epoch {ep:>3}: нет валидных молекул после RDKit, пропускаем")
            continue
        distributions[ep] = np.array(max_sims, dtype=np.float32)
        print(f"  epoch {ep:>3}: {len(max_sims)} molecules, mean={distributions[ep].mean():.3f}")
    return distributions

# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------

def plot_grid_histograms(distributions: dict, out_path: str,
                         title: str = "Per-epoch Tanimoto distribution"):
    epochs = sorted(distributions.keys())
    n = len(epochs)
    if n == 0:
        print("[WARN] нет распределений.")
        return
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)

    n_bins = 30
    x_min, x_max = 0.0, 1.0
    ymax = 0
    for ep in epochs:
        arr = distributions[ep]
        if len(arr):
            counts, _ = np.histogram(arr, bins=n_bins, range=(x_min, x_max))
            ymax = max(ymax, int(counts.max()))

    for idx, ep in enumerate(epochs):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        arr = distributions[ep]
        if len(arr):
            ax.hist(arr, bins=n_bins, range=(x_min, x_max),
                    color="steelblue", edgecolor="white", alpha=0.85)
            mean_v = arr.mean()
            ax.axvline(mean_v, color="red", linestyle="--", label=f"mean={mean_v:.3f}")
            ax.legend(fontsize=8)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, ymax if ymax > 0 else 1)
        ax.set_title(f"epoch {ep}  (n={len(arr)})")
        ax.set_xlabel("Max Tanimoto")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)

    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] сохранено: {os.path.abspath(out_path)}")

# ---------------------------------------------------------------------------
# CLI и запуск из IDE
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Гистограммы Танимото по эпохам REINVENT.")
    p.add_argument("--train", required=True, help="Путь к train.smi или train.csv")
    p.add_argument("--results_dir", required=True, help="Папка с samples_epoch_*.csv")
    p.add_argument("--summary_csv", default="tanimoto_summary.csv")
    p.add_argument("--out_dir", default="hist_plots")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n_bits", type=int, default=2048)
    p.add_argument("--max_per_epoch", type=int, default=5000)
    p.add_argument("--encoding", default="utf-8")
    p.add_argument("--sep", default=",")

    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    print("[INFO] загружаю train fingerprints ...")
    train_fps = load_train_fps(args.train, args.radius, args.n_bits,
                               encoding=args.encoding, sep=args.sep)
    print(f"  train molecules: {len(train_fps)}")

    print("[INFO] собираю распределения по эпохам ...")
    dists = collect_epoch_distributions(
        args.results_dir, train_fps, args.radius, args.n_bits,
        max_per_epoch=args.max_per_epoch,
        encoding=args.encoding, sep=args.sep,
    )
    dists = {ep: arr for ep, arr in dists.items() if len(arr) > 0}
    if not dists:
        print("[WARN] распределения пусты — нечего рисовать.")
        return

    npz_path = os.path.join(args.out_dir, "tanimoto_distributions.npz")
    np.savez(npz_path, **{f"epoch_{ep}": arr for ep, arr in dists.items()})
    print(f"[OK] распределения сохранены: {os.path.abspath(npz_path)}")

    out_files = []

    grid_path = os.path.join(args.out_dir, "hist_grid.png")
    plot_grid_histograms(dists, grid_path)
    out_files.append(grid_path)

    print("\n" + "=" * 60)
    print("ВСЕ СОЗДАННЫЕ ФАЙЛЫ:")
    for f in out_files:
        print(f"  {os.path.abspath(f)}")
    print("=" * 60)
    print("[DONE]")


if __name__ == "__main__":
    main([
        "--train", "data/train.smi",
        "--results_dir", "./samples_by_epoch",
        "--summary_csv", "./epochs/tanimoto_summary.csv",
        "--out_dir", "./epochs",
        "--radius", "2",
        "--n_bits", "2048",
        "--max_per_epoch", "5000",
    ])