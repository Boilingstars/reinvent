"""
vizualization_hist.py — гистограммы и boxplot'ы для сравнения эпох
предобучения REINVENT по индексу Танимото.

Строит:
  1) Распределение максимального Tanimoto к обучающей выборке по эпохам.
  2) Boxplot тех же распределений.
  3) Линейный график метрик (mean_intra, mean_max_to_train) от эпохи.
  4) Матрицу "эпоха × эпоха" среднего Tanimoto между эпохами.

Вход: результат работы tanimoto.py (summary CSV) и
       pickle/npz с распределениями max_sim_to_train.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    raise SystemExit("Нужен RDKit: pip install rdkit-pypi")


# ---------------------------------------------------------------------------
# Вспомогательные функции (повторяют tanimoto.py, но с упором на распределения)
# ---------------------------------------------------------------------------

def _canon(smi: str):
    m = Chem.MolFromSmiles(smi)
    return None if m is None else Chem.MolToSmiles(m)


def _fp(smi: str, radius: int = 2, n_bits: int = 2048):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)


def load_train_fps(path: str, radius: int, n_bits: int):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        col = "smiles" if "smiles" in df.columns else df.columns[0]
        smis = df[col].dropna().astype(str).tolist()
    else:
        with open(path) as f:
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


def collect_epoch_distributions(results_dir: str, train_fps, radius: int, n_bits: int,
                                max_per_epoch: int = 5000):
    """Собирает распределения (max_sim_to_train) по эпохам."""
    import glob
    from collections import defaultdict

    files = sorted(glob.glob(os.path.join(results_dir, "samples_epoch_*.csv")))
    epoch_to_files = defaultdict(list)
    for f in files:
        name = os.path.basename(f)
        digits = "".join(ch for ch in name if ch.isdigit())
        if not digits:
            continue
        epoch = int(digits)
        if 0 < epoch < 10000:
            epoch_to_files[epoch].append(f)

    distributions = {}
    for ep in sorted(epoch_to_files):
        smiles = []
        for f in epoch_to_files[ep]:
            df = pd.read_csv(f)
            col = "smiles" if "smiles" in df.columns else df.columns[0]
            smiles.extend(df[col].dropna().astype(str).tolist())
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
        distributions[ep] = np.array(max_sims, dtype=np.float32)
        print(f"  epoch {ep:>3}: {len(max_sims)} molecules, "
              f"mean={max_sims.mean() if len(max_sims) else 0:.3f}")
    return distributions


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------

def plot_overlay_histograms(distributions: dict, out_path: str,
                            title: str = "Max Tanimoto to training set per epoch"):
    """Наложенные гистограммы распределений max_sim_to_train по эпохам."""
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("viridis")
    epochs = sorted(distributions.keys())
    if not epochs:
        print("[WARN] нет распределений для построения гистограммы.")
        return
    norm = plt.Normalize(min(epochs), max(epochs))

    for ep in epochs:
        arr = distributions[ep]
        if len(arr) == 0:
            continue
        color = cmap(norm(ep))
        plt.hist(arr, bins=40, alpha=0.45, density=True,
                 color=color, label=f"epoch {ep}")

    plt.xlabel("Tanimoto (max to nearest train neighbor)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend(title="Epoch", loc="upper left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] сохранено: {out_path}")


def plot_grid_histograms(distributions: dict, out_path: str,
                         title: str = "Per-epoch Tanimoto distribution"):
    """Сетка гистограмм — по одной на эпоху."""
    epochs = sorted(distributions.keys())
    n = len(epochs)
    if n == 0:
        print("[WARN] нет распределений.")
        return
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)

    for idx, ep in enumerate(epochs):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        arr = distributions[ep]
        if len(arr):
            ax.hist(arr, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
            mean_v = arr.mean()
            ax.axvline(mean_v, color="red", linestyle="--", label=f"mean={mean_v:.3f}")
            ax.legend(fontsize=8)
        ax.set_title(f"epoch {ep}  (n={len(arr)})")
        ax.set_xlabel("Max Tanimoto")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)

    # пустые подграфики
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] сохранено: {out_path}")


def plot_boxplots(distributions: dict, out_path: str,
                  title: str = "Boxplot: max Tanimoto to train by epoch"):
    """Boxplot распределений по эпохам."""
    epochs = sorted(distributions.keys())
    data = [distributions[e] for e in epochs if len(distributions[e])]
    labels = [f"e{e}" for e in epochs if len(distributions[e])]
    if not data:
        return
    plt.figure(figsize=(max(8, 0.7 * len(labels) + 4), 6))
    bp = plt.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                     meanline=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#a3c4f3")
    plt.ylabel("Max Tanimoto to nearest train neighbor")
    plt.xlabel("Epoch")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] сохранено: {out_path}")


def plot_metric_curves(summary_csv: str, out_path: str):
    """Линейные графики mean_intra_tanimoto и mean_max_sim_to_train от эпохи."""
    if not os.path.isfile(summary_csv):
        print(f"[WARN] {summary_csv} не найден — пропускаю metric curves.")
        return
    df = pd.read_csv(summary_csv)
    if df.empty:
        return
    df = df.sort_values("epoch")
    fig, ax1 = plt.subplots(figsize=(9, 5))

    color1 = "tab:blue"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean intra-Tanimoto", color=color1)
    ax1.plot(df["epoch"], df["mean_intra_tanimoto"], "o-", color=color1,
             label="mean_intra_tanimoto")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Mean max Tanimoto → train", color=color2)
    ax2.plot(df["epoch"], df["mean_max_sim_to_train"], "s-", color=color2,
             label="mean_max_sim_to_train")
    ax2.tick_params(axis="y", labelcolor=color2)

    # маркер лучшей эпохи по max_sim_to_train
    if df["mean_max_sim_to_train"].notna().any():
        best_idx = df["mean_max_sim_to_train"].idxmax()
        ax2.axvline(df.loc[best_idx, "epoch"], color="green",
                    linestyle=":", alpha=0.6,
                    label=f"best: epoch {df.loc[best_idx, 'epoch']}")

    fig.suptitle("Tanimoto metrics vs epoch")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] сохранено: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Гистограммы Танимото по эпохам REINVENT.")
    p.add_argument("--train", required=True, help="train.smi")
    p.add_argument("--results_dir", required=True, help="папка с samples_epoch_*.csv")
    p.add_argument("--summary_csv", default="tanimoto_summary.csv")
    p.add_argument("--out_dir", default="hist_plots")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n_bits", type=int, default=2048)
    p.add_argument("--max_per_epoch", type=int, default=5000)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("[INFO] загружаю train fingerprints ...")
    train_fps = load_train_fps(args.train, args.radius, args.n_bits)
    print(f"  train molecules: {len(train_fps)}")

    print("[INFO] собираю распределения по эпохам ...")
    dists = collect_epoch_distributions(
        args.results_dir, train_fps, args.radius, args.n_bits,
        max_per_epoch=args.max_per_epoch,
    )
    if not dists:
        print("[WARN] распределения пусты — нечего рисовать.")
        return

    # сохраним распределения для последующего использования (например, в UMAP)
    npz_path = os.path.join(args.out_dir, "tanimoto_distributions.npz")
    np.savez(npz_path, **{f"epoch_{ep}": arr for ep, arr in dists.items()})
    print(f"[OK] распределения сохранены: {npz_path}")

    plot_overlay_histograms(
        dists,
        os.path.join(args.out_dir, "hist_overlay.png"),
    )
    plot_grid_histograms(
        dists,
        os.path.join(args.out_dir, "hist_grid.png"),
    )
    plot_boxplots(
        dists,
        os.path.join(args.out_dir, "boxplot.png"),
    )
    plot_metric_curves(
        args.summary_csv,
        os.path.join(args.out_dir, "metric_curves.png"),
    )

    print("[DONE]")


if __name__ == "__main__":
    main()
