"""
vizualization_mol.py — 2D-облака молекул (scatter) для разных эпох
предобучения REINVENT.

Использует:
  * Morgan fingerprints (radius=2, 2048 бит) -> UMAP / t-SNE для проекции в 2D
  * Цвет = значение max Tanimoto к обучающей выборке
  * Разные эпохи — на одном графике, разные маркеры
  * Train-набор тоже отображается для контекста

Вход:
  --train   train.smi
  --results_dir  папка с samples_epoch_<N>.csv
"""

import argparse
import os
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    raise SystemExit("Нужен RDKit: pip install rdkit-pypi")

try:
    from umap import UMAP
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False
    from sklearn.manifold import TSNE


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def _canon(smi: str):
    m = Chem.MolFromSmiles(smi)
    return None if m is None else Chem.MolToSmiles(m)


def smiles_to_fp_matrix(smiles_list, radius: int = 2, n_bits: int = 2048):
    """SMILES -> np.ndarray shape (N, n_bits) — матрица Morgan FP."""
    fps = []
    valid = []
    for s in smiles_list:
        c = _canon(s)
        if c is None:
            continue
        m = Chem.MolFromSmiles(c)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        valid.append(c)
    if not fps:
        return np.zeros((0, n_bits), dtype=np.uint8), []
    return np.vstack(fps), valid


def tanimoto_distance_matrix(fps: np.ndarray) -> np.ndarray:
    """Матрица дистанций Танимото (1 - Tanimoto) на матрице FP.
    Считается эффективно через bit-операции."""
    if fps.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    fps = fps.astype(np.uint8)
    # bits intersection = A & B
    intersect = fps @ fps.T  # bit counts пересечения (uint8 0/1)
    a_sum = fps.sum(axis=1)
    b_sum = fps.sum(axis=1)
    union = a_sum[:, None] + b_sum[None, :] - intersect
    # защита от деления на 0
    union = np.where(union == 0, 1, union)
    tanimoto = intersect / union
    np.fill_diagonal(tanimoto, 1.0)
    distance = 1.0 - tanimoto
    return distance.astype(np.float32)


# ---------------------------------------------------------------------------
# Сбор данных по эпохам
# ---------------------------------------------------------------------------

def collect_epoch_data(results_dir: str, max_per_epoch: int = 5000):
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

    out = {}
    for ep in sorted(epoch_to_files):
        smis = []
        for f in epoch_to_files[ep]:
            df = pd.read_csv(f)
            col = "smiles" if "smiles" in df.columns else df.columns[0]
            smis.extend(df[col].dropna().astype(str).tolist())
        if max_per_epoch and len(smis) > max_per_epoch:
            smis = list(np.random.RandomState(42).choice(
                smis, size=max_per_epoch, replace=False))
        out[ep] = smis
        print(f"  epoch {ep:>3}: {len(smis)} molecules loaded")
    return out


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------

def _project_2d(distance_matrix: np.ndarray, method: str = "umap"):
    """Проецирует в 2D через UMAP (precomputed distance) или t-SNE."""
    if method == "umap" and HAVE_UMAP:
        reducer = UMAP(
            n_components=2,
            metric="precomputed",
            random_state=42,
            n_neighbors=min(15, max(2, distance_matrix.shape[0] - 1)),
            min_dist=0.1,
        )
        return reducer.fit_transform(distance_matrix)
    # fallback — t-SNE
    n = distance_matrix.shape[0]
    perplexity = min(30, max(5, n // 5 - 1))
    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        random_state=42,
        perplexity=perplexity,
        init="random",
    )
    return tsne.fit_transform(distance_matrix)


def plot_single_epoch(
    coords, max_sims, epoch, out_path,
    title=None,
):
    """Одно облако молекул одной эпохи с цветом по Tanimoto к train."""
    plt.figure(figsize=(8, 7))
    sc = plt.scatter(
        coords[:, 0], coords[:, 1],
        c=max_sims, cmap="viridis",
        s=14, alpha=0.75, edgecolors="none",
    )
    plt.colorbar(sc, label="Max Tanimoto → train")
    plt.title(title or f"Epoch {epoch}: molecular cloud")
    plt.xlabel("UMAP-1" if HAVE_UMAP else "t-SNE-1")
    plt.ylabel("UMAP-2" if HAVE_UMAP else "t-SNE-2")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] сохранено: {out_path}")


def plot_all_epochs_overlay(
    epoch_data: dict, out_path, max_sims_per_epoch: dict,
    title: str = "All epochs: molecular clouds",
):
    """Все эпохи на одном графике, разные цвета; train — звёздочками."""
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap("plasma")
    epochs = sorted(epoch_data.keys())
    if not epochs:
        return
    norm = plt.Normalize(min(epochs), max(epochs))

    for ep in epochs:
        coords, sims = epoch_data[ep]
        if len(coords) == 0:
            continue
        color = cmap(norm(ep))
        plt.scatter(
            coords[:, 0], coords[:, 1],
            c=[color], s=12, alpha=0.55, edgecolors="none",
            label=f"epoch {ep}",
        )

    # train — поверх
    if "_train" in epoch_data:
        coords_tr, _ = epoch_data["_train"]
        if len(coords_tr):
            plt.scatter(
                coords_tr[:, 0], coords_tr[:, 1],
                c="red", s=40, marker="*", edgecolors="black",
                linewidths=0.6, label="train",
            )

    plt.legend(loc="best", fontsize=9, framealpha=0.9, ncol=2)
    plt.title(title)
    plt.xlabel("UMAP-1" if HAVE_UMAP else "t-SNE-1")
    plt.ylabel("UMAP-2" if HAVE_UMAP else "t-SNE-2")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] сохранено: {out_path}")


def plot_colored_by_epoch(
    all_coords: np.ndarray, all_epochs: np.ndarray,
    out_path, title: str = "All molecules colored by epoch",
):
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap("plasma")
    if len(np.unique(all_epochs)) > 1:
        norm = plt.Normalize(all_epochs.min(), all_epochs.max())
    else:
        norm = plt.Normalize(0, 1)
    sc = plt.scatter(
        all_coords[:, 0], all_coords[:, 1],
        c=all_epochs, cmap=cmap, norm=norm,
        s=12, alpha=0.6, edgecolors="none",
    )
    cbar = plt.colorbar(sc, label="Epoch")
    tick_epochs = np.linspace(all_epochs.min(), all_epochs.max(), num=min(10, len(np.unique(all_epochs))))
    cbar.set_ticks(tick_epochs)
    cbar.set_ticklabels([f"{int(t)}" for t in tick_epochs])
    plt.title(title)
    plt.xlabel("UMAP-1" if HAVE_UMAP else "t-SNE-1")
    plt.ylabel("UMAP-2" if HAVE_UMAP else "t-SNE-2")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] сохранено: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Облака молекул по эпохам REINVENT (UMAP/t-SNE).")
    p.add_argument("--train", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--out_dir", default="mol_clouds")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n_bits", type=int, default=2048)
    p.add_argument("--max_per_epoch", type=int, default=2000)
    p.add_argument("--max_train", type=int, default=2000)
    p.add_argument("--method", choices=["umap", "tsne"], default="umap")
    p.add_argument("--per_epoch_plots", action="store_true",
                   help="Рисовать облако для каждой эпохи отдельно")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # train
    print("[INFO] загружаю train ...")
    if args.train.endswith(".csv"):
        df_tr = pd.read_csv(args.train)
        col = "smiles" if "smiles" in df_tr.columns else df_tr.columns[0]
        train_smiles = df_tr[col].dropna().astype(str).tolist()
    else:
        with open(args.train) as f:
            train_smiles = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    if args.max_train and len(train_smiles) > args.max_train:
        train_smiles = list(np.random.RandomState(0).choice(
            train_smiles, size=args.max_train, replace=False))

    train_fp, train_valid = smiles_to_fp_matrix(train_smiles, args.radius, args.n_bits)
    print(f"  train molecules: {len(train_valid)}")

    train_fps_list = [AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(s), args.radius, nBits=args.n_bits
    ) for s in train_valid]

    # данные по эпохам
    print("[INFO] собираю молекулы по эпохам ...")
    epoch_smiles = collect_epoch_data(args.results_dir, args.max_per_epoch)

    # для каждой эпохи считаем FP и max Tanimoto к train
    epoch_data = {}
    max_sims_per_epoch = {}
    for ep, smis in epoch_smiles.items():
        fp_mat, valid = smiles_to_fp_matrix(smis, args.radius, args.n_bits)
        if fp_mat.shape[0] == 0:
            print(f"  [WARN] epoch {ep}: 0 валидных молекул")
            continue

        # max Tanimoto к train
        max_sims = []
        for row in fp_mat:
            fp_obj = AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles(_canon("C")), args.radius, nBits=args.n_bits
            )  # placeholder
        # пересчитаем max_sims напрямую через FP-объекты (быстрее)
        fps_obj = []
        max_sims = []
        for s in valid:
            mol = Chem.MolFromSmiles(s)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, args.radius, nBits=args.n_bits)
            fps_obj.append(fp)
            sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps_list)
            max_sims.append(float(max(sims)))
        max_sims = np.array(max_sims, dtype=np.float32)

        # считаем дистанции для проекции
        # объединяем с train, чтобы проекция была согласованной
        if train_fp.shape[0]:
            combined = np.vstack([fp_mat, train_fp])
        else:
            combined = fp_mat
        dist = tanimoto_distance_matrix(combined)

        coords_all = _project_2d(dist, method=args.method)
        coords = coords_all[:fp_mat.shape[0]]
        coords_train = coords_all[fp_mat.shape[0]:] if train_fp.shape[0] else np.zeros((0, 2))

        epoch_data[ep] = (coords, max_sims)
        max_sims_per_epoch[ep] = max_sims
        # train запомним один раз — координаты для совмещённого графика
        if "_train" not in epoch_data and train_fp.shape[0]:
            epoch_data["_train"] = (coords_train, np.zeros(coords_train.shape[0]))

        if args.per_epoch_plots:
            plot_single_epoch(
                coords, max_sims, ep,
                os.path.join(args.out_dir, f"cloud_epoch_{ep:03d}.png"),
            )

    if not epoch_data:
        print("[WARN] нет данных ни одной эпохи.")
        return

    # объединённый scatter с цветом по эпохе
    all_coords = []
    all_eps = []
    for ep, (coords, sims) in epoch_data.items():
        if ep == "_train":
            continue
        all_coords.append(coords)
        all_eps.append(np.full(len(coords), ep, dtype=int))
    all_coords = np.vstack(all_coords)
    all_eps = np.concatenate(all_eps)
    plot_colored_by_epoch(
        all_coords, all_eps,
        os.path.join(args.out_dir, "cloud_all_epochs.png"),
    )

    # наложение эпох
    plot_all_epochs_overlay(
        epoch_data,
        os.path.join(args.out_dir, "cloud_overlay.png"),
        max_sims_per_epoch,
    )

    # сохраним координаты и метрики для последующего анализа
    np.savez(
        os.path.join(args.out_dir, "cloud_data.npz"),
        coords=all_coords,
        epochs=all_eps,
    )
    print(f"[OK] данные облака сохранены: {os.path.join(args.out_dir, 'cloud_data.npz')}")
    print("[DONE]")


if __name__ == "__main__":
    main()
