"""
viz_chemspace_alt.py — alternative 2D views of chemical space (not plain PCA scatter).

Why plain PCA often looks weak for Morgan FPs:
  * PC1/PC2 explain little variance (bits are sparse/high-dim)
  * references + many epochs overplot into a grey cloud

This script builds one shared embedding (or shared PCA pre-step) and exports
several complementary views under epochs/eval/chemspace_alt/:

  1. umap_overview.png          — UMAP of ChEMBL + chromophores + all epochs
  2. tsne_overview.png          — t-SNE (PCA→50 then t-SNE)
  3. density_hexbin.png         — hexbin density of refs; epoch points on top
  4. kde_contours.png           — KDE contours for ChEMBL / chromophores + epochs
  5. facets_by_epoch.png        — small multiples: one panel per epoch
  6. hulls_and_centroids.png    — convex hulls + centroid trajectory (UMAP)
  7. similarity_colored_umap.png— epoch points colored by max-Tanimoto→chromophores

Usage (from repo root):
  python epoch_eval/viz_chemspace_alt.py
  python epoch_eval/viz_chemspace_alt.py --max-ref 1500 --method both
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    DEFAULT_CHEMBL,
    DEFAULT_OUT,
    DEFAULT_SAMPLES,
    DEFAULT_TRAIN,
    ensure_dir,
    fps_to_numpy,
    load_epoch_smiles,
    max_sims_to_ref,
    read_smiles,
    smiles_to_fps,
)

try:
    from umap import UMAP

    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False


REF_STYLES = {
    "chembl": dict(color="#9e9e9e", label="ChEMBL (prior)", z=1),
    "chromophore": dict(color="#2ca02c", label="chromophores", z=2),
}


def build_blocks(
    chembl: list[str],
    train: list[str],
    epochs: dict[int, list[str]],
    radius: int,
    n_bits: int,
) -> tuple[np.ndarray, np.ndarray, list, dict[int, list]]:
    """Return X, labels, chemophore fps (for coloring), epoch_fps map."""
    blocks: list[np.ndarray] = []
    labels: list[str] = []
    epoch_fps: dict[int, list] = {}

    _, chembl_fps, _ = smiles_to_fps(chembl, radius, n_bits)
    blocks.append(fps_to_numpy(chembl_fps))
    labels.extend(["chembl"] * len(chembl_fps))

    _, train_fps, _ = smiles_to_fps(train, radius, n_bits)
    blocks.append(fps_to_numpy(train_fps))
    labels.extend(["chromophore"] * len(train_fps))

    for ep in sorted(epochs):
        _, fps, _ = smiles_to_fps(epochs[ep], radius, n_bits)
        if not fps:
            continue
        epoch_fps[ep] = fps
        blocks.append(fps_to_numpy(fps))
        labels.extend([f"epoch_{ep}"] * len(fps))

    return np.vstack(blocks), np.asarray(labels), train_fps, epoch_fps


def embed(
    X: np.ndarray,
    method: str,
    seed: int,
    pca_dim: int = 50,
) -> tuple[np.ndarray, str]:
    """Return (N,2) coordinates and method tag used."""
    n = X.shape[0]
    # binary FP → float; optional PCA compress for speed / stability
    Xp = X
    if X.shape[1] > pca_dim and n > pca_dim + 1:
        Xp = PCA(n_components=min(pca_dim, n - 1), random_state=seed).fit_transform(X)

    method = method.lower()
    if method == "umap":
        if not HAVE_UMAP:
            raise RuntimeError("umap-learn is not installed (pip install umap-learn)")
        reducer = UMAP(
            n_components=2,
            n_neighbors=min(30, max(5, n // 20)),
            min_dist=0.15,
            metric="euclidean",
            random_state=seed,
        )
        return reducer.fit_transform(Xp), "UMAP (PCA→%d prep)" % Xp.shape[1]

    if method == "tsne":
        perplexity = min(40, max(5, (n - 1) // 4))
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return reducer.fit_transform(Xp), f"t-SNE (perplexity={perplexity}, PCA prep)"

    raise ValueError(method)


def epoch_list(labels: np.ndarray) -> list[str]:
    return sorted(
        {lb for lb in np.unique(labels) if str(lb).startswith("epoch_")},
        key=lambda s: int(str(s).split("_")[1]),
    )


def epoch_colors(epochs: list[str], cmap_name: str = "plasma"):
    cmap = plt.get_cmap(cmap_name)
    if len(epochs) <= 1:
        return {epochs[0]: cmap(0.5)} if epochs else {}
    return {lb: cmap(i / (len(epochs) - 1)) for i, lb in enumerate(epochs)}


def plot_scatter_overview(
    Z: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, style in REF_STYLES.items():
        m = labels == name
        if m.any():
            ax.scatter(
                Z[m, 0],
                Z[m, 1],
                c=style["color"],
                s=10 if name == "chembl" else 14,
                alpha=0.25 if name == "chembl" else 0.35,
                marker=".",
                linewidths=0,
                label=style["label"],
                zorder=style["z"],
            )

    eps = epoch_list(labels)
    colors = epoch_colors(eps)
    for lb in eps:
        m = labels == lb
        ax.scatter(
            Z[m, 0],
            Z[m, 1],
            c=[colors[lb]],
            s=42,
            alpha=0.9,
            edgecolors="k",
            linewidths=0.35,
            label=lb.replace("epoch_", "ep "),
            zorder=5,
        )

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(fontsize=8, markerscale=1.3, framealpha=0.92, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_hexbin_density(
    Z: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    method_tag: str,
) -> None:
    """Background = hexbin of pooled references; epochs as markers."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    ref_masks = {
        "ChEMBL density": labels == "chembl",
        "Chromophore density": labels == "chromophore",
    }
    eps = epoch_list(labels)
    colors = epoch_colors(eps)

    for ax, (title, mask) in zip(axes, ref_masks.items()):
        if mask.sum() < 5:
            continue
        hb = ax.hexbin(
            Z[mask, 0],
            Z[mask, 1],
            gridsize=35,
            cmap="Greys",
            mincnt=1,
            alpha=0.95,
        )
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="count")
        for lb in eps:
            m = labels == lb
            ax.scatter(
                Z[m, 0],
                Z[m, 1],
                c=[colors[lb]],
                s=36,
                alpha=0.85,
                edgecolors="k",
                linewidths=0.3,
                label=lb.replace("epoch_", "ep "),
                zorder=5,
            )
        ax.set_title(title)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.grid(True, alpha=0.2)

    axes[1].legend(fontsize=7, loc="best", framealpha=0.9)
    fig.suptitle(f"Hexbin reference density + epoch points ({method_tag})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _kde_grid(xy: np.ndarray, grid_n: int = 100):
    if len(xy) < 5:
        return None
    try:
        kde = gaussian_kde(xy.T)
    except Exception:
        return None
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    pad_x = 0.08 * (xmax - xmin + 1e-9)
    pad_y = 0.08 * (ymax - ymin + 1e-9)
    xs = np.linspace(xmin - pad_x, xmax + pad_x, grid_n)
    ys = np.linspace(ymin - pad_y, ymax + pad_y, grid_n)
    xx, yy = np.meshgrid(xs, ys)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, zz


def plot_kde_contours(
    Z: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    method_tag: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, color, levels in (
        ("chembl", "#616161", 5),
        ("chromophore", "#2ca02c", 5),
    ):
        m = labels == name
        grid = _kde_grid(Z[m])
        if grid is None:
            continue
        xx, yy, zz = grid
        ax.contour(xx, yy, zz, levels=levels, colors=color, linewidths=1.2, alpha=0.8)
        ax.contourf(xx, yy, zz, levels=levels, colors=color, alpha=0.08)

    eps = epoch_list(labels)
    colors = epoch_colors(eps)
    for lb in eps:
        m = labels == lb
        ax.scatter(
            Z[m, 0],
            Z[m, 1],
            c=[colors[lb]],
            s=40,
            alpha=0.9,
            edgecolors="k",
            linewidths=0.3,
            label=lb.replace("epoch_", "ep "),
            zorder=5,
        )

    legend_extra = [
        Patch(facecolor="#616161", edgecolor="#616161", alpha=0.35, label="ChEMBL KDE"),
        Patch(facecolor="#2ca02c", edgecolor="#2ca02c", alpha=0.35, label="chromophore KDE"),
    ]
    handles, labs = ax.get_legend_handles_labels()
    ax.legend(handles + legend_extra, labs + [p.get_label() for p in legend_extra], fontsize=8)
    ax.set_title(f"KDE contours of references + epoch points ({method_tag})")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_facets(
    Z: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    method_tag: str,
) -> None:
    eps = epoch_list(labels)
    if not eps:
        return
    n = len(eps)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    colors = epoch_colors(eps)

    chem = labels == "chembl"
    chrom = labels == "chromophore"
    for ax, lb in zip(axes, eps):
        ax.scatter(Z[chem, 0], Z[chem, 1], c="#cfcfcf", s=6, alpha=0.3, linewidths=0, zorder=1)
        ax.scatter(Z[chrom, 0], Z[chrom, 1], c="#2ca02c", s=8, alpha=0.25, linewidths=0, zorder=2)
        m = labels == lb
        ax.scatter(
            Z[m, 0],
            Z[m, 1],
            c=[colors[lb]],
            s=45,
            alpha=0.95,
            edgecolors="k",
            linewidths=0.35,
            zorder=5,
        )
        ax.set_title(lb.replace("epoch_", "epoch "))
        ax.grid(True, alpha=0.2)

    for ax in axes[len(eps) :]:
        ax.axis("off")

    fig.suptitle(
        f"One epoch per panel (grey=ChEMBL, green=chromophores) · {method_tag}",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_hulls_centroids(
    Z: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    method_tag: str,
) -> None:
    from scipy.spatial import ConvexHull

    fig, ax = plt.subplots(figsize=(9, 7))
    # light refs
    for name, style in REF_STYLES.items():
        m = labels == name
        if m.any():
            ax.scatter(
                Z[m, 0],
                Z[m, 1],
                c=style["color"],
                s=8,
                alpha=0.15,
                linewidths=0,
                zorder=1,
            )

    eps = epoch_list(labels)
    colors = epoch_colors(eps)
    cents = []
    for lb in eps:
        m = labels == lb
        pts = Z[m]
        color = colors[lb]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=[color],
            s=28,
            alpha=0.55,
            edgecolors="none",
            zorder=3,
        )
        c = pts.mean(axis=0)
        cents.append(c)
        ax.scatter(c[0], c[1], c=[color], s=140, marker="*", edgecolors="k", linewidths=0.6, zorder=6)
        ax.annotate(lb.replace("epoch_", "e"), c, fontsize=9, fontweight="bold")
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                closed = np.vstack([hull_pts, hull_pts[0]])
                ax.plot(closed[:, 0], closed[:, 1], color=color, lw=1.6, alpha=0.9, zorder=4)
            except Exception:
                pass

    cents = np.asarray(cents)
    if len(cents) >= 2:
        ax.plot(cents[:, 0], cents[:, 1], "--", color="#333333", lw=1.2, alpha=0.7, zorder=5)

    # reference centroids
    for name, marker, color in (("chembl", "s", "#616161"), ("chromophore", "D", "#2ca02c")):
        m = labels == name
        if m.any():
            c = Z[m].mean(axis=0)
            ax.scatter(c[0], c[1], c=color, s=120, marker=marker, edgecolors="k", label=f"{name} centroid", zorder=7)

    ax.set_title(f"Convex hulls + centroid trajectory ({method_tag})")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_similarity_colored(
    Z: np.ndarray,
    labels: np.ndarray,
    epoch_fps: dict[int, list],
    train_fps: list,
    out_path: Path,
    method_tag: str,
) -> None:
    """Epoch molecules colored by max Tanimoto to chromophore train set."""
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, style in REF_STYLES.items():
        m = labels == name
        if m.any():
            ax.scatter(
                Z[m, 0],
                Z[m, 1],
                c=style["color"],
                s=8,
                alpha=0.18,
                linewidths=0,
                zorder=1,
                label=style["label"],
            )

    # collect epoch coords in label order
    sims_all = []
    xy_all = []
    for ep, fps in sorted(epoch_fps.items()):
        m = labels == f"epoch_{ep}"
        pts = Z[m]
        if len(pts) != len(fps):
            # unique fps path should match; if not, recompute length-safe
            n = min(len(pts), len(fps))
            pts = pts[:n]
            fps = fps[:n]
        sims = max_sims_to_ref(fps, train_fps)
        xy_all.append(pts)
        sims_all.append(sims)

    if not xy_all:
        plt.close(fig)
        return

    xy = np.vstack(xy_all)
    sims = np.concatenate(sims_all)
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=sims,
        cmap="viridis",
        s=48,
        alpha=0.95,
        edgecolors="k",
        linewidths=0.3,
        norm=Normalize(vmin=0.0, vmax=max(0.6, float(np.percentile(sims, 95)))),
        zorder=5,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("max Tanimoto → chromophores")

    ax.set_title(f"Generated points colored by chromophore similarity ({method_tag})")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_coords(Z: np.ndarray, labels: np.ndarray, method: str, out_path: Path) -> None:
    pd.DataFrame({"dim1": Z[:, 0], "dim2": Z[:, 1], "label": labels, "method": method}).to_csv(
        out_path, index=False
    )


def run_method(
    method: str,
    X: np.ndarray,
    labels: np.ndarray,
    train_fps: list,
    epoch_fps: dict[int, list],
    out_dir: Path,
    seed: int,
    pca_dim: int,
) -> None:
    print(f"[INFO] Embedding with {method} …")
    Z, tag = embed(X, method, seed=seed, pca_dim=pca_dim)
    sub = ensure_dir(out_dir / method)
    save_coords(Z, labels, method, sub / f"{method}_coordinates.csv")

    plot_scatter_overview(Z, labels, f"{tag}: ChEMBL + chromophores + epochs", sub / f"{method}_overview.png")
    plot_hexbin_density(Z, labels, sub / f"{method}_hexbin_density.png", tag)
    plot_kde_contours(Z, labels, sub / f"{method}_kde_contours.png", tag)
    plot_facets(Z, labels, sub / f"{method}_facets_by_epoch.png", tag)
    plot_hulls_centroids(Z, labels, sub / f"{method}_hulls_centroids.png", tag)
    plot_similarity_colored(Z, labels, epoch_fps, train_fps, sub / f"{method}_similarity_colored.png", tag)
    print(f"[OK] {method} plots → {sub}")


def main() -> None:
    p = argparse.ArgumentParser(description="Alternative chemical-space 2D visualizations")
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--chembl", type=Path, default=DEFAULT_CHEMBL)
    p.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT / "chemspace_alt")
    p.add_argument("--max-ref", type=int, default=1500)
    p.add_argument("--max-per-epoch", type=int, default=None)
    p.add_argument("--method", choices=["umap", "tsne", "both"], default="both")
    p.add_argument("--pca-dim", type=int, default=50, help="PCA prep dims before UMAP/t-SNE")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = ensure_dir(args.out_dir)
    chembl = read_smiles(args.chembl, max_n=args.max_ref, seed=args.seed)
    train = read_smiles(args.train, max_n=args.max_ref, seed=args.seed + 1)
    epochs = load_epoch_smiles(args.samples_dir, max_per_epoch=args.max_per_epoch, seed=args.seed)
    if not epochs:
        raise SystemExit(f"No epoch CSVs in {args.samples_dir}")

    print("[INFO] Building fingerprint matrix …")
    X, labels, train_fps, epoch_fps = build_blocks(
        chembl, train, epochs, args.radius, args.n_bits
    )
    print(f"[INFO] Matrix {X.shape}; labels={dict(zip(*np.unique(labels, return_counts=True)))}")

    methods = ["umap", "tsne"] if args.method == "both" else [args.method]
    if "umap" in methods and not HAVE_UMAP:
        print("[WARN] umap-learn missing — skipping UMAP (pip install umap-learn)")
        methods = [m for m in methods if m != "umap"]
    if not methods:
        raise SystemExit("No embedding methods available")

    for m in methods:
        run_method(m, X, labels, train_fps, epoch_fps, out, args.seed, args.pca_dim)

    # short guide
    guide = out / "README.md"
    guide.write_text(
        "\n".join(
            [
                "# Alternative chemical-space views",
                "",
                "Plain PCA often fails for Morgan fingerprints (low variance on PC1/PC2, overplotting).",
                "This folder compares embeddings that preserve local neighbourhoods better:",
                "",
                "| File pattern | Idea |",
                "|---|---|",
                "| `*_overview.png` | Classic scatter on UMAP / t-SNE |",
                "| `*_hexbin_density.png` | Density of ChEMBL or chromophores; epochs as points |",
                "| `*_kde_contours.png` | Smooth density contours of both references |",
                "| `*_facets_by_epoch.png` | One panel per epoch (avoids overplotting epochs) |",
                "| `*_hulls_centroids.png` | Convex hull + centroid path of each epoch |",
                "| `*_similarity_colored.png` | Epoch points colored by max Tc → chromophores |",
                "",
                "Prefer **facets** + **similarity-colored UMAP** for interpreting TL drift;",
                "prefer **hexbin/KDE** when reference clouds dominate the plot.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[OK] guide → {guide}")


if __name__ == "__main__":
    main()
