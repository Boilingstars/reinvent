"""
Chemical-space shift of RL TL sweep across epochs vs ChEMBL drugs + chromophores.

Discovers any number of files matching:
  rl_tl_sweep_<run>_epXX_*.csv

Joint embedding (ChEMBL + chromophores + all epochs), then plots:
  01_umap_facets_by_epoch.png   — one panel per epoch (main shift view)
  02_umap_hulls_centroids.png   — hulls + centroid trajectory
  03_umap_overview_by_epoch.png — all epochs overlaid, colored by epoch

Usage:
  .venv_eval\\Scripts\\python.exe results_rl_unimol_abs\\curated\\scripts\\viz_chemspace_rl_sweep_epochs.py
  .venv_eval\\Scripts\\python.exe ...\\viz_chemspace_rl_sweep_epochs.py --run-id 20260801_090230
  .venv_eval\\Scripts\\python.exe ...\\viz_chemspace_rl_sweep_epochs.py --glob ".../data/rl_tl_sweep_*_ep*_1.csv"
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.decomposition import PCA

CURATED = Path(__file__).resolve().parents[1]
ROOT = CURATED.parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "epoch_eval"))
sys.path.insert(0, str(SCRIPTS))

from utils import ensure_dir, fps_to_numpy, read_smiles, smiles_to_fps  # noqa: E402
from viz_chemspace_compare import load_rl_smiles  # noqa: E402

try:
    from umap import UMAP

    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

DEFAULT_DATA = CURATED / "data"
DEFAULT_CHEMBL = CURATED / "data" / "refs" / "chembl_drugs.smi"
DEFAULT_TRAIN = CURATED / "data" / "refs" / "train.smi"
DEFAULT_OUT = CURATED / "figures" / "chemspace_sweep_epochs"

EP_RE = re.compile(r"_ep(\d+)_", re.IGNORECASE)
RUN_RE = re.compile(r"rl_tl_sweep_(\d+_\d+)_ep", re.IGNORECASE)

REF_STYLES = {
    "chembl": dict(color="#9e9e9e", label="ChEMBL drugs", z=1),
    "chromophore": dict(color="#2ca02c", label="Chromophores", z=2),
}


def discover_epoch_files(
    data_dir: Path,
    pattern: str,
    run_id: str | None,
) -> dict[int, Path]:
    """Return {epoch: path}, one file per epoch (latest by mtime if duplicates)."""
    files = sorted(Path(data_dir).glob(pattern) if "*" in pattern else Path(pattern).parent.glob(Path(pattern).name))
    # also allow absolute glob via Path
    if not files and "*" in pattern:
        files = sorted(Path().glob(pattern))
    chosen: dict[int, Path] = {}
    for f in files:
        if not f.is_file():
            continue
        m = EP_RE.search(f.name)
        if not m:
            continue
        if run_id:
            rm = RUN_RE.search(f.name)
            if not rm or rm.group(1) != run_id:
                continue
        ep = int(m.group(1))
        prev = chosen.get(ep)
        if prev is None or f.stat().st_mtime >= prev.stat().st_mtime:
            chosen[ep] = f
    return dict(sorted(chosen.items()))


def embed_umap(X: np.ndarray, seed: int, pca_dim: int = 50) -> tuple[np.ndarray, str]:
    if not HAVE_UMAP:
        raise RuntimeError("umap-learn required")
    n = X.shape[0]
    Xp = X
    if X.shape[1] > pca_dim and n > pca_dim + 1:
        Xp = PCA(n_components=min(pca_dim, n - 1), random_state=seed).fit_transform(X)
    reducer = UMAP(
        n_components=2,
        n_neighbors=min(30, max(5, n // 20)),
        min_dist=0.15,
        metric="euclidean",
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = reducer.fit_transform(Xp)
    return Z, f"UMAP (PCA→{Xp.shape[1]})"


def epoch_labels(labels: np.ndarray) -> list[str]:
    return sorted(
        {lb for lb in np.unique(labels) if str(lb).startswith("epoch_")},
        key=lambda s: int(str(s).split("_")[1]),
    )


def epoch_colors(eps: list[str], cmap_name: str = "plasma"):
    cmap = plt.get_cmap(cmap_name)
    if len(eps) <= 1:
        return {eps[0]: cmap(0.55)} if eps else {}
    return {lb: cmap(i / (len(eps) - 1)) for i, lb in enumerate(eps)}


def plot_facets(Z: np.ndarray, labels: np.ndarray, out: Path, tag: str) -> None:
    eps = epoch_labels(labels)
    if not eps:
        return
    n = len(eps)
    ncols = min(5, n) if n > 4 else min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.3 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    colors = epoch_colors(eps)
    chem = labels == "chembl"
    chrom = labels == "chromophore"

    for ax, lb in zip(axes, eps):
        ax.scatter(Z[chem, 0], Z[chem, 1], c="#cfd8dc", s=4, alpha=0.22, linewidths=0, zorder=1)
        ax.scatter(Z[chrom, 0], Z[chrom, 1], c="#2ca02c", s=5, alpha=0.22, linewidths=0, zorder=2)
        m = labels == lb
        ax.scatter(
            Z[m, 0],
            Z[m, 1],
            c=[colors[lb]],
            s=14,
            alpha=0.75,
            edgecolors="k",
            linewidths=0.12,
            zorder=5,
        )
        ep = lb.replace("epoch_", "")
        ax.set_title(f"epoch {ep}  (n={int(m.sum())})", fontsize=10)
        ax.grid(True, alpha=0.2)

    for ax in axes[len(eps) :]:
        ax.axis("off")

    # shared legend proxy
    handles = [
        Patch(facecolor="#cfd8dc", edgecolor="none", label="ChEMBL drugs"),
        Patch(facecolor="#2ca02c", edgecolor="none", label="Chromophores"),
        Patch(facecolor=plt.get_cmap("plasma")(0.7), edgecolor="k", label="RL (this epoch)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"Chemical-space shift vs drugs & chromophores by epoch\n{tag}",
        fontsize=12,
        y=1.06,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out.name}")


def plot_hulls_centroids(Z: np.ndarray, labels: np.ndarray, out: Path, tag: str) -> None:
    from scipy.spatial import ConvexHull

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    for name, style in REF_STYLES.items():
        m = labels == name
        if m.any():
            ax.scatter(
                Z[m, 0],
                Z[m, 1],
                c=style["color"],
                s=6,
                alpha=0.14,
                linewidths=0,
                zorder=1,
                label=style["label"],
            )

    eps = epoch_labels(labels)
    colors = epoch_colors(eps)
    cents = []
    for lb in eps:
        m = labels == lb
        pts = Z[m]
        color = colors[lb]
        ax.scatter(pts[:, 0], pts[:, 1], c=[color], s=10, alpha=0.35, linewidths=0, zorder=3)
        c = pts.mean(axis=0)
        cents.append(c)
        ax.scatter(c[0], c[1], c=[color], s=160, marker="*", edgecolors="k", linewidths=0.55, zorder=6)
        ax.annotate(lb.replace("epoch_", "e"), (c[0], c[1]), fontsize=9, fontweight="bold",
                    xytext=(4, 4), textcoords="offset points")
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                hp = pts[hull.vertices]
                closed = np.vstack([hp, hp[0]])
                ax.plot(closed[:, 0], closed[:, 1], color=color, lw=1.5, alpha=0.85, zorder=4)
            except Exception:
                pass

    cents = np.asarray(cents)
    if len(cents) >= 2:
        ax.plot(cents[:, 0], cents[:, 1], "--", color="#212121", lw=1.3, alpha=0.75, zorder=5, label="centroid path")

    for name, marker, color, lab in (
        ("chembl", "s", "#616161", "ChEMBL centroid"),
        ("chromophore", "D", "#1b5e20", "Chromophore centroid"),
    ):
        m = labels == name
        if m.any():
            c = Z[m].mean(axis=0)
            ax.scatter(c[0], c[1], c=color, s=90, marker=marker, edgecolors="k", linewidths=0.5, zorder=7, label=lab)

    ax.set_title(f"Epoch hulls + centroid trajectory vs refs\n{tag}")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(fontsize=8, loc="best", framealpha=0.92)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out.name}")


def plot_overview(Z: np.ndarray, labels: np.ndarray, out: Path, tag: str) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    for name, style in REF_STYLES.items():
        m = labels == name
        if m.any():
            ax.scatter(
                Z[m, 0],
                Z[m, 1],
                c=style["color"],
                s=7 if name == "chembl" else 8,
                alpha=0.22 if name == "chembl" else 0.28,
                linewidths=0,
                zorder=style["z"],
                label=style["label"],
            )
    eps = epoch_labels(labels)
    colors = epoch_colors(eps)
    for lb in eps:
        m = labels == lb
        ax.scatter(
            Z[m, 0],
            Z[m, 1],
            c=[colors[lb]],
            s=12,
            alpha=0.65,
            edgecolors="k",
            linewidths=0.1,
            zorder=5,
            label=lb.replace("epoch_", "ep "),
        )
    ax.set_title(f"All epochs overlaid on drugs & chromophores\n{tag}")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(fontsize=7.5, loc="best", ncol=2, framealpha=0.92)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out.name}")


def main() -> None:
    p = argparse.ArgumentParser(description="UMAP shift of RL sweep epochs vs ChEMBL/chromophores")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument(
        "--glob",
        type=str,
        default="rl_tl_sweep_*_ep*_1.csv",
        help="Glob under --data-dir (or absolute glob)",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional filter, e.g. 20260801_090230. If omitted, uses latest file per epoch across runs.",
    )
    p.add_argument("--chembl", type=Path, default=DEFAULT_CHEMBL)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-ref", type=int, default=4000)
    p.add_argument("--max-per-epoch", type=int, default=None, help="Subsample RL SMILES per epoch")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--pca-prep", type=int, default=50)
    p.add_argument(
        "--plots",
        type=str,
        default="facets,hulls,overview",
        help="Comma list: facets,hulls,overview",
    )
    p.add_argument(
        "--epochs",
        type=str,
        default=None,
        help="Optional subset, e.g. 0,4,8. Applies to discovery and to --from-coords redraw.",
    )
    p.add_argument(
        "--from-coords",
        type=Path,
        default=None,
        help="Redraw plots from existing coordinates_umap_epochs.csv (skip FP/UMAP).",
    )
    args = p.parse_args()

    out = ensure_dir(args.out)
    plots = {x.strip().lower() for x in args.plots.split(",") if x.strip()}
    epoch_keep: set[int] | None = None
    if args.epochs:
        epoch_keep = {int(x.strip()) for x in args.epochs.split(",") if x.strip()}

    if args.from_coords is not None:
        coords_path = args.from_coords
        if coords_path.is_dir():
            coords_path = coords_path / "coordinates_umap_epochs.csv"
        coords = pd.read_csv(coords_path)
        if epoch_keep is not None:
            keep_labels = {"chembl", "chromophore"} | {f"epoch_{e}" for e in epoch_keep}
            coords = coords[coords["label"].isin(keep_labels)].copy()
        Z = coords[["umap1", "umap2"]].to_numpy(dtype=float)
        labels_a = coords["label"].to_numpy()
        tag = "UMAP (redraw from saved coords)"
        if "facets" in plots:
            plot_facets(Z, labels_a, out / "01_umap_facets_by_epoch.png", tag)
        if "hulls" in plots:
            plot_hulls_centroids(Z, labels_a, out / "02_umap_hulls_centroids.png", tag)
        if "overview" in plots:
            plot_overview(Z, labels_a, out / "03_umap_overview_by_epoch.png", tag)
        print(f"[DONE] redraw → {out}  epochs={sorted(epoch_keep) if epoch_keep else 'all'}")
        return

    # Prefer newest multi-epoch run if run_id not set
    epoch_files = discover_epoch_files(args.data_dir, args.glob, args.run_id)
    if not epoch_files and args.run_id is None:
        # try pick the run_id with most epochs
        all_files = list(args.data_dir.glob(args.glob))
        by_run: dict[str, list[Path]] = {}
        for f in all_files:
            rm = RUN_RE.search(f.name)
            if rm:
                by_run.setdefault(rm.group(1), []).append(f)
        if by_run:
            best = max(by_run, key=lambda r: len(by_run[r]))
            print(f"[INFO] Auto-selected run-id={best} ({len(by_run[best])} files)")
            epoch_files = discover_epoch_files(args.data_dir, args.glob, best)

    if epoch_keep is not None:
        epoch_files = {e: pth for e, pth in epoch_files.items() if e in epoch_keep}

    if not epoch_files:
        raise SystemExit(f"No epoch CSVs found in {args.data_dir} matching {args.glob}")

    print(f"[INFO] Epochs found: {list(epoch_files)}")
    for ep, path in epoch_files.items():
        print(f"  ep{ep:02d} ← {path.name}")

    print("[INFO] Loading references…")
    chembl = read_smiles(args.chembl, max_n=None, seed=args.seed)
    chromo = read_smiles(args.train, max_n=args.max_ref, seed=args.seed)
    print(f"  chembl={len(chembl)}, chromophores={len(chromo)}")

    blocks: list[np.ndarray] = []
    labels: list[str] = []
    smiles_all: list[str] = []
    meta_rows = []

    for name, smis in (("chembl", chembl), ("chromophore", chromo)):
        valid, fps, n_inv = smiles_to_fps(smis, radius=args.radius, n_bits=args.n_bits, unique=True)
        print(f"  {name}: {len(valid)} (invalid≈{n_inv})")
        if not fps:
            continue
        blocks.append(fps_to_numpy(fps))
        labels.extend([name] * len(fps))
        smiles_all.extend(valid)

    for ep, path in epoch_files.items():
        smi, scores = load_rl_smiles(path, max_n=args.max_per_epoch, seed=args.seed)
        valid, fps, n_inv = smiles_to_fps(smi, radius=args.radius, n_bits=args.n_bits, unique=True)
        # align scores loosely by first occurrence map
        score_map = dict(zip(smi, scores))
        sc = np.array([score_map.get(s, np.nan) for s in valid], dtype=float)
        print(f"  epoch_{ep}: {len(valid)} unique valid (invalid≈{n_inv})")
        if not fps:
            continue
        blocks.append(fps_to_numpy(fps))
        labels.extend([f"epoch_{ep}"] * len(fps))
        smiles_all.extend(valid)
        for s, v in zip(valid, sc):
            meta_rows.append({"smiles": s, "epoch": ep, "score": v, "source": f"epoch_{ep}"})

    X = np.vstack(blocks)
    labels_a = np.asarray(labels)
    print(f"[INFO] Joint matrix {X.shape}")

    print("[INFO] Fitting UMAP…")
    Z, tag = embed_umap(X, args.seed, pca_dim=args.pca_prep)

    coords = pd.DataFrame(
        {
            "smiles": smiles_all,
            "label": labels_a,
            "umap1": Z[:, 0],
            "umap2": Z[:, 1],
        }
    )
    coords.to_csv(out / "coordinates_umap_epochs.csv", index=False)
    if meta_rows:
        pd.DataFrame(meta_rows).to_csv(out / "rl_epochs_meta.csv", index=False)

    if "facets" in plots:
        plot_facets(Z, labels_a, out / "01_umap_facets_by_epoch.png", tag)
    if "hulls" in plots:
        plot_hulls_centroids(Z, labels_a, out / "02_umap_hulls_centroids.png", tag)
    if "overview" in plots:
        plot_overview(Z, labels_a, out / "03_umap_overview_by_epoch.png", tag)

    # tiny report
    run_note = args.run_id or "auto"
    lines = [
        "# RL TL sweep — chemical space vs epochs",
        "",
        f"- run filter: `{run_note}`",
        f"- epochs: {', '.join(str(e) for e in epoch_files)}",
        f"- embedding: {tag}",
        f"- refs: ChEMBL drugs + chromophore train",
        "",
        "## Figures",
        "- `01_umap_facets_by_epoch.png` — shift panel-by-panel (main)",
        "- `02_umap_hulls_centroids.png` — hulls + centroid trajectory",
        "- `03_umap_overview_by_epoch.png` — all epochs overlaid",
        "",
        "Grey = ChEMBL drugs, green = chromophores, colored = RL at that epoch.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] → {out}")


if __name__ == "__main__":
    main()
