#!/usr/bin/env python3
"""Analyze REINVENT transfer-learning metrics to choose the best epoch.

Reads TensorBoard scalars from ``tb_TL/`` and summary lines from ``logs/tl.log``,
then plots loss / sample-quality curves and prints a stop-epoch recommendation.

Usage (from ``reinvent/`` root):

    python tl_epoch_analysis.py
    python tl_epoch_analysis.py --out-dir epochs/tl_analysis
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Need tensorboard: pip install tensorboard"
    ) from exc


# Tag names as written by REINVENT 4.x into TensorBoard
TAG_TRAIN = "A_Mean NLL loss"
TAG_VAL = "A_Mean NLL loss"
TAG_SAMPLE = "A_Mean NLL loss"
TAG_VALID_FRAC = "B_Fraction valid SMILES"
TAG_DUP_FRAC = "C_Fraction duplicate SMILES"
TAG_DIVERSITY = "D_Internal Diversity of sample"

TB_SUBDIRS = {
    "train": "A_Mean NLL loss_Training Loss",
    "val": "A_Mean NLL loss_Validation Loss",
    "sample": "A_Mean NLL loss_Sample Loss",
}


def reinvent_root() -> Path:
    return Path(__file__).resolve().parent


def load_tb_series(tb_dir: Path, subdir: str, tag: str) -> list[tuple[int, float]]:
    """Load scalar series from the longest run under ``tb_dir/subdir``."""
    subpath = tb_dir / subdir
    if not subpath.is_dir():
        return []

    best: list[tuple[int, float]] = []
    for event_file in subpath.glob("events.out.tfevents.*"):
        ea = EventAccumulator(str(event_file))
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            continue
        series = [(int(e.step), float(e.value)) for e in ea.Scalars(tag)]
        if len(series) > len(best):
            best = series
    return sorted(best, key=lambda x: x[0])


def load_tb_root_series(tb_dir: Path, tag: str) -> list[tuple[int, float]]:
    """Load sample-quality scalars stored directly under ``tb_dir/``."""
    best: list[tuple[int, float]] = []
    for event_file in tb_dir.glob("events.out.tfevents.*"):
        ea = EventAccumulator(str(event_file))
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            continue
        series = [(int(e.step), float(e.value)) for e in ea.Scalars(tag)]
        if len(series) > len(best):
            best = series
    return sorted(best, key=lambda x: x[0])


def parse_tl_log(log_path: Path) -> dict[str, float | int | None]:
    """Extract REINVENT summary lines from ``tl.log``."""
    out: dict[str, float | int | None] = {
        "best_val_loss": None,
        "best_val_epoch": None,
    }
    if not log_path.is_file():
        return out

    text = log_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r"Best validation loss \(([\d.]+)\) was at epoch (\d+)",
        text,
    )
    if m:
        out["best_val_loss"] = float(m.group(1))
        out["best_val_epoch"] = int(m.group(2))
    return out


def load_tanimoto_summary(path: Path) -> list[dict[str, str | float]]:
    if not path.is_file():
        return []
    rows: list[dict[str, str | float]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed: dict[str, str | float] = {}
            for key, val in row.items():
                if key == "epoch":
                    parsed[key] = int(val)
                else:
                    try:
                        parsed[key] = float(val)
                    except (TypeError, ValueError):
                        parsed[key] = val
            rows.append(parsed)
    return sorted(rows, key=lambda r: int(r["epoch"]))  # type: ignore[arg-type]


def pick_best_epoch(
    train: list[tuple[int, float]],
    val: list[tuple[int, float]],
    log_info: dict[str, float | int | None],
) -> tuple[int, str]:
    """Return recommended epoch and a short rationale."""
    if not val:
        if log_info.get("best_val_epoch") is not None:
            ep = int(log_info["best_val_epoch"])
            return ep, f"log reports minimum validation loss at epoch {ep}"
        raise RuntimeError("No validation-loss series found in TensorBoard or log.")

    val_dict = dict(val)
    best_ep = min(val_dict, key=val_dict.get)  # type: ignore[arg-type]
    best_loss = val_dict[best_ep]

    train_dict = dict(train)
    if best_ep in train_dict:
        gap = best_loss - train_dict[best_ep]
        if best_ep < max(val_dict):
            later_eps = [e for e in val_dict if e > best_ep]
            if later_eps and all(val_dict[e] > best_loss for e in later_eps):
                return best_ep, (
                    f"validation loss minimum ({best_loss:.3f}) at epoch {best_ep}; "
                    f"later epochs overfit (train–val gap {gap:+.2f})"
                )

    return best_ep, f"validation loss minimum ({best_loss:.3f}) at epoch {best_ep}"


def _plot_series(
    ax: plt.Axes,
    series: list[tuple[int, float]],
    label: str,
    marker: str,
    color: str,
) -> None:
    if not series:
        return
    xs, ys = zip(*series)
    ax.plot(xs, ys, marker=marker, label=label, color=color, linewidth=2)


def plot_losses(
    train: list[tuple[int, float]],
    val: list[tuple[int, float]],
    sample: list[tuple[int, float]],
    best_epoch: int,
    out_path: Path,
    log_info: dict[str, float | int | None],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_series(ax, train, "Training NLL", "o", "#1f77b4")
    _plot_series(ax, val, "Validation NLL", "s", "#ff7f0e")
    _plot_series(ax, sample, "Sample NLL", "^", "#2ca02c")

    ax.axvline(best_epoch, color="crimson", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(
        best_epoch,
        ax.get_ylim()[1],
        f"  stop @ {best_epoch}",
        color="crimson",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    if log_info.get("best_val_loss") is not None:
        ax.set_title(
            "Transfer learning — NLL loss\n"
            f"(log: best val = {log_info['best_val_loss']:.3f} @ epoch {log_info['best_val_epoch']})"
        )
    else:
        ax.set_title("Transfer learning — NLL loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean NLL loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_quality(
    valid_frac: list[tuple[int, float]],
    dup_frac: list[tuple[int, float]],
    diversity: list[tuple[int, float]],
    best_epoch: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)

    panels = [
        (axes[0], valid_frac, "Fraction valid SMILES", "#9467bd"),
        (axes[1], dup_frac, "Fraction duplicate SMILES", "#8c564b"),
        (axes[2], diversity, "Internal diversity", "#17becf"),
    ]
    for ax, series, title, color in panels:
        if series:
            xs, ys = zip(*series)
            ax.plot(xs, ys, marker="o", color=color, linewidth=2)
        ax.axvline(best_epoch, color="crimson", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Sample quality during transfer learning", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tanimoto(summary_rows: list[dict[str, str | float]], best_epoch: int, out_path: Path) -> None:
    if not summary_rows:
        return

    epochs = [int(r["epoch"]) for r in summary_rows]
    mean_max = [float(r["mean_max_tanimoto_to_reference"]) for r in summary_rows]
    frac_high = [float(r["fraction_ge_0_85"]) for r in summary_rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, mean_max, "o-", color="#d62728", label="mean max Tanimoto to train")
    ax1.axvline(best_epoch, color="crimson", linestyle="--", linewidth=1.2, alpha=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean max Tanimoto")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, frac_high, "s--", color="#2ca02c", label="fraction Tanimoto ≥ 0.85")
    ax2.set_ylabel("Fraction ≥ 0.85")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Novelty vs train set (sampled molecules)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_summary_csv(
    out_path: Path,
    train: list[tuple[int, float]],
    val: list[tuple[int, float]],
    sample: list[tuple[int, float]],
    valid_frac: list[tuple[int, float]],
    dup_frac: list[tuple[int, float]],
    diversity: list[tuple[int, float]],
    best_epoch: int,
) -> None:
    epochs = sorted(
        {e for series in (train, val, sample, valid_frac, dup_frac, diversity) for e, _ in series}
    )
    lookup = {
        "train_nll": dict(train),
        "val_nll": dict(val),
        "sample_nll": dict(sample),
        "valid_frac": dict(valid_frac),
        "dup_frac": dict(dup_frac),
        "diversity": dict(diversity),
    }
    fieldnames = list(lookup) + ["recommended_stop"]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", *fieldnames])
        writer.writeheader()
        for ep in epochs:
            row = {"epoch": ep, "recommended_stop": int(ep == best_epoch)}
            for key, d in lookup.items():
                row[key] = f"{d[ep]:.6f}" if ep in d else ""
            writer.writerow(row)


def print_table(
    train: list[tuple[int, float]],
    val: list[tuple[int, float]],
    sample: list[tuple[int, float]],
    valid_frac: list[tuple[int, float]],
) -> None:
    epochs = sorted({e for series in (train, val, sample, valid_frac) for e, _ in series})
    lookup = {
        "train": dict(train),
        "val": dict(val),
        "sample": dict(sample),
        "valid": dict(valid_frac),
    }
    print("\nepoch\ttrain_nll\tval_nll\tsample_nll\tvalid_frac")
    for ep in epochs:
        cols = [str(ep)]
        for key in ("train", "val", "sample", "valid"):
            cols.append(f"{lookup[key][ep]:.4f}" if ep in lookup[key] else "-")
        print("\t".join(cols))


def main() -> int:
    root = reinvent_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tb-dir", type=Path, default=root / "tb_TL")
    parser.add_argument("--log", type=Path, default=root / "logs" / "tl.log")
    parser.add_argument(
        "--tanimoto-summary",
        type=Path,
        default=root / "epochs" / "per_epoch_summary.csv",
        help="Optional per-epoch Tanimoto summary for a second figure",
    )
    parser.add_argument("--out-dir", type=Path, default=root / "epochs" / "tl_analysis")
    args = parser.parse_args()

    train = load_tb_series(args.tb_dir, TB_SUBDIRS["train"], TAG_TRAIN)
    val = load_tb_series(args.tb_dir, TB_SUBDIRS["val"], TAG_VAL)
    sample = load_tb_series(args.tb_dir, TB_SUBDIRS["sample"], TAG_SAMPLE)
    valid_frac = load_tb_root_series(args.tb_dir, TAG_VALID_FRAC)
    dup_frac = load_tb_root_series(args.tb_dir, TAG_DUP_FRAC)
    diversity = load_tb_root_series(args.tb_dir, TAG_DIVERSITY)
    log_info = parse_tl_log(args.log)
    tanimoto_rows = load_tanimoto_summary(args.tanimoto_summary)

    if not val and log_info.get("best_val_epoch") is None:
        print("FAIL: no validation metrics in TensorBoard or log", file=sys.stderr)
        return 1

    best_epoch, rationale = pick_best_epoch(train, val, log_info)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loss_png = args.out_dir / "tl_loss_curves.png"
    quality_png = args.out_dir / "tl_sample_quality.png"
    tanimoto_png = args.out_dir / "tl_tanimoto_vs_epoch.png"
    summary_csv = args.out_dir / "tl_metrics_by_epoch.csv"

    plot_losses(train, val, sample, best_epoch, loss_png, log_info)
    plot_quality(valid_frac, dup_frac, diversity, best_epoch, quality_png)
    plot_tanimoto(tanimoto_rows, best_epoch, tanimoto_png)
    write_summary_csv(
        summary_csv, train, val, sample, valid_frac, dup_frac, diversity, best_epoch
    )

    print_table(train, val, sample, valid_frac)
    print(f"\nRecommended checkpoint epoch: {best_epoch}")
    print(f"Reason: {rationale}")
    print(f"Use checkpoint: checkpoints/TL_reinvent.model.{best_epoch}.chkpt")
    print("\nSaved:")
    print(f"  {loss_png}")
    print(f"  {quality_png}")
    if tanimoto_rows:
        print(f"  {tanimoto_png}")
    print(f"  {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
