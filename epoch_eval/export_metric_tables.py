"""
export_metric_tables.py — turn all epoch_eval metric CSVs into readable tables.

Writes:
  * epochs/eval/tables/metrics_tables.html   — styled multi-table report
  * epochs/eval/tables/<name>.md             — one Markdown file per CSV
  * epochs/eval/tables/all_metrics.md        — single Markdown with all tables
  * epochs/eval/tables/all_metrics.xlsx      — Excel workbook (one sheet per table)
    if openpyxl / xlsxwriter is available; otherwise skipped

Skips huge coordinate dumps (e.g. pca_coordinates.csv).

Usage (from repo root):
  python epoch_eval/export_metric_tables.py
  python epoch_eval/export_metric_tables.py --eval-root epochs/eval --out-dir epochs/eval/tables
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DEFAULT_OUT, ensure_dir

# Raw point clouds — not summary metric tables
SKIP_NAMES = {
    "pca_coordinates.csv",
}

# Pretty section titles for known files
TITLES = {
    "tanimoto_by_epoch.csv": "Tanimoto — by epoch (set-level)",
    "tanimoto_consecutive_epochs.csv": "Tanimoto — consecutive epochs",
    "fcd_by_epoch.csv": "Fréchet ChemNet / Fingerprint Distance",
    "mahalanobis_by_epoch.csv": "Mahalanobis distance",
    "hitrate_by_epoch.csv": "Hit-rate",
    "novelty_diversity_by_epoch.csv": "Novelty & diversity",
    "metrics_summary_by_epoch.csv": "Summary — all metrics + regime",
}


def discover_csvs(eval_root: Path) -> list[Path]:
    files = sorted(eval_root.rglob("*.csv"))
    return [p for p in files if p.name not in SKIP_NAMES and "tables" not in p.parts]


def round_df(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(digits)
    return out


def load_tables(eval_root: Path, digits: int) -> list[tuple[str, Path, pd.DataFrame]]:
    tables: list[tuple[str, Path, pd.DataFrame]] = []
    for path in discover_csvs(eval_root):
        title = TITLES.get(path.name, path.stem.replace("_", " ").title())
        rel = path.relative_to(eval_root).as_posix()
        df = round_df(pd.read_csv(path), digits=digits)
        tables.append((f"{title} ({rel})", path, df))
    return tables


def df_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        # fallback without tabulate
        cols = list(df.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = []
        for _, row in df.iterrows():
            rows.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
        return "\n".join([header, sep, *rows])


def write_markdown(tables: list[tuple[str, Path, pd.DataFrame]], out_dir: Path) -> Path:
    parts = ["# Epoch evaluation — metric tables", ""]
    for title, path, df in tables:
        safe = path.stem + ".md"
        single = out_dir / safe
        body = f"# {title}\n\nSource: `{path}`\n\n{df_to_markdown(df)}\n"
        single.write_text(body, encoding="utf-8")
        parts.append(f"## {title}")
        parts.append("")
        parts.append(f"Source: `{path}`")
        parts.append("")
        parts.append(df_to_markdown(df))
        parts.append("")
    all_path = out_dir / "all_metrics.md"
    all_path.write_text("\n".join(parts), encoding="utf-8")
    return all_path


def write_html(tables: list[tuple[str, Path, pd.DataFrame]], out_dir: Path) -> Path:
    css = """
    :root { color-scheme: light dark; }
    body { font-family: ui-sans-serif, system-ui, Segoe UI, sans-serif;
           margin: 2rem; line-height: 1.4; }
    h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
    h2 { font-size: 1.15rem; margin-top: 2rem; border-bottom: 1px solid #8884; padding-bottom: 0.3rem; }
    .meta { color: #888; font-size: 0.85rem; margin-bottom: 0.75rem; }
    .toc a { margin-right: 1rem; }
    table { border-collapse: collapse; width: 100%; font-size: 0.85rem; margin-bottom: 1rem; }
    th, td { border: 1px solid #8885; padding: 0.35rem 0.55rem; text-align: right; }
    th { background: #8882; text-align: left; position: sticky; top: 0; }
    td:first-child, th:first-child { text-align: left; }
    tr:nth-child(even) { background: #8881; }
    .wrap { overflow-x: auto; }
    """
    toc = []
    sections = []
    for i, (title, path, df) in enumerate(tables):
        anchor = f"t{i}"
        toc.append(f'<a href="#{anchor}">{html.escape(title)}</a>')
        table_html = df.to_html(index=False, border=0, classes=None, na_rep="")
        sections.append(
            f'<h2 id="{anchor}">{html.escape(title)}</h2>\n'
            f'<p class="meta">Source: <code>{html.escape(str(path))}</code> · '
            f'{len(df)} rows × {df.shape[1]} cols</p>\n'
            f'<div class="wrap">{table_html}</div>\n'
        )

    doc = (
        "<!DOCTYPE html>\n<html lang='ru'><head><meta charset='utf-8'>\n"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        "<title>Epoch evaluation — metric tables</title>\n"
        f"<style>{css}</style>\n</head><body>\n"
        "<h1>Epoch evaluation — metric tables</h1>\n"
        f"<p class='meta'>{len(tables)} tables</p>\n"
        f"<nav class='toc'>{' '.join(toc)}</nav>\n"
        + "\n".join(sections)
        + "\n</body></html>\n"
    )
    out = out_dir / "metrics_tables.html"
    out.write_text(doc, encoding="utf-8")
    return out


def write_excel(tables: list[tuple[str, Path, pd.DataFrame]], out_dir: Path) -> Path | None:
    out = out_dir / "all_metrics.xlsx"
    try:
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            used: set[str] = set()
            for title, path, df in tables:
                sheet = path.stem[:31]
                base = sheet
                k = 1
                while sheet in used:
                    sheet = f"{base[:28]}_{k}"
                    k += 1
                used.add(sheet)
                df.to_excel(writer, sheet_name=sheet, index=False)
        return out
    except Exception as exc:
        print(f"[WARN] Excel export skipped: {exc}")
        print("       Optional: pip install openpyxl")
        return None


def print_console(tables: list[tuple[str, Path, pd.DataFrame]]) -> None:
    for title, path, df in tables:
        print("\n" + "=" * 72)
        print(title)
        print(path)
        print("=" * 72)
        print(df.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description="Export metric CSVs as HTML / Markdown / Excel tables")
    p.add_argument("--eval-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--digits", type=int, default=4)
    p.add_argument("--no-console", action="store_true")
    args = p.parse_args()

    out_dir = ensure_dir(args.out_dir or args.eval_root / "tables")
    tables = load_tables(args.eval_root, digits=args.digits)
    if not tables:
        raise SystemExit(f"No metric CSVs found under {args.eval_root}")

    html_path = write_html(tables, out_dir)
    md_path = write_markdown(tables, out_dir)
    xlsx_path = write_excel(tables, out_dir)

    if not args.no_console:
        print_console(tables)

    print(f"\n[OK] HTML  → {html_path}")
    print(f"[OK] MD    → {md_path}")
    if xlsx_path:
        print(f"[OK] Excel → {xlsx_path}")
    print(f"[OK] Per-table .md files in {out_dir}")


if __name__ == "__main__":
    main()
