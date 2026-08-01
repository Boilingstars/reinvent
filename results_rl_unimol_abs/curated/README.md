# Curated pack — RL UniMol abs

Единственный сохранённый набор артефактов из анализа RL UniMol abs.
Всё остальное из `results_rl_unimol_abs/` удалено как дубли / шум.

## Структура

```
curated/
  figures/   # ключевые картинки (01–21 + 06b + исходные имена)
  docs/      # интерпретации
  tables/    # CSV (+ embeddings/)
  data/      # входные CSV / SMI для пересборки
  scripts/   # копии генераторов
```

## Figures

| # | Файл | Смысл |
|---|------|--------|
| 01 | `01_property_histograms.png` | Распределения Score / λ / QED |
| 02 | `02_score_lambda_qed_scatter.png` | Score ≈ окно λ 450–480 nm |
| 03 | `03_metrics_vs_step.png` | Динамика RL |
| 04 | `04_top_molecules_grid.png` | Топ структуры |
| 05 | `05_umap_joint_top5_densities_rl_score.png` | Joint UMAP top-5 + RL |
| 06 | `06_distance_to_refs_by_score_bin.png` | NN-дистанции по Score-бинам |
| 06b | `06b_centroid_distance_by_bin.png` | Центроиды бинов vs refs |
| 07 | `07_rl_vs_chromophore_class_proximity.png` | RL vs train по классам |
| 08 | `08_nearest_class_assignment.png` | Доли nearest top-5 |
| 09 | `09_structure_interpretation_dashboard.png` | Дашборд top-5 |
| 10 | `10_highscore_class_profile.png` | Профиль Score ≥ 0.8 |
| 11 | `11_reps_all_classes.png` | Репрезентативные молекулы top-5 |
| 12 | `12_molecules_scaffold_interpretation.png` | vs molecules.csv |
| 13 | `13_umap_rl_by_nearest_molecules_scaffold.png` | UMAP nearest scaffold_id |
| 14–16 | `14/15/16_*_rl_sweep_vs_unimol.png` | UMAP / PCA / t-SNE: два RL |
| 17 | `17_correlation_heatmap.png` | Корреляции свойств |
| 18 | `18_lambda_windows.png` | Окна λ_abs |
| 19 | `19_umap_rl_by_nearest_class.png` | UMAP RL по nearest top-5 |
| 20 | `20_highscore_molecules_scaffolds.png` | High-Score vs molecules.csv |
| 21 | `21_reps_molecules_scaffolds.png` | Репы по molecules.csv |

Копии с исходными именами: `06_correlation_heatmap.png`, `08_top_molecules_grid.png`, `10_lambda_windows.png`.

## Data (вход)

- `rl_unimol_abs_1.csv`, `rl_tl_sweep_20260731_004743_ep00_1.csv`
- `molecules.csv`, `chembl_scaffold_classes_top5_rl.csv`
- `refs/chembl_drugs.smi`, `refs/train.smi`

## Scripts

Архивные копии. Пути внутри скриптов рассчитаны на старую структуру папок —
для пересборки правьте пути или восстанавливайте layout. Основные:

- `visualize_rl_unimol_abs.py` → 01–04, 17, 18
- `viz_umap_joint.py` → 05
- `viz_chemspace_bins_density.py` → 06, 06b
- `viz_class_proximity.py` → 07–08, 19
- `viz_structure_interpretation.py` → 09–11
- `viz_structure_vs_molecules.py` / `viz_umap_structure_molecules.py` → 12–13, 20–21
- `viz_chemspace_rl_compare.py` → 14–16

## Docs

- `INTERPRETATION_top5.md`, `INTERPRETATION_molecules.md`
- `SUMMARY_rl.md`, `SUMMARY_bins_density.md`
