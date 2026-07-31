# Visualization of RL + Uni-Mol absorption results

Source CSV (repo root): `rl_unimol_abs_1.csv`

## Run

```bash
# from repo root, using the eval venv if present
.\.venv_eval\Scripts\python.exe results_rl_unimol_abs\visualize_rl_unimol_abs.py
```

Needs: `pandas`, `matplotlib`, `seaborn`, `numpy`, `rdkit`, optionally `umap-learn`.

## Outputs

| File | Meaning |
|------|---------|
| `01_property_histograms.png` | Distributions of Score, λ, QED, SA, step |
| `02_score_lambda_qed_scatter.png` | Score–λ and λ–QED relationships |
| `03_metrics_vs_step.png` | RL learning curves by step |
| `04_lambda_score_boxplot_by_step.png` | Per-step boxplots |
| `05_pairwise_properties.png` | Pairplot of key properties |
| `06_correlation_heatmap.png` | Correlations |
| `07_nll_panels.png` | Agent/Prior NLL vs properties |
| `08_top_molecules_grid.png` | Top-scoring clean molecules |
| `09_chemspace_lambda_score.png` | UMAP/t-SNE of chemical space |
| `10_lambda_windows.png` | Optical wavelength windows |
| `SUMMARY.md` + `tables/` | Numeric summaries |
