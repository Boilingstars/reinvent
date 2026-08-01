# Generator check vs training set

**Source:** `rl_tl_sweep_20260801_090230_ep04_1.csv` (epoch 4 TL sweep)

**Conditions:** `valid & clean & λ>0 & λ∈[450.0,480.0] & Score≥0.8`

## Summary
- RL rows: **3200**
- Pass conditions: **370** (11.6%)
- Pass & novel (not exact train): **370**
- Mean max-Tc to train (pass): **0.365**
- Mean Score (pass): **0.922**

## Figures
- `01_score_colored_overview.png` — Score / λ / novelty / funnel
- `02_novelty_vs_train.png` — Tanimoto to train
- `03_pass_rate_by_step.png` — условия по шагам RL
- `04_pca_train_rl_by_score.png` — PCA: все RL | только pass (цвет = Score)
- `05_umap_train_rl_by_score.png` — UMAP то же

Train set = REINVENT transfer-learning SMILES (`train.smi`).
