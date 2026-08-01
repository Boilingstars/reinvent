# Generator check vs training set

**Conditions:** `valid & clean & λ>0 & λ∈[450.0,480.0] & Score≥0.8`

## Summary
- RL rows: **1984**
- Pass conditions: **334** (16.8%)
- Pass & novel (not exact train): **334**
- Mean max-Tc to train (pass): **0.510**
- Mean Score (pass): **0.923**

## Figures
- `01_score_colored_overview.png` — Score / λ / novelty / funnel
- `02_novelty_vs_train.png` — Tanimoto to train
- `03_pass_rate_by_step.png` — условия по шагам RL
- `04_pca_train_rl_by_score.png` — PCA: все RL | только pass (цвет = Score)
- `05_umap_train_rl_by_score.png` — UMAP то же

Train set = REINVENT transfer-learning SMILES (`train.smi`).
