# RL chemspace compare: TL sweep vs UniMol abs

Joint PCA / UMAP / t-SNE on ChEMBL + chromophores + both RL sets.
For each method: two plots (sweep / unimol) with identical reference coordinates,
plus a side-by-side panel.

- Sweep: `rl_tl_sweep_20260731_004743_ep00_1.csv`
- UniMol: `rl_unimol_abs_1.csv`
- ChEMBL n=2021, chromophore n=4000
- RL sweep n=2896, RL unimol n=1792

Files: `{pca,umap,tsne}_01_rl_sweep.png`, `{pca,umap,tsne}_02_rl_unimol.png`,
`{method}_00_side_by_side.png`, `coordinates_*.csv`.
