# RL TL sweep — chemical space vs epochs

- run filter: `20260801_090230`
- epochs: 0, 1, 2, 3, 4, 5, 6, 7, 8
- embedding: UMAP (PCA→50)
- refs: ChEMBL drugs + chromophore train

## Figures
- `01_umap_facets_by_epoch.png` — shift panel-by-panel (main)
- `02_umap_hulls_centroids.png` — hulls + centroid trajectory
- `03_umap_overview_by_epoch.png` — all epochs overlaid

Grey = ChEMBL drugs, green = chromophores, colored = RL at that epoch.
