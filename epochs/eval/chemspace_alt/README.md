# Alternative chemical-space views

Plain PCA often fails for Morgan fingerprints (low variance on PC1/PC2, overplotting).
This folder compares embeddings that preserve local neighbourhoods better:

| File pattern | Idea |
|---|---|
| `*_overview.png` | Classic scatter on UMAP / t-SNE |
| `*_hexbin_density.png` | Density of ChEMBL or chromophores; epochs as points |
| `*_kde_contours.png` | Smooth density contours of both references |
| `*_facets_by_epoch.png` | One panel per epoch (avoids overplotting epochs) |
| `*_hulls_centroids.png` | Convex hull + centroid path of each epoch |
| `*_similarity_colored.png` | Epoch points colored by max Tc → chromophores |

Prefer **facets** + **similarity-colored UMAP** for interpreting TL drift;
prefer **hexbin/KDE** when reference clouds dominate the plot.
