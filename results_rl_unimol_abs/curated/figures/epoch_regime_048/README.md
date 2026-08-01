# Epoch regime: underfit (0) / optimal (4) / overfit (8)

RL TL sweep `20260801_090230`, epochs **0, 4, 8** only.

## Verdict
| epoch | regime | composite |
|---|---|---:|
| 0 | underfit / early | 3.20 |
| **4** | **recommended** | **4.87** |
| 8 | overfit / late | 4.30 |

See `summary/epoch_recommendation.md`.

## Figures
| Plot | Path |
|---|---|
| Tanimoto max / cross-mean | `tanimoto/tanimoto_max_median_and_crossmean.png` |
| Max-Tc boxplots | `tanimoto/tanimoto_max_boxplots.png` |
| Cross-epoch Tanimoto | `tanimoto/tanimoto_cross_epoch.png` |
| Novelty & diversity | `novelty_diversity/diversity_novelty_trajectories.png` |
| Hit-rate | `hitrate/hitrate_curves.png` |
| FCD / FFD | `fcd/fcd_curves.png` |
| Mahalanobis curves | `mahalanobis/mahalanobis_curves.png` |
| Mahalanobis boxplots | `mahalanobis/mahalanobis_boxplots.png` |
| Chemspace UMAP (0/4/8) | `../chemspace_sweep_epochs/01_umap_facets_by_epoch.png` |
| Verdict | `summary/epoch_recommendation.md` |

## How to read (expected story)
- **ep0 underfit**: far from chromophores (high FCD/FFD, low chromo hit-rate), still drug-like (low FFD to ChEMBL), high diversity
- **ep4 optimal**: better chromophore proximity + hit-rate, diversity still healthy, novelty intact
- **ep8 overfit**: even closer to chromophores / higher hit-rate, but diversity↓ and distribution moves away from ChEMBL (FFD_chembl↑) — late specialization / collapse of variety
