# Epoch recommendation (set-level metrics)

Recommended epoch: **4**

How to read regimes:
- underfit / early — still close to drug-like prior; chromophore match weak
- recommended — best composite of chromophore proximity, novelty, diversity, hit-rate
- overfit / late — memorising train set (↑ max-Tc / ↓ novelty / ↓ diversity)

Note: comparisons use full set statistics (cross-mean / distributional distances),
not single nearest-neighbour pairs.

 epoch           regime  composite_score  tanimoto__mean_max_tanimoto_vs_chromophores  tanimoto__cross_mean_vs_chromophores  fcd__ffd_vs_chromophores  novelty__novelty_vs_chromophores  novelty__internal_diversity  hitrate__hitrate_chromophore_like
     0 underfit / early         3.200000                                     0.239873                              0.093464                 10.816898                           1.00000                     0.900691                           0.013437
     4      recommended         4.865932                                     0.333881                              0.109237                  9.519869                           1.00000                     0.876046                           0.187188
     8   overfit / late         4.300000                                     0.375257                              0.121253                  8.056163                           0.99774                     0.852381                           0.321250
