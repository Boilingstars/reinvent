# Epoch recommendation (set-level metrics)

Recommended epoch: **2**

How to read regimes:
- underfit / early — still close to drug-like prior; chromophore match weak
- recommended — best composite of chromophore proximity, novelty, diversity, hit-rate
- overfit / late — memorising train set (↑ max-Tc / ↓ novelty / ↓ diversity)

Note: comparisons use full set statistics (cross-mean / distributional distances),
not single nearest-neighbour pairs.

 epoch           regime  composite_score  tanimoto__mean_max_tanimoto_vs_chromophores  tanimoto__cross_mean_vs_chromophores  fcd__ffd_vs_chromophores  novelty__novelty_vs_chromophores  novelty__internal_diversity  hitrate__hitrate_chromophore_like
     0 underfit / early         3.200000                                     0.233149                              0.089298                 14.150052                          1.000000                     0.895886                           0.010204
     2      recommended         5.804927                                     0.379234                              0.117320                  6.515747                          1.000000                     0.885770                           0.388235
     4   late (monitor)         5.802630                                     0.377162                              0.118414                  5.822998                          1.000000                     0.884898                           0.329545
     6   overfit / late         4.824127                                     0.411501                              0.123641                  6.141358                          0.988235                     0.872639                           0.458824
     8   overfit / late         4.290895                                     0.446419                              0.124015                  5.956779                          0.976471                     0.873817                           0.482353
    10   late (monitor)         4.819026                                     0.409916                              0.120566                  6.358327                          0.988235                     0.878783                           0.470588
