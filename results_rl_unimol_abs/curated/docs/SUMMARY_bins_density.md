# Score-bin distances in chemical space

Bins: [0.0,0.2), [0.2,0.4), [0.4,0.6), [0.6,0.8), [0.8,1.0]

Mean nearest-neighbor distance of RL molecules in each Score bin to the nearest chromophore / ChEMBL molecule in the embedding.

## PCA

| Score bin | n | mean Score | mean NN to chromo | mean NN to ChEMBL | centroid to chromo | centroid to ChEMBL |
|---|---:|---:|---:|---:|---:|---:|
| [0.0,0.2) | 726 | 0.025 | 0.048 | 0.639 | 0.397 | 1.668 |
| [0.2,0.4) | 90 | 0.305 | 0.080 | 1.271 | 1.167 | 2.625 |
| [0.4,0.6) | 101 | 0.494 | 0.075 | 1.306 | 1.221 | 2.700 |
| [0.6,0.8) | 148 | 0.713 | 0.119 | 1.749 | 1.677 | 3.175 |
| [0.8,1.0] | 727 | 0.902 | 0.148 | 1.960 | 1.922 | 3.413 |

## UMAP

| Score bin | n | mean Score | mean NN to chromo | mean NN to ChEMBL | centroid to chromo | centroid to ChEMBL |
|---|---:|---:|---:|---:|---:|---:|
| [0.0,0.2) | 726 | 0.025 | 0.076 | 1.283 | 1.076 | 4.556 |
| [0.2,0.4) | 90 | 0.305 | 0.093 | 1.855 | 0.936 | 6.233 |
| [0.4,0.6) | 101 | 0.494 | 0.102 | 2.027 | 1.308 | 6.614 |
| [0.6,0.8) | 148 | 0.713 | 0.122 | 2.492 | 2.063 | 7.400 |
| [0.8,1.0] | 727 | 0.902 | 0.166 | 2.788 | 2.450 | 7.812 |

## t-SNE

| Score bin | n | mean Score | mean NN to chromo | mean NN to ChEMBL | centroid to chromo | centroid to ChEMBL |
|---|---:|---:|---:|---:|---:|---:|
| [0.0,0.2) | 726 | 0.025 | 1.275 | 12.565 | 6.867 | 52.416 |
| [0.2,0.4) | 90 | 0.305 | 1.387 | 21.809 | 29.353 | 74.839 |
| [0.4,0.6) | 101 | 0.494 | 1.596 | 24.595 | 32.352 | 77.746 |
| [0.6,0.8) | 148 | 0.713 | 2.066 | 31.090 | 39.375 | 84.062 |
| [0.8,1.0] | 727 | 0.902 | 2.793 | 35.169 | 43.158 | 87.954 |
