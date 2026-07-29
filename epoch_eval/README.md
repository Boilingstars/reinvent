# Epoch evaluation for chromophore TL

Separate scripts — each computes one metric family and writes plots/CSV under `epochs/eval/`.

| Script | What it does |
|--------|----------------|
| `eval_tanimoto.py` | Intra-epoch, consecutive-epoch, vs chromophores, vs ChEMBL (cross-mean + max/median) |
| `eval_fcd.py` | Fréchet ChemNet Distance (+ fingerprint Fréchet fallback) |
| `eval_mahalanobis.py` | Mahalanobis distance to chromophore / ChEMBL PCA spaces |
| `eval_hitrate.py` | RDKit / REINVENT validity + chromophore-like oracle hit-rate |
| `eval_novelty_diversity.py` | Novelty, uniqueness, internal diversity trajectory |
| `eval_pca.py` | Joint 2D PCA: ChEMBL + chromophores + all epochs |
| `eval_summary_table.py` | Merges numeric CSVs → table + underfit/optimal/overfit advice |
| `run_all.py` | Runs everything in order |

## Quick start

```bash
# from repo root
python epoch_eval/eval_tanimoto.py
python epoch_eval/eval_fcd.py          # optional: pip install fcd
python epoch_eval/eval_mahalanobis.py
python epoch_eval/eval_hitrate.py --threshold 0.4
python epoch_eval/eval_novelty_diversity.py
python epoch_eval/eval_pca.py
python epoch_eval/eval_summary_table.py

# or all at once
python epoch_eval/run_all.py --max-ref 2000
```

Defaults:
- train chromophores: `data/train.smi`
- ChEMBL drugs: `data/chembl_drugs.smi`
- samples: `samples_by_epoch/samples_epoch_*.csv`
- outputs: `epochs/eval/<metric>/`

All set comparisons use **full pairwise / distributional** statistics (cross-mean Tanimoto, Fréchet, Mahalanobis), not a single nearest neighbour.
