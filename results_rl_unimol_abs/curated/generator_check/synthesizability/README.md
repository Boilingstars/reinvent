# Synthesizability pipeline

**Source:** `rl_tl_sweep_20260801_090230_ep04_1.csv` (epoch 4)

## Methods
1. **SA score (Ertl)** — fragment heuristic, 1–10, ↓ easier
2. **SCScore (Coley)** — NN trained on ~12M **Reaxys** reactions, 1–5, ↓ easier
3. **RAscore** — CASP proxy (AiZynth success probability) for highly-novel set

> «SA на базе Reaxys» в литературе = **SCScore**, не классический Ertl SA.

## Results (baseline λ∈[450,480], Score≥0.8)
- Baseline: **370**
- SA≤4.0: **338** (91.4%)
- SCScore≤3.0: **126** (34.1%)
- SA ∩ SCScore: **125** (33.8%)

Mean SA (baseline): 3.10 · Mean SCScore (baseline): 3.52

## Novelty of SCScore-pass (n=126)
- exact in train / ChEMBL / molecules.csv: **0**
- exact in PubChem: **5**
- highly_novel (Tc&lt;0.40 local): **14**
- RAscore on highly_novel: **13** likely_accessible, **1** likely_hard

## Figures
- `01_synth_methods_overview.png`
- `02_synth_filter_funnel.png`
- `03_threshold_sweeps.png`
- `04_scscore_pass_structures.png`
- `05_umap_top5_scscore_pass.png` — ep4 UMAP highlight
- `06_scscore_pass_novelty.png`
- `07_scscore_pass_novelty_structures.png`
- `08_highly_novel_structures.png`
- `09_highly_novel_casp_proxy_rascore.png`

## Tables
- `tables/rl_with_synth_scores.csv` — все скоры
- `tables/scscore_pass_sorted.csv` — SCScore≤3
- `tables/synth_pass_molecules.csv` — SA∩SCScore
- `tables/scscore_pass_novelty.csv` — novelty tiers + PubChem
- `tables/highly_novel_askcos_proxy_rascore.csv`
