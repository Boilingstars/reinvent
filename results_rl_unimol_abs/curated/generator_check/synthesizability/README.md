# Synthesizability pipeline

## Methods
1. **SA score (Ertl)** — fragment heuristic, 1–10, ↓ easier
2. **SCScore (Coley)** — NN trained on ~12M **Reaxys** reactions, 1–5, ↓ easier
3. **RAscore** — optional; predicts AiZynthFinder success

> «SA на базе Reaxys» в литературе = **SCScore**, не классический Ertl SA.

## Results (baseline λ∈[450.0,480.0], Score≥0.8)
- Baseline: **334**
- SA≤4.0: **317** (94.9%)
- SCScore≤3.0: **25** (7.5%)
- SA ∩ SCScore: **25** (7.5%)

Mean SA (baseline): 3.18
Mean SCScore (baseline): 4.21

## Figures
- `01_synth_methods_overview.png`
- `02_synth_filter_funnel.png`
- `03_threshold_sweeps.png`

## Tables
- `tables/rl_with_synth_scores.csv` — все скоры
- `tables/synth_pass_molecules.csv` — прошедшие фильтр
