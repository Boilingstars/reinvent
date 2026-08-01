# Structural interpretation of RL UniMol generation

## Method
Each RL molecule is assigned to the **nearest scaffold family** among top-5
(stilbene, benzothiazole, chalcone, azobenzene, thiazolidinedione)
by **maximum Morgan Tanimoto** to molecules of that family.

This is a soft structural label — not a hard SMARTS/substructure match.

## Key numbers
- RL molecules classified: **1792**
- High-Score (S≥0.8): **727** (40.6%)

### Nearest-class share (all RL)
- **stilbene** (стильбен): 37.4%
- **benzothiazole** (бензотиазол): 24.2%
- **chalcone** (халкон): 11.8%
- **azobenzene** (азобензол): 19.0%
- **thiazolidinedione** (тиазолидиндион): 7.5%

### Nearest-class share (Score ≥ 0.8)
- **stilbene** (стильбен): 46.6%
- **benzothiazole** (бензотиазол): 27.5%
- **chalcone** (халкон): 8.0%
- **azobenzene** (азобензол): 14.3%
- **thiazolidinedione** (тиазолидиндион): 3.6%

### Shift vs train chromophores (Δ mean max-Tc)
- **stilbene**: Δ=+0.024 (RL 0.275 vs train 0.250)
- **benzothiazole**: Δ=+0.028 (RL 0.255 vs train 0.226)
- **chalcone**: Δ=+0.006 (RL 0.248 vs train 0.242)
- **azobenzene**: Δ=+0.010 (RL 0.243 vs train 0.233)
- **thiazolidinedione**: Δ=+0.001 (RL 0.231 vs train 0.230)

## How to read the figures
- `01_…dashboard`: UMAP by class + composition + Δ vs train + Score–Tc scatter
- `02_reps_*`: example structures for each family (high Score preferred)
- `03_…profile`: donut of high-Score families + mean Score by class

## Chemical takeaway
Generation concentrates on **stilbene-like** and **benzothiazole-like** neighborhoods;
high-Score molecules amplify this bias. Chalcone/thiazolidinedione remain secondary.
Relative to the training chromophore set, RL **over-weights stilbene/benzothiazole**
and under-weights some other families — a signature of the λ_abs reward landscape.
