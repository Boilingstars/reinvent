# RL vs molecules.csv scaffolds

Each RL molecule is assigned to the nearest `scaffold_id` group in `molecules.csv`
by maximum Morgan Tanimoto.

- RL molecules: **1792**
- High-Score (S≥0.8): **727**

## Top nearest scaffolds (all RL)
- **stilbene** (Stilbene): 18.7% of RL, mean Score=0.65, high-Score share=26.4%
- **chalcone** (Chalcone): 7.8% of RL, mean Score=0.43, high-Score share=6.7%
- **benzothiazole** (Benzothiazole): 7.4% of RL, mean Score=0.62, high-Score share=9.4%

## Broader scaffold_class among nearest hits
- **chromophore**: 62.5% all RL; high-Score 65.7%
- **heterocycle**: 17.7% all RL; high-Score 17.9%
- **natural_product**: 10.4% all RL; high-Score 8.9%
- **scaffold**: 7.5% all RL; high-Score 6.6%
- **drug_class**: 1.9% all RL; high-Score 0.8%

## Why use molecules.csv in addition to top-5 ChEMBL classes?
- `molecules.csv` has **finer** families (fluorescein, cyanine, BODIPY, rhodamine, …)
  that are more dye/chromophore-specific than stilbene/chalcone alone.
- Good for asking: *does RL rediscover classic dye scaffolds or invent stilbene-like generics?*
- Top-5 ChEMBL classes answer *broad chemotype neighborhood*;
  molecules.csv answers *which named dye/drug scaffolds are closest*.

Compare with `structure_interpretation/` (top-5 ChEMBL families).
