# Challenge report

Five exams of the RL generator. One figure per exam.

## Is this set enough?
Yes for a first challenge battery, with Coverage defined against **molecules.csv** as the
target chemical space (not just the λ window). Together they probe: reach of target chemistry,
mode collapse, generalization, joint novelty×quality, and reward/prior pathologies.

## Results (short)

| # | Exam | Key number | Figure |
|---|---|---|---|
| 1 | Coverage / Recall | Coverage@0.50=1.1%; Recall@0.50=1.7% | `01_coverage_recall.png` |
| 2 | Scaffold collapse | dominance=0.014, H_norm=0.981, N_eff=543 | `02_scaffold_collapse.png` |
| 3 | Memorization vs extrapolation | extrapolate=23.4%, memorize=9.5% of high-Score | `03_memorization_vs_extrapolation.png` |
| 4 | Novelty–Precision frontier | true successes=75 (22.5% of on-target) | `04_novelty_precision_frontier.png` |
| 5 | Prior drift / hacking | ΔPrior(high−low)=5.5; cheap QED rate=72.8% | `05_prior_drift_reward_hacking.png` |

## Reading guide
- **Low Coverage/Recall** → generator does not populate literature chromophore space.
- **Dominance↑ / H_norm↓ with Score** → scaffold collapse under RL.
- **Few extrapolation points** → no generalization beyond train neighborhood.
- **Empty success quadrant** on frontier → cannot be novel *and* on-target at once.
- **Prior↑ with Score + cheap QED** → drifted from prior and/or reward hacking the surrogate.
