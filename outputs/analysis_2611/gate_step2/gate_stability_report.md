# CausalGate stability diagnostics (descriptive, seed=42)

This is a ranking/allocation analysis, **not** an assessment of causal effects or predictive feature importance.

## Pairwise summary

| Group | Pair count | Mean Spearman rank correlation | Mean top-5 Jaccard |
|---|---:|---:|---:|
| LOEO_site | 66 | 0.5878 | 0.5089 |
| LORO_region | 3 | 0.2234 | 0.3889 |

## Common LORO top-5 features

age, if_use_mobile_phone

## Caveats

- All results are seed=42 and use overlapping source training samples between folds.
- Rank consistency is relative, even when all gate values remain close to 0.5.
- A score above 0.5 reflects allocation to z_c; it is not a hard selected feature.
- The V5 joint prediction head uses z_c and z_s, so gate scores are not feature importances.
- Threshold sensitivity is a display diagnostic, not hyperparameter tuning.
- Do not select a threshold or algorithm from held-out test performance.
