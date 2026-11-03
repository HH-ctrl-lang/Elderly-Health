# Experiment 2611, step 3: permutation-based feature reliance

- Fixed existing V5 checkpoints; no retraining, tuning or model selection.
- Features permuted independently within each original site.
- Source validation and held-out test are reported separately.
- Main head uses both z_c and z_s; zc head uses z_c only.
- Positive AUC/AP drop means worse predictions after perturbation.
- A negative drop means shuffling improved the evaluated metric.
- SD across permutations reflects shuffling randomness only.
- Correlated features can make permutation effects hard to interpret.
- This is not proof of feature importance as a causal effect.
- Never select parameters or report an optimized model from held-out test results.
