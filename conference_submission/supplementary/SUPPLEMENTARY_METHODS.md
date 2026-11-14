# Supplementary Methods and Results

## Dataset handling

The exact cleaned CSV used by the completed run is identified by SHA-256 in `run_config.json`. Exact duplicate rows were removed before splitting. Continuous columns (`age`, `trestbps`, `chol`, `thalachh`, `oldpeak`) were standardized using only the relevant training partition. Categorical fields (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`) were mapped using the project loader. The loader preserves `ca=4`; the five retained `ca=4` rows are not missing-value imputations.

## Model equations

For pathway representations `h_m`, `h_e`, and `h_c`, and mismatch scalar `m_e`, each pair has forward and reverse transformed products:

```text
F_ij = T_i(h_i) * T_j(h_j)
R_ij = T_j(h_i) * T_i(h_j)
g_ij = sigmoid(s_gate * W_ij [h_i, h_j, m_e])
c_ij = g_forward * F_ij + g_reverse * R_ij
```

The learned gates are split into forward and reverse halves. The legacy control uses only the forward flow; it is a single-flow control, not a symmetric interaction. The directional arm learns separate transforms for the reversed ordering.

The physiology-derived scalar is implemented as:

```text
HR_pred = 208 - 0.7 * age
deficit = max(1 - thalachh / max(HR_pred, 1), 0)
ischemia = softplus(a1*oldpeak + a2*exang + a3*oldpeak*exang)
m_e = deficit * ischemia
```

Coefficients `a1..a3` are clamped nonnegative. This deterministic engineered quantity is not a patient-specific physiological measurement and has not been clinically validated in this experiment.

Separate linear projections of `m_e + h_c`, the sum of pairwise interactions, and the scaled sum of pathway representations form `z_mech`, `z_inter`, and `z_res`; `z_total = z_mech + z_inter + z_res`. Probabilities are sigmoid outputs, with a fold-specific scalar temperature fitted on inner OOF predictions for the nested arms.

The implemented objective combines binary cross entropy, an association-conditioned physiology hinge, coupling-weight regularization, gate regularization, residual-scale regularization, and optimizer weight decay. The configured `l1_lambda` is not used by the loss implementation. The empirical `ca` association is computed but is not used in the hinge. Across the 100 inner training partitions, positive association signs occurred for oldpeak in 4 and exang in 9; all 25 outer-refit partitions had negative signs for each, so these two feature-specific hinge terms were not active in the outer refits.

## Nested protocol

For each seed in `[21, 42, 123, 2026, 3407]`, outer assignments use stratified five-fold splitting with shuffle and that seed. For outer fold `k`, inner splitting uses four folds and `random_state = seed + 1009*k`. Three candidates vary dropout, learning rate, optimizer weight decay, physiology weight, gate weight, and residual weight (see `run_config.json`). Candidate selection uses mean inner ROC-AUC + 0.5 × MCC, with epoch selection based on inner validation loss. The selected candidate's inner OOF probabilities alone determine temperature and Youden-J threshold. The final model is refit on the complete outer-training partition and its held-out fold is scored once. The complete OOF set has 602 observations per seed per arm.

## Outcome definitions and statistical comparison

Discrimination includes ROC-AUC and average precision. Thresholded outcomes include accuracy, precision, sensitivity, specificity, F1, and MCC. Calibration includes Brier score and 10-bin expected calibration error. Per-fold and per-seed values are reported in `fold_metrics.csv` and `seed_metrics.csv`; raw row-level OOF predictions are deliberately not copied into this submission package.

Two-sided exact Wilcoxon signed-rank tests compare the five seed-level metrics. They are exploratory: the five estimates repeatedly use the same dataset and overlapping folds, are not five independent patient samples, and no multiplicity correction was applied. All nominal comparisons with the historical Stage-5 baseline are invalid as controlled inference because its outer validation fold selected the checkpoint/epoch. Those p-values are retained in `statistical_comparison_nominal.csv` with that warning; do not cite them as evidence of superiority or noninferiority.

## Supplementary files

- `fold_metrics.csv`: 25 fold summaries for each nested arm.
- `seed_metrics.csv`: five seed summaries for each nested arm and the historical baseline.
- `statistical_comparison_nominal.csv`: all run-generated paired tests, with the historical-baseline limitation.
- `run_config.json`: complete recorded configuration and dataset hash.
- `resource_usage_summary.json`: recorded run telemetry.
