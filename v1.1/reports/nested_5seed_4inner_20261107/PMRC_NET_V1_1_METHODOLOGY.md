# PMRC-Net v1.1 Methodology

## Versioning and preservation

The v1.0 source and historical outputs remain at the repository root and under their existing `results/`, `experiments/`, `checkpoints/`, `reports/`, and `final_status/` paths. v1.1 code, configurations, checkpoints, OOF predictions, figures, and reports are isolated under `v1.1/`. The pre-implementation SHA-256 inventory is `v1.1/reports/v1_0_preservation_manifest.json`.

Dataset: 602 exact-deduplicated rows; SHA-256 `a786b4a01e306cd892293c2671e15f29a29a17d06c69b16b19906a337ad42b90`. Patient IDs are unavailable; rows are not assumed independent patients. The continuous scaler is fit separately on each inner-training or outer-training partition. Categorical source mappings are explicit and unknown codes are rejected. No imputation is performed; non-finite continuous values are rejected before model computation.

## Nested evaluation

Five outer stratified folds are generated independently for seeds [21, 42, 123, 2026, 3407] and saved in the run split manifest. Each outer-training partition is split into four stratified inner folds. Inner validation binary cross-entropy selects a training duration: the ceiling of the median of inner-fold best epochs under the fixed patience and minimum-delta rules. A fresh model is then fit on all outer-training rows for exactly that selected duration. The outer test partition is evaluated once and is not passed to epoch selection, early stopping, scaler fitting, or checkpoint selection.

The fixed model hyperparameters are held constant in `v1.1/configs/nested_cv.yaml`. Runs are sequential, batch size 8, zero workers, two CPU threads, FP32, and AdamW with weight decay `0.0001`. No outer-test-driven hyperparameter selection or threshold optimization is performed. Seed 42 and all other configured seeds are preserved in the split manifest.

## Objective and architecture

The unsubstantiated empirical-direction hinge is removed rather than reoriented. The v1.1 objective is binary cross-entropy with logits:

$$L_{BCE}=-\frac{1}{N}\sum_{i=1}^N\left[y_i\log\sigma(z_i)+(1-y_i)\log(1-\sigma(z_i))\right].$$

AdamW weight decay remains an optimizer regularizer, not a second hand-written loss. No `l1_lambda` is exposed because the previous coefficient was unused and no scientifically specified L1 target was established. Mixed precision is disabled.

The pathway encoders, physiological mismatch representation, and additive logit heads are retained. For pathway pair $(l,r)$, v1.0's two products were identical and its second gate half was discarded. v1.1 uses ordered transforms:

$$f_{l\to r}=T_l(l)\odot T_r(r),\qquad f_{r\to l}=T_r(l)\odot T_l(r),$$
$$c_{lr}=\sigma(g_{lr})_1\odot f_{l\to r}+\sigma(g_{lr})_2\odot f_{r\to l}.$$

Both gate halves now contribute to the interaction output; each uses a distinct ordered transformation product. This corrects the named asymmetric construction without adding modules or increasing hidden dimension. It remains a computational design, not a proven biological mechanism. The prediction still decomposes as `z_total = z_mech + z_inter + z_res`.

The retained raw-scale mismatch is $m_e=I\max(1-u,0)$, where $u=HR_{max}/\max(208-0.7\,age,1)$ and $I=\operatorname{softplus}(\max(a_1,0)\,oldpeak+\max(a_2,0)\,exang+\max(a_3,0)\,oldpeak\cdot exang)$. Its parameters remain learned. No monotonic constraint is inferred from target correlations; there is no v1.1 physiological hinge term.

## Metrics and inference limits

Reported metrics are accuracy, precision, sensitivity/recall, specificity, F1, MCC, ROC-AUC, average precision, Brier score, and ten-bin equal-width ECE. Classification threshold is fixed at 0.5. Seed summary is mean and sample SD (`ddof=1`) of each seed's pooled outer OOF metrics. Paired percentile intervals resample sample IDs stratified by target while retaining all seeds' predictions for each sampled ID. They are descriptive row-level intervals; unknown patient clustering prevents patient-level inference.
