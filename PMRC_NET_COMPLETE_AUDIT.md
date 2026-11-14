# PMRC-Net Complete Audit

Audit date: 2026-11-07  
Dataset: 602 exact-row-deduplicated observations; SHA-256 `a786b4a01e306cd892293c2671e15f29a29a17d06c69b16b19906a337ad42b90`  
Overall verdict: **INCONCLUSIVE for research generalization; NOT clinically ready.** The categorical-index defect is fixed and its CPU/CUDA regression tests pass. Internal saved-artifact integrity and metric calculations pass; patient independence, external validity, and an untouched nested estimate remain unresolved.

## Preservation and Execution

- **PASS** — No model training or retraining was run. Existing five-seed PMRC-Net and baseline OOF predictions were sufficient for audit-level reproduction.
- **PASS** — No architecture layers, model dimensions, training hyperparameters, historical checkpoints, dataset rows, OOF predictions, or historical experiment outputs were changed. The only code changes are input validation and its tests; this report is a new root-level artifact.
- **PASS** — Configured runtime selected CUDA (`torch 2.11.0+cu128`, CUDA 12.8). Existing setup test completed model forward/backward and checkpoint round-trip on CUDA.
- **PASS** — Full unit suite after the fix: 21 tests run, 0 failures, 0 errors, 0 skipped. This includes the 18 pre-existing tests and three categorical-input regression tests.
- **PASS** — Focused categorical suite: CPU, CUDA, and source-code normalization rejection tests all passed.
- **PASS** — Static diagnostics found no errors in the modified Python files.
- **WARNING** — The full suite emitted two scikit-learn `StandardScaler` feature-name warnings from a test that passes an ndarray after fitting with a DataFrame. They did not fail execution and do not indicate validation leakage in `make_fold_data`.

## Categorical Input Fix

`PMRCNet.forward` now checks categorical tensor shape, rejects complex values, and verifies that every value is finite, integral, and inside the configured embedding range `[0, 10)` before any embedding lookup. Invalid values raise a `ValueError` naming the categorical feature, row, supplied value, and accepted range. No clamping or embedding-time remapping is performed. The loader still applies its documented canonical mappings for known `thal`, `slope`, and `ca` codes; missing or unknown values in those mappings now raise `ValueError` instead of falling back to 0 or 4.

**PASS** — Valid categories produce finite outputs on CPU and CUDA.  
**PASS** — Negative, `10`, fractional, and NaN category values raise `ValueError` on CPU and CUDA.  
**PASS** — Unknown source `thal` code is rejected by the data loader.  
**PASS** — The pre-fix CUDA failure was reproduced: an invalid index reached `nn.Embedding` and triggered the device-side `srcIndex < srcSelectDimSize` assertion. Post-fix CUDA tests reject it before that operation.  
**LIMIT** — Model-side bounds are the shared embedding range, not a feature-specific clinical codebook for every category. Known source mappings are checked, but arbitrary values within `[0, 10)` are not universally checked against each feature's semantic domain.

## Model and Checkpoint Checks

| Component/check | Status | Result |
|---|---|---|
| Metabolic pathway encoder | **PASS** | Finite projected representation on saved fold-1 checkpoint inference |
| Exercise-response pathway encoder | **PASS** | Finite projected representation |
| Coronary/electrical pathway encoder | **PASS** | Finite projected representation |
| Mismatch physiology module and projection | **PASS** | Finite `m_e` output |
| Three coupling gate layers | **PASS** | Finite gate logits for `W_me`, `W_ec`, and `W_mc` |
| Mechanistic, interaction, residual logits | **PASS** | All finite on saved fold-1 inference |
| Total logit decomposition | **PASS** | `z_total == z_mech + z_inter + z_res` |
| Probability output | **PASS** | Finite and bounded in `[0, 1]` |
| Existing fold-1 checkpoint load/inference | **PASS** | CUDA inference matches saved validation probabilities; maximum absolute difference `1.11e-16` |
| Checkpoint round-trip | **PASS** | Covered by setup and unit tests |

These checks establish computational behavior only, not physiological validity or causal interpretation.

## Recalculated PMRC-Net Metrics

All metrics use threshold 0.5 for thresholded classification metrics. ECE uses ten equal-width bins on OOF probabilities, with the final bin closed at 1.0. Five-seed values are mean and sample standard deviation (`ddof=1`) over seed-level pooled OOF metrics; they are recalculated from saved predictions, not newly trained models.

### Historical Seed 42

| Metric | Recalculated value |
|---|---:|
| Accuracy | 0.825581 |
| Precision | 0.830389 |
| Recall / sensitivity | 0.804795 |
| Specificity | 0.845161 |
| F1 | 0.817391 |
| MCC | 0.650830 |
| ROC-AUC | 0.895526 |
| Average precision | 0.890900 |
| Brier score | 0.130724 |
| ECE-10 | 0.038328 |
| TN / FP / FN / TP | 262 / 48 / 57 / 235 |

**PASS** — The 602-row OOF file has unique sample indices `0..601`, labels matching the loaded dataset, fold IDs matching the seed-42 manifest, finite probabilities in `[0,1]`, and metrics matching the stored aggregate and all five fold metric rows.

### Five Seeds, Five Folds

Seeds: `21, 42, 123, 2026, 3407`. All five OOF files contain 602 unique rows and match deterministic stratified fold assignments, target labels, and valid probability ranges. The 25 saved fold metric rows and seed-level summary values were recalculated and matched.

| Metric | Mean | Sample SD |
|---|---:|---:|
| Accuracy | 0.814950 | 0.008519 |
| Precision | 0.818277 | 0.011030 |
| Recall / sensitivity | 0.795205 | 0.012204 |
| Specificity | 0.833548 | 0.012198 |
| F1 | 0.806521 | 0.008977 |
| MCC | 0.629579 | 0.017056 |
| ROC-AUC | 0.896613 | 0.004981 |
| Average precision | 0.892303 | 0.005773 |
| Brier score | 0.131309 | 0.002471 |
| ECE-10 | 0.042256 | 0.004987 |

**PASS** — Saved-prediction reproduction for all five seeds and all five folds. **NOT RUN** — Fresh 25-fold training replay; intentionally omitted to honor the no-retraining requirement and preserve completed experiments.

## Data, Preprocessing, and Leakage

- **PASS** — The raw CSV has 1,888 rows and no missing values in the eight categorical columns. Exact-row deduplication produces the documented 602 rows.
- **PASS** — Fold validation indices cover every row exactly once and are disjoint from training indices. Target labels and fold assignments match the OOF files.
- **PASS** — `StandardScaler` is fit on each training fold and then applied to validation data. Empirical association directions used by the loss are calculated from each training partition.
- **PASS** — Audit source checks find no use of OOF CSVs in the Stage 2 training path.
- **FAIL for an unbiased generalization estimate** — The same fold validation labels drive early stopping/checkpoint selection and are then used to score that selected checkpoint. The OOF rows are aligned and reproducible, but this is not a fully untouched nested-CV estimate; selection optimism is possible.
- **INCONCLUSIVE** — The 602 unique feature rows are not confirmed independent patients. No patient identifier or patient-level provenance is available, so row-level bootstrap intervals cannot account for patient clustering.
- **BLOCKED** — No independent external validation was performed or verified; candidate source overlap remains unresolved.

## Baselines, Ablations, and Figures

**Baselines: PASS.** Ten additional models have five complete seed OOF sets each (3,010 rows/model). For each model, seed fold assignments, sample IDs, labels, probability bounds, and per-seed metrics including ECE were checked against the saved outputs. All ten seed summaries match recalculation. Models: Logistic Regression, SVM, Random Forest, MLP, KNN, Decision Tree, XGBoost, LightGBM, CatBoost, and TabPFN.

**Seed-42 curve data: PASS.** The verified 11-model figure-data check confirms matching sample IDs, labels, and fold IDs for every model. ECE recalculations match exactly; Brier differs from direct mean squared error by at most `2.78e-17`. Saved ROC, precision-recall, calibration, five-seed, and comparison figures exist under `reports/figures/verified_11_model_comparison/`; their documented generation path consumes the verified OOF data. No figures were regenerated during this audit.

**Paired 11-model statistics: PASS.** The ten baseline comparisons in `reports/VERIFIED_11_MODEL_COMPARISON.md` were recalculated from the verified seed-42 OOF files using the report's fixed 2,000-resample paired bootstrap (`20261107`). All 30 displayed point estimates and interval bounds match to the report's four-decimal precision; no bootstrap replicates were skipped.

**Ablations: PASS.** All four ablation OOF files match the seed-42 rows/folds/labels and their stored aggregate metrics.

| Variant | ROC-AUC |
|---|---:|
| Complete | 0.895526 |
| No physiology | 0.895647 |
| No pathways | 0.532601 |
| No mechanistic output | 0.896067 |
| No residual output | 0.898266 |

Interpretation is limited: the physiology penalty was inactive; the `no_pathways` variant still retains the physiology-derived mechanistic logit while zeroing interaction and residual logits. The small differences for no-mechanistic/no-residual do not establish component irrelevance or clinical value.

**Paired Random Forest statistics: PASS.** The 602-row sample/fold/label merge and all five 2,000-resample paired-bootstrap rows were independently recalculated and match the saved values. PMRC-Net minus Random Forest: ROC-AUC `+0.015759` (95% interval `[0.000909, 0.031264]`), AP `+0.014524` (`[-0.001846, 0.031594]`), Brier `-0.012161` (`[-0.022279, -0.002185]`). These are descriptive row-level intervals, not a superiority claim.

**Historical comparison discrepancy: FAIL as an authoritative source.** Root `results/final_model_comparison.csv` reports PMRC-Net Full accuracy `0.807309` and ROC-AUC `0.875939`, which do not match the verified seed-42 OOF values `0.825581` and `0.895526`. It also includes a Tabular Transformer score although the maintained Stage 2 runner records that implementation as unavailable. Its split/prediction provenance is not sufficient to reconcile it; preserve it as historical, but do not cite it as the verified comparison. The later verified 11-model report and its OOF inputs are the reproducible comparison source.

## Detected Implementation and Research Limitations

- **FAIL** — `l1_lambda` is configured and stored but unused by `PMRCConsistencyLoss`.
- **FAIL** — The positive-direction physiological hinge penalty is inactive in all recorded seed-42 folds: observed `oldpeak`, `exang`, and `ca` associations are negative, while only positive `oldpeak` and `exang` branches activate the penalty. `ca` has no penalty branch.
- **FAIL** — `mixed_precision: true` is present in config, but the maintained training path does not implement autocast/mixed precision.
- **WARNING** — In the asymmetric coupling branch, both concatenated elementwise products are algebraically identical (`a*b == b*a`) and only the first half of the gated tensor is retained. The second gate-output half therefore does not contribute to the prediction. This audit documents the behavior; no architecture change was made.
- **FAIL** — A NaN in a continuous input propagates to non-finite model outputs. The raw dataset has no missing values, but inference does not impute or reject missing continuous values.
- **PASS, narrow stress case** — Replacing the raw physiology batch with values of magnitude `1e6` remained finite in the tested forward pass. This does not constitute a clinical input-range or broad numerical-stability guarantee.
- **INCONCLUSIVE** — Clinical utility, calibration transportability, causality, confirmed patient independence, and generalization to an independent cohort are not established. The model is not clinically ready.

## Final Verdict

**Implementation fix: PASS.** The invalid categorical-index crash is replaced by a clear Python `ValueError` on CPU and CUDA, with no silent embedding clamp/remap. All existing and new tests pass, actual checkpoint inference remains unchanged for valid inputs, and historical experiments were not retrained or overwritten.

**Research audit: PARTIAL / INCONCLUSIVE.** Internal saved predictions, metrics, seed/fold alignment, baselines, ablations, verified curve data, and paired bootstrap calculations are reproducible. The validation design's checkpoint selection, unresolved patient identity, inactive/unused loss components, missing-value inference behavior, legacy comparison discrepancy, and lack of external validation prevent an unqualified completion or clinical-readiness verdict.
