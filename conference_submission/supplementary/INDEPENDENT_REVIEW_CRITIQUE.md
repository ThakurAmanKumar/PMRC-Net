# Independent-Style Reviewer Critique

This is a critical pre-submission simulation, not an external peer review.

## Reviewer 1 — Machine-learning methods

**Recommendation: Major revision before submission.**

1. The Stage-5 baseline is not an unbiased nested-CV comparator: the outer validation fold selects the baseline checkpoint/epoch. Its apparent score and paired p-values cannot support an improvement claim.
2. The manuscript must show the exact data lineage and split construction, including that the same 602 rows are reused across five seeds and the dataset has no verified patient identifier.
3. Five seed pairs on one dataset provide weak inferential evidence; report uncertainty descriptively, avoid significance claims, and do not treat folds as independent samples.
4. The claimed physiology-informed loss needs implementation-level scrutiny. The outer-refit association signs make the oldpeak/exang hinge terms inactive; `ca` is computed but not used, and `l1_lambda` is unused.
5. A head-to-head baseline must be rerun under identical outer/inner splits, inner-only selection, calibration, thresholds, and one-time outer test evaluation.

**Response in this package:** The historical baseline is explicitly labeled selection-contaminated and descriptive only; no superiority claim is made. The inactive loss behavior and unused components are disclosed. The nested directional-vs-single-flow comparison remains exploratory.

## Reviewer 2 — Clinical prediction and data governance

**Recommendation: Major revision; clinical claims are not currently supportable.**

1. The source and collection process for the merged CSV, target definition, patient-level unit, and licensing/consent are not established.
2. There is no external or temporal validation, clinical-utility analysis, prospective study, or evidence that a selected threshold reflects a clinical decision.
3. Calibration on repeated internal folds does not establish calibration in a deployment population.
4. The engineered heart-rate/ischemia score is not a validated physiological measure, and the named feature groups should not be called causal or clinically validated pathways.
5. Ethics review/exemption and data rights require author confirmation.

**Response in this package:** The work is framed as an exploratory binary benchmark, not a screening or diagnostic tool. Ethics, data rights, source provenance, target semantics, and patient independence are listed as unresolved author checks.

## Reviewer 3 — Statistics and reporting

**Recommendation: Major revision / reject if claims are not narrowed.**

1. The five seed-level values are correlated because they reuse one small dataset and overlapping splits. A Wilcoxon test over five seed pairs is very low-powered and does not provide population-level uncertainty.
2. Ten metrics and multiple comparisons are reported without multiplicity adjustment. Interpret all p-values as exploratory and avoid dichotomous claims.
3. The ROC and precision-recall figures average across seeds; their bands represent between-seed variability, not confidence intervals.
4. The historical baseline p-values must not be interpreted due to outer-fold checkpoint selection.
5. Reporting should retain fold-level and seed-level outputs, exact hashes, runtime versions, and complete calibration/threshold isolation.

**Response in this package:** The manuscript labels p-values as unadjusted exploratory comparisons, discloses the shared dataset, retains aggregate fold/seed data and run configuration, and flags baseline p-values as nominal only.

## Overall disposition

The completed run is suitable as a transparent, exploratory engineering benchmark. It is not ready to support clinical claims or a controlled claim of improvement over the historical baseline. The most important next experiment is a fully nested, protocol-matched baseline followed by external evaluation with verified patient-level independence.
