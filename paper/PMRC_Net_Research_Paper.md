# PMRC-Net: A Physiologically Structured Neural Network with Mismatch and Residual Consistency for Heart Disease Risk Assessment

## Abstract

PMRC-Net is a tabular neural network organized into metabolic, exercise-response, and coronary/electrical pathways, with a physiology-inspired mismatch representation, gated interactions, and residual output. On 602 deduplicated observations from a merged heart-disease dataset, the historical seed-42 five-fold pooled OOF ROC-AUC was 0.8955. Across five seeds, mean ROC-AUC was 0.8966 (SD 0.0050). These are internal row-level results, not clinical validation. Patient-level independence and independent external validation remain unverified.

## Keywords

Heart disease risk assessment; structured neural networks; tabular learning; out-of-fold validation; reproducibility; physiological mismatch.

## Introduction and research gap

Clinical tabular datasets contain heterogeneous measurements and categorical encodings. PMRC-Net explores whether an explicit pathway organization and additive logit decomposition provide a useful computational inductive bias. The study does not claim that names such as metabolic or coronary establish biological validity. The research gap is treated as an engineering hypothesis: whether structured computation can be evaluated transparently against standard baselines under leakage-controlled validation.

## Methodology

The maintained implementation uses five continuous and eight categorical inputs. Continuous values are standardized separately inside each training fold. Categorical values are passed through learned embeddings. Pathway representations are projected to a shared hidden dimension. A raw-scale physiology module computes an expected heart rate, heart-rate deficit, a softplus ischemia score, and their product as mismatch. Asymmetric coupling gates combine pathway states pairwise. The mechanistic, interaction, and residual logits are added to form the final logit, and the sigmoid gives the probability.

The loss is binary cross-entropy plus a conditional physiological hinge and coupling L2 regularization. The configured L1 parameter is unused in the actual implementation. The observed physiological hinge was zero because empirical directions were negative while the original hinge only activates positive directions.

## Dataset and provenance

The local notebook records 1,888 merged rows and 602 exact unique rows. There is no verified patient identifier. The source clues reference Cleveland, Kaggle, and merged data, but source cohort membership and patient-level independence cannot be established. Rows are therefore described as deduplicated observations.

## Experimental protocol

The primary internal evaluation is stratified five-fold OOF validation with fold-local preprocessing. Stage 5 repeats this protocol for seeds 21, 42, 123, 2026, and 3407. The historical seed-42 result is preserved separately. Comparisons use saved aligned OOF artifacts. Threshold 0.5 is reported descriptively.

## Results

The historical PMRC-Net pooled seed-42 results were ROC-AUC 0.8955, accuracy 0.8256, F1 0.8174, MCC 0.6508, and Brier 0.1307. The five-seed mean values were ROC-AUC 0.8966, accuracy 0.8150, F1 0.8065, MCC 0.6296, and Brier 0.1313. The keyed Stage 4.2 comparison gives Random Forest ROC-AUC 0.8798, a PMRC-Net difference of +0.0158, and a descriptive paired bootstrap interval of [0.0009, 0.0313]. AP difference is +0.0145 and Brier difference is -0.0122. No superiority claim is made.

Detailed recalculation is in `reports/final_metric_recalculation.csv` and five-seed values are in `paper/tables/five_seed_summary.csv`.

## Ablation and robustness

Removing pathways produced a major ROC-AUC reduction to 0.5326. Removing physiology produced 0.8956, while removing mechanistic and residual outputs produced 0.8961 and 0.8983. These results indicate computational sensitivity patterns under this dataset, not causal necessity or physiological truth. Stage 4 perturbation and robustness analyses are descriptive model-sensitivity tests.

## Calibration and uncertainty

The saved pooled Brier score was 0.1307 and equal-width ten-bin ECE was 0.0383. Row-level bootstrap intervals do not model unknown patient clustering. Calibration was not externally validated and no external calibrator was fit.

## External validation

External validation is pending. UCI Heart Disease and Statlog Heart were screened from official UCI documentation, but their possible overlap with the local merged source cannot be ruled out. No external observations were used, and no external metrics are reported.

## Discussion

The evidence supports an implementation with reproducible internal OOF behavior and finite gradient flow through inspected components. It does not establish clinical usefulness, diagnostic readiness, causality, or clinical superiority. The modest seed variation supports numerical stability at the row level, while the lower mean threshold metrics relative to the historical seed-42 run illustrate the importance of reporting seed variability.

## Limitations and ethics

The lack of patient identifiers prevents patient-level independence verification. Source-derived duplicates may represent repeated measurements or repeated patients. The data are small, merged, and not a prospective clinical cohort. Predictions must not be used for clinical decisions. Any future external evaluation requires independent patient-level provenance, prespecified feature mapping, frozen preprocessing/model parameters, and supervisor review.

## Conclusion and future work

PMRC-Net is implementation-complete and internally validated under the saved protocol. It is not externally validated. Future work should obtain a genuinely independent, patient-level dataset; preregister feature and target mapping; evaluate clustering-aware uncertainty; and seek domain review of the physiological assumptions before making clinical claims.

## Reproducibility

Run `python test_setup.py`, `python -m unittest discover -s tests -v`, and `python scripts/finalize_project.py` from the repository root. Do not start training automatically. The original notebook and historical Stage 1–4 artifacts are preserved.

## References

See `paper/references.bib` for the verified UCI dataset records. The manuscript intentionally does not add unsupported literature claims.
