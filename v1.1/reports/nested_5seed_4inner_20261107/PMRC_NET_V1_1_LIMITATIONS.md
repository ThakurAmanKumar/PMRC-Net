# PMRC-Net v1.1 Limitations

- No patient IDs or source-level linkage are available. The analysis unit is the deduplicated row, not a verified independent patient.
- No independent external dataset was established or evaluated. External generalization is unknown.
- Nested outer/inner CV avoids using outer labels for epoch selection, but repeated stratified folds on one small dataset are not a substitute for patient-level or external validation.
- v1.0 predictions remain historical and use validation folds for checkpoint selection; paired comparisons are not methodologically symmetric even when outer fold IDs match.
- The physiology representation is retained as a design hypothesis. Its prior target-correlation hinge is removed because it was inactive and its direction was not independently prespecified. No causal or physiological validity is established.
- The corrected ordered coupling has a mathematical directional form and testable information flow; it does not establish biologically meaningful directionality.
- ECE is bin-dependent and finite-sample; calibration transportability is unknown. No post-hoc calibration or threshold selection was fitted.
- Row-level bootstrap intervals cannot account for unknown patient clustering. No clinical superiority claim follows.
- The evaluation data retain potential source/cohort overlap and provenance uncertainty; exact deduplication does not resolve it.
- FP32 was used. Mixed precision was not implemented or evaluated.
- Inputs with missing or non-finite continuous values are rejected. No imputation strategy is provided.
- Baseline predictions are re-used from completed historical artifacts, not retrained under the new nested model-selection workflow. This preserves v1.0 and enables split-aligned descriptive comparison but does not standardize every baseline's selection procedure.
- The experiment is research-only. Diagnostic utility, clinical safety, prospective performance, and deployment readiness are not established.
