# Final Validation Report

Overall project status: **INCONCLUSIVE**.
Patient-level independence: **INCONCLUSIVE**. Independent external validation: **BLOCKED**. Clinical readiness: **NOT ESTABLISHED**.

## Verification checks

| Check | Evidence | Status |
|---|---|---|
| `dataset_hash` | `dataset/cleaned_merged_heart_dataset.csv` | **PASS** |
| `dataset_counts_match_documented_expectations` | `dataset/cleaned_merged_heart_dataset.csv; PATIENT_DUPLICATE_ANALYSIS.md` | **PASS** |
| `near_duplicate_calculation` | `scripts/build_final_status.py; PATIENT_DUPLICATE_ANALYSIS.md` | **PASS** |
| `stage4_2_alignment` | `results/stage2_validation/oof_predictions.csv; results/stage4_2_baseline_comparison/random_forest_oof_aligned.csv` | **PASS** |
| `corrected_metrics` | `final_metric_recalculation.csv; paired OOF predictions` | **PASS** |
| `checkpoint_hash` | `checkpoints/controlled_pmrc_seed42/best.pt; reproducibility_manifest.json` | **PASS** |
| `stage5_dataset_hash` | `results/stage5_final_validation/reproducibility_manifest.json` | **PASS** |
| `five_seed_summary` | `results/stage5_final_validation/multi_seed_summary.csv; multi_seed_results.csv` | **PASS** |
| `automated_test_log` | `test_suite_stdout.log; test_suite_stderr.log` | **PASS** |
| `automated_test_exit_code` | `test_execution_metadata.json` | **PASS** |
| `automated_test_duration_recorded` | `captured unittest summary; test_execution_metadata.json` | **PASS** |
| `legacy_setup_log` | `test_setup_direct.log; test_execution_metadata.json` | **PASS** |
| `external_not_evaluated` | `absence of external_validation_predictions.csv` | **BLOCKED** |
| `external_dataset_screening_evidence` | `external_dataset_provenance.csv; external_dataset_feature_mapping.csv` | **PASS** |
| `required_output_files` | `final_status required output list` | **PASS** |
| `json_outputs_valid` | `all required JSON outputs` | **PASS** |
| `csv_outputs_valid` | `all required CSV outputs` | **PASS** |
| `referenced_evidence_paths` | `FINAL_RESEARCH_STATUS.json evidence_paths` | **PASS** |
| `inventory_integrity` | `final_experiment_inventory.csv` (self-exclusion, filters, and per-file hashes) | **PASS** |

## Captured test execution

- Suite: **PASS**; command: `C:\Users\AMAN KUMAR THAKUR\Downloads\PMRC-Net-Research\.venv\Scripts\python.exe -m unittest discover -s tests -v`.
- Actual process exit code: `0`; unittest-reported tests: `13`.
- unittest-reported duration: `4.105` seconds; measured process elapsed: `17.038375` seconds.
- Legacy setup command: `C:\Users\AMAN KUMAR THAKUR\Downloads\PMRC-Net-Research\.venv\Scripts\python.exe test_setup.py`; actual exit code: `0` (PASS).
- Raw captured output is stored in the corresponding stdout/stderr logs.

## Dataset duplicate measurements

| Measurement | Actual | Previous expected |
|---|---:|---:|
| `raw_rows` | 1888 | 1888 |
| `exact_duplicate_rows` | 1286 | 1286 |
| `exact_unique_rows` | 602 | 602 |
| `feature_duplicate_rows` | 1588 | 1588 |
| `feature_duplicate_groups` | 302 | 302 |
| `conflicting_target_groups` | 0 | 0 |
| `near_duplicate_candidate_pairs` | 3453 | not assumed; calculated from rows |

Method: Column-wise z-score: subtract the full dataset column mean and divide by population standard deviation (ddof=0); zero-variance columns map to zero. Rows are compared only when all eight categorical columns match; Euclidean distance threshold is 0.1. All previously documented exact/feature duplicate counts match the dataset. The earlier near-duplicate count was hard-coded rather than measured; this run replaces it with the computed pair count.

## Remaining limitations

- Patient identity cannot be inferred from the available identifiers or source linkage; patient independence remains **INCONCLUSIVE**.
- No independent external dataset has been established; independent external validation remains **BLOCKED**.
- Clinical readiness is **NOT ESTABLISHED**.
- Overall project status remains **INCONCLUSIVE**, regardless of internal evidence-check results.

The inventory is generated after all listed outputs and references are verified; its self-entry is excluded, as are `.venv` and `__pycache__` paths.

---

## Authoritative current status

- Overall project status: **INCONCLUSIVE**.
- Patient-level independence: **INCONCLUSIVE**.
- Independent external validation: **BLOCKED**.
- Clinical readiness: **NOT ESTABLISHED**.
