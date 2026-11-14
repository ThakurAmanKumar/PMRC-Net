# QA Gate Report

Date: 2026-11-14

## Test command

.\.venv\Scripts\python.exe -m unittest tests.test_smoke tests.test_metrics tests.test_plots

## Result

Ran 26 tests in 3.449s

OK

## Summary

- Passed: 26
- Failed: 0
- Skipped: 0

## Included suites

- PMRC architecture smoke tests
- Metrics regression tests
- Evaluation helper tests
- Plotting tests
- Preprocessing and dataset checks
- Checkpoint save/reload checks

## Regression verification

The metrics regression fix was verified with the dedicated targeted suite:

.\.venv\Scripts\python.exe -m unittest tests.test_metrics -v

This suite passed 7/7 tests, including:

- supplied predicted-label arrays are treated as the source of truth
- missing thresholds remain None when not recomputed
- no float(None) conversion occurs
- confusion matrices use supplied predictions
- probability metrics use supplied probabilities
- threshold selection remains restricted to validation/inner-fold data
- undefined metric handling remains explicit and non-zero-coercive

## Evaluation-output verification

Verified in the passing suite:

- raw confusion matrices
- normalized confusion matrices
- sensitivity and specificity
- ROC-AUC and Average Precision
- Brier Score and ECE
- undefined metric handling
- ROC, PR, and calibration plot generation
- output file generation from plotting helpers

## Plot generation status

Plot tests passed and generated output files under the temporary test environment for:

- ROC curves
- precision-recall curves
- calibration curves
- confusion matrices

## Remaining warnings

The suite emitted the existing scikit-learn StandardScaler feature-name warning during smoke tests:

- X does not have valid feature names, but StandardScaler was fitted with feature names

This warning is present but non-fatal and does not block the QA gate.

## Gate decision

The required QA gate passed. No required test failed.

This QA gate does not include a full nested cross-validation run, and no full nested CV was launched in this iteration.

The project is ready for the next full nested-CV experiment only after that next experiment is started deliberately and its own preflight and results checks are passed.
