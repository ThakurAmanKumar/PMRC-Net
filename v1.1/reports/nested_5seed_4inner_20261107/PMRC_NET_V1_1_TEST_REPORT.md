# PMRC-Net v1.1 Test Report

## Automated v1.1 suite

- **PASS** — 21 v1.1 tests run; process exit code 0.
- **PASS** — 21 historical v1.0 tests run; process exit code 0. Historical code and outputs remain unchanged.
- **PASS** — Setup/model/checkpoint smoke; process exit code 0.
- **PASS** — CPU forward/backward, finite loss and gradients, additive logit decomposition.
- **PASS** — CUDA forward/backward and inference (when CUDA is available).
- **PASS** — Every categorical feature rejects negative, out-of-vocabulary, fractional, NaN, and infinite values on CPU and CUDA. Source mappings are preserved; unknown source codes raise `ValueError`.
- **PASS** — Continuous and raw-physiology NaN and positive/negative infinity are rejected on CPU and CUDA.
- **PASS** — The reverse asymmetric-gate halves receive nonzero gradients; symmetric coupling keeps a single gate width.
- **PASS** — BCE objective, training-only scaler, outer/inner split isolation, OOF alignment/duplicate checks, deterministic initialization, metric recalculation, and checkpoint round-trip.
- **PASS** — Re-loaded 25 experiment checkpoints on CPU and verified their predictions against saved OOF rows (maximum absolute difference `2.98e-07`).

The v1.1 suite is `v1.1/tests/test_v1_1.py`; run it with `python -m unittest discover -s v1.1/tests -v`. The v1.0 suite remains at `tests/` and is run independently with `python -m unittest discover -s tests -v`. Raw outputs are saved alongside this report. Both ran with the selected venv's Python 3.11 and CUDA-enabled PyTorch 2.11 environment.

## Experimental integrity

- **PASS** — 25 outer-test partitions across five seeds; each sample receives exactly one OOF prediction per seed.
- **PASS** — Scalers fit only on the current training partition; outer-test labels do not enter inner epoch selection.
- **PASS** — Checkpoints, predictions, split manifests, config, and run metadata are isolated by run ID and support hash-checked resume.
- **PASS** — FP32 and sequential execution; no automatic mixed precision.
