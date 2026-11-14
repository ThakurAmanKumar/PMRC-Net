# PMRC-Net Research Baseline v1.1

The repository root is the preserved v1.0 baseline. v1.1 is separate and writes only below this directory. Do not move or regenerate the root v1.0 predictions, checkpoints, configurations, or reports.

## Run

From the project root, run the fixed nested-CV configuration sequentially:

```powershell
.\.venv\Scripts\python.exe .\v1.1\run_nested_cv.py --run-id nested_YYYYMMDD
```

The run uses five fixed seeds, five outer folds, and four inner folds. Inner-fold validation BCE selects a median epoch count; each outer model is then refit only on its outer-training partition. OOF predictions, split manifests, fold checkpoints, and reports are stored in separate v1.1 directories.

To continue an interrupted run with the same configuration and dataset hashes:

```powershell
.\.venv\Scripts\python.exe .\v1.1\run_nested_cv.py --run-id nested_YYYYMMDD --resume
```

An existing run ID is never reused without `--resume`; resume rejects changed config or dataset hashes and skips completed fold artifacts.

## Test and report

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s v1.1/tests -v
.\.venv\Scripts\python.exe .\v1.1\build_reports.py --run-id nested_YYYYMMDD
```

The report builder verifies saved OOF alignment before recalculating metrics or plotting. Reports are written once under `v1.1/reports/<run-id>/` and figures under `v1.1/figures/<run-id>/`; existing report directories are not overwritten. It reuses existing baseline predictions and does not train or modify v1.0 experiments.

See `reports/PMRC_NET_V1_1_METHODOLOGY.md` and `reports/PMRC_NET_V1_1_LIMITATIONS.md` for method and scope.