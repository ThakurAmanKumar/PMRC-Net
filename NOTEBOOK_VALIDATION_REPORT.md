# Notebook Validation Report

## Rebuild and source review

- Rebuilt the root notebook as a self-contained Google Colab notebook. The prior notebook's 56 cells (32 code, 24 markdown) were inspected; stale outputs and execution counts were removed.
- Architecture source: v1.1/models/pmrc_net.py.
- Data source mapping, category validation, and fold scaling: v1.1/data/dataset.py and v1.1/configs/nested_cv.yaml.
- Nested evaluation protocol: v1.1/run_nested_cv.py and v1.1/reports/nested_5seed_4inner_20261107/PMRC_NET_V1_1_METHODOLOGY.md.
- Scientific limitations: root ARCHITECTURE.md, METHODOLOGY.md, LIMITATIONS.md, PMRC_NET_COMPLETE_AUDIT.md, and v1.1 limitations/report.
- Version is consistently PMRC-Net v1.1; the v1.0 objective/evaluation workflow is not mixed into it.

## Corrections made

- Removed duplicate notebook benchmarks, obsolete baseline/ablation code, forced target-derived physiological directions, local/Drive paths, and synthetic report claims.
- Calibration now consumes only quick held-out test predictions or verified outer-fold OOF probabilities from trained v1.1 models.
- All scalers are fit on training partitions only. Invalid category codes and missing/non-finite inputs are rejected.
- Quick mode is the default. Nested mode is opt-in and protects each outer test fold from inner selection.
- Every run uses a unique output directory and does not touch historical experiment artifacts.

## Validation results

- Notebook JSON parsed successfully: 33 cells total (18 markdown, 15 code); every code cell passed Python compilation. All 18 requested sections are present, all saved outputs/execution counts are cleared, and the embedded model matches v1.1/models/pmrc_net.py.
- The project CSV loaded and validated: 1,888 rows, 1,286 exact duplicates, 602 exact-row-deduplicated rows, no missing cells, and target counts 310/292. Invalid source categories, fractional categories, non-finite continuous values, and invalid targets were rejected.
- A training-only scaler check confirmed that fitted means equal the training partition means and differ from full-dataset means.
- Synthetic software checks passed on CPU and CUDA: model forward/backward, additive-logit identity, sigmoid bounds, finite gradients, and checkpoint round-trip.
- A one-epoch synthetic quick-mode run passed its stratified 64/16/20 split, held-out prediction, metrics, plots, checkpoint, and ZIP export checks.
- A one-epoch synthetic nested-mode run completed all 5 seeds × 5 outer folds × 4 inner folds. The 500 generated OOF rows were checked for row alignment and uniqueness; 25 checkpoints, fold/seed metrics, calibration, figures, and ZIP export were verified.
- Synthetic metric checks returned ROC-AUC 1.0 and Brier score 0.085 for the known four-row test vector. These and all synthetic-mode results are software checks only, not dataset experiment results.

## Not performed

- The notebook was not executed in a live Google Colab runtime; the Google Colab upload/download UI therefore was not exercised. The notebook uses the standard Colab file-upload/download APIs and has no local-project imports or paths.
- No new training or clinical performance evaluation was run on the uploaded/project data. No experiment metrics are claimed here.
- The local project virtual environment does not include IPython; the isolated local smoke harness supplied a display shim for that notebook-only import. IPython is included in COLAB_REQUIREMENTS.txt and is part of the Colab notebook environment.
