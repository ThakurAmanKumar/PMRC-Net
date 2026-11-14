# PMRC-Net Google Colab Execution Guide

## Running the notebook

1. Upload Physiological_Mismatch_and_Residual_Consistency_Network.ipynb to Google Colab.
2. Select a GPU runtime if available (Runtime > Change runtime type > T4 GPU). The notebook detects CUDA automatically and also supports CPU.
3. Use Run All. Upload exactly one CSV at the prompted file-upload cell, preferably heart_disease_merged.csv.
4. Leave EXPERIMENT_MODE = "quick" for the default stratified train/validation/test experiment. To run research mode, set it to "nested" before Run All.
5. After the run, download the generated ZIP. To download the selected trained checkpoint separately, execute download_checkpoint().

The notebook needs no local modules, local configuration, Windows path, Drive mount, or virtual environment. Colab's existing packages are used without automatic upgrades. Training is FP32.

## Dataset and preprocessing

The CSV must include age, sex, cp, trestbps, chol, fbs, restecg, thalachh, exang, oldpeak, slope, ca, thal, and target. The target must be binary 0/1. The notebook reports the raw shape, columns, first five rows, missing-value counts, and exact duplicate count. It rejects missing/non-finite continuous values and unsupported category codes instead of imputing or silently remapping them.

Documented v1.1 source harmonization is applied: thal {0:0, 1:6, 2:3, 3:7, 6:6, 7:7}, slope {0:0, 1:1, 2:2, 3:3}, ca {0:0, 1:1, 2:2, 3:3, 4:4}. Other categorical vocabularies are checked against the v1.1 configuration. Exact raw rows are deduplicated before mapping. The established reference data have 1,888 raw rows and 602 exact-row-deduplicated observations; for other uploads, the notebook reports the actual count. The internal row_index is not a patient identifier.

StandardScaler is fit only on each training partition. Physiology-module inputs remain on their raw scale, as in the v1.1 implementation.

## Version and model

The notebook embeds the authoritative PMRC-Net v1.1 architecture from v1.1/models/pmrc_net.py. It includes the metabolic, exercise-response, and coronary/electrical pathways; the clinical physiology mismatch module; ordered asymmetric coupling; mechanistic, interaction, and residual logits; additive final logit; and sigmoid probability. The v1.1 objective is BCE-with-logits with AdamW weight decay. The v1.0 empirical-direction hinge is intentionally not combined with v1.1. Existing project artifacts are not read, modified, or overwritten.

## Evaluation modes

### Quick (default)

One stratified split reserves 20% for the final test. The remaining 80% is split into training and validation (64%/16% of total). Early stopping and checkpoint selection use validation BCE. The scaler is fit only on the training partition. The final test partition is evaluated after training and is not used for selection. Metrics and plots are based on that held-out test prediction set; this mode does not produce OOF predictions.

### Nested research

Five configured seeds (21, 42, 123, 2026, 3407), five stratified outer folds per seed, and four stratified inner folds are used. Inner validation BCE selects the ceiling of the median best epoch across the four inner folds. A fresh model is trained on all outer-training rows for exactly that duration. The outer test partition is excluded from scaling, early stopping, checkpoint selection, and tuning. The OOF table is checked for one aligned prediction per row per seed before metrics are calculated. Per-fold metrics and per-seed pooled OOF metrics are saved; mean and sample SD (ddof=1) summarize the five seed values. The displayed nested curves use seed-42 OOF predictions.

The nested run may exceed a single Colab session's runtime. Batch size, max_epochs, patience, and min_delta are configurable. If changing these from v1.1 defaults, record that change and avoid presenting the run as directly comparable to the documented reference experiment. The requested five-seed/five-outer/four-inner counts are validated and are not silently reduced.

## Metrics, calibration, and figures

Accuracy, precision, sensitivity, specificity, F1, MCC, ROC-AUC, average precision, Brier score, ECE-10, and confusion matrix are computed from actual held-out predictions. Average precision is not trapezoidal PR-AUC. ROC, PR, calibration, and confusion-matrix figures use the same true labels and prediction probabilities. Calibration uses ten equal-width bins and is descriptive; no untrained model, evaluation-fitted scaler, post-hoc calibration, or threshold optimization is used.

The synthetic checks are software checks only and are not clinical results. Nested seed variability is not a patient-level confidence interval.

## Output files

Each run gets a unique ID below PMRC_Net_Colab_Results/. The ZIP includes configuration.json, training_history.json, metrics.csv, predictions (test_predictions.csv or oof_predictions.csv), per-fold/per-seed metrics in nested mode, evaluation_report.md, calibration_results.json, calibration_bins.csv, calibration_by_seed.csv, confusion_matrix.csv, figures, and checkpoint file(s). No historical v1.0/v1.1 output is overwritten.

## Limitations

The reference dataset has 602 exact-row-deduplicated observations, not 602 confirmed independent patients. Patient independence and source provenance are unverified. There is no independent external validation. The physiological representation and directional coupling are model hypotheses, not causal findings. Clinical utility, safety, readiness, and superiority are not established.
