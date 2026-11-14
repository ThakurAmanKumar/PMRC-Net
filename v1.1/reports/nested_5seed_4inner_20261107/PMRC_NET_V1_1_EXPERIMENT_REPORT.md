# PMRC-Net v1.1 Experiment Report

Run: `nested_5seed_4inner_20261107`  
Run status: **COMPLETED**  
Device: `cuda`; GPU: `NVIDIA GeForce RTX 2050`  
Seeds: [21, 42, 123, 2026, 3407]; outer folds: 5; inner folds: 4; completed outer checkpoints: 25  
Dataset SHA-256: `a786b4a01e306cd892293c2671e15f29a29a17d06c69b16b19906a337ad42b90`

## Mean and sample SD across seeds

Each value is `mean +/- sample SD (ddof=1)` across five seed-level pooled OOF metric values.

| Model | accuracy | precision | sensitivity | specificity | f1 | mcc | roc_auc | average_precision | brier_score | ece |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PMRC-Net v1.0 | 0.8150 +/- 0.0085 | 0.8183 +/- 0.0110 | 0.7952 +/- 0.0122 | 0.8335 +/- 0.0122 | 0.8065 +/- 0.0090 | 0.6296 +/- 0.0171 | 0.8966 +/- 0.0050 | 0.8923 +/- 0.0058 | 0.1313 +/- 0.0025 | 0.0423 +/- 0.0050 |
| PMRC-Net v1.1 | 0.8027 +/- 0.0109 | 0.8113 +/- 0.0167 | 0.7733 +/- 0.0115 | 0.8303 +/- 0.0188 | 0.7918 +/- 0.0104 | 0.6052 +/- 0.0220 | 0.8841 +/- 0.0046 | 0.8748 +/- 0.0107 | 0.1415 +/- 0.0040 | 0.0694 +/- 0.0075 |
| Logistic Regression | 0.7568 +/- 0.0030 | 0.7504 +/- 0.0034 | 0.7473 +/- 0.0092 | 0.7658 +/- 0.0059 | 0.7488 +/- 0.0043 | 0.5132 +/- 0.0062 | 0.8185 +/- 0.0044 | 0.7628 +/- 0.0074 | 0.1732 +/- 0.0023 | 0.0474 +/- 0.0080 |
| SVM | 0.7748 +/- 0.0073 | 0.7777 +/- 0.0067 | 0.7500 +/- 0.0135 | 0.7981 +/- 0.0071 | 0.7636 +/- 0.0088 | 0.5490 +/- 0.0146 | 0.8489 +/- 0.0050 | 0.8421 +/- 0.0047 | 0.1572 +/- 0.0029 | 0.0426 +/- 0.0061 |
| Random Forest | 0.8023 +/- 0.0081 | 0.8128 +/- 0.0114 | 0.7699 +/- 0.0078 | 0.8329 +/- 0.0119 | 0.7907 +/- 0.0081 | 0.6045 +/- 0.0165 | 0.8797 +/- 0.0022 | 0.8728 +/- 0.0028 | 0.1429 +/- 0.0011 | 0.0582 +/- 0.0052 |
| MLP | 0.7003 +/- 0.0435 | 0.6980 +/- 0.0644 | 0.7007 +/- 0.0887 | 0.7000 +/- 0.1385 | 0.6935 +/- 0.0352 | 0.4070 +/- 0.0754 | 0.7605 +/- 0.0491 | 0.7277 +/- 0.0604 | 0.2042 +/- 0.0157 | 0.0808 +/- 0.0182 |
| KNN | 0.7681 +/- 0.0075 | 0.7773 +/- 0.0075 | 0.7315 +/- 0.0110 | 0.8026 +/- 0.0070 | 0.7537 +/- 0.0086 | 0.5359 +/- 0.0150 | 0.8606 +/- 0.0055 | 0.8477 +/- 0.0057 | 0.1514 +/- 0.0030 | 0.0421 +/- 0.0117 |
| Decision Tree | 0.7542 +/- 0.0145 | 0.7398 +/- 0.0261 | 0.7630 +/- 0.0180 | 0.7458 +/- 0.0372 | 0.7508 +/- 0.0104 | 0.5092 +/- 0.0282 | 0.8141 +/- 0.0171 | 0.7879 +/- 0.0174 | 0.1850 +/- 0.0110 | 0.1213 +/- 0.0156 |
| XGBoost | 0.7674 +/- 0.0097 | 0.7663 +/- 0.0130 | 0.7493 +/- 0.0148 | 0.7845 +/- 0.0160 | 0.7576 +/- 0.0102 | 0.5344 +/- 0.0194 | 0.8373 +/- 0.0007 | 0.8239 +/- 0.0067 | 0.1701 +/- 0.0021 | 0.0869 +/- 0.0058 |
| LightGBM | 0.7774 +/- 0.0105 | 0.7757 +/- 0.0156 | 0.7616 +/- 0.0082 | 0.7923 +/- 0.0185 | 0.7685 +/- 0.0093 | 0.5544 +/- 0.0210 | 0.8497 +/- 0.0051 | 0.8415 +/- 0.0081 | 0.1615 +/- 0.0038 | 0.0747 +/- 0.0064 |
| CatBoost | 0.8000 +/- 0.0067 | 0.8023 +/- 0.0118 | 0.7801 +/- 0.0098 | 0.8187 +/- 0.0143 | 0.7910 +/- 0.0063 | 0.5996 +/- 0.0135 | 0.8768 +/- 0.0053 | 0.8770 +/- 0.0034 | 0.1515 +/- 0.0052 | 0.0934 +/- 0.0051 |
| TabPFN | 0.8306 +/- 0.0071 | 0.8388 +/- 0.0081 | 0.8055 +/- 0.0112 | 0.8542 +/- 0.0083 | 0.8218 +/- 0.0079 | 0.6610 +/- 0.0143 | 0.9008 +/- 0.0036 | 0.8939 +/- 0.0044 | 0.1245 +/- 0.0027 | 0.0324 +/- 0.0084 |

## Paired comparisons

The v1.1 versus v1.0 estimates and paired intervals below use same sample IDs and seed-specific outer folds. See `paired_comparisons.csv` for v1.1 versus all baselines. Intervals are descriptive and not superiority tests.

| Metric | v1.0 mean | v1.1 mean | v1.1 - v1.0 | 95% paired interval |
|---|---:|---:|---:|---:|
| roc_auc | 0.8966 | 0.8841 | -0.0126 | [-0.0191, -0.0058] |
| average_precision | 0.8923 | 0.8748 | -0.0175 | [-0.0309, -0.0047] |
| brier_score | 0.1313 | 0.1415 | +0.0102 | [+0.0052, +0.0154] |

## Seed-42 confusion matrix

Threshold `0.5`; rows are observed labels and columns are predicted labels:

| Observed \ Predicted | 0 | 1 |
|---|---:|---:|
| 0 | 263 | 47 |
| 1 | 69 | 223 |

## OOF audit

**PASS** — Each seed has one outer-test prediction per sample, with saved sample index, fold, target, and probability. All predictions were validated for fold identity, target alignment, uniqueness, finiteness, and `[0, 1]` bounds before metrics were computed. All 25 checkpoints were reloaded on CPU and reproduced their OOF probabilities (maximum absolute difference `2.98e-07`). Predictions and checkpoints are stored under `v1.1/predictions/nested_5seed_4inner_20261107/` and `v1.1/checkpoints/nested_5seed_4inner_20261107/`.

The ROC, precision-recall, calibration, and confusion-matrix figures in `v1.1/figures/nested_5seed_4inner_20261107/` use actual saved seed-42 outer OOF probabilities. No curve was inferred from aggregate metrics.

## Interpretation

This is a nested internal evaluation on the available row-level dataset. It does not establish independent patients, external validity, clinical utility, or clinical readiness. The v1.0 outputs retain their historical checkpoint-selection limitation; v1.1's inner-fold duration selection avoids that specific outer-test reuse.
