# Final Statistical Validation

The PMRC-Net Stage 2 OOF and Stage 4.2 Random Forest OOF were merged on `sample_index`, `fold`, and `target`. The merge has `602` rows, unique sample indices on both sides, identical folds and labels: **PASS**.

Corrected values from the actual metric CSV: PMRC-Net ROC-AUC `0.89552585`; Random Forest ROC-AUC `0.87976690`; difference `+0.01575895`; AP difference `+0.01452427`; Brier difference `-0.01216060`; saved ROC-AUC paired bootstrap interval `[0.000909, 0.031264]`.

Five-seed results were read from `results/stage5_final_validation/multi_seed_summary.csv`: ROC-AUC `0.8966 +/- 0.0050`, accuracy `0.8150 +/- 0.0085`, F1 `0.8065 +/- 0.0090`, MCC `0.6296 +/- 0.0171`, Brier `0.1313 +/- 0.0025`.

Bootstrap results are row-level and cannot account for unknown patient clustering. No statistical superiority or clinical superiority claim is made.

---

## Authoritative current status

- Overall project status: **INCONCLUSIVE**.
- Patient-level independence: **INCONCLUSIVE**.
- Independent external validation: **BLOCKED**.
- Clinical readiness: **NOT ESTABLISHED**.
