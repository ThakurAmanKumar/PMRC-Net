# Final Baseline Comparison

The authoritative comparison uses `results/stage2_validation/oof_predictions.csv` and `results/stage4_2_baseline_comparison/random_forest_oof_aligned.csv`, merged on `sample_index`, `fold`, and `target`.

- PMRC-Net ROC-AUC: `0.89552585`
- Random Forest ROC-AUC: `0.87976690`
- Difference: `+0.01575895`
- AP difference: `+0.01452427`
- Brier difference: `-0.01216060`
- ROC-AUC bootstrap interval: `[0.000909, 0.031264]`

The comparison is descriptive and row-level; no superiority claim is made.

---

## Authoritative current status

- Overall project status: **INCONCLUSIVE**.
- Patient-level independence: **INCONCLUSIVE**.
- Independent external validation: **BLOCKED**.
- Clinical readiness: **NOT ESTABLISHED**.
