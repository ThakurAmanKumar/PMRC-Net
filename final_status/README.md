# PMRC-Net Final Status

This folder is the centralized final verification and supervisor-readiness package for PMRC-Net. It summarizes verified evidence without replacing historical experiments.

## Status meanings

- **PASS:** verified successfully with saved evidence.
- **FAIL:** a verified requirement was tested and failed.
- **INCONCLUSIVE:** available evidence is insufficient.
- **BLOCKED:** a required external resource or provenance decision is unavailable.

## Authoritative evidence

- Historical PMRC-Net OOF: `results/stage2_validation/oof_predictions.csv`
- Stage 4.2 aligned Random Forest OOF: `results/stage4_2_baseline_comparison/random_forest_oof_aligned.csv`
- Five-seed validation: `results/stage5_final_validation/`
- Maintained implementation: `src/`
- Model configuration: `configs/model_config.yaml`
- Checkpoints: `checkpoints/`
- Paper: `paper/PMRC_Net_Research_Paper.md`

Machine-readable outputs in this folder are `FINAL_RESEARCH_STATUS.json`, `final_validation_summary.json`, `final_metric_recalculation.csv`, `final_alignment_verification.json`, `final_experiment_inventory.csv`, the patient-independence JSON, test execution metadata, and the external-dataset CSV files. `FINAL_VALIDATION_REPORT.md` records the check-by-check evidence and limitations.

## Reproduce verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe test_setup.py
.\.venv\Scripts\python.exe scripts/build_final_status.py
```

These commands do not retrain the model. Independent external validation remains blocked and patient-level independence remains inconclusive.

---

## Authoritative current status

- Overall project status: **INCONCLUSIVE**.
- Patient-level independence: **INCONCLUSIVE**.
- Independent external validation: **BLOCKED**.
- Clinical readiness: **NOT ESTABLISHED**.
