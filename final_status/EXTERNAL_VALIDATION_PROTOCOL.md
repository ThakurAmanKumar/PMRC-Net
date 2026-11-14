# External Validation Protocol

External validation remains **BLOCKED** until an independent dataset with documented source provenance and compatible feature definitions is obtained.

1. Obtain institutional or repository permission and the official raw dataset.
2. Record source, population, collection period, target definition, missingness, identifiers, and license.
3. Demonstrate that the source cohort is not one of the local merged development cohorts, or obtain source-level linkage sufficient to exclude overlap.
4. Freeze `checkpoints/controlled_pmrc_seed42/best.pt` and the configuration/hash in `results/stage5_final_validation/reproducibility_manifest.json`.
5. Apply only training-fitted preprocessing and documented feature order. Do not fit a scaler, calibrator, threshold, or model on external observations.
6. Exclude incompatible observations with recorded reasons; never invent feature mappings.
7. Save external predictions, metadata, hashes, sample counts, metrics, calibration, and confidence intervals under a new `results/external_validation_<dataset>/` directory.
8. Report external results separately from internal OOF results and retain BLOCKED if independence remains uncertain.

---

## Authoritative current status

- Overall project status: **INCONCLUSIVE**.
- Patient-level independence: **INCONCLUSIVE**.
- Independent external validation: **BLOCKED**.
- Clinical readiness: **NOT ESTABLISHED**.
