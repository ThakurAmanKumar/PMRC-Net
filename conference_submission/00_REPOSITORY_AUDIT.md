# Repository and Experiment Audit

## Scope and provenance

- Dataset: `dataset/cleaned_merged_heart_dataset.csv` (the requested `(5).csv` variant was absent during the approved run).
- Dataset SHA-256: `a786b4a01e306cd892293c2671e15f29a29a17d06c69b16b19906a337ad42b90`.
- Raw rows/columns: 1888 / 14; exact duplicates: 1286; post-deduplication rows: 602.
- The CSV has no patient identifier or verified source provenance. A deduplicated row is not assumed to be a unique patient.
- The stored dataset and original baseline outputs were not edited by paper-package generation.

## Experiment status

- Status: `COMPLETE`; five seeds × five outer folds × two nested arms; all 50 arm/fold checkpoints completed.
- Each arm has 3,010 OOF prediction records (602 rows repeated once per seed). Fold metrics and seed metrics are preserved in `supplementary/`.
- Recorded duration: 8831.9 seconds; peak process RSS 1696.4 MiB; maximum system RAM 91.3%; maximum GPU temperature 60.0 °C.
- The run recorded no CUDA OOM or failed/skipped outer fold. The run's post-processing failure was fixed and resumed from checkpoints before the COMPLETE status was written.

## Dataset and preprocessing findings

- Exact duplicate rows are removed before CV; no missing cells were present in the raw CSV.
- Five continuous columns are standardized using the relevant training partition; eight categorical columns use fixed source-code maps.
- The loader maps `ca` values 0–4 identically. It retains `ca=4` (28 raw records; five after exact deduplication); the value is not converted to missing and is not imputed. This corrects an earlier prompt/documentation ambiguity.
- `thal` maps `{0:0, 1:6, 2:3, 3:7, 6:6, 7:7}`; `slope` codes are retained.
- Raw target counts: 0 = 911, 1 = 977; after deduplication: 0 = 310, 1 = 292.

## Model, loss, and nested protocol

- The nested experiment uses `OptimizedPMRCNet`, hidden size 16, three named feature-group encoders, a projected engineered physiology mismatch, pairwise coupling gates, and additive mechanistic/interaction/residual logits.
- `HR_pred = 208 - 0.7 × age` is an age-predicted maximum-heart-rate equation; its use here is an engineered score, not a validated patient-specific physiological measurement. The mismatch multiplies a nonnegative heart-rate deficit by a softplus score from `oldpeak`, `exang`, and their interaction.
- For each of five seeds, outer assignments are stratified five-fold splits. Within each outer-training partition, four inner folds and the configured candidate grid select hyperparameters/epoch; temperature and Youden threshold are fitted only from selected-candidate inner OOF predictions. The outer test fold is evaluated once.
- The directionality arm differs from the legacy single-flow control by using separate forward and reversed feature transforms. The control is not a symmetric-coupling model.
- The loss configuration includes `l1_lambda`, but the implemented objective does not use it. The empirical association of `ca` is computed but not used in the physiology hinge.
- Across 100 inner-training partitions (five seeds × five outer folds × four inner folds), positive association signs occurred for `oldpeak` in 4, `exang` in 9, and `ca` in 5 partitions. All 25 outer-refit partitions had negative association signs for `oldpeak` and `exang`; therefore, the feature-specific physiology hinge did not activate for those features in outer refits. This implementation detail limits how strongly the run tests a physiology-informed loss.

## Historical baseline protocol: not a controlled comparison

The Stage-5 baseline uses aligned sample indices, targets, seeds, and outer-fold assignments, but it does not share the nested evaluation protocol:

1. It uses the original `PMRCNet` implementation rather than the optimized architecture.
2. Its Stage-5 training procedure selects the best checkpoint/epoch using the fold-specific `valid_loader`. That validation fold is the nominal outer test fold, so its labels influence model selection.
3. It uses a fixed 0.5 decision threshold and does not apply the inner-only temperature calibration / threshold-selection process.

Consequently, its OOF performance is selection-contaminated and is shown only as historical context. The experiment's nominal paired p-values against this baseline are recorded for transparency but are **not valid confirmatory comparisons**. No improvement, superiority, or noninferiority claim against that baseline is supported. A clean baseline comparison would require a new, nested evaluation of a baseline model under the same split, selection, calibration, and threshold protocol.

## Files examined

- `src/models/pmrc_net.py`, `src/models/pmrc_net_optimized.py`
- `src/data/dataset.py`, `src/training/losses.py`, `src/evaluation/metrics.py`
- `scripts/run_pmrc_nested_optimization.py` (read-only for this paper-generation task)
- `results/PMRC_NET_FINAL_RESEARCH/run_config.json`, `FINAL_RESEARCH_REPORT.md`, `statistical_comparison.csv`, `resource_usage_summary.json`
- `results/PMRC_NET_FINAL_RESEARCH/{directional_coupling,legacy_coupling_control}/` fold, seed, and OOF outputs
- `results/stage5_final_validation/stage5_validation.py` and its saved per-seed OOF predictions

## Source-control / preservation note

The submission generator is separate at `scripts/generate_conference_submission.py`. No PMRC-Net source, dataset, baseline, or completed experiment artifact is changed by the generator. The package excludes row-level OOF predictions; it includes only aggregate fold/seed metrics and the recorded configuration.
