from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"
STAGE2 = ROOT / "results" / "stage2_validation"
STAGE4 = ROOT / "results" / "stage4_statistical_validation"
STAGE5 = ROOT / "results" / "stage5_final_validation"
DATA = ROOT / "dataset" / "cleaned_merged_heart_dataset.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def setup() -> None:
    for directory in (REPORTS, PAPER, FIGURES, TABLES):
        directory.mkdir(parents=True, exist_ok=True)


def inventory() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        rows.append({"path": relative, "suffix": path.suffix, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(rows).to_csv(REPORTS / "final_experiment_inventory.csv", index=False)
    python_files = sum(row["suffix"] == ".py" for row in rows)
    notebooks = sum(row["suffix"] == ".ipynb" for row in rows)
    (REPORTS / "final_repository_inventory.md").write_text(f"""# Final Repository Inventory

Generated from the repository at audit time. Historical files were not deleted or rewritten.

- Files indexed: `{len(rows)}`
- Python files: `{python_files}`
- Notebooks: `{notebooks}`
- Dataset SHA-256: `{sha256(DATA)}`
- Primary dataset: `dataset/cleaned_merged_heart_dataset.csv`
- Original notebook preserved: `{(ROOT / 'Physiological_Mismatch_and_Residual_Consistency_Network.ipynb').exists()}`
- Stage 5 seed files: `{len(list((STAGE5 / 'seed_oof_predictions').glob('seed_*_oof.csv')))}`

The complete path, size, and hash inventory is in `final_experiment_inventory.csv`. Generated reports and paper files are included in this inventory on subsequent runs.
""", encoding="utf-8")


def implementation_audit() -> None:
    config = read_json(STAGE5 / "reproducibility_manifest.json")
    (REPORTS / "final_implementation_audit.md").write_text(f"""# Final Implementation Audit

## Maintained implementation

- Dataset loading, exact deduplication, categorical harmonization, and local sample indices are implemented in `src/data/dataset.py`.
- Fold preprocessing uses `StandardScaler.fit` on training rows and `transform` on validation rows.
- The model implementation is in `src/models/pmrc_net.py`; the original notebook remains preserved.
- The implemented loss is in `src/training/losses.py` and uses BCE, optional physiological hinge terms, and coupling L2 regularization.
- Checkpoint round trips are implemented in `src/training/checkpoints.py`.

## Findings

- `loss.l1_lambda` is declared and stored but is not applied by the loss implementation. This is documented, not silently corrected.
- The physiological hinge is conditional on positive empirical directions. Existing fold associations are negative, so the recorded physiological penalty is zero.
- The notebook contains executed historical cells with environment-specific imports and at least one ordering error (`y_true`/`oof_df` referenced after a failed cell). The maintained Python modules and saved artifacts are the reproducibility path.
- Notebook-local baseline classes are not importable project modules; they are treated as historical evidence only.
- No patient identifier exists in the cleaned data, and no claim of patient independence is made.
- External validation is not performed because UCI Heart Disease is a documented source family that may overlap the merged development data.

## Reproducibility evidence

- Dataset hash: `{config['dataset_hash']}`
- Configuration hash: `{config['config_hash']}`
- Checkpoint identity is recorded in `results/stage5_final_validation/reproducibility_manifest.json`.
- Five seed OOF files and fold assignments are retained.
""", encoding="utf-8")


def limitations() -> None:
    (REPORTS / "final_known_limitations.md").write_text("""# Final Known Limitations

1. The 602 rows are deduplicated observations, not confirmed independent patients.
2. The source merge includes clues for Cleveland/Kaggle/UCI-derived data, but source-level patient provenance is unavailable.
3. UCI Heart Disease and Statlog Heart have compatible-looking fields but are not demonstrably independent from the merged development source; no external validation claim is made.
4. Bootstrap intervals resample rows and cannot account for unknown patient clustering.
5. The positive-direction-only physiological hinge was inactive in the observed training folds.
6. `l1_lambda` is configured but unused.
7. Ablations demonstrate numerical behavior under this dataset and protocol, not causal or physiological validity.
8. Threshold 0.5 results are fixed descriptive results; threshold selection and calibration tuning require a separate pre-specified nested protocol.
9. Exact bitwise reproducibility across hardware is not claimed.
10. No clinical utility, diagnostic readiness, causality, or clinical superiority is established.
""", encoding="utf-8")


def external_screening() -> None:
    candidates = [
        {"candidate": "UCI Heart Disease (ID 45; Cleveland/Hungary/Switzerland/VA)", "source": "UCI ML Repository", "url": "https://archive.ics.uci.edu/dataset/45/heart+disease", "observations": 303, "features": 13, "target": "goal/num; commonly binary presence vs absence", "missing_values": "Yes", "patient_ids": "Names and social-security numbers removed/replaced in repository files", "overlap_risk": "High: local notebook explicitly describes a merged UCI/Kaggle-like source family", "status": "INCONCLUSIVE; not used"},
        {"candidate": "Statlog (Heart) (ID 145)", "source": "UCI ML Repository", "url": "https://archive.ics.uci.edu/dataset/145/statlog+heart", "observations": 270, "features": 13, "target": "absence (1) / presence (2)", "missing_values": "No", "patient_ids": "Not established from downloaded data; no file downloaded", "overlap_risk": "Unknown-to-high: UCI documents similarity to Heart Disease databases", "status": "INCONCLUSIVE; not used"},
    ]
    pd.DataFrame(candidates).to_csv(REPORTS / "external_dataset_provenance.csv", index=False)
    mapping = []
    for candidate in ["UCI Heart Disease", "Statlog Heart"]:
        for local, external, decision in [("age", "age", "name-level match only"), ("sex", "sex", "definition/coding not verified"), ("cp", "chest-pain", "coding not verified"), ("trestbps", "rest-bp", "units/source not verified"), ("chol", "serum-chol", "units/source not verified"), ("fbs", "fasting-blood-sugar", "coding not verified"), ("restecg", "electrocardiographic", "coding not verified"), ("thalachh", "max-heart-rate", "local rename only; source identity not verified"), ("exang", "angina", "coding not verified"), ("oldpeak", "oldpeak", "name-level match only"), ("slope", "slope", "coding not verified"), ("ca", "major-vessels", "missingness/definition not verified"), ("thal", "thal", "coding not verified")]:
            mapping.append({"candidate": candidate, "local_feature": local, "candidate_feature": external, "compatibility": decision, "verified_from_downloaded_data": False, "used_for_validation": False})
    pd.DataFrame(mapping).to_csv(REPORTS / "external_dataset_feature_mapping.csv", index=False)
    (REPORTS / "external_dataset_screening.md").write_text("""# External Dataset Screening

Official UCI pages were inspected for the Heart Disease collection (ID 45) and Statlog Heart (ID 145). The UCI page confirms that ID 45 contains four databases and that its common 13-feature representation includes the local-looking clinical fields. It also states that patient names and social-security numbers were removed or replaced. Statlog Heart is documented as similar to the Heart Disease databases.

These facts make both candidates scientifically relevant but do not establish independence from the local 1,888-row merged source. The local notebook explicitly describes mixed Cleveland/Kaggle/merged data, and no source manifest or patient identifiers are available for overlap resolution. No external files were downloaded, mapped, fitted, or evaluated. The external-validation stage is therefore **INCONCLUSIVE/BLOCKED**, and no external metrics are reported.

Sources:

- UCI Heart Disease, dataset ID 45: https://archive.ics.uci.edu/dataset/45/heart+disease, DOI `10.24432/C52P4X`, CC BY 4.0 as reported by UCI.
- Statlog (Heart), dataset ID 145: https://archive.ics.uci.edu/dataset/145/statlog+heart, DOI `10.24432/C57303`, CC BY 4.0 as reported by UCI.

See `external_dataset_provenance.csv` and `external_dataset_feature_mapping.csv` for candidate-level evidence and compatibility limits.
""", encoding="utf-8")


def statistics() -> pd.DataFrame:
    oof = pd.read_csv(STAGE2 / "oof_predictions.csv")
    forest = pd.read_csv(ROOT / "results" / "stage4_2_baseline_comparison" / "random_forest_oof_aligned.csv")
    key = ["sample_index", "fold", "target"]
    required = set(key + ["probability"])
    if not required.issubset(oof.columns) or not required.issubset(forest.columns):
        raise ValueError("Authoritative PMRC-Net and Random Forest OOF files have incompatible schemas")
    if not oof["sample_index"].is_unique or not forest["sample_index"].is_unique:
        raise ValueError("Authoritative OOF files contain duplicate sample_index values")
    aligned = oof.merge(forest, on=key, how="inner", validate="one_to_one", suffixes=("_pmrc", "_random_forest"))
    alignment = {
        "pmrc_source": "results/stage2_validation/oof_predictions.csv",
        "random_forest_source": "results/stage4_2_baseline_comparison/random_forest_oof_aligned.csv",
        "alignment_key": key,
        "pmrc_rows": int(len(oof)),
        "random_forest_rows": int(len(forest)),
        "aligned_rows": int(len(aligned)),
        "sample_index_sets_identical": bool(set(oof["sample_index"]) == set(forest["sample_index"])),
        "folds_and_targets_key_aligned": bool(len(aligned) == len(oof) == len(forest)),
        "status": "PASS" if len(aligned) == 602 else "INCONCLUSIVE",
    }
    (REPORTS / "final_alignment_verification.json").write_text(json.dumps(alignment, indent=2), encoding="utf-8")
    if alignment["status"] != "PASS":
        raise ValueError("Authoritative OOF files could not be aligned on sample_index, fold, and target")
    y = aligned["target"].to_numpy(int)
    p = aligned["probability_pmrc"].to_numpy(float)
    from src.evaluation.metrics import calculate_metrics
    forest_probability = aligned["probability_random_forest"].to_numpy(float)
    pmrc_metrics = calculate_metrics(y, p)
    forest_metrics = calculate_metrics(y, forest_probability)
    rows = [
        {"model": "PMRC-Net", "source": alignment["pmrc_source"], "alignment_key": "sample_index,fold,target", **pmrc_metrics},
        {"model": "Random Forest", "source": alignment["random_forest_source"], "alignment_key": "sample_index,fold,target", **forest_metrics},
    ]
    result = pd.DataFrame(rows)
    result.to_csv(REPORTS / "final_metric_recalculation.csv", index=False)
    result.to_csv(TABLES / "baseline_comparison.csv", index=False)
    comparison = pd.DataFrame([
        {"metric": metric, "pmrc_estimate": pmrc_metrics[metric], "random_forest_estimate": forest_metrics[metric], "difference_pmrc_minus_random_forest": pmrc_metrics[metric] - forest_metrics[metric], "source": "keyed merge on sample_index, fold, target"}
        for metric in ("roc_auc", "average_precision", "brier_score", "accuracy", "precision", "sensitivity", "specificity", "f1", "mcc")
    ])
    comparison.to_csv(TABLES / "baseline_comparison_metrics.csv", index=False)
    (REPORTS / "final_statistical_validation.md").write_text("""# Final Statistical Validation

The authoritative PMRC-Net OOF file (`results/stage2_validation/oof_predictions.csv`) and Stage 4.2 Random Forest file (`results/stage4_2_baseline_comparison/random_forest_oof_aligned.csv`) were independently verified and merged on `sample_index`, `fold`, and `target`. Row ordering was not used as the pairing key. The keyed merge contains 602 one-to-one observations with identical sample-index sets, folds, and labels.

The corrected pooled results are PMRC-Net ROC-AUC `0.8955`, Random Forest ROC-AUC `0.8798`, and difference `+0.0158`. The saved Stage 4.2 paired bootstrap ROC-AUC interval is `[0.0009, 0.0313]`. The AP difference is `+0.0145` and Brier difference is `-0.0122`. These are descriptive row-level results; no superiority claim is made because patient dependence is unknown.

Five-seed results are in `results/stage5_final_validation/multi_seed_summary.csv`: mean ROC-AUC `0.8966`, standard deviation `0.0050`; mean accuracy `0.8150`, standard deviation `0.0085`; mean F1 `0.8065`, standard deviation `0.0090`; mean MCC `0.6296`, standard deviation `0.0171`; mean Brier `0.1313`, standard deviation `0.0025`.

Calibration, robustness, and ablation values are retained from Stage 4/Stage 2 artifacts and interpreted descriptively. No new threshold, calibrator, or hyperparameter was selected from outer OOF labels.
""", encoding="utf-8")
    return result


def scientific_docs() -> None:
    (REPORTS / "final_reproducibility_audit.md").write_text("""# Final Reproducibility Audit

PASS: dataset SHA-256, configuration hash, software versions, checkpoint hash, random seed set, fold manifest, preprocessing description, and Stage 5 OOF integrity are recorded in `results/stage5_final_validation/reproducibility_manifest.json` and `final_research_audit.json`.

PASS: checkpoint loading was executed. PASS: five seed OOF files contain 602 unique sample indices, complete fold labels, and finite probabilities.

INCONCLUSIVE: patient-level independence and exact bitwise reproducibility across hardware. BLOCKED: independent external validation because source overlap cannot be ruled out.

The reproducible path is the maintained Python source and explicit commands in `REPRODUCIBILITY.md`; the original notebook is preserved as historical research context.
""", encoding="utf-8")
    (REPORTS / "final_scientific_claims_audit.md").write_text("""# Final Scientific Claims Audit

| Claim | Status | Boundary |
|---|---|---|
| PMRC-Net produces valid internal OOF predictions | PASS | Row-level, deduplicated observations |
| Performance is stable across five seeds | PASS | Seed variability only; patient clustering unknown |
| PMRC-Net is superior to Random Forest | INCONCLUSIVE | Saved interval is descriptive; no superiority claim |
| Pathway modules have numerical computational pathways | PASS | Gradient and ablation evidence; not causal validity |
| Physiological mismatch is clinically valid | INCONCLUSIVE | Biological interpretation remains a hypothesis |
| Dataset rows are independent patients | INCONCLUSIVE | No patient identifiers or source provenance |
| Independent external validation is complete | BLOCKED | Candidate sources may overlap development data |
| Model is clinically ready | FAIL | No clinical validation or utility study |
""", encoding="utf-8")
    (REPORTS / "final_architecture_documentation.md").write_text((ROOT / "results/stage5_final_validation/architecture_justification.md").read_text(encoding="utf-8") + "\nThe documentation describes implemented computation, not clinical validity.\n", encoding="utf-8")
    (REPORTS / "final_loss_function_documentation.md").write_text((ROOT / "results/stage5_final_validation/loss_function_analysis.md").read_text(encoding="utf-8") + "\nNo architecture or loss changes were made during finalization.\n", encoding="utf-8")
    (REPORTS / "final_ablation_interpretation.md").write_text("""# Final Ablation Interpretation

Saved pooled Stage 2 ROC-AUC values are: no physiology `0.8956`, no pathways `0.5326`, no mechanistic output `0.8961`, and no residual output `0.8983`. Removing pathways causes a large degradation under this protocol, while removing mechanistic or residual outputs changes ROC-AUC only slightly. These results establish sensitivity of this implementation on this dataset; they do not prove causal necessity, physiological validity, or generalization. The no-physiology result must be interpreted with the inactive positive-direction hinge and unused `l1_lambda`.
""", encoding="utf-8")


def figures() -> None:
    seed = pd.read_csv(STAGE5 / "multi_seed_results.csv")
    summary = pd.read_csv(STAGE5 / "multi_seed_summary.csv")
    seed_means = seed.groupby("seed", sort=True)["roc_auc"].mean()
    plt.figure(figsize=(7, 4)); plt.bar(seed_means.index.astype(str), seed_means.to_numpy()); plt.ylabel("Mean fold ROC-AUC"); plt.xlabel("Seed"); plt.title("PMRC-Net seed variability (saved OOF runs)"); plt.tight_layout(); savefig("five_seed_roc_auc")
    history_path = ROOT / "experiments" / "controlled_pmrc_seed42" / "history.json"
    if history_path.exists():
        history = pd.DataFrame(read_json(history_path))
        plt.figure(figsize=(7, 4)); plt.plot(history["epoch"], history["train_loss"], label="train"); plt.plot(history["epoch"], history["valid_loss"], label="validation"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Controlled seed-42 loss curves"); plt.tight_layout(); savefig("training_validation_loss")
    oof = pd.read_csv(ROOT / "results" / "oof_predictions.csv")
    y = oof["target_true"].to_numpy(int)
    model_columns = [("PMRC-Net", "PMRC-Net_Full"), ("Random Forest", "RandomForest"), ("Logistic Regression", "LogisticRegression"), ("SVM", "SVM")]
    plt.figure(figsize=(6, 5))
    for name, column in model_columns:
        if column in oof: fpr, tpr, _ = roc_curve(y, oof[column]); plt.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y, oof[column]):.3f})")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(); plt.title("Saved OOF ROC curves"); plt.tight_layout(); savefig("baseline_roc_curves")
    plt.figure(figsize=(6, 5))
    for name, column in model_columns:
        if column in oof: precision, recall, _ = precision_recall_curve(y, oof[column]); plt.plot(recall, precision, label=f"{name} ({average_precision_score(y, oof[column]):.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.title("Saved OOF precision-recall curves"); plt.tight_layout(); savefig("precision_recall_curves")
    pmrc = pd.read_csv(STAGE2 / "oof_predictions.csv"); cm = confusion_matrix(pmrc["target"], pmrc["probability"] >= .5, labels=[0, 1]); plt.figure(figsize=(4, 4)); plt.imshow(cm, cmap="Blues"); plt.colorbar(); plt.xticks([0, 1], ["0", "1"]); plt.yticks([0, 1], ["0", "1"]); plt.xlabel("Predicted"); plt.ylabel("Observed"); plt.title("PMRC-Net confusion matrix"); [plt.text(j, i, cm[i, j], ha="center", va="center") for i in range(2) for j in range(2)]; plt.tight_layout(); savefig("confusion_matrix")
    frac, mean = calibration_curve(y, oof["PMRC-Net_Full"], n_bins=10); plt.figure(figsize=(5, 5)); plt.plot(mean, frac, "o-", label="PMRC-Net"); plt.plot([0, 1], [0, 1], "k--", label="Ideal"); plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction positive"); plt.legend(); plt.title("Saved OOF calibration"); plt.tight_layout(); savefig("calibration")
    ablation = {"Full": read_json(STAGE2 / "aggregate_metrics.json")["roc_auc"], "No physiology": read_json(STAGE2 / "aggregate_metrics_no_physiology.json")["roc_auc"], "No pathways": read_json(STAGE2 / "aggregate_metrics_no_pathways.json")["roc_auc"], "No mechanistic": read_json(STAGE2 / "aggregate_metrics_no_mechanistic.json")["roc_auc"], "No residual": read_json(STAGE2 / "aggregate_metrics_no_residual.json")["roc_auc"]}; plt.figure(figsize=(7, 4)); plt.bar(list(ablation), list(ablation.values())); plt.ylabel("Pooled OOF ROC-AUC"); plt.xticks(rotation=25, ha="right"); plt.title("Saved ablation results"); plt.tight_layout(); savefig("ablation_comparison")
    robust_path = STAGE4 / "robustness_analysis.csv"
    if robust_path.exists():
        robust = pd.read_csv(robust_path); numeric = robust.select_dtypes(include="number"); plt.figure(figsize=(7, 4)); numeric.mean().plot(kind="bar"); plt.ylabel("Mean recorded value"); plt.title("Saved robustness summary"); plt.tight_layout(); savefig("robustness_summary")


def savefig(name: str) -> None:
    plt.savefig(FIGURES / f"{name}.png", dpi=300); plt.savefig(FIGURES / f"{name}.pdf"); plt.close()


def paper(metric_table: pd.DataFrame) -> None:
    metric_markdown = metric_table.to_markdown(index=False) if "tabulate" in sys.modules else metric_table.to_string(index=False)
    (TABLES / "five_seed_summary.csv").write_bytes((STAGE5 / "multi_seed_summary.csv").read_bytes())
    if not (TABLES / "baseline_comparison.csv").exists():
        metric_table.to_csv(TABLES / "baseline_comparison.csv", index=False)
    (PAPER / "references.bib").write_text("""@misc{uci_heart_disease,
  author = {Janosi, Andras and Steinbrunn, William and Pfisterer, Matthias and Detrano, Robert},
  title = {Heart Disease Dataset},
  year = {1989},
  publisher = {UCI Machine Learning Repository},
  doi = {10.24432/C52P4X},
  url = {https://archive.ics.uci.edu/dataset/45/heart+disease}
}

@misc{uci_statlog_heart,
  title = {Statlog (Heart) Dataset},
  publisher = {UCI Machine Learning Repository},
  url = {https://archive.ics.uci.edu/dataset/145/statlog+heart},
  doi = {10.24432/C57303}
}
""", encoding="utf-8")
    (PAPER / "PMRC_Net_Research_Paper.md").write_text(f"""# PMRC-Net: A Physiologically Structured Neural Network with Mismatch and Residual Consistency for Heart Disease Risk Assessment

## Abstract

PMRC-Net is a tabular neural network organized into metabolic, exercise-response, and coronary/electrical pathways, with a physiology-inspired mismatch representation, gated interactions, and residual output. On 602 deduplicated observations from a merged heart-disease dataset, the historical seed-42 five-fold pooled OOF ROC-AUC was 0.8955. Across five seeds, mean ROC-AUC was 0.8966 (SD 0.0050). These are internal row-level results, not clinical validation. Patient-level independence and independent external validation remain unverified.

## Keywords

Heart disease risk assessment; structured neural networks; tabular learning; out-of-fold validation; reproducibility; physiological mismatch.

## Introduction and research gap

Clinical tabular datasets contain heterogeneous measurements and categorical encodings. PMRC-Net explores whether an explicit pathway organization and additive logit decomposition provide a useful computational inductive bias. The study does not claim that names such as metabolic or coronary establish biological validity. The research gap is treated as an engineering hypothesis: whether structured computation can be evaluated transparently against standard baselines under leakage-controlled validation.

## Methodology

The maintained implementation uses five continuous and eight categorical inputs. Continuous values are standardized separately inside each training fold. Categorical values are passed through learned embeddings. Pathway representations are projected to a shared hidden dimension. A raw-scale physiology module computes an expected heart rate, heart-rate deficit, a softplus ischemia score, and their product as mismatch. Asymmetric coupling gates combine pathway states pairwise. The mechanistic, interaction, and residual logits are added to form the final logit, and the sigmoid gives the probability.

The loss is binary cross-entropy plus a conditional physiological hinge and coupling L2 regularization. The configured L1 parameter is unused in the actual implementation. The observed physiological hinge was zero because empirical directions were negative while the original hinge only activates positive directions.

## Dataset and provenance

The local notebook records 1,888 merged rows and 602 exact unique rows. There is no verified patient identifier. The source clues reference Cleveland, Kaggle, and merged data, but source cohort membership and patient-level independence cannot be established. Rows are therefore described as deduplicated observations.

## Experimental protocol

The primary internal evaluation is stratified five-fold OOF validation with fold-local preprocessing. Stage 5 repeats this protocol for seeds 21, 42, 123, 2026, and 3407. The historical seed-42 result is preserved separately. Comparisons use saved aligned OOF artifacts. Threshold 0.5 is reported descriptively.

## Results

The historical PMRC-Net pooled seed-42 results were ROC-AUC 0.8955, accuracy 0.8256, F1 0.8174, MCC 0.6508, and Brier 0.1307. The five-seed mean values were ROC-AUC 0.8966, accuracy 0.8150, F1 0.8065, MCC 0.6296, and Brier 0.1313. The keyed Stage 4.2 comparison gives Random Forest ROC-AUC 0.8798, a PMRC-Net difference of +0.0158, and a descriptive paired bootstrap interval of [0.0009, 0.0313]. AP difference is +0.0145 and Brier difference is -0.0122. No superiority claim is made.

Detailed recalculation is in `reports/final_metric_recalculation.csv` and five-seed values are in `paper/tables/five_seed_summary.csv`.

## Ablation and robustness

Removing pathways produced a major ROC-AUC reduction to 0.5326. Removing physiology produced 0.8956, while removing mechanistic and residual outputs produced 0.8961 and 0.8983. These results indicate computational sensitivity patterns under this dataset, not causal necessity or physiological truth. Stage 4 perturbation and robustness analyses are descriptive model-sensitivity tests.

## Calibration and uncertainty

The saved pooled Brier score was 0.1307 and equal-width ten-bin ECE was 0.0383. Row-level bootstrap intervals do not model unknown patient clustering. Calibration was not externally validated and no external calibrator was fit.

## External validation

External validation is pending. UCI Heart Disease and Statlog Heart were screened from official UCI documentation, but their possible overlap with the local merged source cannot be ruled out. No external observations were used, and no external metrics are reported.

## Discussion

The evidence supports an implementation with reproducible internal OOF behavior and finite gradient flow through inspected components. It does not establish clinical usefulness, diagnostic readiness, causality, or clinical superiority. The modest seed variation supports numerical stability at the row level, while the lower mean threshold metrics relative to the historical seed-42 run illustrate the importance of reporting seed variability.

## Limitations and ethics

The lack of patient identifiers prevents patient-level independence verification. Source-derived duplicates may represent repeated measurements or repeated patients. The data are small, merged, and not a prospective clinical cohort. Predictions must not be used for clinical decisions. Any future external evaluation requires independent patient-level provenance, prespecified feature mapping, frozen preprocessing/model parameters, and supervisor review.

## Conclusion and future work

PMRC-Net is implementation-complete and internally validated under the saved protocol. It is not externally validated. Future work should obtain a genuinely independent, patient-level dataset; preregister feature and target mapping; evaluate clustering-aware uncertainty; and seek domain review of the physiological assumptions before making clinical claims.

## Reproducibility

Run `python test_setup.py`, `python -m unittest discover -s tests -v`, and `python scripts/finalize_project.py` from the repository root. Do not start training automatically. The original notebook and historical Stage 1–4 artifacts are preserved.

## References

See `paper/references.bib` for the verified UCI dataset records. The manuscript intentionally does not add unsupported literature claims.
""", encoding="utf-8")


def completion(metric_table: pd.DataFrame) -> None:
    audit = read_json(STAGE5 / "final_research_audit.json")
    status = {"software_completion": "PASS", "internal_validation": "PASS", "statistical_validation": "PASS", "external_validation": "BLOCKED", "patient_level_provenance": "INCONCLUSIVE", "paper_preparation": "PASS", "automated_tests": "INCONCLUSIVE", "overall": "INCONCLUSIVE"}
    (REPORTS / "FINAL_RESEARCH_STATUS.json").write_text(json.dumps({"status": status, "evidence": audit}, indent=2), encoding="utf-8")
    (REPORTS / "FINAL_DELIVERABLES_INDEX.md").write_text("""# Final Deliverables Index

## Audit and status

- `final_repository_inventory.md`
- `final_implementation_audit.md`
- `final_experiment_inventory.csv`
- `final_known_limitations.md`
- `FINAL_PROJECT_COMPLETION_REPORT.md`
- `FINAL_RESEARCH_STATUS.json`

## Scientific validation

- `final_statistical_validation.md`
- `final_metric_recalculation.csv`
- `final_alignment_verification.json`
- `final_reproducibility_audit.md`
- `final_scientific_claims_audit.md`
- `external_dataset_screening.md`
- `external_dataset_provenance.csv`
- `external_dataset_feature_mapping.csv`

The authoritative Stage 4.2 source remains `results/stage4_2_baseline_comparison/random_forest_oof_aligned.csv`; legacy Stage 4 summary files were preserved and are not silently rewritten.

## Documentation and paper

- `final_architecture_documentation.md`
- `final_loss_function_documentation.md`
- `final_ablation_interpretation.md`
- `paper/PMRC_Net_Research_Paper.md`
- `paper/references.bib`
- `paper/tables/`
- `paper/figures/` (PNG and PDF figures generated from saved artifacts)

## Tests

- `tests/test_smoke.py`
- `final_test_results.md`
""", encoding="utf-8")
    (REPORTS / "FINAL_PROJECT_COMPLETION_REPORT.md").write_text("""# Final Project Completion Report

## Overall status

Software and internal research finalization are complete. Independent external validation is **BLOCKED** because no candidate dataset can be demonstrated to be independent of the local merged development source. The project is suitable for supervisor review as a transparent internal-validation manuscript, not as a clinically validated system.

## Completed stages

- Repository inventory and implementation audit.
- External dataset source screening without fabricating external evidence.
- Independent metric recalculation and five-seed integrity review.
- Corrected Stage 4.2 PMRC-Net versus Random Forest comparison using a one-to-one merge on `sample_index`, `fold`, and `target`; row ordering was not used.
- Architecture, loss, ablation, provenance, and scientific-claims documentation.
- Lightweight smoke-test source covering data, preprocessing, model, loss, gradients, optimizer updates, checkpoint round-trip, metrics, OOF alignment, CPU, and CUDA when available. The dataset test passed; the full unittest runner remains inconclusive because it terminates during the checkpoint test without a traceback.
- Publication figures, manuscript, tables, references, reproducibility audit, and deliverables index.

## Exact commands

```powershell
.\\.venv\\Scripts\\python.exe test_setup.py
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
.\\.venv\\Scripts\\python.exe scripts\\finalize_project.py
.\\.venv\\Scripts\\python.exe results\\stage5_final_validation\\stage5_validation.py --task reports
```

Training is intentionally not started by finalization commands.

## Scientific decision

PASS: implementation complete, internal validation complete, statistical validation complete, reproducibility evidence recorded.

PASS: Stage 4.2 alignment verified for 602 observations. Corrected values are PMRC-Net ROC-AUC 0.8955, Random Forest ROC-AUC 0.8798, difference +0.0158, ROC-AUC bootstrap interval [0.0009, 0.0313], AP difference +0.0145, and Brier difference -0.0122.

INCONCLUSIVE: full automated test-suite execution in the current terminal session; direct checkpoint serialization and loading succeeded outside unittest.

INCONCLUSIVE: patient-level independence and causal/physiological interpretation.

BLOCKED: independent external validation and any clinical-readiness claim.

Supervisor approval remains required for source provenance resolution, external-data selection, and interpretation of the physiological assumptions.
""", encoding="utf-8")


def main() -> None:
    setup(); inventory(); implementation_audit(); limitations(); external_screening(); metric_table = statistics(); scientific_docs(); figures(); paper(metric_table); completion(metric_table)
    print(json.dumps({"status": "completed", "reports": str(REPORTS), "paper": str(PAPER)}, indent=2))


if __name__ == "__main__":
    main()
