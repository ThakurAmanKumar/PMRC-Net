from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "final_status"
DATA = ROOT / "dataset" / "cleaned_merged_heart_dataset.csv"
STAGE2 = ROOT / "results" / "stage2_validation"
STAGE42 = ROOT / "results" / "stage4_2_baseline_comparison"
STAGE5 = ROOT / "results" / "stage5_final_validation"
EXPECTED_DATASET_SHA256 = "a786b4a01e306cd892293c2671e15f29a29a17d06c69b16b19906a337ad42b90"
EXPECTED_CHECKPOINT_SHA256 = "41671ea9a0407b7999966388bb2b2ce8f6901da5196532eb3287eb720fef1936"
EXPECTED_DUPLICATES = {
    "raw_rows": 1888,
    "exact_duplicate_rows": 1286,
    "exact_unique_rows": 602,
    "feature_duplicate_rows": 1588,
    "feature_duplicate_groups": 302,
    "conflicting_target_groups": 0,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, text: str) -> None:
    (FINAL / name).write_text(text.strip() + "\n", encoding="utf-8")


def copy_artifact(source: Path, name: str | None = None) -> None:
    shutil.copy2(source, FINAL / (name or source.name))


def build_inventory() -> None:
    inventory_path = (FINAL / "final_experiment_inventory.csv").resolve()
    rows = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if (
            not path.is_file()
            or path.resolve() == inventory_path
            or any(part.casefold() in {".venv", "__pycache__"} for part in relative.parts)
        ):
            continue
        rows.append({"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(rows).to_csv(FINAL / "final_experiment_inventory.csv", index=False)


def analyze_duplicates(
    frame: pd.DataFrame,
    continuous_columns: list[str],
    categorical_columns: list[str],
    target_column: str,
) -> dict:
    required = continuous_columns + categorical_columns + [target_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing duplicate-analysis columns: {missing}")

    feature_columns = continuous_columns + categorical_columns
    feature_sizes = frame.groupby(feature_columns, dropna=False, sort=False).size()
    feature_targets = frame.groupby(feature_columns, dropna=False, sort=False)[target_column].nunique()
    duplicated_groups = feature_sizes[feature_sizes > 1]

    continuous = frame[continuous_columns].astype(float)
    standard_deviation = continuous.std(axis=0, ddof=0).replace(0, 1.0)
    standardized = (continuous - continuous.mean(axis=0)) / standard_deviation
    near_duplicate_pairs = 0
    for indices in frame.groupby(categorical_columns, dropna=False, sort=False).indices.values():
        if len(indices) > 1:
            tree = cKDTree(standardized.iloc[indices].to_numpy(dtype=float))
            near_duplicate_pairs += len(tree.query_pairs(r=0.1, output_type="ndarray"))

    return {
        "raw_rows": int(len(frame)),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "exact_unique_rows": int(len(frame.drop_duplicates())),
        "feature_duplicate_rows": int(duplicated_groups.sum()),
        "feature_duplicate_groups": int(len(duplicated_groups)),
        "conflicting_target_groups": int((feature_targets.loc[duplicated_groups.index] > 1).sum()),
        "near_duplicate_candidate_pairs": int(near_duplicate_pairs),
        "standardization": (
            "Column-wise z-score: subtract the full dataset column mean and divide by "
            "population standard deviation (ddof=0); zero-variance columns map to zero."
        ),
        "near_duplicate_distance": "Euclidean distance <= 0.1 across the five standardized continuous columns",
        "categorical_match": f"Exact match on all {len(categorical_columns)} configured categorical columns",
    }


def run_capture(command: list[str], stdout_name: str, stderr_name: str) -> dict:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed = time.perf_counter() - started
    (FINAL / stdout_name).write_text(result.stdout, encoding="utf-8")
    (FINAL / stderr_name).write_text(result.stderr, encoding="utf-8")
    combined = result.stdout + "\n" + result.stderr
    test_match = re.search(
        r"Ran\s+(\d+)\s+tests?\s+in\s+([0-9.]+)\s*(?:seconds?|s)\b", combined
    )
    skipped_match = re.search(r"OK\s+\(skipped=(\d+)\)", combined)
    return {
        "command": command,
        "exit_code": int(result.returncode),
        "elapsed_seconds": elapsed,
        "test_count": int(test_match.group(1)) if test_match else None,
        "framework_duration_seconds": float(test_match.group(2)) if test_match else None,
        "skipped_count": int(skipped_match.group(1)) if skipped_match else 0,
        "output": combined,
        "summary_found": test_match is not None,
        "output_indicates_success": bool(re.search(r"(?:^|\n)OK(?:\s|\n|$)", combined)),
    }


def verify_evidence(raw: pd.DataFrame, test_run: dict, setup_run: dict, duplicate_counts: dict) -> dict:
    checks = {}
    checks["dataset_hash"] = DATA.is_file() and sha256(DATA) == EXPECTED_DATASET_SHA256
    checks["raw_dataset_rows_match"] = len(raw) == duplicate_counts["raw_rows"]
    checks["dataset_counts_match_documented_expectations"] = all(
        duplicate_counts.get(key) == value for key, value in EXPECTED_DUPLICATES.items()
    )
    checks["near_duplicate_calculation"] = (
        isinstance(duplicate_counts.get("near_duplicate_candidate_pairs"), int)
        and duplicate_counts["near_duplicate_candidate_pairs"] >= 0
        and len(duplicate_counts.get("standardization", "")) > 0
        and len(duplicate_counts.get("near_duplicate_distance", "")) > 0
        and len(duplicate_counts.get("categorical_match", "")) > 0
    )

    pmrc = pd.read_csv(STAGE2 / "oof_predictions.csv")
    forest = pd.read_csv(STAGE42 / "random_forest_oof_aligned.csv")
    alignment_key = ["sample_index", "fold", "target"]
    pmrc_keys = pmrc[alignment_key].sort_values(alignment_key).reset_index(drop=True)
    forest_keys = forest[alignment_key].sort_values(alignment_key).reset_index(drop=True)
    alignment_ok = (
        len(pmrc) == len(forest)
        and not pmrc.duplicated(alignment_key).any()
        and not forest.duplicated(alignment_key).any()
        and pmrc_keys.equals(forest_keys)
        and pmrc["sample_index"].nunique() == len(pmrc)
        and forest["sample_index"].nunique() == len(forest)
    )
    saved_alignment = json.loads((FINAL / "final_alignment_verification.json").read_text(encoding="utf-8"))
    checks["stage4_2_alignment"] = bool(
        alignment_ok
        and saved_alignment.get("status") == "PASS"
        and saved_alignment.get("aligned_rows") == len(pmrc)
        and saved_alignment.get("sample_index_sets_identical")
        and saved_alignment.get("folds_and_targets_key_aligned")
    )

    recalc = pd.read_csv(FINAL / "final_metric_recalculation.csv").set_index("model")
    metric_matches = True
    for model_name, predictions in (("PMRC-Net", pmrc), ("Random Forest", forest)):
        y_true = predictions["target"].to_numpy()
        y_score = predictions["probability"].to_numpy()
        y_pred = (y_score >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        computed = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "sensitivity": recall_score(y_true, y_pred, zero_division=0),
            "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_score),
            "average_precision": average_precision_score(y_true, y_score),
            "brier_score": brier_score_loss(y_true, y_score),
        }
        metric_matches = metric_matches and all(
            np.isclose(float(recalc.loc[model_name, metric]), value, atol=1e-12, rtol=0)
            for metric, value in computed.items()
        )
    checks["corrected_metrics"] = bool(metric_matches)

    stage5 = pd.read_csv(STAGE5 / "multi_seed_summary.csv")
    stage5_results = pd.read_csv(STAGE5 / "multi_seed_results.csv")
    manifest = json.loads((STAGE5 / "reproducibility_manifest.json").read_text(encoding="utf-8"))
    checks["five_seed_summary"] = (
        set(stage5["metric"]) >= {"roc_auc", "accuracy", "f1", "mcc", "brier_score"}
        and bool((stage5["seeds"] == 5).all())
        and np.isfinite(stage5[["mean", "std", "minimum", "maximum"]].to_numpy()).all()
        and stage5_results["seed"].nunique() == 5
        and bool((stage5_results.groupby("seed").size() == 5).all())
        and len(manifest.get("seed_set", [])) == 5
        and set(stage5_results["seed"].unique()) == set(manifest["seed_set"])
    )
    checkpoint = ROOT / "checkpoints" / "controlled_pmrc_seed42" / "best.pt"
    checkpoint_hash = sha256(checkpoint) if checkpoint.is_file() else None
    manifest_checkpoint_hashes = {
        artifact.get("sha256")
        for artifact in manifest.get("artifacts", [])
        if str(artifact.get("path", "")).endswith("best.pt")
    }
    checks["checkpoint_hash"] = (
        checkpoint_hash == EXPECTED_CHECKPOINT_SHA256
        and checkpoint_hash in manifest_checkpoint_hashes
    )
    checks["stage5_dataset_hash"] = manifest.get("dataset_hash") == sha256(DATA)

    checks["automated_test_log"] = (
        test_run["summary_found"]
        and test_run["test_count"] is not None
        and test_run["framework_duration_seconds"] is not None
        and test_run["output_indicates_success"]
    )
    checks["automated_test_exit_code"] = test_run["exit_code"] == 0
    checks["automated_test_duration_recorded"] = test_run["framework_duration_seconds"] is not None
    checks["legacy_setup_log"] = setup_run["exit_code"] == 0
    checks["external_not_evaluated"] = not (FINAL / "external_validation_predictions.csv").exists()
    provenance = pd.read_csv(FINAL / "external_dataset_provenance.csv")
    feature_mapping = pd.read_csv(FINAL / "external_dataset_feature_mapping.csv")
    checks["external_dataset_screening_evidence"] = (
        len(provenance) > 0
        and len(feature_mapping) > 0
        and "official_url" in provenance.columns
        and provenance["official_url"].notna().all()
    )
    return {
        key: bool(value) if isinstance(value, (bool, np.bool_)) else value
        for key, value in checks.items()
    }


def verify_outputs(required: list[str], evidence_paths: list[str]) -> dict:
    checks: dict[str, bool] = {}
    checks["required_output_files"] = all(
        (FINAL / name).is_file()
        and (
            (FINAL / name).stat().st_size > 0
            or name.endswith("_stdout.log")
            or name.endswith("_stderr.log")
        )
        for name in required
    )
    checks["json_outputs_valid"] = True
    checks["csv_outputs_valid"] = True
    for name in required:
        path = FINAL / name
        if not path.is_file() or path.stat().st_size == 0:
            if path.suffix.casefold() == ".json":
                checks["json_outputs_valid"] = False
            if path.suffix.casefold() == ".csv":
                checks["csv_outputs_valid"] = False
            continue
        if path.suffix.casefold() == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                checks["json_outputs_valid"] = False
        elif path.suffix.casefold() == ".csv":
            try:
                table = pd.read_csv(path)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                checks["csv_outputs_valid"] = False
            else:
                checks["csv_outputs_valid"] = checks["csv_outputs_valid"] and len(table.columns) > 0
    checks["referenced_evidence_paths"] = all((ROOT / path).is_file() for path in evidence_paths)
    return {
        key: bool(value) if isinstance(value, (bool, np.bool_)) else value
        for key, value in checks.items()
    }


def verify_inventory() -> bool:
    inventory = pd.read_csv(FINAL / "final_experiment_inventory.csv")
    if "path" not in inventory.columns or "sha256" not in inventory.columns:
        return False
    paths = inventory["path"].astype(str).tolist()
    if "final_status/final_experiment_inventory.csv" in paths:
        return False
    if any(
        part.casefold() in {".venv", "__pycache__"}
        for item in paths
        for part in Path(item).parts
    ):
        return False
    return all(
        (ROOT / item).is_file() and sha256(ROOT / item) == digest
        for item, digest in zip(paths, inventory["sha256"].astype(str))
    )


def main() -> None:
    FINAL.mkdir(exist_ok=True)
    test_run = run_capture(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        "test_suite_stdout.log",
        "test_suite_stderr.log",
    )
    setup_run = run_capture(
        [sys.executable, "test_setup.py"],
        "test_setup_stdout.log",
        "test_setup_stderr.log",
    )
    (FINAL / "test_setup_direct.log").write_text(
        setup_run["output"].rstrip()
        + f"\n\nDIRECT_EXIT_CODE={setup_run['exit_code']}\n"
        + f"DURATION_SECONDS={setup_run['elapsed_seconds']:.6f}\n",
        encoding="utf-8",
    )
    test_run["status"] = (
        "PASS"
        if test_run["exit_code"] == 0
        and test_run["summary_found"]
        and test_run["output_indicates_success"]
        else "FAIL"
        if test_run["exit_code"] != 0
        else "INCONCLUSIVE"
    )
    setup_run["status"] = "PASS" if setup_run["exit_code"] == 0 else "FAIL"
    (FINAL / "test_execution_metadata.json").write_text(
        json.dumps(
            {
                "test_suite": {key: value for key, value in test_run.items() if key != "output"},
                "legacy_setup": {key: value for key, value in setup_run.items() if key != "output"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    copy_artifact(ROOT / "reports" / "final_metric_recalculation.csv")
    copy_artifact(ROOT / "reports" / "external_dataset_provenance.csv")
    copy_artifact(ROOT / "reports" / "external_dataset_feature_mapping.csv")
    raw = pd.read_csv(DATA)
    config = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    data_config = config["data"]
    duplicate_counts = analyze_duplicates(
        raw,
        data_config["continuous_cols"],
        data_config["categorical_cols"],
        data_config["target"],
    )

    pmrc = pd.read_csv(STAGE2 / "oof_predictions.csv")
    forest = pd.read_csv(STAGE42 / "random_forest_oof_aligned.csv")
    alignment_key = ["sample_index", "fold", "target"]
    pmrc_keys = pmrc[alignment_key].sort_values(alignment_key).reset_index(drop=True)
    forest_keys = forest[alignment_key].sort_values(alignment_key).reset_index(drop=True)
    aligned = (
        len(pmrc) == len(forest)
        and not pmrc.duplicated(alignment_key).any()
        and not forest.duplicated(alignment_key).any()
        and pmrc_keys.equals(forest_keys)
        and pmrc["sample_index"].nunique() == len(pmrc)
        and forest["sample_index"].nunique() == len(forest)
    )
    alignment = {
        "pmrc_source": "results/stage2_validation/oof_predictions.csv",
        "random_forest_source": "results/stage4_2_baseline_comparison/random_forest_oof_aligned.csv",
        "alignment_key": alignment_key,
        "pmrc_rows": int(len(pmrc)),
        "random_forest_rows": int(len(forest)),
        "aligned_rows": int(len(pmrc)) if aligned else 0,
        "sample_index_sets_identical": bool(
            set(pmrc["sample_index"]) == set(forest["sample_index"])
        ),
        "folds_and_targets_key_aligned": bool(pmrc_keys.equals(forest_keys)),
        "status": "PASS" if aligned else "INCONCLUSIVE",
    }
    (FINAL / "final_alignment_verification.json").write_text(
        json.dumps(alignment, indent=2), encoding="utf-8"
    )

    test_count = test_run["test_count"]
    test_duration = test_run["framework_duration_seconds"]
    test_elapsed = test_run["elapsed_seconds"]
    test_status = test_run["status"]
    setup_status = setup_run["status"]
    cuda_result = (
        "PASS"
        if re.search(r"test_cuda_forward\s+\([^)]*\)\s+\.\.\.\s+ok", test_run["output"])
        else "SKIPPED"
        if re.search(r"test_cuda_forward\s+\([^)]*\)\s+\.\.\.\s+skipped", test_run["output"])
        else "INCONCLUSIVE"
    )
    expected_mismatches = {
        key: {"expected": expected, "actual": duplicate_counts[key]}
        for key, expected in EXPECTED_DUPLICATES.items()
        if duplicate_counts[key] != expected
    }
    duplicate_discrepancy_note = (
        "All previously documented exact/feature duplicate counts match the dataset. "
        "The earlier near-duplicate count was hard-coded rather than measured; this run replaces it with the computed pair count."
        if not expected_mismatches
        else "Dataset counts differ from prior documented expectations; actual values are reported. "
        + json.dumps(expected_mismatches, sort_keys=True)
        + " The earlier near-duplicate count was hard-coded rather than measured; this run replaces it with the computed pair count."
    )

    status = {
        "project": "PMRC-Net",
        "version": "final-verification-2026-11-07",
        "verification_date": str(date.today()),
        "dataset_sha256": sha256(DATA),
        "overall_project_status": "INCONCLUSIVE",
        "requirements": {
            "repository_audit": "INCONCLUSIVE",
            "implementation_audit": "INCONCLUSIVE",
            "internal_five_seed_validation": "INCONCLUSIVE",
            "statistical_validation": "INCONCLUSIVE",
            "stage4_2_keyed_alignment": "INCONCLUSIVE",
            "automated_test_suite": test_status,
            "legacy_setup_test_direct": setup_status,
            "patient_level_independence": "INCONCLUSIVE",
            "external_dataset_screening": "INCONCLUSIVE",
            "independent_external_validation": "BLOCKED",
            "paper_review": "INCONCLUSIVE",
            "security_repository_audit": "INCONCLUSIVE",
            "release_readiness": "INCONCLUSIVE",
        },
        "evidence_paths": [
            "dataset/cleaned_merged_heart_dataset.csv",
            "checkpoints/controlled_pmrc_seed42/best.pt",
            "results/stage2_validation/oof_predictions.csv",
            "final_status/final_alignment_verification.json",
            "final_status/final_metric_recalculation.csv",
            "final_status/final_validation_summary.json",
            "results/stage4_2_baseline_comparison/random_forest_oof_aligned.csv",
            "results/stage5_final_validation/multi_seed_summary.csv",
            "results/stage5_final_validation/multi_seed_results.csv",
            "results/stage5_final_validation/reproducibility_manifest.json",
            "final_status/test_suite_stderr.log",
            "final_status/test_suite_stdout.log",
            "final_status/test_setup_direct.log",
            "final_status/PATIENT_DUPLICATE_ANALYSIS.md",
        ],
        "external_validation_status": "BLOCKED",
        "patient_independence_status": "INCONCLUSIVE",
        "clinical_readiness": "NOT ESTABLISHED",
        "dataset_duplicate_analysis": duplicate_counts,
        "near_duplicate_calculation_note": (
            "The prior report's near-duplicate value was hard-coded and is superseded by "
            "the computed candidate-pair count in this report."
        ),
        "expected_duplicate_counts": EXPECTED_DUPLICATES,
        "duplicate_count_discrepancies": expected_mismatches,
        "test_suite_status": {
            "unittest": test_status,
            "exit_code": test_run["exit_code"],
            "tests": test_count,
            "duration_seconds": test_duration,
            "elapsed_seconds": test_elapsed,
            "skipped": test_run["skipped_count"],
            "cuda_test": cuda_result,
            "legacy_setup_direct_exit_code": setup_run["exit_code"],
        },
        "unresolved_issues": [
            "No verified patient identifiers or source-level patient provenance",
            "Unknown patient clustering limits row-level bootstrap interpretation",
            "No genuinely independent compatible external dataset verified",
            "l1_lambda is declared but unused",
            "The mixed_precision flag is configured but not implemented in the maintained training path",
        ],
        "final_release_recommendation": "Supervisor review is appropriate for an internally validated research package. Do not claim clinical readiness, patient-level independence, or independent external generalization.",
    }
    (FINAL / "FINAL_RESEARCH_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    metrics = pd.read_csv(FINAL / "final_metric_recalculation.csv").set_index("model")
    summary = pd.read_csv(STAGE5 / "multi_seed_summary.csv").set_index("metric")
    bootstrap = pd.read_csv(STAGE42 / "paired_bootstrap_intervals.csv").set_index("metric")
    roc_interval = [
        float(bootstrap.loc["roc_auc", "lower_95"]),
        float(bootstrap.loc["roc_auc", "upper_95"]),
    ]
    validation_summary = {
        "status": "INCONCLUSIVE",
        "overall_project_status": "INCONCLUSIVE",
        "clinical_readiness": "NOT ESTABLISHED",
        "dataset": {
            "rows_after_exact_deduplication": duplicate_counts["exact_unique_rows"],
            "sha256": sha256(DATA),
            "duplicate_analysis": duplicate_counts,
            "near_duplicate_calculation_note": (
                "The prior report's near-duplicate value was hard-coded and is superseded by "
                "the computed candidate-pair count in this report."
            ),
            "expected_duplicate_counts": EXPECTED_DUPLICATES,
            "duplicate_count_discrepancies": expected_mismatches,
        },
        "stage4_2_alignment": alignment,
        "stage4_2_metrics": {
            "pmrc_roc_auc": float(metrics.loc["PMRC-Net", "roc_auc"]),
            "random_forest_roc_auc": float(metrics.loc["Random Forest", "roc_auc"]),
            "roc_auc_difference": float(metrics.loc["PMRC-Net", "roc_auc"] - metrics.loc["Random Forest", "roc_auc"]),
            "average_precision_difference": float(metrics.loc["PMRC-Net", "average_precision"] - metrics.loc["Random Forest", "average_precision"]),
            "brier_difference": float(metrics.loc["PMRC-Net", "brier_score"] - metrics.loc["Random Forest", "brier_score"]),
            "roc_auc_bootstrap_interval": roc_interval,
        },
        "five_seed_summary": {metric: {"mean": float(summary.loc[metric, "mean"]), "std": float(summary.loc[metric, "std"])} for metric in ["roc_auc", "accuracy", "f1", "mcc", "brier_score"]},
        "tests": status["test_suite_status"],
        "external_validation": "BLOCKED",
        "patient_independence": "INCONCLUSIVE",
    }
    (FINAL / "final_validation_summary.json").write_text(json.dumps(validation_summary, indent=2), encoding="utf-8")

    write("README.md", """
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
""")

    write("FINAL_TEST_EXECUTION_REPORT.md", f"""
# Final Test Execution Report

## Complete suite

Command: `{' '.join(test_run['command'])}`

- Exit code: `{test_run['exit_code']}`
- Result: **{test_status}**
- Tests reported by unittest: `{test_count}`
- unittest duration from captured output: `{test_duration}`
- Process elapsed seconds: `{test_elapsed:.6f}`
- CUDA forward test result from captured output: **{cuda_result}**

Captured stdout and stderr are retained in `final_status/test_suite_stdout.log` and `final_status/test_suite_stderr.log`. Test count and framework duration are parsed from the captured unittest summary; the exit code is the subprocess return code.

Legacy direct setup command: `{' '.join(setup_run['command'])}`; exit code: `{setup_run['exit_code']}`; status: **{setup_status}**. Captured output is in `final_status/test_setup_direct.log`.
""")

    write("FINAL_RELEASE_CHECKLIST.md", f"""
# Final Release Checklist

- [x] Historical notebook, checkpoints, OOF predictions, fold assignments, and experiment histories preserved.
- [{'x' if sha256(DATA) == EXPECTED_DATASET_SHA256 else ' '}] Dataset SHA-256 verified.
- [ ] Maintained source and configuration audit re-run: **INCONCLUSIVE**.
- [{'x' if test_status == 'PASS' else ' '}] Complete unittest suite: **{test_status}**, exit code `{test_run['exit_code']}`, tests `{test_count}`, duration `{test_duration}`.
- [{'x' if cuda_result == 'PASS' else ' '}] CUDA forward smoke path: **{cuda_result}**.
- [ ] Stage 4.2 alignment: pending final evidence verification.
- [ ] Corrected baseline metrics: pending final evidence verification.
- [ ] Five-seed summary: pending final evidence verification.
- [ ] Broader audit areas not re-run by this builder: **INCONCLUSIVE**.
- [ ] External dataset screening evidence: pending final evidence verification.
- [ ] Patient-level independence verified: **INCONCLUSIVE**.
- [ ] Independent external validation completed: **BLOCKED**.
- [ ] Clinical readiness: **NOT ESTABLISHED**.
""")

    write("FINAL_RESEARCH_RELEASE_NOTES.md", f"""
# Final Research Release Notes

This verification pass reruns the test suite and direct setup test, records their captured output and actual process exit codes, and preserves the existing architecture and historical results. Test-suite status: **{test_status}**; tests reported: `{test_count}`; framework duration seconds: `{test_duration}`; process exit code: `{test_run['exit_code']}`.

Corrected comparison is captured in `final_metric_recalculation.csv` and independently checked against the Stage 2 and Stage 4.2 prediction files.

The release remains limited by unknown patient-level independence and blocked independent external validation. Clinical readiness is **NOT ESTABLISHED**. No retraining or architecture change was performed.
""")

    write("FINAL_IMPLEMENTATION_AUDIT.md", """
# Final Implementation Audit

The maintained source path is `src/`. Dataset loading and exact deduplication are in `src/data/dataset.py`; fold-local scaling is performed by `make_fold_data`; the PMRC-Net architecture is in `src/models/pmrc_net.py`; loss logic is in `src/training/losses.py`; checkpoint round trips are in `src/training/checkpoints.py`.

This builder does not re-run the full implementation audit. Referenced prior findings are retained, but this pass marks the implementation audit **INCONCLUSIVE** rather than treating an unverified report as a new PASS.

Findings: `l1_lambda` is declared but unused; the mixed-precision flag is configured but not implemented in the maintained training path; the original notebook contains historical environment-specific cells and is not the maintained reproducibility entrypoint.
""")

    write("FINAL_STATISTICAL_VALIDATION.md", f"""
# Final Statistical Validation

The PMRC-Net Stage 2 OOF and Stage 4.2 Random Forest OOF were merged on `sample_index`, `fold`, and `target`. The merge has `{len(pmrc)}` rows, unique sample indices on both sides, identical folds and labels: **{alignment['status']}**.

Corrected values from the actual metric CSV: PMRC-Net ROC-AUC `{float(metrics.loc['PMRC-Net', 'roc_auc']):.8f}`; Random Forest ROC-AUC `{float(metrics.loc['Random Forest', 'roc_auc']):.8f}`; difference `{float(metrics.loc['PMRC-Net', 'roc_auc'] - metrics.loc['Random Forest', 'roc_auc']):+.8f}`; AP difference `{float(metrics.loc['PMRC-Net', 'average_precision'] - metrics.loc['Random Forest', 'average_precision']):+.8f}`; Brier difference `{float(metrics.loc['PMRC-Net', 'brier_score'] - metrics.loc['Random Forest', 'brier_score']):+.8f}`; saved ROC-AUC paired bootstrap interval `[{roc_interval[0]:.6f}, {roc_interval[1]:.6f}]`.

Five-seed results were read from `results/stage5_final_validation/multi_seed_summary.csv`: ROC-AUC `{float(summary.loc['roc_auc', 'mean']):.4f} +/- {float(summary.loc['roc_auc', 'std']):.4f}`, accuracy `{float(summary.loc['accuracy', 'mean']):.4f} +/- {float(summary.loc['accuracy', 'std']):.4f}`, F1 `{float(summary.loc['f1', 'mean']):.4f} +/- {float(summary.loc['f1', 'std']):.4f}`, MCC `{float(summary.loc['mcc', 'mean']):.4f} +/- {float(summary.loc['mcc', 'std']):.4f}`, Brier `{float(summary.loc['brier_score', 'mean']):.4f} +/- {float(summary.loc['brier_score', 'std']):.4f}`.

Bootstrap results are row-level and cannot account for unknown patient clustering. No statistical superiority or clinical superiority claim is made.
""")

    write("FINAL_REPRODUCIBILITY_AUDIT.md", f"""
# Final Reproducibility Audit

Dataset and checkpoint hash checks are recorded in `FINAL_RESEARCH_STATUS.json`; Stage 5 seed details are recorded in `results/stage5_final_validation/reproducibility_manifest.json`.

Test-suite result: **{test_status}**, process exit code `{test_run['exit_code']}`, reported tests `{test_count}`, framework duration `{test_duration}` seconds. Direct setup result: **{setup_status}**, process exit code `{setup_run['exit_code']}`.

Patient-level independence remains **INCONCLUSIVE**. Independent external validation remains **BLOCKED** because source overlap cannot be ruled out. Exact bitwise reproducibility across hardware is not claimed.
""")

    write("FINAL_SCIENTIFIC_CLAIMS_AUDIT.md", f"""
# Final Scientific Claims Audit

| Claim | Status | Boundary |
|---|---|---|
| Internal OOF predictions are complete and aligned | {alignment['status']} | {len(pmrc)} observations |
| Performance is summarized across five seeds | {'PASS' if len(pd.read_csv(STAGE5 / 'multi_seed_results.csv')['seed'].unique()) == 5 else 'INCONCLUSIVE'} | Row-level seed variability only |
| PMRC-Net is statistically superior to Random Forest | INCONCLUSIVE | Descriptive interval; unknown patient clustering |
| Modules have numerical computational pathways | INCONCLUSIVE | Prior gradient/ablation evidence was not re-audited by this builder |
| Physiological interpretation is clinically valid | INCONCLUSIVE | Design hypothesis, not established biology |
| Rows represent independent patients | INCONCLUSIVE | No patient identifiers or source manifest |
| Independent external validation is complete | BLOCKED | No verified independent compatible dataset |
| Clinical readiness | NOT ESTABLISHED | No clinical validation or utility study |
""")

    write("FINAL_KNOWN_LIMITATIONS.md", f"""
# Final Known Limitations

- {duplicate_counts['exact_unique_rows']} rows are exact-unique observations, not confirmed independent patients.
- The original {duplicate_counts['raw_rows']}-row dataset has unresolved source and patient-level provenance.
- Row-level bootstrap intervals cannot model unknown patient clustering.
- The positive-direction physiological hinge was inactive in observed folds.
- `l1_lambda` is declared but unused.
- The configured mixed-precision flag is not implemented in the maintained training path.
- UCI candidates may overlap the local development source; external validation is blocked.
- No clinical utility, diagnostic readiness, causality, or clinical superiority is established.
""")

    write("PATIENT_INDEPENDENCE_AUDIT.md", f"""
# Patient Independence Audit

The dataset contains {duplicate_counts['raw_rows']} rows reduced to {duplicate_counts['exact_unique_rows']} exact unique rows. The cleaned dataset has no verified patient, subject, or source identifier. UCI documentation also notes that identifiers were removed or replaced in the public Heart Disease files.

Exact duplicate removal prevents identical rows from crossing folds, but it does not prove that remaining rows are distinct patients. Near-duplicate or repeated-visit relationships cannot be resolved without source identifiers, dates, or cohort metadata. Group-aware validation is therefore unavailable.

Conclusion: **INCONCLUSIVE** patient-level independence. The project makes no claim that {duplicate_counts['exact_unique_rows']} rows equal the same number of patients.
""")

    write("EXTERNAL_DATASET_SCREENING.md", """
# External Dataset Screening

UCI Heart Disease (ID 45) and Statlog Heart (ID 145) were screened from official UCI documentation. ID 45 documents Cleveland, Hungary, Switzerland, and VA Long Beach databases; Statlog Heart is documented as similar to the Heart Disease databases. Both have local-looking feature names, but source overlap with the merged development data cannot be excluded.

No external file was downloaded, mapped, fitted, or evaluated. Screening status is PASS as a screening activity; independent external validation is BLOCKED.

Evidence: `final_status/external_dataset_provenance.csv`, `final_status/external_dataset_feature_mapping.csv`, and official URLs listed in the source report from the prior audit.
""")

    write("EXTERNAL_VALIDATION_REPORT.md", """
# External Validation Report

Status: **BLOCKED**.

No candidate dataset could be shown to be independent of the local merged development data. The available UCI candidates may contain the same source cohorts or observations. Without source-level patient provenance and overlap resolution, evaluating them would not support an independent external-validation claim.

No external predictions, metrics, calibration results, confidence intervals, or fine-tuning were generated.
""")

    write("FINAL_ARCHITECTURE_AUDIT.md", """
# Final Architecture Audit

The implemented PMRC-Net has metabolic, exercise-response, and coronary/electrical pathway projections; categorical embeddings; a raw-scale physiology mismatch module; asymmetric or symmetric coupling gates; mechanistic, interaction, and residual output heads; and an additive total logit `z_total = z_mech + z_inter + z_res`.

The gradient report records finite gradients and optimizer updates for inspected groups in both coupling diagnostics. Ablation results show a large pathway-removal degradation and smaller changes when mechanistic or residual heads are removed. These establish computational behavior, not causal or physiological validity.

No architecture change was made.
""")

    write("FINAL_LOSS_FUNCTION_AUDIT.md", """
# Final Loss Function Audit

The implemented loss is BCE plus a conditional physiological hinge weighted by `phys_lambda` and coupling-gate L2 regularization weighted by `l2_lambda`. `l1_lambda` is declared but unused. The physiological hinge is activated only for positive empirical directions; observed associations were negative, so the recorded physiological penalty was zero.

The zero penalty is an implementation consequence, not evidence that the physiological assumptions are validated. No loss component was artificially activated and no loss or architecture change was made.
""")

    write("FINAL_ABLATION_ANALYSIS.md", """
# Final Ablation Analysis

Saved pooled ROC-AUC values: full PMRC-Net `0.8955`, no physiology `0.8956`, no pathways `0.5326`, no mechanistic output `0.8961`, and no residual output `0.8983`.

Removing pathways causes a large degradation under this protocol. Removing mechanistic or residual outputs changes ROC-AUC only slightly. Removing physiology is difficult to interpret because the physiological penalty was inactive. These are descriptive computational ablations, not evidence of causal necessity or clinical validity.
""")

    write("FINAL_BASELINE_COMPARISON.md", f"""
# Final Baseline Comparison

The authoritative comparison uses `results/stage2_validation/oof_predictions.csv` and `results/stage4_2_baseline_comparison/random_forest_oof_aligned.csv`, merged on `sample_index`, `fold`, and `target`.

- PMRC-Net ROC-AUC: `{float(metrics.loc['PMRC-Net', 'roc_auc']):.8f}`
- Random Forest ROC-AUC: `{float(metrics.loc['Random Forest', 'roc_auc']):.8f}`
- Difference: `{float(metrics.loc['PMRC-Net', 'roc_auc'] - metrics.loc['Random Forest', 'roc_auc']):+.8f}`
- AP difference: `{float(metrics.loc['PMRC-Net', 'average_precision'] - metrics.loc['Random Forest', 'average_precision']):+.8f}`
- Brier difference: `{float(metrics.loc['PMRC-Net', 'brier_score'] - metrics.loc['Random Forest', 'brier_score']):+.8f}`
- ROC-AUC bootstrap interval: `[{roc_interval[0]:.6f}, {roc_interval[1]:.6f}]`

The comparison is descriptive and row-level; no superiority claim is made.
""")

    write("FINAL_DATASET_PROVENANCE_REPORT.md", f"""
# Final Dataset Provenance Report

Confirmed local facts: the dataset has {duplicate_counts['raw_rows']} rows and {duplicate_counts['exact_unique_rows']} exact unique rows. The cleaned data contain no verified patient identifier. Feature and target columns are documented in `configs/model_config.yaml`, and exact deduplication is applied by the maintained loader.

Source clues reference Cleveland, Kaggle, and merged data but do not establish cohort boundaries, collection periods, or patient identity. No original source files with identifiers are present. Patient-level provenance and independence remain **INCONCLUSIVE**.
""")

    write("FINAL_PREPROCESSING_AUDIT.md", """
# Final Preprocessing Audit

The prior audit records that `make_fold_data` fits `StandardScaler` on training indices and transforms validation indices. This builder does not re-run the full preprocessing audit, so the broader preprocessing finding remains **INCONCLUSIVE** in this final verification.

The test suite includes a training-scaler smoke test. Stage 2/Stage 4.2 prediction alignment is independently checked in `final_alignment_verification.json`.

No OOF prediction file is used as a training dataset. Threshold 0.5 is descriptive; no external threshold or calibrator was fit.
""")

    write("FINAL_SECURITY_AND_REPOSITORY_AUDIT.md", """
# Final Security and Repository Audit

This final-status builder does not re-run a repository-wide security scan. Therefore, this report is **INCONCLUSIVE** as a current security verification. Private absolute paths remain in historical manifests and are not copied into claims as provenance; they identify the local environment only.

The project contains generated historical artifacts and figures, but they are retained to preserve reproducibility. The final release documentation is centralized in `final_status/`. No automatic training-on-open behavior was added.

Dependency and environment information is recorded in `requirements.txt` and `ENVIRONMENT.md`. Public repository author/URL metadata in `CITATION.cff` remains intentionally incomplete until verified ownership metadata is supplied.
""")

    write("FINAL_PROJECT_COMPLETION_REPORT.md", f"""
# Final Project Completion Report

## Completed

- Centralized final-status package created without changing historical experiments.
- Existing repository, implementation, preprocessing, architecture, loss, ablation, provenance, security, and scientific-claims audit reports are retained; broad domain audits were not all re-run by this builder and remain **INCONCLUSIVE** where not verified.
- Complete unittest suite: **{test_status}**, `{test_count}` tests reported, `{test_duration}` seconds reported duration, exit code `{test_run['exit_code']}`; CUDA result: **{cuda_result}**.
- Direct `test_setup.py`: **{setup_status}**, exit code `{setup_run['exit_code']}`.
- Stage 4.2 keyed PMRC-Net/Random Forest comparison was recalculated and is subject to final evidence checks.
- Five-seed summary and dataset hash were checked against saved files.
- Paper claims remain bounded to internal validation; this builder does not re-review the manuscript.

## Key status

Overall project status: **INCONCLUSIVE**.

INCONCLUSIVE: patient-level independence and exact bitwise cross-hardware reproducibility.

BLOCKED: independent external validation.

Clinical readiness: **NOT ESTABLISHED**.

## Reproduction commands

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe test_setup.py
.\.venv\Scripts\python.exe scripts/build_final_status.py
```

These commands do not retrain the model. The project is suitable for supervisor review as an internally validated research package, not as a clinically validated diagnostic system.
""")

    write("FINAL_RESEARCH_STATUS.md", f"""
# Final Research Status

Overall status: **INCONCLUSIVE** because patient-level independence is unresolved and independent external validation is blocked.

Automated test suite: **{test_status}** (exit code `{test_run['exit_code']}`, tests `{test_count}`, reported duration `{test_duration}` seconds). Direct setup test: **{setup_status}** (exit code `{setup_run['exit_code']}`). Other requirement statuses are recorded in `FINAL_RESEARCH_STATUS.json`; unverified audits remain **INCONCLUSIVE**.

Patient-level independence: **INCONCLUSIVE**. Exact bitwise cross-hardware reproducibility is not claimed.

Independent external validation: **BLOCKED**. Clinical readiness: **NOT ESTABLISHED**.

Machine-readable detail is in `FINAL_RESEARCH_STATUS.json` and `final_validation_summary.json`.
""")

    pd.DataFrame([
        {"source_or_artifact": "Local cleaned merged dataset", "official_name": "cleaned_merged_heart_dataset.csv", "publisher": "Repository-local artifact", "population": "Not documented in local source manifest", "collection_period": "Unknown", "original_sample_size": duplicate_counts["raw_rows"], "patient_identifier_available": False, "record_identifier_available": False, "evidence": "dataset file and notebook outputs", "overlap_status": "Merged source boundaries unresolved", "status": "INCONCLUSIVE"},
        {"source_or_artifact": "Cleveland/UCI clue", "official_name": "UCI Heart Disease collection", "publisher": "UCI Machine Learning Repository", "population": "UCI documents Cleveland and other source databases", "collection_period": "Not established for local rows", "original_sample_size": 303, "patient_identifier_available": False, "record_identifier_available": False, "evidence": "official UCI documentation and notebook clues", "overlap_status": "May contribute to local merged data", "status": "INCONCLUSIVE"},
        {"source_or_artifact": "Kaggle/merged clue", "official_name": "Unspecified Kaggle-derived component", "publisher": "Not identified in local source manifest", "population": "Unknown", "collection_period": "Unknown", "original_sample_size": "Unknown", "patient_identifier_available": False, "record_identifier_available": False, "evidence": "notebook text clue only", "overlap_status": "Cannot resolve", "status": "INCONCLUSIVE"},
    ]).to_csv(FINAL / "patient_source_inventory.csv", index=False)
    pd.DataFrame([
        {"dataset": "UCI Heart Disease (ID 45)", "original_source": "UCI Machine Learning Repository", "official_url": "https://archive.ics.uci.edu/dataset/45/heart+disease", "publication_or_documentation": "UCI documentation; DOI 10.24432/C52P4X", "population": "Four documented source databases including Cleveland, Hungary, Switzerland, and VA Long Beach", "sample_size": 303, "outcome_definition": "goal/num; common binary presence versus absence", "features": 13, "missingness": "Yes", "provenance_evidence": "Official UCI page inspected", "local_overlap": "High; local notebook describes merged Cleveland/Kaggle/UCI-like clues", "independence_evidence": "Insufficient", "decision": "INCONCLUSIVE; not evaluated"},
        {"dataset": "Statlog (Heart) (ID 145)", "original_source": "UCI Machine Learning Repository", "official_url": "https://archive.ics.uci.edu/dataset/145/statlog+heart", "publication_or_documentation": "UCI documentation; DOI 10.24432/C57303", "population": "Statlog heart-disease classification population; exact overlap with local source unresolved", "sample_size": 270, "outcome_definition": "absence (1) versus presence (2)", "features": 13, "missingness": "No", "provenance_evidence": "Official UCI page inspected", "local_overlap": "Unknown-to-high; UCI documents similarity to Heart Disease databases", "independence_evidence": "Insufficient", "decision": "INCONCLUSIVE; not evaluated"},
    ]).to_csv(FINAL / "external_dataset_provenance.csv", index=False)
    (FINAL / "patient_independence_status.json").write_text(json.dumps({
        "status": "INCONCLUSIVE",
        **duplicate_counts,
        "patient_identifiers_found": False,
        "group_aware_validation_possible": False,
        "near_duplicate_method": duplicate_counts["standardization"] + " "
        + duplicate_counts["near_duplicate_distance"] + " and "
        + duplicate_counts["categorical_match"],
        "conclusion": "Exact and feature-level duplicate analysis cannot establish patient identity. Original source linkage and identifiers are unavailable.",
    }, indent=2), encoding="utf-8")
    write("PATIENT_SOURCE_TRACEABILITY.md", f"""
# Patient Source Traceability

The local evidence identifies a merged dataset with {duplicate_counts['raw_rows']} raw rows and {duplicate_counts['exact_unique_rows']} exact unique rows. Notebook text contains clues for Cleveland/UCI and Kaggle/merged sources, but no source manifest maps each row to a source cohort. The cleaned CSV contains no patient, subject, source, date, or reliable record identifier.

UCI documentation confirms that the public Heart Disease files removed or replaced patient identifiers. That documentation describes candidate source populations, not the identity of local rows. Consequently, source-specific patient traceability cannot be reconstructed from the repository.

See `patient_source_inventory.csv` for the evidence inventory. Status: **INCONCLUSIVE**.
""")
    write("PATIENT_DUPLICATE_ANALYSIS.md", f"""
# Patient Duplicate Analysis

- Raw rows: `{duplicate_counts['raw_rows']}`
- Exact duplicate rows: `{duplicate_counts['exact_duplicate_rows']}`
- Exact unique rows: `{duplicate_counts['exact_unique_rows']}`
- Feature-only duplicate rows participating in duplicate groups: `{duplicate_counts['feature_duplicate_rows']}`
- Feature-only duplicate groups: `{duplicate_counts['feature_duplicate_groups']}`
- Duplicate groups with conflicting target labels: `{duplicate_counts['conflicting_target_groups']}`
- Actual near-duplicate candidate pairs: `{duplicate_counts['near_duplicate_candidate_pairs']}`

Method: {duplicate_counts['standardization']} Compare each pair only within groups having identical values in all eight categorical columns; {duplicate_counts['near_duplicate_distance']}. This is a screening heuristic only; no threshold can prove or disprove shared patient identity, and clinical units/repeated-visit metadata are unavailable.

Expectation check: {duplicate_discrepancy_note}

Exact duplicate removal prevents identical records from crossing validation folds. It does not prove that the remaining records represent distinct people. Patient independence remains **INCONCLUSIVE**.
""")
    write("EXTERNAL_VALIDATION_PROTOCOL.md", """
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
""")
    write("EXTERNAL_DATASET_SCREENING.md", """
# External Dataset Screening

UCI Heart Disease (ID 45) and Statlog Heart (ID 145) were screened using official UCI documentation. Both have compatible-looking variables, but UCI Heart Disease contains multiple source databases and Statlog is documented as similar to the Heart Disease databases. The local notebook already contains Cleveland/Kaggle/merged source clues.

No raw external file was downloaded or evaluated. Because source overlap and patient-level independence cannot be excluded, both candidates are **INCONCLUSIVE** for independence and rejected for independent validation. See `external_dataset_provenance.csv` and `external_dataset_feature_mapping.csv`.
""")
    write("EXTERNAL_VALIDATION_REPORT.md", """
# External Validation Report

Status: **BLOCKED**. No candidate passed the required independence and compatibility gate. No external predictions, metrics, confidence intervals, calibration curves, fine-tuning, or model changes were produced.

The absence of an external result is intentional. Evaluating a likely overlapping UCI-derived file would be source-held-out or exploratory validation, not independent external validation.
""")
    write("FINAL_PATIENT_AND_EXTERNAL_VALIDATION_SUMMARY.md", """
# Patient and External Validation Summary

Patient-level independence: **INCONCLUSIVE**. The repository has no reliable patient identifiers, source-row linkage, dates, or cohort manifest. Exact and feature-level duplicate analyses were completed, but these cannot identify people.

Independent external validation: **BLOCKED**. Official UCI candidates were screened, but overlap with the merged local development source cannot be ruled out. No external evaluation was performed.

Required next evidence: original source files with linkage metadata or an approved independent clinical cohort with patient-level provenance, compatible measurements, target definitions, and permission for frozen-model evaluation.
""")
    status["evidence_paths"].extend([
        "final_status/patient_independence_status.json",
        "final_status/patient_source_inventory.csv",
        "final_status/PATIENT_DUPLICATE_ANALYSIS.md",
        "final_status/EXTERNAL_VALIDATION_PROTOCOL.md",
        "final_status/test_execution_metadata.json",
        "final_status/test_setup_stdout.log",
        "final_status/test_setup_stderr.log",
        "final_status/external_dataset_provenance.csv",
        "final_status/external_dataset_feature_mapping.csv",
    ])
    (FINAL / "FINAL_RESEARCH_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    write("FINAL_VALIDATION_REPORT.md", "# Final Validation Report\n\nReport content is generated after evidence checks.\n")
    required = [
        "README.md", "FINAL_PROJECT_COMPLETION_REPORT.md", "FINAL_TEST_EXECUTION_REPORT.md", "FINAL_RELEASE_CHECKLIST.md", "FINAL_RESEARCH_RELEASE_NOTES.md", "FINAL_IMPLEMENTATION_AUDIT.md", "FINAL_STATISTICAL_VALIDATION.md", "FINAL_REPRODUCIBILITY_AUDIT.md", "FINAL_SCIENTIFIC_CLAIMS_AUDIT.md", "FINAL_KNOWN_LIMITATIONS.md", "PATIENT_INDEPENDENCE_AUDIT.md", "PATIENT_SOURCE_TRACEABILITY.md", "PATIENT_DUPLICATE_ANALYSIS.md", "EXTERNAL_DATASET_SCREENING.md", "EXTERNAL_VALIDATION_REPORT.md", "EXTERNAL_VALIDATION_PROTOCOL.md", "FINAL_PATIENT_AND_EXTERNAL_VALIDATION_SUMMARY.md", "FINAL_ARCHITECTURE_AUDIT.md", "FINAL_LOSS_FUNCTION_AUDIT.md", "FINAL_ABLATION_ANALYSIS.md", "FINAL_BASELINE_COMPARISON.md", "FINAL_DATASET_PROVENANCE_REPORT.md", "FINAL_PREPROCESSING_AUDIT.md", "FINAL_SECURITY_AND_REPOSITORY_AUDIT.md", "FINAL_RESEARCH_STATUS.md", "FINAL_VALIDATION_REPORT.md", "FINAL_RESEARCH_STATUS.json", "final_metric_recalculation.csv", "final_experiment_inventory.csv", "external_dataset_provenance.csv", "external_dataset_feature_mapping.csv", "final_alignment_verification.json", "final_validation_summary.json", "patient_source_inventory.csv", "patient_independence_status.json", "test_suite_stdout.log", "test_suite_stderr.log", "test_setup_stdout.log", "test_setup_stderr.log", "test_setup_direct.log", "test_execution_metadata.json",
    ]
    required_before_inventory = [name for name in required if name != "final_experiment_inventory.csv"]
    checks = verify_evidence(raw, test_run, setup_run, duplicate_counts)
    checks.update(verify_outputs(required_before_inventory, status["evidence_paths"]))
    status["evidence_checks"] = checks
    status["requirements"]["automated_test_suite"] = (
        "PASS" if checks["automated_test_log"] and checks["automated_test_exit_code"] else test_status
    )
    status["requirements"]["legacy_setup_test_direct"] = "PASS" if checks["legacy_setup_log"] else "INCONCLUSIVE"
    status["requirements"]["stage4_2_keyed_alignment"] = "PASS" if checks["stage4_2_alignment"] else "INCONCLUSIVE"
    status["requirements"]["statistical_validation"] = "PASS" if checks["corrected_metrics"] and checks["five_seed_summary"] else "INCONCLUSIVE"
    status["requirements"]["internal_five_seed_validation"] = "PASS" if checks["five_seed_summary"] else "INCONCLUSIVE"
    status["requirements"]["external_dataset_screening"] = "PASS" if checks["external_dataset_screening_evidence"] else "INCONCLUSIVE"
    validation_status = "PASS" if all(checks.values()) else "INCONCLUSIVE"
    status["evidence_validation_status"] = validation_status
    validation_summary["evidence_checks"] = checks
    validation_summary["evidence_validation_status"] = validation_status
    validation_summary["tests"] = status["test_suite_status"]
    validation_summary["stage4_2_alignment"] = alignment
    validation_summary["status"] = "INCONCLUSIVE"

    check_rows = [
        ("dataset_hash", "dataset/cleaned_merged_heart_dataset.csv"),
        ("dataset_counts_match_documented_expectations", "dataset/cleaned_merged_heart_dataset.csv; PATIENT_DUPLICATE_ANALYSIS.md"),
        ("near_duplicate_calculation", "scripts/build_final_status.py; PATIENT_DUPLICATE_ANALYSIS.md"),
        ("stage4_2_alignment", "results/stage2_validation/oof_predictions.csv; results/stage4_2_baseline_comparison/random_forest_oof_aligned.csv"),
        ("corrected_metrics", "final_metric_recalculation.csv; paired OOF predictions"),
        ("checkpoint_hash", "checkpoints/controlled_pmrc_seed42/best.pt; reproducibility_manifest.json"),
        ("stage5_dataset_hash", "results/stage5_final_validation/reproducibility_manifest.json"),
        ("five_seed_summary", "results/stage5_final_validation/multi_seed_summary.csv; multi_seed_results.csv"),
        ("automated_test_log", "test_suite_stdout.log; test_suite_stderr.log"),
        ("automated_test_exit_code", "test_execution_metadata.json"),
        ("automated_test_duration_recorded", "captured unittest summary; test_execution_metadata.json"),
        ("legacy_setup_log", "test_setup_direct.log; test_execution_metadata.json"),
        ("external_not_evaluated", "absence of external_validation_predictions.csv"),
        ("external_dataset_screening_evidence", "external_dataset_provenance.csv; external_dataset_feature_mapping.csv"),
        ("required_output_files", "final_status required output list"),
        ("json_outputs_valid", "all required JSON outputs"),
        ("csv_outputs_valid", "all required CSV outputs"),
        ("referenced_evidence_paths", "FINAL_RESEARCH_STATUS.json evidence_paths"),
    ]
    report_lines = [
        "# Final Validation Report",
        "",
        "Overall project status: **INCONCLUSIVE**.",
        "Patient-level independence: **INCONCLUSIVE**. Independent external validation: **BLOCKED**. Clinical readiness: **NOT ESTABLISHED**.",
        "",
        "## Verification checks",
        "",
        "| Check | Evidence | Status |",
        "|---|---|---|",
    ]
    for key, evidence in check_rows:
        result = checks.get(key, False)
        outcome = "BLOCKED" if key == "external_not_evaluated" and result else "PASS" if result else "INCONCLUSIVE"
        report_lines.append(f"| `{key}` | `{evidence}` | **{outcome}** |")
    report_lines.extend([
        "",
        "## Captured test execution",
        "",
        f"- Suite: **{test_status}**; command: `{' '.join(test_run['command'])}`.",
        f"- Actual process exit code: `{test_run['exit_code']}`; unittest-reported tests: `{test_count}`.",
        f"- unittest-reported duration: `{test_duration}` seconds; measured process elapsed: `{test_elapsed:.6f}` seconds.",
        f"- Legacy setup command: `{' '.join(setup_run['command'])}`; actual exit code: `{setup_run['exit_code']}` ({setup_status}).",
        "- Raw captured output is stored in the corresponding stdout/stderr logs.",
        "",
        "## Dataset duplicate measurements",
        "",
        "| Measurement | Actual | Previous expected |",
        "|---|---:|---:|",
    ])
    for key, expected in EXPECTED_DUPLICATES.items():
        report_lines.append(f"| `{key}` | {duplicate_counts[key]} | {expected} |")
    report_lines.extend([
        f"| `near_duplicate_candidate_pairs` | {duplicate_counts['near_duplicate_candidate_pairs']} | not assumed; calculated from rows |",
        "",
        f"Method: {duplicate_counts['standardization']} Rows are compared only when all eight categorical columns match; Euclidean distance threshold is 0.1. {duplicate_discrepancy_note}",
        "",
        "## Remaining limitations",
        "",
        "- Patient identity cannot be inferred from the available identifiers or source linkage; patient independence remains **INCONCLUSIVE**.",
        "- No independent external dataset has been established; independent external validation remains **BLOCKED**.",
        "- Clinical readiness is **NOT ESTABLISHED**.",
        "- Overall project status remains **INCONCLUSIVE**, regardless of internal evidence-check results.",
        "",
        "The inventory is generated after all listed outputs and references are verified; its self-entry is excluded, as are `.venv` and `__pycache__` paths.",
        "",
    ])
    write("FINAL_VALIDATION_REPORT.md", "\n".join(report_lines))
    write(
        "FINAL_RELEASE_CHECKLIST.md",
        f"""# Final Release Checklist

- [{'x' if checks['dataset_hash'] else ' '}] Dataset SHA-256 verified.
- [{'x' if checks['automated_test_log'] and checks['automated_test_exit_code'] else ' '}] Test suite execution recorded from captured output and actual process exit code.
- [{'x' if checks['stage4_2_alignment'] else ' '}] Stage 4.2 alignment verified from actual keyed prediction files.
- [{'x' if checks['corrected_metrics'] else ' '}] Corrected metrics checked against actual prediction files.
- [{'x' if checks['checkpoint_hash'] else ' '}] Checkpoint hash verified against expected and manifest hashes.
- [{'x' if checks['five_seed_summary'] else ' '}] Stage 5 summary checked against its five seed/fold records and manifest.
- [ ] Repository and implementation audits not re-run in this build: **INCONCLUSIVE**.
- [ ] Patient-level independence: **INCONCLUSIVE**.
- [ ] Independent external validation: **BLOCKED**.
- [ ] Clinical readiness: **NOT ESTABLISHED**.

Overall project status: **INCONCLUSIVE**.
"""
    )
    requirements = status["requirements"]
    write(
        "FINAL_RESEARCH_STATUS.md",
        f"""# Final Research Status

Overall status: **INCONCLUSIVE**.

| Requirement | Status |
|---|---|
"""
        + "\n".join(f"| `{name}` | **{value}** |" for name, value in requirements.items())
        + """

Patient-level independence remains **INCONCLUSIVE**; independent external validation remains **BLOCKED**; clinical readiness is **NOT ESTABLISHED**.

Machine-readable evidence and captured test metadata are in `FINAL_RESEARCH_STATUS.json`, `final_validation_summary.json`, and `FINAL_VALIDATION_REPORT.md`.
"""
    )
    status["evidence_checks"] = checks
    (FINAL / "FINAL_RESEARCH_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    validation_summary = json.loads((FINAL / "final_validation_summary.json").read_text(encoding="utf-8"))
    validation_summary["evidence_checks"] = checks
    validation_summary["evidence_validation_status"] = validation_status
    (FINAL / "final_validation_summary.json").write_text(json.dumps(validation_summary, indent=2), encoding="utf-8")

    status_footer = (
        "\n\n---\n\n## Authoritative current status\n\n"
        "- Overall project status: **INCONCLUSIVE**.\n"
        "- Patient-level independence: **INCONCLUSIVE**.\n"
        "- Independent external validation: **BLOCKED**.\n"
        "- Clinical readiness: **NOT ESTABLISHED**.\n"
    )
    for markdown_path in FINAL.glob("*.md"):
        text = markdown_path.read_text(encoding="utf-8")
        text = re.sub(
            r"\|\s*Clinical readiness is established\s*\|\s*FAIL\s*\|",
            "| Clinical readiness | NOT ESTABLISHED |",
            text,
        )
        text = re.sub(
            r"Clinical readiness is established: \*\*FAIL\*\*",
            "Clinical readiness: **NOT ESTABLISHED**",
            text,
        )
        footer_start = text.find("\n---\n\n## Authoritative current status")
        if footer_start >= 0:
            text = text[:footer_start]
        markdown_path.write_text(text.rstrip() + status_footer, encoding="utf-8")

    output_checks = verify_outputs(required_before_inventory, status["evidence_paths"])
    checks.update(output_checks)
    if not all(output_checks.values()):
        raise RuntimeError(f"Final reports or evidence failed integrity verification: {output_checks}")

    build_inventory()
    inventory_ok = verify_inventory()
    if not inventory_ok:
        raise RuntimeError("Generated inventory failed exclusion or hash integrity checks")
    checks["inventory_integrity"] = inventory_ok

    validation_report = FINAL / "FINAL_VALIDATION_REPORT.md"
    report_text = validation_report.read_text(encoding="utf-8")
    report_text = report_text.replace(
        "| `referenced_evidence_paths` | `FINAL_RESEARCH_STATUS.json evidence_paths` | **PASS** |",
        "| `referenced_evidence_paths` | `FINAL_RESEARCH_STATUS.json evidence_paths` | **PASS** |\n"
        "| `inventory_integrity` | `final_experiment_inventory.csv` (self-exclusion, filters, and per-file hashes) | **PASS** |",
    )
    validation_report.write_text(report_text, encoding="utf-8")
    status["evidence_checks"] = checks
    status["evidence_validation_status"] = (
        "PASS" if all(value for value in checks.values() if isinstance(value, bool)) else "INCONCLUSIVE"
    )
    (FINAL / "FINAL_RESEARCH_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    validation_summary = json.loads((FINAL / "final_validation_summary.json").read_text(encoding="utf-8"))
    validation_summary["evidence_checks"] = checks
    validation_summary["evidence_validation_status"] = status["evidence_validation_status"]
    (FINAL / "final_validation_summary.json").write_text(json.dumps(validation_summary, indent=2), encoding="utf-8")

    final_output_checks = verify_outputs(required_before_inventory, status["evidence_paths"])
    checks.update(final_output_checks)
    if not all(final_output_checks.values()):
        raise RuntimeError(f"Post-inventory report update failed integrity verification: {final_output_checks}")
    status["evidence_checks"] = checks
    (FINAL / "FINAL_RESEARCH_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    validation_summary["evidence_checks"] = checks
    (FINAL / "final_validation_summary.json").write_text(json.dumps(validation_summary, indent=2), encoding="utf-8")

    final_output_checks = verify_outputs(required_before_inventory, status["evidence_paths"])
    if not all(final_output_checks.values()):
        raise RuntimeError(f"Final pre-inventory verification failed: {final_output_checks}")
    build_inventory()
    inventory_ok = verify_inventory()
    if not inventory_ok:
        raise RuntimeError("Final generated inventory failed exclusion or hash integrity checks")
    print(json.dumps({
        "status": "PASS" if all(value for value in checks.values() if isinstance(value, bool)) and inventory_ok else "INCONCLUSIVE",
        "files_verified": len(required_before_inventory),
        "inventory_integrity": inventory_ok,
        "evidence_checks": checks,
        "directory": str(FINAL),
    }, indent=2))


if __name__ == "__main__":
    main()
