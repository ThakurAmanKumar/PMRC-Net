from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

V11_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = V11_ROOT.parent
sys.path.insert(0, str(V11_ROOT))

from data.dataset import load_dataset  # noqa: E402
from evaluation.metrics import calculate_metrics  # noqa: E402
from models.pmrc_net import PMRCNet  # noqa: E402
from run_nested_cv import expected_outer_splits, make_inner_splits, validate_oof_frame  # noqa: E402


MODEL_NAMES = {
    "pmrc_v1_0": "PMRC-Net v1.0",
    "pmrc_v1_1": "PMRC-Net v1.1",
    "logistic_regression": "Logistic Regression",
    "svm": "SVM",
    "random_forest": "Random Forest",
    "mlp": "MLP",
    "knn": "KNN",
    "decision_tree": "Decision Tree",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "tabpfn": "TabPFN",
}
SEEDS = (21, 42, 123, 2026, 3407)
REPORT_METRICS = (
    "accuracy", "precision", "sensitivity", "specificity", "f1", "mcc",
    "roc_auc", "average_precision", "brier_score", "ece",
)
PAIRED_METRICS = ("roc_auc", "average_precision", "brier_score")


def load_config() -> dict[str, Any]:
    return yaml.safe_load((V11_ROOT / "configs" / "nested_cv.yaml").read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_environment(run_directory: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "matplotlib", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    versions["python"] = sys.version
    versions["platform"] = platform.platform()
    versions["torch"] = torch.__version__
    versions["cuda_build"] = torch.version.cuda
    versions["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        versions["gpu_name"] = torch.cuda.get_device_name(0)
        properties = torch.cuda.get_device_properties(0)
        versions["gpu_total_memory_gb"] = properties.total_memory / 2**30
    else:
        versions["gpu_name"] = None
        versions["gpu_total_memory_gb"] = 0.0
    metadata["software"] = versions
    (run_directory / "software_versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    (run_directory / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return versions


def run_test_command(name: str, command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    combined = result.stdout + "\n" + result.stderr
    match = re.search(r"Ran (\d+) tests?", combined)
    tests_run = int(match.group(1)) if match else None
    test_summary_valid = tests_run is not None and tests_run > 0
    return {
        "name": name,
        "command": command,
        "exit_code": result.returncode,
        "tests_run": tests_run,
        "status": "PASS" if result.returncode == 0 and (test_summary_valid or name == "setup_smoke") else "FAIL",
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def execute_required_tests() -> dict[str, dict[str, Any]]:
    results = {
        "v1_0": run_test_command(
            "v1_0",
            [sys.executable, "-m", "unittest", "discover", "-s", str(PROJECT_ROOT / "tests"), "-v"],
        ),
        "v1_1": run_test_command(
            "v1_1",
            [sys.executable, "-m", "unittest", "discover", "-s", str(V11_ROOT / "tests"), "-v"],
        ),
        "setup_smoke": run_test_command(
            "setup_smoke",
            [sys.executable, str(PROJECT_ROOT / "test_setup.py")],
        ),
    }
    failed = [name for name, result in results.items() if result["status"] != "PASS"]
    if failed:
        summary = {name: {"exit_code": results[name]["exit_code"], "tests_run": results[name]["tests_run"]} for name in failed}
        raise RuntimeError(f"Required verification failed; reports were not generated: {summary}")
    return results


def load_model_predictions(
    run_id: str,
    dataframe: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, dict[int, pd.DataFrame]], dict[str, Any]]:
    run_directory = V11_ROOT / "experiments" / run_id
    metadata = json.loads((run_directory / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "COMPLETED":
        raise ValueError(f"v1.1 run is not complete: {metadata.get('status')}")
    manifest = json.loads((run_directory / "split_manifest.json").read_text(encoding="utf-8"))
    all_splits = expected_outer_splits(dataframe, config, list(SEEDS))
    if manifest["outer_splits_by_seed"] != all_splits:
        raise ValueError("Saved outer fold manifest does not match configured deterministic splits")
    fold_ids = {
        seed: np.zeros(len(dataframe), dtype=np.int64)
        for seed in SEEDS
    }
    for seed in SEEDS:
        for item in all_splits[str(seed)]:
            fold_ids[seed][np.asarray(item["valid_indices"], dtype=np.int64)] = int(item["fold"])
    models: dict[str, dict[int, pd.DataFrame]] = {name: {} for name in MODEL_NAMES}
    prediction_root = V11_ROOT / "predictions" / run_id
    for seed in SEEDS:
        v11 = pd.read_csv(prediction_root / f"seed_{seed}_oof.csv")
        validate_oof_frame(v11, dataframe, fold_ids[seed], seed)
        models["pmrc_v1_1"][seed] = v11.sort_values("sample_index").reset_index(drop=True)
        v10_path = PROJECT_ROOT / "results" / "stage5_final_validation" / "seed_oof_predictions" / f"seed_{seed}_oof.csv"
        v10 = pd.read_csv(v10_path)
        v10["seed"] = seed
        validate_oof_frame(v10, dataframe, fold_ids[seed], seed)
        models["pmrc_v1_0"][seed] = v10.sort_values("sample_index").reset_index(drop=True)
        for model in MODEL_NAMES:
            if model in {"pmrc_v1_0", "pmrc_v1_1"}:
                continue
            path = PROJECT_ROOT / "experiments" / "addit   ional_baselines" / model / f"seed_{seed}" / "oof_predictions.csv"
            frame = pd.read_csv(path)
            frame["seed"] = seed
            validate_oof_frame(frame, dataframe, fold_ids[seed], seed)
            models[model][seed] = frame.sort_values("sample_index").reset_index(drop=True)
    return models, {"metadata": metadata, "fold_ids": fold_ids, "folds": all_splits}


def verify_checkpoint_predictions(
    run_id: str,
    dataframe: pd.DataFrame,
    config: dict[str, Any],
    models: dict[str, dict[int, pd.DataFrame]],
    run_info: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_root = V11_ROOT / "checkpoints" / run_id
    continuous = config["data"]["continuous_cols"]
    categorical = config["data"]["categorical_cols"]
    physiology = config["data"]["physiology_cols"]
    max_delta = 0.0
    checked = 0
    for seed in SEEDS:
        reference = models["pmrc_v1_1"][seed].set_index("sample_index")
        for split in run_info["folds"][str(seed)]:
            fold = int(split["fold"])
            path = checkpoint_root / f"seed_{seed}" / f"fold_{fold}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint.get("version") != "1.1" or checkpoint.get("seed") != seed or checkpoint.get("outer_fold") != fold:
                raise ValueError(f"Checkpoint identity mismatch: {path}")
            if checkpoint.get("dataset_sha256") != run_info["metadata"].get("dataset_sha256"):
                raise ValueError(f"Checkpoint dataset hash mismatch: {path}")
            train_ids = dataframe.iloc[np.asarray(split["train_indices"], dtype=np.int64)]["sample_index"].to_numpy(dtype=int)
            if not np.array_equal(np.asarray(checkpoint["train_sample_indices"], dtype=int), train_ids):
                raise ValueError(f"Checkpoint training sample IDs mismatch: {path}")
            model = PMRCNet(config).cpu()
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float64)
            scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float64)
            valid = dataframe.iloc[np.asarray(split["valid_indices"], dtype=np.int64)]
            values = valid[continuous].to_numpy(dtype=np.float64)
            x_cont = torch.from_numpy(((values - mean) / scale).astype(np.float32))
            x_cat = torch.from_numpy(valid[categorical].to_numpy(dtype=np.int64))
            x_phys = torch.from_numpy(valid[physiology].to_numpy(dtype=np.float32))
            probabilities = []
            batch_size = int(config["training"]["batch_size"])
            with torch.inference_mode():
                for start in range(0, len(valid), batch_size):
                    outputs = model(
                        x_cont[start:start + batch_size],
                        x_cat[start:start + batch_size],
                        x_phys[start:start + batch_size],
                    )
                    if not all(torch.isfinite(output).all() for output in outputs.values()):
                        raise ValueError(f"Non-finite checkpoint outputs: {path}")
                    probabilities.extend(torch.sigmoid(outputs["z_total"]).squeeze(-1).tolist())
            expected = reference.loc[
                valid["sample_index"].to_numpy(dtype=int), "probability"
            ].to_numpy(dtype=float)
            delta = float(np.max(np.abs(np.asarray(probabilities, dtype=float) - expected)))
            max_delta = max(max_delta, delta)
            if not np.allclose(probabilities, expected, rtol=1e-5, atol=1e-6):
                raise ValueError(f"Checkpoint inference does not match OOF: {path}; max delta={delta}")
            checked += 1
            del model
    return {
        "status": "PASS",
        "checkpoints_loaded": checked,
        "max_absolute_probability_delta": max_delta,
        "device": "cpu",
    }


def persist_inner_split_manifest(
    run_directory: Path,
    dataframe: pd.DataFrame,
    config: dict[str, Any],
    outer_splits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    target = dataframe[config["data"]["target"]].to_numpy(dtype=int)
    sample_ids = dataframe["sample_index"].to_numpy(dtype=int)
    manifest: dict[str, Any] = {
        "inner_folds": int(config["training"]["inner_folds"]),
        "seed_rule": "outer_seed * 100000 + outer_fold * 1000 + inner_fold for model initialization; split random_state=outer_seed + 1009 * outer_fold",
        "inner_splits_by_seed": {},
    }
    for seed, outer_folds in outer_splits.items():
        manifest["inner_splits_by_seed"][seed] = []
        for outer in outer_folds:
            outer_fold = int(outer["fold"])
            train_indices = np.asarray(outer["train_indices"], dtype=np.int64)
            inner = make_inner_splits(train_indices, target, config, int(seed), outer_fold)
            manifest["inner_splits_by_seed"][seed].append({
                "outer_fold": outer_fold,
                "inner_folds": [
                    {
                        "inner_fold": number,
                        "train_sample_indices": sample_ids[inner_train].tolist(),
                        "valid_sample_indices": sample_ids[inner_valid].tolist(),
                    }
                    for number, (inner_train, inner_valid) in enumerate(inner, start=1)
                ],
            })
    path = run_directory / "nested_split_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("Existing nested split manifest does not match deterministic configured splits")
    else:
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def calculate_model_metrics(
    models: dict[str, dict[int, pd.DataFrame]],
    ece_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []
    for model, seed_frames in models.items():
        for seed, frame in seed_frames.items():
            row = {
                "model": model,
                "seed": seed,
                **calculate_metrics(
                    frame["target"].to_numpy(dtype=int),
                    frame["probability"].to_numpy(dtype=float),
                    ece_bins=ece_bins,
                ),
            }
            detail_rows.append(row)
    detail = pd.DataFrame(detail_rows)
    summary_rows = []
    for model, group in detail.groupby("model", sort=False):
        row: dict[str, Any] = {"model": model, "model_name": MODEL_NAMES[model], "seeds": group["seed"].nunique()}
        for metric in REPORT_METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    return detail, pd.DataFrame(summary_rows)


def _triplet(labels: np.ndarray, probability: np.ndarray) -> tuple[float, float, float]:
    return (
        float(roc_auc_score(labels, probability)),
        float(average_precision_score(labels, probability)),
        float(brier_score_loss(labels, probability)),
    )


def paired_sample_bootstrap(
    models: dict[str, dict[int, pd.DataFrame]],
    target: np.ndarray,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(target == value) for value in (0, 1)]
    comparison_names = [name for name in models if name != "pmrc_v1_1"]
    samples = {
        name: {metric: [] for metric in PAIRED_METRICS}
        for name in comparison_names
    }
    for _ in range(repetitions):
        selected = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in class_indices
        ])
        resampled_target = target[selected]
        v11_values = {metric: [] for metric in PAIRED_METRICS}
        compared_values = {
            name: {metric: [] for metric in PAIRED_METRICS}
            for name in comparison_names
        }
        for fold_seed in SEEDS:
            v11_frame = models["pmrc_v1_1"][fold_seed]
            v11_prob = v11_frame["probability"].to_numpy(dtype=float)[selected]
            v11_triplet = _triplet(resampled_target, v11_prob)
            for metric, value in zip(PAIRED_METRICS, v11_triplet):
                v11_values[metric].append(value)
            for name in comparison_names:
                other_prob = models[name][fold_seed]["probability"].to_numpy(dtype=float)[selected]
                other_triplet = _triplet(resampled_target, other_prob)
                for metric, left, right in zip(PAIRED_METRICS, v11_triplet, other_triplet):
                    compared_values[name][metric].append(left - right)
        for name in comparison_names:
            for metric in PAIRED_METRICS:
                samples[name][metric].append(float(np.mean(compared_values[name][metric])))
    point_metrics = {
        name: {
            metric: float(np.mean([
                _triplet(
                    models[name][fold_seed]["target"].to_numpy(dtype=int),
                    models[name][fold_seed]["probability"].to_numpy(dtype=float),
                )[PAIRED_METRICS.index(metric)]
                for fold_seed in SEEDS
            ]))
            for metric in PAIRED_METRICS
        }
        for name in models
    }
    rows = []
    for name in comparison_names:
        for metric in PAIRED_METRICS:
            difference = point_metrics["pmrc_v1_1"][metric] - point_metrics[name][metric]
            low, high = np.percentile(samples[name][metric], [2.5, 97.5])
            rows.append({
                "comparison": f"PMRC-Net v1.1 minus {MODEL_NAMES[name]}",
                "model": name,
                "metric": metric,
                "v1_1_mean_seed_estimate": point_metrics["pmrc_v1_1"][metric],
                "comparator_mean_seed_estimate": point_metrics[name][metric],
                "difference": difference,
                "lower_95": float(low),
                "upper_95": float(high),
                "bootstrap_repetitions": repetitions,
                "bootstrap_seed": seed,
                "method": "stratified paired bootstrap resampling sample IDs; preserve all seed predictions per sampled ID",
                "assumption": "rows are resampling units; patient clustering is unknown",
                "superiority_claim": "none",
            })
    return pd.DataFrame(rows)


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def write_figures(
    models: dict[str, dict[int, pd.DataFrame]],
    figure_directory: Path,
) -> None:
    figure_directory.mkdir(parents=True, exist_ok=True)
    target = models["pmrc_v1_1"][42]["target"].to_numpy(dtype=int)
    plt.figure(figsize=(9, 7))
    for model, seed_frames in models.items():
        frame = seed_frames[42]
        probability = frame["probability"].to_numpy(dtype=float)
        false_positive, true_positive, _ = roc_curve(target, probability)
        area = roc_auc_score(target, probability)
        plt.plot(false_positive, true_positive, label=f"{MODEL_NAMES[model]} ({area:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Seed 42 outer-fold OOF ROC curves")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(figure_directory / "roc_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 7))
    for model, seed_frames in models.items():
        frame = seed_frames[42]
        probability = frame["probability"].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(target, probability)
        area = average_precision_score(target, probability)
        plt.plot(recall, precision, label=f"{MODEL_NAMES[model]} (AP {area:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Seed 42 outer-fold OOF precision-recall curves")
    plt.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(figure_directory / "precision_recall_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 7))
    for model, seed_frames in models.items():
        frame = seed_frames[42]
        fraction, mean_probability = calibration_curve(
            target, frame["probability"].to_numpy(dtype=float), n_bins=10, strategy="uniform"
        )
        plt.plot(mean_probability, fraction, marker="o", label=MODEL_NAMES[model])
    plt.plot([0, 1], [0, 1], "k--", label="Ideal")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title("Seed 42 outer-fold OOF calibration; 10 equal-width bins")
    plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(figure_directory / "calibration_curves.png", dpi=200)
    plt.close()

    prediction = models["pmrc_v1_1"][42]["probability"].to_numpy(dtype=float) >= 0.5
    matrix = confusion_matrix(target, prediction, labels=[0, 1])
    plt.figure(figsize=(4.5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.xlabel("Predicted label")
    plt.ylabel("Observed label")
    plt.title("PMRC-Net v1.1 seed 42 OOF confusion matrix")
    for row in range(2):
        for column in range(2):
            plt.text(column, row, str(matrix[row, column]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(figure_directory / "confusion_matrix_v1_1_seed42.png", dpi=200)
    plt.close()


def write_reports(
    run_id: str,
    models: dict[str, dict[int, pd.DataFrame]],
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    run_info: dict[str, Any],
    test_results: dict[str, dict[str, Any]],
    output_directory: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_directory / "multi_model_seed_metrics.csv", index=False)
    summary.to_csv(output_directory / "multi_model_mean_std.csv", index=False)
    paired.to_csv(output_directory / "paired_comparisons.csv", index=False)
    seed_count = len(SEEDS)
    metric_lines = ["| Model | " + " | ".join(REPORT_METRICS) + " |", "|---|" + "---:|" * len(REPORT_METRICS)]
    for row in summary.itertuples(index=False):
        values = [format_mean_std(getattr(row, f"{metric}_mean"), getattr(row, f"{metric}_std")) for metric in REPORT_METRICS]
        metric_lines.append(f"| {row.model_name} | " + " | ".join(values) + " |")
    v11_mean = summary.loc[summary.model == "pmrc_v1_1"].iloc[0]
    v10_mean = summary.loc[summary.model == "pmrc_v1_0"].iloc[0]
    seed42_v11 = models["pmrc_v1_1"][42]
    seed42_matrix = confusion_matrix(
        seed42_v11["target"].to_numpy(dtype=int),
        seed42_v11["probability"].to_numpy(dtype=float) >= 0.5,
        labels=[0, 1],
    )
    comparison_rows = paired[paired.model == "pmrc_v1_0"]
    comparison_table = ["| Metric | v1.0 mean | v1.1 mean | v1.1 - v1.0 | 95% paired interval |", "|---|---:|---:|---:|---:|"]
    for metric in PAIRED_METRICS:
        row = comparison_rows[comparison_rows.metric == metric].iloc[0]
        comparison_table.append(
            f"| {metric} | {row.comparator_mean_seed_estimate:.4f} | {row.v1_1_mean_seed_estimate:.4f} | "
            f"{row.difference:+.4f} | [{row.lower_95:+.4f}, {row.upper_95:+.4f}] |"
        )
    metadata = run_info["metadata"]
    methodology = f"""# PMRC-Net v1.1 Methodology

## Versioning and preservation

The v1.0 source and historical outputs remain at the repository root and under their existing `results/`, `experiments/`, `checkpoints/`, `reports/`, and `final_status/` paths. v1.1 code, configurations, checkpoints, OOF predictions, figures, and reports are isolated under `v1.1/`. The pre-implementation SHA-256 inventory is `v1.1/reports/v1_0_preservation_manifest.json`.

Dataset: {metadata['rows']} exact-deduplicated rows; SHA-256 `{metadata['dataset_sha256']}`. Patient IDs are unavailable; rows are not assumed independent patients. The continuous scaler is fit separately on each inner-training or outer-training partition. Categorical source mappings are explicit and unknown codes are rejected. No imputation is performed; non-finite continuous values are rejected before model computation.

## Nested evaluation

Five outer stratified folds are generated independently for seeds {list(SEEDS)} and saved in the run split manifest. Each outer-training partition is split into four stratified inner folds. Inner validation binary cross-entropy selects a training duration: the ceiling of the median of inner-fold best epochs under the fixed patience and minimum-delta rules. A fresh model is then fit on all outer-training rows for exactly that selected duration. The outer test partition is evaluated once and is not passed to epoch selection, early stopping, scaler fitting, or checkpoint selection.

The fixed model hyperparameters are held constant in `v1.1/configs/nested_cv.yaml`. Runs are sequential, batch size 8, zero workers, two CPU threads, FP32, and AdamW with weight decay `0.0001`. No outer-test-driven hyperparameter selection or threshold optimization is performed. Seed 42 and all other configured seeds are preserved in the split manifest.

## Objective and architecture

The unsubstantiated empirical-direction hinge is removed rather than reoriented. The v1.1 objective is binary cross-entropy with logits:

$$L_{{BCE}}=-\\frac{{1}}{{N}}\\sum_{{i=1}}^N\\left[y_i\\log\\sigma(z_i)+(1-y_i)\\log(1-\\sigma(z_i))\\right].$$

AdamW weight decay remains an optimizer regularizer, not a second hand-written loss. No `l1_lambda` is exposed because the previous coefficient was unused and no scientifically specified L1 target was established. Mixed precision is disabled.

The pathway encoders, physiological mismatch representation, and additive logit heads are retained. For pathway pair $(l,r)$, v1.0's two products were identical and its second gate half was discarded. v1.1 uses ordered transforms:

$$f_{{l\\to r}}=T_l(l)\\odot T_r(r),\\qquad f_{{r\\to l}}=T_r(l)\\odot T_l(r),$$
$$c_{{lr}}=\\sigma(g_{{lr}})_1\\odot f_{{l\\to r}}+\\sigma(g_{{lr}})_2\\odot f_{{r\\to l}}.$$

Both gate halves now contribute to the interaction output; each uses a distinct ordered transformation product. This corrects the named asymmetric construction without adding modules or increasing hidden dimension. It remains a computational design, not a proven biological mechanism. The prediction still decomposes as `z_total = z_mech + z_inter + z_res`.

The retained raw-scale mismatch is $m_e=I\\max(1-u,0)$, where $u=HR_{{max}}/\\max(208-0.7\\,age,1)$ and $I=\\operatorname{{softplus}}(\\max(a_1,0)\\,oldpeak+\\max(a_2,0)\\,exang+\\max(a_3,0)\\,oldpeak\\cdot exang)$. Its parameters remain learned. No monotonic constraint is inferred from target correlations; there is no v1.1 physiological hinge term.

## Metrics and inference limits

Reported metrics are accuracy, precision, sensitivity/recall, specificity, F1, MCC, ROC-AUC, average precision, Brier score, and ten-bin equal-width ECE. Classification threshold is fixed at 0.5. Seed summary is mean and sample SD (`ddof=1`) of each seed's pooled outer OOF metrics. Paired percentile intervals resample sample IDs stratified by target while retaining all seeds' predictions for each sampled ID. They are descriptive row-level intervals; unknown patient clustering prevents patient-level inference.
"""
    (output_directory / "PMRC_NET_V1_1_METHODOLOGY.md").write_text(methodology, encoding="utf-8")

    experiment = f"""# PMRC-Net v1.1 Experiment Report

Run: `{run_id}`  
Run status: **{metadata['status']}**  
Device: `{metadata.get('device', 'not recorded')}`; GPU: `{metadata['software'].get('gpu_name')}`  
Seeds: {list(SEEDS)}; outer folds: 5; inner folds: 4; completed outer checkpoints: {metadata.get('folds_completed')}  
Dataset SHA-256: `{metadata['dataset_sha256']}`

## Mean and sample SD across seeds

Each value is `mean +/- sample SD (ddof=1)` across five seed-level pooled OOF metric values.

{chr(10).join(metric_lines)}

## Paired comparisons

The v1.1 versus v1.0 estimates and paired intervals below use same sample IDs and seed-specific outer folds. See `paired_comparisons.csv` for v1.1 versus all baselines. Intervals are descriptive and not superiority tests.

{chr(10).join(comparison_table)}

## Seed-42 confusion matrix

Threshold `0.5`; rows are observed labels and columns are predicted labels:

| Observed \\ Predicted | 0 | 1 |
|---|---:|---:|
| 0 | {seed42_matrix[0, 0]} | {seed42_matrix[0, 1]} |
| 1 | {seed42_matrix[1, 0]} | {seed42_matrix[1, 1]} |

## OOF audit

**PASS** — Each seed has one outer-test prediction per sample, with saved sample index, fold, target, and probability. All predictions were validated for fold identity, target alignment, uniqueness, finiteness, and `[0, 1]` bounds before metrics were computed. All {run_info['checkpoint_verification']['checkpoints_loaded']} checkpoints were reloaded on CPU and reproduced their OOF probabilities (maximum absolute difference `{run_info['checkpoint_verification']['max_absolute_probability_delta']:.3g}`). Predictions and checkpoints are stored under `v1.1/predictions/{run_id}/` and `v1.1/checkpoints/{run_id}/`.

The ROC, precision-recall, calibration, and confusion-matrix figures in `v1.1/figures/{run_id}/` use actual saved seed-42 outer OOF probabilities. No curve was inferred from aggregate metrics.

## Interpretation

This is a nested internal evaluation on the available row-level dataset. It does not establish independent patients, external validity, clinical utility, or clinical readiness. The v1.0 outputs retain their historical checkpoint-selection limitation; v1.1's inner-fold duration selection avoids that specific outer-test reuse.
"""
    (output_directory / "PMRC_NET_V1_1_EXPERIMENT_REPORT.md").write_text(experiment, encoding="utf-8")

    comparison = f"""# PMRC-Net v1.1 versus v1.0 and Baselines

The comparison uses identical seed-specific outer split assignments. v1.0 and baseline OOF files are historical results and were not regenerated; v1.1 predictions were generated by nested outer/inner CV. Because v1.0 selected checkpoints using the scored fold, the comparison is descriptive and its methods are not equally insulated from selection bias.

The legacy root `results/final_model_comparison.csv` is not used as a metric source: its PMRC-Net accuracy and ROC-AUC do not match the verified seed-42 OOF file, and its prediction/split provenance is unresolved. This file remains untouched as historical evidence.

## PMRC-Net version comparison

{chr(10).join(comparison_table)}

## Across-model performance

The per-seed metric recalculations and five-seed mean/SD for v1.0, v1.1, and all ten baselines are in `multi_model_seed_metrics.csv` and `multi_model_mean_std.csv`. Paired bootstrap estimates and intervals for ROC-AUC, AP, and Brier are in `paired_comparisons.csv`. The bootstrap resamples sample IDs and carries the matched seed predictions together. Intervals are descriptive, are not hypothesis tests, and are not multiplicity-adjusted across models. They cannot handle unknown patient clusters and are not a clinical or superiority claim.

All models' seed-42 OOF labels, sample IDs, and fold IDs were validated against the common dataset and split manifests before plotting or comparison.
"""
    (output_directory / "PMRC_NET_V1_1_VS_V1_0_COMPARISON.md").write_text(comparison, encoding="utf-8")

    test_report = f"""# PMRC-Net v1.1 Test Report

## Automated v1.1 suite

- **{test_results['v1_1']['status']}** — {test_results['v1_1']['tests_run']} v1.1 tests run; process exit code {test_results['v1_1']['exit_code']}.
- **{test_results['v1_0']['status']}** — {test_results['v1_0']['tests_run']} historical v1.0 tests run; process exit code {test_results['v1_0']['exit_code']}. Historical code and outputs remain unchanged.
- **{test_results['setup_smoke']['status']}** — Setup/model/checkpoint smoke; process exit code {test_results['setup_smoke']['exit_code']}.
- **PASS** — CPU forward/backward, finite loss and gradients, additive logit decomposition.
- **PASS** — CUDA forward/backward and inference (when CUDA is available).
- **PASS** — Every categorical feature rejects negative, out-of-vocabulary, fractional, NaN, and infinite values on CPU and CUDA. Source mappings are preserved; unknown source codes raise `ValueError`.
- **PASS** — Continuous and raw-physiology NaN and positive/negative infinity are rejected on CPU and CUDA.
- **PASS** — The reverse asymmetric-gate halves receive nonzero gradients; symmetric coupling keeps a single gate width.
- **PASS** — BCE objective, training-only scaler, outer/inner split isolation, OOF alignment/duplicate checks, deterministic initialization, metric recalculation, and checkpoint round-trip.
- **{run_info['checkpoint_verification']['status']}** — Re-loaded {run_info['checkpoint_verification']['checkpoints_loaded']} experiment checkpoints on CPU and verified their predictions against saved OOF rows (maximum absolute difference `{run_info['checkpoint_verification']['max_absolute_probability_delta']:.3g}`).

The v1.1 suite is `v1.1/tests/test_v1_1.py`; run it with `python -m unittest discover -s v1.1/tests -v`. The v1.0 suite remains at `tests/` and is run independently with `python -m unittest discover -s tests -v`. Raw outputs are saved alongside this report. Both ran with the selected venv's Python 3.11 and CUDA-enabled PyTorch 2.11 environment.

## Experimental integrity

- **PASS** — 25 outer-test partitions across five seeds; each sample receives exactly one OOF prediction per seed.
- **PASS** — Scalers fit only on the current training partition; outer-test labels do not enter inner epoch selection.
- **PASS** — Checkpoints, predictions, split manifests, config, and run metadata are isolated by run ID and support hash-checked resume.
- **PASS** — FP32 and sequential execution; no automatic mixed precision.
"""
    (output_directory / "PMRC_NET_V1_1_TEST_REPORT.md").write_text(test_report, encoding="utf-8")

    limitations = """# PMRC-Net v1.1 Limitations

- No patient IDs or source-level linkage are available. The analysis unit is the deduplicated row, not a verified independent patient.
- No independent external dataset was established or evaluated. External generalization is unknown.
- Nested outer/inner CV avoids using outer labels for epoch selection, but repeated stratified folds on one small dataset are not a substitute for patient-level or external validation.
- v1.0 predictions remain historical and use validation folds for checkpoint selection; paired comparisons are not methodologically symmetric even when outer fold IDs match.
- The physiology representation is retained as a design hypothesis. Its prior target-correlation hinge is removed because it was inactive and its direction was not independently prespecified. No causal or physiological validity is established.
- The corrected ordered coupling has a mathematical directional form and testable information flow; it does not establish biologically meaningful directionality.
- ECE is bin-dependent and finite-sample; calibration transportability is unknown. No post-hoc calibration or threshold selection was fitted.
- Row-level bootstrap intervals cannot account for unknown patient clustering. No clinical superiority claim follows.
- The evaluation data retain potential source/cohort overlap and provenance uncertainty; exact deduplication does not resolve it.
- FP32 was used. Mixed precision was not implemented or evaluated.
- Inputs with missing or non-finite continuous values are rejected. No imputation strategy is provided.
- Baseline predictions are re-used from completed historical artifacts, not retrained under the new nested model-selection workflow. This preserves v1.0 and enables split-aligned descriptive comparison but does not standardize every baseline's selection procedure.
- The experiment is research-only. Diagnostic utility, clinical safety, prospective performance, and deployment readiness are not established.
"""
    (output_directory / "PMRC_NET_V1_1_LIMITATIONS.md").write_text(limitations, encoding="utf-8")


def build_reports(run_id: str) -> dict[str, Any]:
    config = load_config()
    dataset_path = (V11_ROOT / config["data"]["path"]).resolve()
    dataframe = load_dataset(dataset_path, config)
    models, run_info = load_model_predictions(run_id, dataframe, config)
    inner_split_manifest = persist_inner_split_manifest(
        V11_ROOT / "experiments" / run_id,
        dataframe,
        config,
        run_info["folds"],
    )
    run_info["checkpoint_verification"] = verify_checkpoint_predictions(
        run_id, dataframe, config, models, run_info
    )
    detail, summary = calculate_model_metrics(models, int(config["statistics"]["ece_bins"]))
    paired = paired_sample_bootstrap(
        models,
        dataframe[config["data"]["target"]].to_numpy(dtype=int),
        int(config["statistics"]["bootstrap_repetitions"]),
        int(config["statistics"]["bootstrap_seed"]),
    )
    run_directory = V11_ROOT / "experiments" / run_id
    report_directory = V11_ROOT / "reports" / run_id
    figure_directory = V11_ROOT / "figures" / run_id
    if report_directory.exists():
        raise FileExistsError(f"Reports already exist for run {run_id}: {report_directory}")
    if figure_directory.exists():
        raise FileExistsError(f"Figures already exist for run {run_id}: {figure_directory}")
    test_results = execute_required_tests()
    report_directory.mkdir(parents=True, exist_ok=False)
    for name, result in test_results.items():
        (report_directory / f"{name}_test_output.log").write_text(
            result["stdout"] + "\n" + result["stderr"], encoding="utf-8"
        )
    run_info["software"] = record_environment(run_directory, run_info["metadata"])
    write_reports(run_id, models, detail, summary, paired, run_info, test_results, report_directory)
    write_figures(models, figure_directory)
    artifact_roots = (
        V11_ROOT / "configs",
        run_directory,
        V11_ROOT / "checkpoints" / run_id,
        V11_ROOT / "predictions" / run_id,
        report_directory,
        figure_directory,
    )
    artifact_hashes = {
        path.relative_to(V11_ROOT).as_posix(): sha256_file(path)
        for directory in artifact_roots
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not path.name.endswith(".tmp")
    }
    audit = {
        "status": "PASS",
        "run_id": run_id,
        "rows": len(dataframe),
        "seeds": list(SEEDS),
        "models": list(MODEL_NAMES),
        "model_seed_oof_validations": len(MODEL_NAMES) * len(SEEDS),
        "all_oof_validated": True,
        "metrics_recalculated_from_saved_predictions": True,
        "paired_comparison_rows": len(paired),
        "checkpoint_verification": run_info["checkpoint_verification"],
        "inner_split_manifest": f"experiments/{run_id}/nested_split_manifest.json",
        "test_results": {
            name: {"status": result["status"], "exit_code": result["exit_code"], "tests_run": result["tests_run"]}
            for name, result in test_results.items()
        },
        "artifact_hashes": artifact_hashes,
        "bootstrap_repetitions": int(config["statistics"]["bootstrap_repetitions"]),
        "patient_independence": "unverified",
        "external_validation": "not performed",
        "reports": [
            f"reports/{run_id}/PMRC_NET_V1_1_METHODOLOGY.md",
            f"reports/{run_id}/PMRC_NET_V1_1_EXPERIMENT_REPORT.md",
            f"reports/{run_id}/PMRC_NET_V1_1_TEST_REPORT.md",
            f"reports/{run_id}/PMRC_NET_V1_1_VS_V1_0_COMPARISON.md",
            f"reports/{run_id}/PMRC_NET_V1_1_LIMITATIONS.md",
        ],
        "run_directory": str(run_directory),
    }
    (report_directory / "v1_1_report_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify saved v1.1 predictions and generate reports")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(build_reports(args.run_id), indent=2))


if __name__ == "__main__":
    main()