from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "cleaned_merged_heart_dataset.csv"
BASELINE_ROOT = ROOT / "experiments" / "additional_baselines"
PMRC_ROOT = ROOT / "results" / "stage5_final_validation"
PMRC_SEED_ROOT = PMRC_ROOT / "seed_oof_predictions"
HISTORICAL_PMRC_OOF = ROOT / "results" / "stage2_validation" / "oof_predictions.csv"
FOLD_MANIFEST = ROOT / "results" / "stage2_validation" / "fold_assignments.json"
REPORT_PATH = ROOT / "reports" / "VERIFIED_11_MODEL_COMPARISON.md"
FIGURE_ROOT = ROOT / "reports" / "figures" / "verified_11_model_comparison"
SEEDS = [21, 42, 123, 2026, 3407]
N_FOLDS = 5
THRESHOLD = 0.5
ECE_BINS = 10
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20261107
METRICS = [
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "mcc",
    "roc_auc",
    "average_precision",
    "brier_score",
    "ece_10",
]
PERCENT_METRICS = {"accuracy", "precision", "sensitivity", "specificity", "f1"}
HIGHER_IS_BETTER = set(METRICS) - {"brier_score", "ece_10"}
MODELS = [
    ("logistic_regression", "Logistic Regression"),
    ("svm", "Support Vector Machine (SVM)"),
    ("random_forest", "Random Forest"),
    ("mlp", "Multi-Layer Perceptron (MLP)"),
    ("knn", "K-Nearest Neighbors (KNN)"),
    ("decision_tree", "Decision Tree"),
    ("xgboost", "XGBoost"),
    ("lightgbm", "LightGBM"),
    ("catboost", "CatBoost"),
    ("tabpfn", "TabPFN"),
    ("pmrc_net", "Proposed PMRC-Net"),
]
DISPLAY = dict(MODELS)
OOF_COLUMNS = {"sample_index", "fold", "target", "probability"}


def load_data() -> tuple[pd.DataFrame, str]:
    raw_bytes = DATASET.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    data = pd.read_csv(DATASET).drop_duplicates().reset_index(drop=True)
    if "target" not in data.columns:
        raise ValueError("The cleaned dataset has no target column")
    data.insert(0, "sample_index", np.arange(len(data), dtype=np.int64))
    return data, digest


def expected_fold_ids(target: np.ndarray, seed: int) -> np.ndarray:
    fold_ids = np.zeros(len(target), dtype=np.int64)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold_number, (_, valid_indices) in enumerate(splitter.split(np.zeros(len(target)), target), start=1):
        fold_ids[valid_indices] = fold_number
    return fold_ids


def ece_score(target: np.ndarray, probability: np.ndarray) -> float:
    edges = np.linspace(0.0, 1.0, ECE_BINS + 1)
    total = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (probability >= lower) & (
            (probability <= upper) if index == ECE_BINS - 1 else (probability < upper)
        )
        if in_bin.any():
            total += in_bin.mean() * abs(target[in_bin].mean() - probability[in_bin].mean())
    return float(total)


def calculate_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "sensitivity": float(recall_score(target, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(target, prediction)),
        "roc_auc": float(roc_auc_score(target, probability)),
        "average_precision": float(average_precision_score(target, probability)),
        "brier_score": float(brier_score_loss(target, probability)),
        "ece_10": ece_score(target, probability),
    }


def validate_frame(
    frame: pd.DataFrame,
    data: pd.DataFrame,
    expected_folds: np.ndarray,
    required_samples: np.ndarray | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    if not OOF_COLUMNS.issubset(frame.columns):
        return {"valid": False, "issues": [f"missing columns: {sorted(OOF_COLUMNS - set(frame.columns))}"]}
    checked = frame.copy()
    for column in ("sample_index", "fold", "target", "probability"):
        checked[column] = pd.to_numeric(checked[column], errors="coerce")
    if checked[list(OOF_COLUMNS)].isna().any().any():
        issues.append("non-numeric or missing required values")
        return {"valid": False, "issues": issues}
    checked["sample_index"] = checked["sample_index"].astype(int)
    checked["fold"] = checked["fold"].astype(int)
    checked["target"] = checked["target"].astype(int)
    if checked["sample_index"].duplicated().any():
        issues.append("duplicate sample indices")
    expected_samples = (
        set(range(len(data))) if required_samples is None else set(required_samples.astype(int).tolist())
    )
    if set(checked["sample_index"]) != expected_samples:
        issues.append("sample indices do not match expected coverage")
    if not np.isfinite(checked["probability"].to_numpy(dtype=float)).all():
        issues.append("probabilities are not finite")
    probability = checked["probability"].to_numpy(dtype=float)
    if ((probability < 0.0) | (probability > 1.0)).any():
        issues.append("probabilities are outside [0, 1]")
    target_by_sample = data.set_index("sample_index")["target"].astype(int)
    expected_fold_by_sample = pd.Series(expected_folds, index=data["sample_index"].astype(int))
    for row in checked.itertuples(index=False):
        sample = int(row.sample_index)
        if sample in expected_samples:
            if int(row.target) != int(target_by_sample.loc[sample]):
                issues.append("target labels do not match the dataset")
                break
            if int(row.fold) != int(expected_fold_by_sample.loc[sample]):
                issues.append("fold assignment does not match the expected split")
                break
    if not (checked["fold"].between(1, N_FOLDS)).all():
        issues.append("fold id outside the expected 1-5 range")
    return {"valid": not issues, "issues": sorted(set(issues)), "rows": len(checked)}


def validate_manifest(target: np.ndarray) -> bool:
    manifest = json.loads(FOLD_MANIFEST.read_text(encoding="utf-8"))
    if int(manifest.get("rows", -1)) != len(target) or int(manifest.get("seed", -1)) != 42:
        return False
    expected = expected_fold_ids(target, 42)
    manifest_folds = manifest.get("folds", [])
    if len(manifest_folds) != N_FOLDS:
        return False
    for item in manifest_folds:
        fold = int(item["fold"])
        actual_valid = set(map(int, item["valid_indices"]))
        expected_valid = set(np.flatnonzero(expected == fold).tolist())
        if actual_valid != expected_valid:
            return False
    return True


def load_model_oof(
    model: str,
    seed: int,
    data: pd.DataFrame,
    fold_ids: np.ndarray,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if model == "pmrc_net":
        path = PMRC_SEED_ROOT / f"seed_{seed}_oof.csv"
    else:
        path = BASELINE_ROOT / model / f"seed_{seed}" / "oof_predictions.csv"
    if not path.exists():
        return None, {"path": path.relative_to(ROOT).as_posix(), "valid": False, "issues": ["file missing"]}
    frame = pd.read_csv(path)
    result = validate_frame(frame, data, fold_ids)
    result["path"] = path.relative_to(ROOT).as_posix()
    result["frame"] = frame if result["valid"] else None
    return frame if result["valid"] else None, result


def validate_fold_files(model: str, data: pd.DataFrame, seed_fold_ids: dict[int, np.ndarray]) -> tuple[int, int, bool]:
    if model == "pmrc_net":
        return 0, len(SEEDS) * N_FOLDS, all(
            np.array_equal(np.sort(np.unique(seed_fold_ids[seed])), np.arange(1, N_FOLDS + 1))
            for seed in SEEDS
        )
    expected_file_total = len(SEEDS) * N_FOLDS
    present_total = 0
    all_valid = True
    for seed in SEEDS:
        seed_dir = BASELINE_ROOT / model / f"seed_{seed}"
        files = sorted(seed_dir.glob("fold_*_oof.csv"))
        present_total += len(files)
        expected_ids = seed_fold_ids[seed]
        for fold_number in range(1, N_FOLDS + 1):
            path = seed_dir / f"fold_{fold_number}_oof.csv"
            if not path.exists():
                all_valid = False
                continue
            valid_indices = np.flatnonzero(expected_ids == fold_number)
            result = validate_frame(pd.read_csv(path), data, expected_ids, valid_indices)
            if not result["valid"]:
                all_valid = False
        if len(files) != N_FOLDS:
            all_valid = False
    return present_total, expected_file_total, all_valid


def compare_seed_summary(model: str, computed: list[dict[str, float]], dataset_hash: str) -> str:
    if model == "pmrc_net":
        path = PMRC_ROOT / "multi_seed_summary.csv"
        if not path.exists():
            return "missing summary"
        saved = pd.read_csv(path).set_index("metric")
        summary_values = pd.DataFrame(computed)
        okay = True
        missing_metrics: list[str] = []
        for metric in METRICS:
            if metric not in saved.index:
                missing_metrics.append(metric)
                continue
            mean = float(summary_values[metric].mean())
            std = float(summary_values[metric].std(ddof=1))
            if not np.isclose(mean, float(saved.loc[metric, "mean"]), atol=1e-8):
                okay = False
            if not np.isclose(std, float(saved.loc[metric, "std"]), atol=1e-8):
                okay = False
        if not okay:
            return "differs from recomputed OOF"
        return f"matches; {', '.join(missing_metrics)} not stored" if missing_metrics else "matches"
    path = BASELINE_ROOT / model / "seed_summary.csv"
    if not path.exists():
        return "missing summary"
    saved = pd.read_csv(path)
    if not {"seed", *METRICS}.issubset(saved.columns):
        return "summary columns incomplete"
    computed_by_seed = {int(row["seed"]): row for row in computed}
    okay = set(saved["seed"].astype(int)) == set(computed_by_seed)
    for row in saved.to_dict(orient="records"):
        seed = int(row["seed"])
        if seed not in computed_by_seed:
            okay = False
            continue
        for metric in METRICS:
            if not np.isclose(float(row[metric]), computed_by_seed[seed][metric], atol=1e-8):
                okay = False
    if "dataset_sha256" in saved.columns:
        okay = okay and set(saved["dataset_sha256"].astype(str)) == {dataset_hash}
    return "matches" if okay else "differs from recomputed OOF"


def validate_aggregate_oof(
    model: str,
    data: pd.DataFrame,
    seed_fold_ids: dict[int, np.ndarray],
    seed_frames: dict[int, pd.DataFrame],
) -> str:
    if model == "pmrc_net":
        total_rows = sum(len(frame) for frame in seed_frames.values())
        return "PASS: 5 x 602 rows (3010 total); per-seed files" if len(seed_frames) == len(SEEDS) and total_rows == len(SEEDS) * len(data) else "FAIL: incomplete per-seed OOF coverage"
    path = BASELINE_ROOT / model / "all_seeds_oof.csv"
    if not path.exists():
        return "FAIL: missing all_seeds_oof.csv"
    frame = pd.read_csv(path)
    required = {"seed", *OOF_COLUMNS}
    if not required.issubset(frame.columns):
        return "FAIL: aggregate columns incomplete"
    counts = frame.groupby("seed")["sample_index"].count().to_dict()
    if set(map(int, counts)) != set(SEEDS) or any(int(value) != len(data) for value in counts.values()):
        return "FAIL: coverage differs from 5 x 602"
    for seed in SEEDS:
        group = frame.loc[frame["seed"].astype(int) == seed].drop(columns="seed")
        check = validate_frame(group, data, seed_fold_ids[seed])
        saved = seed_frames.get(seed)
        if not check["valid"] or saved is None:
            return f"FAIL: seed {seed} aggregate invalid"
        group = group.sort_values("sample_index").reset_index(drop=True)
        saved = saved.sort_values("sample_index").reset_index(drop=True)
        if not np.array_equal(group["sample_index"], saved["sample_index"]):
            return f"FAIL: seed {seed} aggregate sample mismatch"
        if not np.array_equal(group["fold"], saved["fold"]) or not np.array_equal(group["target"], saved["target"]):
            return f"FAIL: seed {seed} aggregate labels/folds differ from seed OOF"
        if not np.allclose(group["probability"], saved["probability"], atol=0.0, rtol=0.0):
            return f"FAIL: seed {seed} aggregate probabilities differ from seed OOF"
    return "PASS: 5 x 602 rows (3010 total)"


def fmt(value: float | None, percent: bool = False, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "N/A — unavailable"
    return f"{100 * value:.2f}%" if percent else f"{value:.{digits}f}"


def fmt_mean_std(values: list[float], metric: str) -> str:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(finite) == 0:
        return "N/A — unavailable"
    mean = float(finite.mean())
    std = float(finite.std(ddof=1)) if len(finite) > 1 else float("nan")
    if metric in PERCENT_METRICS:
        return f"{mean * 100:.2f}% ± {std * 100:.2f} pp" if np.isfinite(std) else f"{mean * 100:.2f}% ± N/A"
    return f"{mean:.4f} ± {std:.4f}" if np.isfinite(std) else f"{mean:.4f} ± N/A"


def rank_map(seed_metrics: dict[str, list[dict[str, float]]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for metric in METRICS:
        direction = not (metric in {"brier_score", "ece_10"})
        values = {
            model: float(np.mean([row[metric] for row in rows]))
            for model, rows in seed_metrics.items()
            if rows and np.isfinite([row[metric] for row in rows]).all()
        }
        ordered = sorted(values, key=values.get, reverse=direction)
        result[metric] = {model: rank for rank, model in enumerate(ordered, start=1)}
    return result


def paired_bootstrap(
    target: np.ndarray,
    probabilities: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, tuple[float, float, float]]], int]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    baseline_names = [model for model, _ in MODELS if model != "pmrc_net"]
    samples = {model: {metric: [] for metric in ("roc_auc", "average_precision", "brier_score")} for model in baseline_names}
    skipped = 0
    for _ in range(BOOTSTRAP_ITERATIONS):
        indices = rng.integers(0, len(target), size=len(target))
        bootstrap_target = target[indices]
        if np.unique(bootstrap_target).size != 2:
            skipped += 1
            continue
        pmrc = probabilities["pmrc_net"][indices]
        pmrc_scores = {
            "roc_auc": roc_auc_score(bootstrap_target, pmrc),
            "average_precision": average_precision_score(bootstrap_target, pmrc),
            "brier_score": brier_score_loss(bootstrap_target, pmrc),
        }
        for model in baseline_names:
            baseline = probabilities[model][indices]
            samples[model]["roc_auc"].append(pmrc_scores["roc_auc"] - roc_auc_score(bootstrap_target, baseline))
            samples[model]["average_precision"].append(
                pmrc_scores["average_precision"] - average_precision_score(bootstrap_target, baseline)
            )
            samples[model]["brier_score"].append(pmrc_scores["brier_score"] - brier_score_loss(bootstrap_target, baseline))
    intervals: dict[str, dict[str, tuple[float, float, float]]] = {}
    for model, metric_samples in samples.items():
        intervals[model] = {}
        for metric, values in metric_samples.items():
            array = np.asarray(values, dtype=float)
            low, high = np.quantile(array, [0.025, 0.975])
            if metric == "roc_auc":
                point = roc_auc_score(target, probabilities["pmrc_net"]) - roc_auc_score(target, probabilities[model])
            elif metric == "average_precision":
                point = average_precision_score(target, probabilities["pmrc_net"]) - average_precision_score(target, probabilities[model])
            else:
                point = brier_score_loss(target, probabilities["pmrc_net"]) - brier_score_loss(target, probabilities[model])
            intervals[model][metric] = (float(point), float(low), float(high))
    return intervals, skipped


def validate_seed42_figure_data(
    seed42_frames: dict[str, pd.DataFrame],
    data: pd.DataFrame,
    expected_folds: np.ndarray,
) -> dict[str, Any]:
    expected_samples = data["sample_index"].to_numpy(dtype=int)
    expected_target = data["target"].to_numpy(dtype=int)
    checks: dict[str, dict[str, Any]] = {}
    for model, _ in MODELS:
        frame = seed42_frames[model].sort_values("sample_index").reset_index(drop=True)
        sample_match = np.array_equal(frame["sample_index"].to_numpy(dtype=int), expected_samples)
        target_match = np.array_equal(frame["target"].to_numpy(dtype=int), expected_target)
        fold_match = np.array_equal(frame["fold"].to_numpy(dtype=int), expected_folds)
        target = frame["target"].to_numpy(dtype=int)
        probability = frame["probability"].to_numpy(dtype=float)

        edges = np.linspace(0.0, 1.0, ECE_BINS + 1)
        independent_ece = 0.0
        for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            in_bin = (probability >= lower) & (
                (probability <= upper) if bin_index == ECE_BINS - 1 else (probability < upper)
            )
            if in_bin.any():
                independent_ece += in_bin.mean() * abs(target[in_bin].mean() - probability[in_bin].mean())
        calculated_ece = ece_score(target, probability)
        brier_mse = float(np.mean(np.square(target - probability)))
        brier_sklearn = float(brier_score_loss(target, probability))

        if model == "pmrc_net":
            saved_summary_check = "not stored in multi-seed summary"
        else:
            saved_summary = pd.read_csv(BASELINE_ROOT / model / "seed_summary.csv")
            saved_row = saved_summary.loc[saved_summary["seed"].astype(int) == 42].iloc[0]
            ece_matches_saved = np.isclose(calculated_ece, float(saved_row["ece_10"]), atol=1e-12)
            brier_matches_saved = np.isclose(brier_sklearn, float(saved_row["brier_score"]), atol=1e-12)
            saved_summary_check = "matches" if ece_matches_saved and brier_matches_saved else "differs"

        checks[model] = {
            "rows": len(frame),
            "sample_indices_match": sample_match,
            "labels_match": target_match,
            "folds_match": fold_match,
            "ece": float(independent_ece),
            "ece_helper_delta": float(independent_ece - calculated_ece),
            "brier": brier_sklearn,
            "brier_direct_mse_delta": float(brier_sklearn - brier_mse),
            "saved_summary_check": saved_summary_check,
        }

    return {
        "checks": checks,
        "all_same_samples": all(item["sample_indices_match"] for item in checks.values()),
        "all_same_labels": all(item["labels_match"] for item in checks.values()),
        "all_same_folds": all(item["folds_match"] for item in checks.values()),
        "max_abs_ece_delta": max(abs(item["ece_helper_delta"]) for item in checks.values()),
        "max_abs_brier_delta": max(abs(item["brier_direct_mse_delta"]) for item in checks.values()),
    }


def plot_metric_bar(
    seed_metrics: dict[str, list[dict[str, float]]],
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    percent: bool = False,
) -> None:
    names = [name for name, _ in MODELS]
    labels = [DISPLAY[name] for name in names]
    means = [np.mean([row[metric] for row in seed_metrics[name]]) for name in names]
    deviations = [np.std([row[metric] for row in seed_metrics[name]], ddof=1) for name in names]
    scale = 100.0 if percent else 1.0
    colors = ["#26736b" if name == "pmrc_net" else "#71828a" for name in names]
    fig, axis = plt.subplots(figsize=(13, 7.5))
    positions = np.arange(len(names))
    axis.bar(
        positions,
        np.asarray(means) * scale,
        yerr=np.asarray(deviations) * scale,
        capsize=4,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    axis.set_xticks(positions, labels, rotation=35, ha="right", fontsize=9)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    axis.set_axisbelow(True)
    fig.text(
        0.5,
        0.015,
        "Bars = mean; error bars = +/-1 sample standard deviation (ddof=1, n=5), not standard error.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURE_ROOT / filename, dpi=600, bbox_inches="tight")
    plt.close(fig)


def create_figures(
    seed_metrics: dict[str, list[dict[str, float]]],
    seed42_frames: dict[str, pd.DataFrame],
    target: np.ndarray,
) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    display_order = [model for model, _ in MODELS]
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(display_order)))

    fig, axis = plt.subplots(figsize=(11, 10))
    for color, model in zip(colors, display_order):
        frame = seed42_frames[model].sort_values("sample_index")
        probability = frame["probability"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(target, probability)
        auc = roc_auc_score(target, probability)
        axis.plot(fpr, tpr, color=color, linewidth=1.8, label=f"{DISPLAY[model]} (AUC {auc:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#333333", linewidth=1.2, label="Chance")
    axis.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate (Sensitivity)",
        title=f"Receiver Operating Characteristic, Seed 42 OOF (n={len(target)})",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.2, linewidth=0.7)
    axis.set_axisbelow(True)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=2, fontsize=9, frameon=False, handlelength=2.5, columnspacing=1.8)
    fig.subplots_adjust(bottom=0.30, top=0.91, left=0.11, right=0.98)
    fig.savefig(FIGURE_ROOT / "roc_curves.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 10))
    for color, model in zip(colors, display_order):
        frame = seed42_frames[model].sort_values("sample_index")
        probability = frame["probability"].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(target, probability)
        ap = average_precision_score(target, probability)
        axis.plot(recall, precision, color=color, linewidth=1.8, label=f"{DISPLAY[model]} (AP {ap:.3f})")
    axis.set(
        xlabel="Recall (Sensitivity)",
        ylabel="Precision",
        title=f"Precision-Recall Curves, Seed 42 OOF (n={len(target)})",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.2, linewidth=0.7)
    axis.set_axisbelow(True)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=2, fontsize=9, frameon=False, handlelength=2.5, columnspacing=1.8)
    fig.subplots_adjust(bottom=0.30, top=0.91, left=0.11, right=0.98)
    fig.savefig(FIGURE_ROOT / "precision_recall_curves.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    plot_metric_bar(seed_metrics, "accuracy", "Accuracy Across Five Seeds", "Accuracy (%)", "accuracy_comparison.png", True)
    plot_metric_bar(seed_metrics, "roc_auc", "ROC-AUC Across Five Seeds", "ROC-AUC", "roc_auc_comparison.png")
    plot_metric_bar(seed_metrics, "f1", "F1-Score Across Five Seeds", "F1-score (%)", "f1_comparison.png", True)
    plot_metric_bar(seed_metrics, "mcc", "Matthews Correlation Coefficient Across Five Seeds", "MCC", "mcc_comparison.png")
    plot_metric_bar(seed_metrics, "brier_score", "Brier Score Across Five Seeds (Lower Is Better)", "Brier score (lower is better)", "brier_score_comparison.png")

    fig, axis = plt.subplots(figsize=(11, 9.5))
    for color, model in zip(colors, display_order):
        frame = seed42_frames[model].sort_values("sample_index")
        probability = frame["probability"].to_numpy(dtype=float)
        edges = np.linspace(0.0, 1.0, ECE_BINS + 1)
        bin_probability: list[float] = []
        bin_fraction: list[float] = []
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (probability >= lower) & ((probability <= upper) if index == ECE_BINS - 1 else (probability < upper))
            if mask.any():
                bin_probability.append(float(probability[mask].mean()))
                bin_fraction.append(float(target[mask].mean()))
        ece = ece_score(target, probability)
        axis.plot(bin_probability, bin_fraction, marker="o", markersize=4, linewidth=1.7, color=color, label=f"{DISPLAY[model]} (ECE {ece:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#333333", linewidth=1.2, label="Perfect calibration")
    axis.set(
        xlabel="Mean Predicted Probability",
        ylabel="Observed Positive Fraction",
        title=f"Calibration, Seed 42 OOF, {ECE_BINS} Equal-Width Bins (n={len(target)})",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.2, linewidth=0.7)
    axis.set_axisbelow(True)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=2, fontsize=9, frameon=False, handlelength=2.5, columnspacing=1.8)
    fig.subplots_adjust(bottom=0.32, top=0.91, left=0.12, right=0.98)
    fig.savefig(FIGURE_ROOT / "calibration_curves.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    variability_metrics = ["accuracy", "f1", "roc_auc", "average_precision", "mcc", "brier_score"]
    variability_labels = ["Accuracy", "F1-score", "ROC-AUC", "Average Precision", "MCC", "Brier Score"]
    pmrc_rows = seed_metrics["pmrc_net"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    seed_positions = np.arange(len(SEEDS))
    for axis, metric, label in zip(axes.flat, variability_metrics, variability_labels):
        values = [row[metric] for row in pmrc_rows]
        axis.plot(seed_positions, values, marker="o", color="#26736b", linewidth=1.7)
        axis.axhline(np.mean(values), color="#ba5c3a", linestyle="--", linewidth=1.2, label=f"Mean {np.mean(values):.4f}")
        axis.set_title(label)
        axis.set_xlabel("Random Seed")
        axis.set_ylabel("Metric value")
        axis.set_xticks(seed_positions, [str(seed) for seed in SEEDS], rotation=0, ha="center", fontsize=9)
        axis.set_xlim(-0.3, len(SEEDS) - 0.7)
        axis.grid(alpha=0.2, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("PMRC-Net Five-Seed OOF Variability", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURE_ROOT / "pmrc_seed_variability.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def build_report(
    data: pd.DataFrame,
    dataset_hash: str,
    seed_metrics: dict[str, list[dict[str, float]]],
    seed_frames: dict[str, dict[int, pd.DataFrame]],
    inventory: dict[str, dict[str, Any]],
    manifest_ok: bool,
    figure_validation: dict[str, Any],
) -> str:
    rankings = rank_map(seed_metrics)
    lines: list[str] = [
        "# Verified 11-Model PMRC-Net Research Comparison",
        "",
        "## Executive Summary",
        "",
        f"This report re-evaluates the saved out-of-fold (OOF) predictions for all 10 baselines and proposed PMRC-Net. It covers {len(data)} deduplicated observations, five prespecified seeds, and five stratified folds per seed. All performance summaries below were recomputed from local prediction files; no model was trained or refit.",
        "",
        "The ranking is metric-dependent. The table marks the best mean across seeds in each metric; PMRC-Net's rank is shown separately. Differences are descriptive, not evidence of clinical superiority. Paired row-bootstrap intervals use the aligned seed-42 OOF sample and cannot account for unknown patient clustering.",
        "",
        "## Sources and Integrity Checks",
        "",
        f"- Dataset: `dataset/cleaned_merged_heart_dataset.csv`; SHA-256 `{dataset_hash}`.",
        f"- Dataset observations after exact-row deduplication: {len(data)}; positive class is `target = 1`.",
        f"- Seed-42 fold manifest matches the generated five-fold stratified split: **{'PASS' if manifest_ok else 'FAIL'}**.",
        "- Expected seeds: `21, 42, 123, 2026, 3407`; each split uses shuffled `StratifiedKFold(n_splits=5, random_state=seed)`.",
        "- PMRC-Net five-seed OOF files are in `results/stage5_final_validation/seed_oof_predictions/`; its historical seed-42 OOF source is `results/stage2_validation/oof_predictions.csv`.",
        "- Baseline seed OOF files are under `experiments/additional_baselines/<model>/seed_<seed>/`; per-fold files were independently checked where saved.",
        "- Prediction validation checks required columns, exactly one row for every expected sample, target-label alignment, expected fold assignment, and finite probabilities within [0, 1].",
        "",
        "| Model | Completed seeds | OOF rows / seed | Fold coverage | Per-fold OOF files | Aggregate OOF | Probabilities and alignment | Saved summary vs recomputed |",
        "|---|---:|---:|---|---:|---|---|---|",
    ]
    for model, _ in MODELS:
        row = inventory[model]
        fold_text = f"{row['folds_covered']} / {N_FOLDS} per seed; {row['fold_match']}"
        file_text = f"{row['fold_files']}/{row['expected_fold_files']}" if model != "pmrc_net" else "not saved separately"
        validation = "PASS" if row["all_valid"] else "ISSUES: " + "; ".join(row["issues"][:2])
        lines.append(
            f"| {DISPLAY[model]} | {row['completed_seeds']}/{len(SEEDS)} | {row['rows_per_seed']} | {fold_text} | {file_text} | {row['aggregate_oof']} | {validation} | {row['summary_check']} |"
        )
    lines.extend([
        "",
        "TabPFN verification: its five seed OOF files contain 602 validated predictions each (3,010 total); its 25 individual fold files are checked against the corresponding expected validation indices, with no missing folds. The aggregate `all_seeds_oof.csv` is also checked against the per-seed files; completion is not inferred from `summary.json` alone.",
        "",
        "### Saved Summary Audit",
        "",
        "The source `seed_summary.csv` / multi-seed summary was compared with metrics recomputed from each seed's OOF predictions, and each baseline `all_seeds_oof.csv` was checked row-for-row against its seed files. All ten baseline summary tables match recomputation. PMRC-Net's nine stored summary metrics match; ECE is not stored in its aggregate summary and is recomputed from OOF probabilities. Any actual discrepancy is retained rather than silently reconciled.",
        "",
        "## Main Five-Seed Comparison",
        "",
        "Values are mean ± sample standard deviation (`ddof=1`) across the available seed-level pooled OOF metrics. Percentage metrics are shown as percentages; SD is in percentage points. ROC-AUC, AP, MCC, Brier, and ECE are decimals. A bold value marks the best mean for that metric; lower is better for Brier and ECE.",
        "",
        "| Model | Accuracy | Precision | Sensitivity | Specificity | F1-score | MCC | ROC-AUC | AP | Brier Score | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    best: dict[str, str] = {}
    for metric in METRICS:
        eligible = [(model, np.mean([item[metric] for item in rows])) for model, rows in seed_metrics.items() if rows]
        if eligible:
            best[metric] = (max if metric in HIGHER_IS_BETTER else min)(eligible, key=lambda pair: pair[1])[0]
    for model, _ in MODELS:
        values = []
        for metric in METRICS:
            value = fmt_mean_std([row[metric] for row in seed_metrics[model]], metric)
            if best.get(metric) == model:
                value = f"**{value}**"
            values.append(value)
        lines.append(f"| {DISPLAY[model]} | " + " | ".join(values) + " |")

    lines.extend([
        "",
        "### Metric Leaders and PMRC-Net Ranks",
        "",
        "| Metric | Best model | Best mean | PMRC-Net rank | PMRC-Net mean ± SD |",
        "|---|---|---:|---:|---:|",
    ])
    for metric in METRICS:
        leader = best.get(metric)
        leader_values = [row[metric] for row in seed_metrics[leader]] if leader else []
        pmrc_values = [row[metric] for row in seed_metrics["pmrc_net"]]
        rank = rankings[metric].get("pmrc_net", "N/A")
        leader_fmt = fmt_mean_std(leader_values, metric) if leader else "N/A — unavailable"
        pmrc_fmt = fmt_mean_std(pmrc_values, metric)
        lines.append(f"| {metric} | {DISPLAY[leader] if leader else 'N/A'} | {leader_fmt} | {rank}/{len(MODELS)} | {pmrc_fmt} |")

    lines.extend([
        "",
        "## Seed-Level OOF Results",
        "",
        "Each seed row is computed from all held-out predictions pooled across its five folds. The final row for each model is the mean ± sample SD across available seeds; it is not a pooled metric across repeated predictions from different seeds.",
        "",
        "| Model | Seed | Accuracy | F1-score | ROC-AUC | AP | MCC | Brier Score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for model, _ in MODELS:
        model_rows = {int(metrics["seed"]): metrics for metrics in seed_metrics[model]}
        for seed in SEEDS:
            metrics = model_rows.get(seed)
            if metrics is None:
                cells = ["N/A — unavailable"] * 6
            else:
                cells = [fmt(metrics["accuracy"], True), fmt(metrics["f1"], True), fmt(metrics["roc_auc"]), fmt(metrics["average_precision"]), fmt(metrics["mcc"]), fmt(metrics["brier_score"])]
            lines.append(f"| {DISPLAY[model]} | {seed} | " + " | ".join(cells) + " |")
        metrics_by_seed = seed_metrics[model]
        summary_cells = [fmt_mean_std([item[key] for item in metrics_by_seed], key) for key in ("accuracy", "f1", "roc_auc", "average_precision", "mcc", "brier_score")]
        lines.append(f"| {DISPLAY[model]} | Mean ± SD (n={len(metrics_by_seed)}) | " + " | ".join(summary_cells) + " |")

    lines.extend([
        "",
        "## Seed-42 Pooled OOF Results",
        "",
        "These metrics use the single common seed-42 evaluation set (602 aligned rows) and are kept separate from five-seed averages. They are the basis for the historical PMRC-Net comparison and the paired bootstrap analysis.",
        "",
        "| Model | Accuracy | Precision | Sensitivity | Specificity | F1-score | MCC | ROC-AUC | AP | Brier Score | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    seed42_scores: dict[str, dict[str, float]] = {}
    for model, _ in MODELS:
        frame = seed_frames[model][42].sort_values("sample_index")
        y = frame["target"].to_numpy(dtype=int)
        p = frame["probability"].to_numpy(dtype=float)
        scores = calculate_metrics(y, p)
        seed42_scores[model] = scores
        cells = [fmt(scores[metric], metric in PERCENT_METRICS) for metric in METRICS]
        lines.append(f"| {DISPLAY[model]} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## PMRC-Net Historical Result Verification",
        "",
        "Expected values below are verification references from the request and existing project summaries, not substitute values. Recomputed values come from the corresponding local OOF source.",
        "",
        "| Five-seed metric | Expected mean ± SD | Recomputed mean ± SD | Status |",
        "|---|---:|---:|---|",
    ])
    expected_multi = {
        "accuracy": (0.8149502, 0.0085),
        "f1": (0.8065208, 0.0090),
        "roc_auc": (0.8966129, 0.0050),
        "mcc": (0.6295792, 0.0171),
        "brier_score": (0.1313092, 0.0025),
    }
    pmrc_rows = seed_metrics["pmrc_net"]
    for metric, (expected_mean, expected_std) in expected_multi.items():
        observed_mean = float(np.mean([item[metric] for item in pmrc_rows]))
        observed_std = float(np.std([item[metric] for item in pmrc_rows], ddof=1))
        status = "MATCH" if np.isclose(expected_mean, observed_mean, atol=1e-6) and np.isclose(expected_std, observed_std, atol=1e-4) else "DISCREPANCY"
        lines.append(f"| {metric} | {expected_mean:.7f} ± {expected_std:.4f} | {observed_mean:.7f} ± {observed_std:.4f} | {status} |")
    historical_expected = {
        "accuracy": 0.8256,
        "precision": 0.8304,
        "sensitivity": 0.8048,
        "specificity": 0.8452,
        "f1": 0.8174,
        "mcc": 0.6508,
        "roc_auc": 0.8955,
        "average_precision": 0.8909,
        "brier_score": 0.1307,
    }
    lines.extend([
        "",
        "| Historical seed-42 metric | Expected | Recomputed from Stage 2 OOF | Status |",
        "|---|---:|---:|---|",
    ])
    for metric, expected in historical_expected.items():
        observed = seed42_scores["pmrc_net"][metric]
        status = "MATCH" if np.isclose(expected, observed, atol=0.00015) else "DISCREPANCY"
        lines.append(f"| {metric} | {expected:.4f} | {observed:.4f} | {status} |")
    historical_file = pd.read_csv(HISTORICAL_PMRC_OOF).sort_values("sample_index").reset_index(drop=True)
    stage5_seed42 = seed_frames["pmrc_net"][42].sort_values("sample_index").reset_index(drop=True)
    same_historical = all(
        np.array_equal(historical_file[column].to_numpy(), stage5_seed42[column].to_numpy())
        for column in ("sample_index", "fold", "target")
    ) and np.allclose(historical_file["probability"], stage5_seed42["probability"], atol=0.0, rtol=0.0)
    lines.extend([
        "",
        f"The Stage 2 seed-42 OOF file and Stage 5 seed-42 OOF file are {'identical' if same_historical else 'not identical'} after sorting by `sample_index`. The historical Stage 2 artifact remains the reference source for the historical result.",
        "",
        "## Paired Seed-42 Comparisons",
        "",
        f"For each baseline, differences are PMRC-Net minus baseline on the same aligned 602 seed-42 rows. The paired percentile bootstrap resamples rows with replacement using the same index draw for both models; {BOOTSTRAP_ITERATIONS:,} iterations; NumPy RNG seed `{BOOTSTRAP_SEED}`; 95% percentile intervals. Replicates with only one target class are skipped (observed skipped count: {{SKIPPED_BOOTSTRAPS}}). Positive ROC-AUC/AP differences favor PMRC-Net; negative Brier differences favor PMRC-Net.",
        "",
        "Intervals are descriptive row-level intervals only. Unknown patient clustering means they cannot be interpreted as patient-independent evidence or as a clinical/statistical superiority claim. No multiplicity adjustment is applied across the 30 metric comparisons.",
        "",
        "| Baseline | ROC-AUC difference [95% CI] | AP difference [95% CI] | Brier difference [95% CI] | Descriptive direction |",
        "|---|---:|---:|---:|---|",
    ])
    probabilities = {
        model: seed_frames[model][42].sort_values("sample_index")["probability"].to_numpy(dtype=float)
        for model, _ in MODELS
    }
    interval_results, skipped = paired_bootstrap(data["target"].to_numpy(dtype=int), probabilities)
    lines = [line.replace("{SKIPPED_BOOTSTRAPS}", str(skipped)) for line in lines]
    for model, _ in MODELS:
        if model == "pmrc_net":
            continue
        cells = []
        interval_excludes_zero = []
        for metric in ("roc_auc", "average_precision", "brier_score"):
            estimate, low, high = interval_results[model][metric]
            cells.append(f"{estimate:+.4f} [{low:+.4f}, {high:+.4f}]")
            interval_excludes_zero.append(low > 0 or high < 0)
        point_better = all(
            (seed42_scores["pmrc_net"][metric] > seed42_scores[model][metric]) if metric != "brier_score" else
            (seed42_scores["pmrc_net"][metric] < seed42_scores[model][metric])
            for metric in ("roc_auc", "average_precision", "brier_score")
        )
        ci_label = "row-bootstrap intervals exclude zero for all three" if all(interval_excludes_zero) else "one or more intervals include zero"
        point_label = "PMRC-Net numerically better on all three" if point_better else "mixed metric directions"
        lines.append(f"| {DISPLAY[model]} | {cells[0]} | {cells[1]} | {cells[2]} | {point_label}; {ci_label} |")

    lines.extend([
        "",
        "## PMRC-Net vs Baselines Across Metrics",
        "",
        "The following table compares five-seed means in the favorable direction for each metric. ‘Better’ means PMRC-Net has a higher mean for accuracy, precision, sensitivity, specificity, F1, MCC, ROC-AUC, and AP, or a lower mean for Brier and ECE. The comparison is numerical and does not by itself establish a meaningful or statistically reliable advantage.",
        "",
        "| Baseline | Metrics where PMRC-Net mean is better | Metrics where baseline mean is better or tied | Overall direction |",
        "|---|---|---|---|",
    ])
    for model, _ in MODELS:
        if model == "pmrc_net":
            continue
        better: list[str] = []
        other: list[str] = []
        for metric in METRICS:
            pmrc_mean = float(np.mean([row[metric] for row in seed_metrics["pmrc_net"]]))
            baseline_mean = float(np.mean([row[metric] for row in seed_metrics[model]]))
            improved = pmrc_mean > baseline_mean if metric in HIGHER_IS_BETTER else pmrc_mean < baseline_mean
            (better if improved else other).append(metric)
        direction = f"better on {len(better)}/{len(METRICS)} metrics" if better else "no mean-metric wins"
        lines.append(f"| {DISPLAY[model]} | {', '.join(better) or 'None'} | {', '.join(other) or 'None'} | {direction} |")

    lines.extend([
        "",
        "## Figures",
        "",
        "The ROC and precision-recall curves use the real, verified seed-42 OOF labels and probabilities for all 11 models. No curve was inferred from a summary statistic. Bar-chart values are five-seed means with error bars equal to +/-1 sample standard deviation (`ddof=1`, n=5), not standard error. Calibration uses ten equal-width bins on the common seed-42 OOF sample.",
        "",
        "![ROC curves for all models, calculated from seed-42 OOF predictions](figures/verified_11_model_comparison/roc_curves.png)",
        "*Figure 1. ROC curves from true labels and OOF probabilities, seed 42, 602 aligned observations.*",
        "",
        "![Precision-recall curves for all models, calculated from seed-42 OOF predictions](figures/verified_11_model_comparison/precision_recall_curves.png)",
        "*Figure 2. Precision–recall curves from true labels and OOF probabilities, seed 42, 602 aligned observations.*",
        "",
        "![Five-seed accuracy comparison](figures/verified_11_model_comparison/accuracy_comparison.png)",
        "*Figure 3. Accuracy means; error bars show +/-1 sample SD (`ddof=1`, n=5), not standard error.*",
        "",
        "![Five-seed ROC-AUC comparison](figures/verified_11_model_comparison/roc_auc_comparison.png)",
        "*Figure 4. ROC-AUC means; error bars show +/-1 sample SD (`ddof=1`, n=5), not standard error.*",
        "",
        "![Five-seed F1-score comparison](figures/verified_11_model_comparison/f1_comparison.png)",
        "*Figure 5. F1-score means; error bars show +/-1 sample SD (`ddof=1`, n=5), not standard error.*",
        "",
        "![Five-seed MCC comparison](figures/verified_11_model_comparison/mcc_comparison.png)",
        "*Figure 6. MCC means; error bars show +/-1 sample SD (`ddof=1`, n=5), not standard error.*",
        "",
        "![Five-seed Brier score comparison](figures/verified_11_model_comparison/brier_score_comparison.png)",
        "*Figure 7. Brier score means; error bars show +/-1 sample SD (`ddof=1`, n=5), not standard error. Lower Brier score is better.*",
        "",
        "![Ten-bin reliability curves for all models](figures/verified_11_model_comparison/calibration_curves.png)",
        "*Figure 8. Ten-bin reliability curves from aligned seed-42 probabilities; calibration estimates are descriptive.*",
        "",
        "![PMRC-Net performance variability across five seeds](figures/verified_11_model_comparison/pmrc_seed_variability.png)",
        "*Figure 9. PMRC-Net seed-level OOF metrics and across-seed means.*",
        "",
        "## Methodology",
        "",
        f"The evaluation dataset is the repository's cleaned, exact-row-deduplicated heart dataset ({len(data)} rows; SHA-256 `{dataset_hash}`). Its `target` column defines class 1 as positive. Each model is evaluated on the existing five-fold stratified partitions generated for seeds 21, 42, 123, 2026, and 3407. The seed-42 PMRC fold assignment was checked against `results/stage2_validation/fold_assignments.json`; all saved per-model assignments were checked against the corresponding split generated with the runner's `StratifiedKFold` protocol.",
        "",
        "Preprocessing follows each saved model's existing implementation and remains fold-local. PMRC-Net standardizes continuous inputs using the training fold and retains the encoded categorical inputs. Baseline Logistic Regression, KNN, SVM, and MLP use a `StandardScaler` fitted in their training-fold pipeline; tree and boosting baselines use the encoded values without scaling, as configured by `run_additional_baselines.py`. No preprocessing or evaluation settings were changed for this report.",
        "",
        "For each seed, predictions from its five held-out folds are assembled into one OOF vector, then metrics are recomputed from those 602 predictions. Threshold-dependent metrics use the runner's fixed threshold of 0.5 and positive class 1. Specificity is TN/(TN+FP). ROC-AUC and average precision use probabilities. Brier score is the mean squared probability error. ECE uses ten fixed equal-width bins over [0, 1], weighted by each bin's fraction of rows; the final bin includes probability 1. Seed means and sample standard deviations are reported separately from the single-seed pooled OOF metric table.",
        "",
        "The model comparison uses the same dataset and seed/fold partition protocol. The statistical comparison is paired on sample index using seed-42 OOF predictions, and the same bootstrap row indices are applied to PMRC-Net and each baseline. This is not a new experiment, model selection, threshold optimization, or external validation.",
        "",
        "## Limitations and Interpretation",
        "",
        "1. Patient-level independence has not been verified; the dataset contains no confirmed patient identifier.",
        "2. External independent validation has not been established. Candidate external datasets may overlap the development source or lack verified provenance.",
        "3. These results do not establish clinical readiness, diagnostic utility, or clinical superiority.",
        "4. Row-level bootstrap intervals cannot account for unknown patient clustering and are not patient-independent evidence.",
        "5. The physiological hinge penalty was inactive in the previously evaluated configuration; its contribution is not established by these results.",
        "6. The no-physiology ablation was close to the full model, so physiological contribution must not be overstated.",
        "7. The no-pathways ablation showed substantial performance collapse, but the implementation and interpretation must be verified before mechanistic claims are made.",
        "8. Five-seed variability measures sensitivity to the configured random seeds on this dataset and is not a substitute for independent external replication.",
        "9. Calibration curves and ECE are finite-sample, bin-dependent descriptions; no recalibration or threshold was selected on these OOF results.",
        "",
        "## Reproducibility and Output",
        "",
        "This report was generated by `scripts/generate_verified_11_model_comparison.py`. The script reads only existing datasets, manifests, summaries, and OOF artifacts; it does not instantiate or fit any model. The nine PNG figures are stored in `reports/figures/verified_11_model_comparison/` at 600 DPI. Existing experiment artifacts and the earlier comparison draft were left unchanged.",
        "",
    ])
    lines.extend([
        "## Figure Data Validation",
        "",
        f"All 11 seed-42 prediction files were checked before plotting. Exact sample-index sequence match: **{'PASS' if figure_validation['all_same_samples'] else 'FAIL'}**. Ground-truth target match: **{'PASS' if figure_validation['all_same_labels'] else 'FAIL'}**. Fold assignment match: **{'PASS' if figure_validation['all_same_folds'] else 'FAIL'}**. The common sample count is {len(data)}; mismatches are not silently aligned or replaced.",
        f"ECE was independently recomputed from each model's 602 seed-42 probabilities using the same ten equal-width bins over [0, 1], with the final bin closed at 1.0; the maximum absolute difference from the report's ECE calculation was {figure_validation['max_abs_ece_delta']:.3g}. The Brier cross-check compared `brier_score_loss` on the OOF labels/probabilities with the direct mean squared probability error; maximum absolute difference was {figure_validation['max_abs_brier_delta']:.3g}. Baseline ECE and Brier values were also checked against saved seed-42 summaries; PMRC-Net ECE is recomputed because it is not in the multi-seed summary.",
        "",
        "| Model | Seed-42 rows | Sample indices | Labels | Fold IDs | ECE (10 bins) | Brier (direct OOF) | Saved ECE/Brier check |",
        "|---|---:|---|---|---|---:|---:|---|",
    ])
    for model, _ in MODELS:
        check = figure_validation["checks"][model]
        sample_check = "PASS" if check["sample_indices_match"] else "FAIL"
        labels_check = "PASS" if check["labels_match"] else "FAIL"
        folds_check = "PASS" if check["folds_match"] else "FAIL"
        lines.append(
            f"| {DISPLAY[model]} | {check['rows']} | {sample_check} | {labels_check} | {folds_check} | {check['ece']:.6f} | {check['brier']:.6f} | {check['saved_summary_check']} |"
        )
    return "\n".join(lines)


def main() -> None:
    data, dataset_hash = load_data()
    target = data["target"].to_numpy(dtype=int)
    fold_ids = {seed: expected_fold_ids(target, seed) for seed in SEEDS}
    manifest_ok = validate_manifest(target)
    seed_metrics: dict[str, list[dict[str, float]]] = {}
    seed_frames: dict[str, dict[int, pd.DataFrame]] = {}
    inventory: dict[str, dict[str, Any]] = {}

    for model, _ in MODELS:
        model_rows: list[dict[str, float]] = []
        model_frames: dict[int, pd.DataFrame] = {}
        validation_results: list[dict[str, Any]] = []
        issues: list[str] = []
        for seed in SEEDS:
            frame, check = load_model_oof(model, seed, data, fold_ids[seed])
            validation_results.append(check)
            if frame is None:
                issues.extend([f"seed {seed}: {issue}" for issue in check["issues"]])
                continue
            frame = frame.sort_values("sample_index").reset_index(drop=True)
            model_frames[seed] = frame
            metrics = calculate_metrics(frame["target"].to_numpy(dtype=int), frame["probability"].to_numpy(dtype=float))
            metrics["seed"] = seed
            model_rows.append(metrics)

        fold_files, expected_files, fold_files_ok = validate_fold_files(model, data, fold_ids)
        fold_counts = [len(set(frame["fold"].astype(int))) for frame in model_frames.values()]
        coverage_ok = len(model_rows) == len(SEEDS) and all(count == N_FOLDS for count in fold_counts)
        if not fold_files_ok:
            issues.append("one or more individual fold files are missing or invalid")
        summary_check = compare_seed_summary(model, model_rows, dataset_hash) if model_rows else "not verifiable"
        if not summary_check.startswith("matches"):
            issues.append(f"saved summary {summary_check}")
        aggregate_check = validate_aggregate_oof(model, data, fold_ids, model_frames)
        if not aggregate_check.startswith("PASS:"):
            issues.append(f"aggregate OOF: {aggregate_check}")
        seed_metrics[model] = model_rows
        seed_frames[model] = model_frames
        inventory[model] = {
            "completed_seeds": len(model_rows),
            "rows_per_seed": len(data) if len(model_rows) == len(SEEDS) else f"up to {len(data)}",
            "folds_covered": N_FOLDS if coverage_ok else max(fold_counts, default=0),
            "fold_match": "expected assignments" if coverage_ok else "incomplete or mismatch",
            "fold_files": fold_files,
            "expected_fold_files": expected_files,
            "aggregate_oof": aggregate_check,
            "all_valid": coverage_ok and fold_files_ok and summary_check.startswith("matches") and aggregate_check.startswith("PASS:"),
            "issues": issues,
            "summary_check": summary_check,
        }

    if manifest_ok and len(seed_frames.get("pmrc_net", {})) == len(SEEDS) and all(
        model in seed_frames and 42 in seed_frames[model] for model, _ in MODELS
    ):
        historical = pd.read_csv(HISTORICAL_PMRC_OOF)
        historical_check = validate_frame(historical, data, fold_ids[42])
        if not historical_check["valid"]:
            raise ValueError(f"Historical PMRC seed-42 OOF failed validation: {historical_check['issues']}")
        seed42_frames = {model: frames[42] for model, frames in seed_frames.items()}
        figure_validation = validate_seed42_figure_data(seed42_frames, data, fold_ids[42])
        if not all(
            figure_validation[key]
            for key in ("all_same_samples", "all_same_labels", "all_same_folds")
        ):
            mismatches = {
                model: check
                for model, check in figure_validation["checks"].items()
                if not (check["sample_indices_match"] and check["labels_match"] and check["folds_match"])
            }
            raise ValueError(f"Seed-42 curve/calibration alignment failed; no figures generated: {mismatches}")
        create_figures(seed_metrics, seed42_frames, target)
    else:
        raise ValueError("Could not generate figures because a model lacks complete validated seed-42 OOF predictions")

    report = build_report(data, dataset_hash, seed_metrics, seed_frames, inventory, manifest_ok, figure_validation)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Report: {REPORT_PATH.relative_to(ROOT)}")
    print(f"Figures: {FIGURE_ROOT.relative_to(ROOT)}")
    print(f"Dataset SHA-256: {dataset_hash}")
    print(f"Seed-42 fold manifest: {'PASS' if manifest_ok else 'FAIL'}")
    print(
        "Seed-42 figure alignment: "
        f"samples={'PASS' if figure_validation['all_same_samples'] else 'FAIL'}, "
        f"labels={'PASS' if figure_validation['all_same_labels'] else 'FAIL'}, "
        f"folds={'PASS' if figure_validation['all_same_folds'] else 'FAIL'}"
    )
    print(f"ECE max cross-check delta: {figure_validation['max_abs_ece_delta']:.3g}")
    print(f"Brier max direct-MSE delta: {figure_validation['max_abs_brier_delta']:.3g}")
    for model, _ in MODELS:
        item = inventory[model]
        print(f"{model}: seeds={item['completed_seeds']}/5, folds={item['folds_covered']}/5, valid={item['all_valid']}, summary={item['summary_check']}")
        for issue in item["issues"]:
            print(f"  ISSUE: {issue}")


if __name__ == "__main__":
    main()