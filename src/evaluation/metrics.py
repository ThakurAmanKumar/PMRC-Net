from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

BinaryMetric = float | int | None
ThresholdMethod = Literal["default", "youden", "f1", "sensitivity_constrained"]


def _validate_binary_inputs(labels: Any, probabilities: Any) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or labels.shape != probabilities.shape or labels.size == 0:
        raise ValueError("Labels and probabilities must be non-empty, aligned one-dimensional arrays")
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must contain only finite binary values 0 and 1")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1]")
    return labels.astype(int), probabilities


def expected_calibration_error(
    labels: Any,
    probabilities: Any,
    bins: int = 10,
) -> float:
    """Equal-width ECE; bins are left-closed/right-open, except the last bin."""
    labels, probabilities = _validate_binary_inputs(labels, probabilities)
    if bins < 1:
        raise ValueError("bins must be at least one")
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if in_bin.any():
            error += float(in_bin.mean()) * abs(float(labels[in_bin].mean() - probabilities[in_bin].mean()))
    return float(error)


def maximum_calibration_error(
    labels: Any,
    probabilities: Any,
    bins: int = 10,
) -> float:
    """Maximum absolute confidence-accuracy gap across non-empty equal-width bins."""
    labels, probabilities = _validate_binary_inputs(labels, probabilities)
    if bins < 1:
        raise ValueError("bins must be at least one")
    edges = np.linspace(0.0, 1.0, bins + 1)
    errors = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if in_bin.any():
            errors.append(abs(float(labels[in_bin].mean() - probabilities[in_bin].mean())))
    return float(max(errors)) if errors else float("nan")


def confusion_matrices(labels: Any, predicted: Any) -> dict[str, np.ndarray]:
    """Return 2x2 matrices with rows=actual and columns=predicted, ordered 0 then 1."""
    labels = np.asarray(labels)
    predicted = np.asarray(predicted)
    if labels.ndim != 1 or predicted.ndim != 1 or labels.shape != predicted.shape or labels.size == 0:
        raise ValueError("Labels and predictions must be non-empty, aligned one-dimensional arrays")
    if not np.isin(labels, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("Labels and predictions must contain only binary values 0 and 1")
    raw = confusion_matrix(labels, predicted, labels=[0, 1]).astype(np.int64)
    row_sums = raw.sum(axis=1, keepdims=True)
    column_sums = raw.sum(axis=0, keepdims=True)
    row_normalized = np.divide(
        raw, row_sums, out=np.full(raw.shape, np.nan, dtype=float), where=row_sums != 0,
    )
    column_normalized = np.divide(
        raw, column_sums, out=np.full(raw.shape, np.nan, dtype=float), where=column_sums != 0,
    )
    return {
        "raw": raw,
        "row_normalized": row_normalized,
        "column_normalized": column_normalized,
    }


def calculate_metrics(
    y_true: Any,
    probabilities: Any,
    threshold: float | None = 0.5,
    ece_bins: int = 10,
    predicted_labels: Any | None = None,
) -> dict[str, BinaryMetric]:
    """Calculate binary classification and probability metrics.

    Undefined ratios and one-class ranking metrics are ``None`` rather than
    silently coerced to zero or chance. PR-AUC is trapezoidal area under the
    precision-recall curve; Average Precision is reported separately.
    ECE/MCE use ``ece_bins`` equal-width bins on [0, 1].
    """
    labels, probabilities = _validate_binary_inputs(y_true, probabilities)
    if predicted_labels is None:
        if threshold is None or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be finite and in [0, 1]")
        predicted = (probabilities >= threshold).astype(int)
    else:
        predicted = np.asarray(predicted_labels)
        if predicted.shape != labels.shape or not np.isin(predicted, [0, 1]).all():
            raise ValueError("Predicted labels must be aligned binary values")
        predicted = predicted.astype(int)
        threshold = None
    if ece_bins < 1:
        raise ValueError("ece_bins must be at least one")

    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()

    def ratio(numerator: float, denominator: float) -> float | None:
        return float(numerator / denominator) if denominator else None

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    f2 = ratio(5 * tp, 5 * tp + 4 * fn + fp)
    fpr = ratio(fp, fp + tn)
    fnr = ratio(fn, fn + tp)
    fdr = ratio(fp, fp + tp)
    for_ = ratio(fn, fn + tn)
    npv = ratio(tn, tn + fn)
    balanced_accuracy = (
        float((recall + specificity) / 2)
        if recall is not None and specificity is not None
        else None
    )
    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ratio((tp * tn) - (fp * fn), float(mcc_denominator))
    observed_agreement = (tp + tn) / labels.size
    expected_agreement = (
        (tn + fp) * (tn + fn) + (fn + tp) * (fp + tp)
    ) / labels.size**2
    kappa = (
        float((observed_agreement - expected_agreement) / (1 - expected_agreement))
        if expected_agreement != 1
        else None
    )

    both_classes = np.unique(labels).size == 2
    if both_classes:
        roc_auc = float(roc_auc_score(labels, probabilities))
        average_precision = float(average_precision_score(labels, probabilities))
        precision_curve, recall_curve, _ = precision_recall_curve(labels, probabilities)
        pr_auc = float(auc(recall_curve, precision_curve))
    else:
        roc_auc = average_precision = pr_auc = None

    clipped = np.clip(probabilities, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return {
        "threshold": None if threshold is None else float(threshold),
        "accuracy": float((tp + tn) / labels.size),
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "f2": f2,
        "mcc": mcc,
        "balanced_accuracy": balanced_accuracy,
        "negative_predictive_value": npv,
        "npv": npv,
        "positive_predictive_value": precision,
        "ppv": precision,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "false_discovery_rate": fdr,
        "false_omission_rate": for_,
        "cohen_kappa": kappa,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "pr_auc": pr_auc,
        "log_loss": float(log_loss(labels, np.column_stack((1 - clipped, clipped)), labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities, ece_bins),
        "mce": maximum_calibration_error(labels, probabilities, ece_bins),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold(
    labels: Any,
    probabilities: Any,
    method: ThresholdMethod = "youden",
    minimum_sensitivity: float | None = None,
) -> dict[str, float | str | bool | None]:
    """Select a threshold from supplied validation predictions only.

    Sensitivity-constrained selection requires an explicit target and maximizes
    specificity among thresholds that meet it. Ties are resolved toward 0.5.
    """
    labels, probabilities = _validate_binary_inputs(labels, probabilities)
    if method == "default":
        threshold = 0.5
        score = None
        constraint_met = True
    else:
        if method == "sensitivity_constrained":
            if minimum_sensitivity is None or not 0 < minimum_sensitivity <= 1:
                raise ValueError("sensitivity_constrained requires minimum_sensitivity in (0, 1]")
        elif method not in {"youden", "f1"}:
            raise ValueError(f"Unsupported threshold method: {method}")
        candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
        evaluated = []
        for candidate in candidates:
            metrics = calculate_metrics(labels, probabilities, float(candidate))
            if method == "youden":
                objective = (
                    float(metrics["sensitivity"]) + float(metrics["specificity"]) - 1
                    if metrics["sensitivity"] is not None and metrics["specificity"] is not None
                    else float("-inf")
                )
                feasible = True
            elif method == "f1":
                objective = float(metrics["f1"]) if metrics["f1"] is not None else float("-inf")
                feasible = True
            else:
                sensitivity = metrics["sensitivity"]
                feasible = sensitivity is not None and sensitivity >= minimum_sensitivity
                objective = float(metrics["specificity"]) if feasible and metrics["specificity"] is not None else float("-inf")
            evaluated.append((objective, -abs(float(candidate) - 0.5), float(candidate), feasible))
        best = max(evaluated)
        score, _, threshold, constraint_met = best
        if not np.isfinite(score):
            score = None

    selected_metrics = calculate_metrics(labels, probabilities, threshold)
    return {
        "method": method,
        "threshold": float(threshold),
        "selection_score": float(score) if score is not None else None,
        "minimum_sensitivity": minimum_sensitivity,
        "constraint_met": bool(constraint_met),
        "validation_sensitivity": selected_metrics["sensitivity"],
        "validation_specificity": selected_metrics["specificity"],
        "validation_f1": selected_metrics["f1"],
    }
