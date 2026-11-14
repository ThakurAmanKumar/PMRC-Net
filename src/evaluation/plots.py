from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve


FIGURE_SIZE = (8.0, 6.0)
EXPORT_FORMATS = ("png", "pdf", "svg")


def _style() -> None:
    plt.rcParams.update({
        "figure.figsize": FIGURE_SIZE,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.frameon": False,
        "svg.fonttype": "none",
    })


def _save(fig: plt.Figure, output_dir: str | Path, name: str) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in EXPORT_FORMATS:
        path = output / f"{name}.{extension}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _unavailable_plot(output_dir: str | Path, name: str, title: str, reason: str) -> list[Path]:
    _style()
    fig, axis = plt.subplots()
    axis.axis("off")
    axis.set_title(title, pad=18)
    axis.text(0.5, 0.5, reason, ha="center", va="center", wrap=True, transform=axis.transAxes)
    return _save(fig, output_dir, name)


def plot_roc_curves(
    predictions: Mapping[str, tuple[Sequence[int], Sequence[float]]],
    output_dir: str | Path,
    name: str = "01_roc_curve",
) -> list[Path]:
    _style()
    fig, axis = plt.subplots()
    for model_name, (labels, probabilities) in predictions.items():
        labels_array = np.asarray(labels, dtype=int)
        if np.unique(labels_array).size < 2:
            continue
        fpr, tpr, _ = roc_curve(labels_array, probabilities)
        axis.plot(fpr, tpr, linewidth=1.8, label=f"{model_name} (AUC {auc(fpr, tpr):.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1, label="Chance")
    axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate", title="Outer OOF ROC curves")
    axis.legend(loc="lower right")
    return _save(fig, output_dir, name)


def plot_precision_recall_curves(
    predictions: Mapping[str, tuple[Sequence[int], Sequence[float]]],
    output_dir: str | Path,
    name: str = "02_precision_recall_curve",
) -> list[Path]:
    _style()
    fig, axis = plt.subplots()
    for model_name, (labels, probabilities) in predictions.items():
        labels_array = np.asarray(labels, dtype=int)
        if np.unique(labels_array).size < 2:
            continue
        precision, recall, _ = precision_recall_curve(labels_array, probabilities)
        average_precision = float(average_precision_score(labels_array, probabilities))
        axis.plot(recall, precision, linewidth=1.8, label=f"{model_name} (AP {average_precision:.3f})")
    axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision", title="Outer OOF precision-recall curves")
    axis.legend(loc="best")
    return _save(fig, output_dir, name)


def plot_confusion_matrix(
    matrix: np.ndarray,
    output_dir: str | Path,
    name: str = "03_confusion_matrix",
    normalized: bool = False,
) -> list[Path]:
    _style()
    values = np.asarray(matrix, dtype=float if normalized else int)
    if values.shape != (2, 2):
        raise ValueError("Confusion matrix must be 2x2 with rows actual and columns predicted")
    fig, axis = plt.subplots()
    image = axis.imshow(np.ma.masked_invalid(values), cmap="Blues", vmin=0, vmax=1 if normalized else None)
    fig.colorbar(image, ax=axis, label="Proportion" if normalized else "Count")
    axis.set_xticks([0, 1], labels=["Predicted Negative", "Predicted Positive"])
    axis.set_yticks([0, 1], labels=["Actual Negative", "Actual Positive"])
    axis.set_title("Normalized confusion matrix" if normalized else "Confusion matrix")
    midpoint = float(np.nanmax(values)) / 2 if np.isfinite(values).any() else 0
    for row in range(2):
        for column in range(2):
            value = values[row, column]
            label = "undefined" if not np.isfinite(value) else f"{value:.3f}" if normalized else f"{value:d}"
            axis.text(column, row, label, ha="center", va="center", color="white" if np.isfinite(value) and value > midpoint else "black")
    axis.set_ylabel("Actual class")
    axis.set_xlabel("Predicted class")
    return _save(fig, output_dir, name)


def plot_calibration_curve(
    labels: Sequence[int],
    probabilities: Sequence[float],
    output_dir: str | Path,
    bins: int = 10,
    name: str = "05_calibration_curve",
) -> list[Path]:
    _style()
    labels_array = np.asarray(labels, dtype=int)
    if labels_array.size == 0 or np.unique(labels_array).size < 2:
        return _unavailable_plot(output_dir, name, "Calibration curve", "Both outcome classes are required.")
    observed, predicted = calibration_curve(labels_array, probabilities, n_bins=bins, strategy="uniform")
    fig, axis = plt.subplots()
    axis.plot(predicted, observed, marker="o", linewidth=1.8, label=f"Model ({bins} equal-width bins)")
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.55", label="Perfect calibration")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed event frequency", title="Outer OOF reliability curve")
    axis.legend(loc="best")
    return _save(fig, output_dir, name)


def plot_training_curves(
    history: pd.DataFrame | None,
    output_dir: str | Path,
    name: str = "06_training_curves",
) -> list[Path]:
    if history is None or history.empty or not {"epoch", "train_loss", "valid_loss"}.issubset(history.columns):
        return _unavailable_plot(
            output_dir,
            name,
            "Training and validation loss",
            "Fold-aligned epoch loss history is unavailable for the selected saved OOF run; no curve was inferred from aggregate metrics.",
        )
    _style()
    fig, axis = plt.subplots()
    grouped = history.groupby("epoch")[["train_loss", "valid_loss"]].mean()
    axis.plot(grouped.index, grouped["train_loss"], marker="o", label="Training loss")
    axis.plot(grouped.index, grouped["valid_loss"], marker="o", label="Validation loss")
    axis.set(xlabel="Epoch", ylabel="Loss", title="Mean training history over available folds")
    axis.legend()
    return _save(fig, output_dir, name)


def plot_fold_accuracy(
    fold_metrics: pd.DataFrame,
    output_dir: str | Path,
    name: str = "07_fold_comparison",
) -> list[Path]:
    required = {"seed", "fold", "accuracy"}
    if fold_metrics.empty or not required.issubset(fold_metrics.columns):
        return _unavailable_plot(output_dir, name, "Fold comparison", "Fold-level metrics are unavailable.")
    _style()
    fig, axis = plt.subplots()
    for seed, group in fold_metrics.groupby("seed"):
        axis.plot(group["fold"], group["accuracy"], marker="o", alpha=0.8, label=f"Seed {seed}")
    axis.set(xticks=sorted(fold_metrics["fold"].unique()), xlabel="Outer fold", ylabel="Accuracy", ylim=(0, 1), title="Accuracy across outer folds")
    axis.legend(ncol=2)
    return _save(fig, output_dir, name)


def plot_sensitivity_specificity(
    fold_metrics: pd.DataFrame,
    output_dir: str | Path,
    name: str = "08_sensitivity_specificity",
) -> list[Path]:
    required = {"seed", "fold", "sensitivity", "specificity"}
    if fold_metrics.empty or not required.issubset(fold_metrics.columns):
        return _unavailable_plot(output_dir, name, "Sensitivity and specificity", "Fold-level sensitivity/specificity are unavailable.")
    _style()
    fig, axis = plt.subplots()
    summary = fold_metrics.groupby("fold")[["sensitivity", "specificity"]].agg(["mean", "std"])
    for metric, color in (("sensitivity", "#167D8D"), ("specificity", "#C55A11")):
        means = summary[(metric, "mean")]
        errors = summary[(metric, "std")].fillna(0)
        axis.errorbar(means.index, means, yerr=errors, marker="o", capsize=3, label=metric.title(), color=color)
    axis.set(xticks=summary.index, xlabel="Outer fold", ylabel="Rate", ylim=(0, 1), title="Sensitivity and specificity by fold")
    axis.legend()
    return _save(fig, output_dir, name)


def plot_mcc_comparison(
    fold_metrics: pd.DataFrame,
    output_dir: str | Path,
    name: str = "09_mcc_comparison",
) -> list[Path]:
    if fold_metrics.empty or not {"seed", "mcc"}.issubset(fold_metrics.columns):
        return _unavailable_plot(output_dir, name, "MCC comparison", "Fold- or seed-level MCC values are unavailable.")
    _style()
    fig, axis = plt.subplots()
    groups = [group["mcc"].to_numpy() for _, group in fold_metrics.groupby("seed")]
    labels = [str(seed) for seed in sorted(fold_metrics["seed"].unique())]
    axis.boxplot(groups, labels=labels, showmeans=True)
    axis.set(xlabel="Seed", ylabel="MCC", title="Fold-wise MCC distributions by seed")
    return _save(fig, output_dir, name)


def plot_seed_comparison(
    seed_metrics: pd.DataFrame,
    output_dir: str | Path,
    name: str = "08_seed_comparison",
    metrics: Sequence[str] = ("accuracy", "sensitivity", "specificity", "mcc", "roc_auc"),
) -> list[Path]:
    available = [metric for metric in metrics if metric in seed_metrics]
    if seed_metrics.empty or "seed" not in seed_metrics or not available:
        return _unavailable_plot(output_dir, name, "Seed-wise performance", "Seed-level metrics are unavailable.")
    _style()
    fig, axis = plt.subplots(figsize=(9, 6))
    x = np.arange(len(seed_metrics))
    width = 0.8 / len(available)
    for index, metric in enumerate(available):
        axis.bar(x - 0.4 + width * (index + 0.5), seed_metrics[metric], width, label=metric)
    axis.set(xticks=x, xticklabels=seed_metrics["seed"].astype(str), xlabel="Seed", ylabel="Metric value", ylim=(0, 1), title="Seed-wise pooled OOF performance")
    axis.legend(ncol=2)
    return _save(fig, output_dir, name)


def plot_probability_distribution(
    labels: Sequence[int],
    probabilities: Sequence[float],
    output_dir: str | Path,
    name: str = "12_probability_distribution",
) -> list[Path]:
    labels_array = np.asarray(labels, dtype=int)
    probabilities_array = np.asarray(probabilities, dtype=float)
    if labels_array.size == 0 or labels_array.shape != probabilities_array.shape:
        return _unavailable_plot(output_dir, name, "Predicted probability distribution", "Aligned OOF labels and probabilities are unavailable.")
    _style()
    fig, axis = plt.subplots()
    for target, label, color in ((0, "Actual negative", "#167D8D"), (1, "Actual positive", "#C55A11")):
        values = probabilities_array[labels_array == target]
        if values.size:
            axis.hist(values, bins=np.linspace(0, 1, 21), alpha=0.58, density=True, label=label, color=color)
    axis.set(xlim=(0, 1), xlabel="Predicted probability", ylabel="Density", title="OOF probability distribution by actual class")
    axis.legend()
    return _save(fig, output_dir, name)


def plot_component_contributions(
    predictions: pd.DataFrame | None,
    output_dir: str | Path,
    name: str = "11_component_analysis",
) -> list[Path]:
    columns = ("z_mech", "z_inter", "z_res")
    if predictions is None or predictions.empty or not set(columns).issubset(predictions.columns):
        return _unavailable_plot(
            output_dir,
            name,
            "Component logit contributions",
            "Component logits are not present in the selected five-seed OOF artifact; no values were reconstructed from total logits.",
        )
    _style()
    fig, axis = plt.subplots()
    axis.boxplot([predictions[column].to_numpy() for column in columns], labels=["Mechanistic", "Interaction", "Residual"], showmeans=True)
    axis.axhline(0, color="0.4", linewidth=1)
    axis.set(ylabel="Logit contribution", title="Saved OOF component logit distributions")
    return _save(fig, output_dir, name)


def plot_metric_comparison(
    frame: pd.DataFrame,
    model_column: str,
    output_dir: str | Path,
    metrics: Sequence[str],
    name: str,
    title: str,
) -> list[Path]:
    available = [metric for metric in metrics if metric in frame]
    if frame.empty or model_column not in frame or not available:
        return _unavailable_plot(output_dir, name, title, "Comparable model metrics are unavailable.")
    _style()
    fig, axis = plt.subplots(figsize=(10, 6))
    x = np.arange(len(frame))
    width = 0.85 / len(available)
    for index, metric in enumerate(available):
        axis.bar(x - 0.425 + width * (index + 0.5), frame[metric], width, label=metric)
    axis.set(xticks=x, xticklabels=frame[model_column].astype(str), ylabel="Metric value", ylim=(0, 1), title=title)
    axis.tick_params(axis="x", labelrotation=35)
    axis.legend(ncol=3)
    return _save(fig, output_dir, name)


def plot_ablation_comparison(
    ablations: pd.DataFrame,
    output_dir: str | Path,
    name: str = "09_ablation_comparison",
) -> list[Path]:
    return plot_metric_comparison(
        ablations, "variant", output_dir,
        ("accuracy", "sensitivity", "specificity", "f1", "mcc", "roc_auc", "average_precision", "brier_score", "ece"),
        name, "Archived PMRC-Net ablation metrics (scope labeled in the accompanying report)",
    )


def plot_baseline_comparison(
    baselines: pd.DataFrame,
    output_dir: str | Path,
    name: str = "10_baseline_comparison",
) -> list[Path]:
    return plot_metric_comparison(
        baselines, "model_name", output_dir,
        ("accuracy", "sensitivity", "specificity", "f1", "mcc", "roc_auc", "average_precision", "brier_score"),
        name, "Archived five-seed nested model comparison",
    )