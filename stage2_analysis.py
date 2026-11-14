from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from src.evaluation.metrics import calculate_metrics


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        if mask.any():
            total += mask.mean() * abs(y_true[mask].mean() - probability[mask].mean())
    return float(total)


def main() -> None:
    root = Path(__file__).resolve().parent
    stage = root / "results" / "stage2_validation"
    oof = pd.read_csv(stage / "oof_predictions.csv")
    y_true = oof["target"].to_numpy(dtype=int)
    probability = oof["probability"].to_numpy(dtype=float)
    (stage / "calibration").mkdir(exist_ok=True)
    (stage / "explainability").mkdir(exist_ok=True)
    (stage / "perturbation_tests").mkdir(exist_ok=True)
    (stage / "plots").mkdir(exist_ok=True)

    rows = []
    for threshold in np.arange(0.20, 0.81, 0.05):
        rows.append(calculate_metrics(y_true, probability, float(threshold)))
    pd.DataFrame(rows).to_csv(stage / "calibration" / "threshold_sensitivity.csv", index=False)
    calibration = {"model": "PMRC-Net", "ece_10_equal_width": expected_calibration_error(y_true, probability),
                   "brier_score": float(np.mean((probability - y_true) ** 2)),
                   "note": "ECE is descriptive and bin-dependent; no calibration model was fit on outer folds."}
    (stage / "calibration" / "calibration_metrics.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    fraction, mean_probability = calibration_curve(y_true, probability, n_bins=10, strategy="uniform")
    calibration_frame = pd.DataFrame({"mean_predicted_probability": mean_probability, "fraction_positive": fraction})
    calibration_frame.to_csv(stage / "calibration" / "reliability_diagram.csv", index=False)
    plt.figure(figsize=(5, 5))
    plt.plot(mean_probability, fraction, marker="o", label="PMRC-Net")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Ideal")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction positive")
    plt.legend()
    plt.tight_layout()
    plt.savefig(stage / "plots" / "reliability_diagram.png", dpi=150)
    plt.close()

    (stage / "explainability" / "README.md").write_text(
        """# Explainability outputs\n\n"
        "The current validated artifact is an OOF prediction audit. SHAP and Integrated Gradients are optional follow-up analyses and are not represented by fabricated values here.\n\n"
        "Feature attribution must be computed from each fold's training-fitted model and preprocessing, then summarized across folds. Any attribution is associative/model-based evidence, not causal evidence.\n""",
        encoding="utf-8")
    (stage / "perturbation_tests" / "README.md").write_text(
        """# Perturbation and monotonicity tests\n\n"
        "The original physiological hinge loss is inactive in this experiment because all training-fold empirical directions are negative and the implementation activates only positive-direction constraints. This is recorded rather than overridden.\n\n"
        "Perturbation results must be generated from fold-specific checkpoints, with the perturbation defined before inspecting outer validation outcomes. They should be reported as model sensitivity analyses, never as causal or guaranteed clinical-safety evidence.\n""",
        encoding="utf-8")
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()