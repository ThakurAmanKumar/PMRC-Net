from __future__ import annotations

import unittest

import numpy as np

from src.evaluation.metrics import (
    calculate_metrics,
    confusion_matrices,
    expected_calibration_error,
    maximum_calibration_error,
    select_threshold,
)


class BinaryMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = np.array([0, 0, 1, 1])
        self.probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    def test_metrics_match_hand_calculated_perfect_classification(self) -> None:
        metrics = calculate_metrics(self.labels, self.probabilities)
        for key in (
            "accuracy", "precision", "recall", "sensitivity", "specificity", "f1", "f2",
            "mcc", "balanced_accuracy", "npv", "ppv", "cohen_kappa", "roc_auc",
            "average_precision", "pr_auc",
        ):
            self.assertAlmostEqual(metrics[key], 1.0, msg=key)
        self.assertAlmostEqual(metrics["brier_score"], 0.085)
        self.assertAlmostEqual(metrics["ece"], 0.25)
        self.assertAlmostEqual(metrics["mce"], 0.4)
        self.assertAlmostEqual(metrics["false_positive_rate"], 0.0)
        self.assertAlmostEqual(metrics["false_negative_rate"], 0.0)
        self.assertAlmostEqual(metrics["false_discovery_rate"], 0.0)
        self.assertAlmostEqual(metrics["false_omission_rate"], 0.0)
        self.assertEqual((metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]), (2, 0, 0, 2))

    def test_confusion_matrix_orientation_and_normalization(self) -> None:
        matrices = confusion_matrices(self.labels, np.array([0, 1, 0, 1]))
        np.testing.assert_array_equal(matrices["raw"], [[1, 1], [1, 1]])
        np.testing.assert_allclose(matrices["row_normalized"], [[0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_allclose(matrices["column_normalized"], [[0.5, 0.5], [0.5, 0.5]])

    def test_metrics_accept_fold_specific_threshold_predictions(self) -> None:
        metrics = calculate_metrics(
            self.labels,
            self.probabilities,
            predicted_labels=np.array([0, 1, 1, 1]),
        )
        self.assertIsNone(metrics["threshold"])
        self.assertEqual((metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]), (1, 1, 0, 2))

    def test_undefined_metrics_are_not_coerced_to_zero(self) -> None:
        metrics = calculate_metrics(np.array([0, 0]), np.array([0.2, 0.3]))
        self.assertIsNone(metrics["precision"])
        self.assertIsNone(metrics["sensitivity"])
        self.assertIsNone(metrics["balanced_accuracy"])
        self.assertIsNone(metrics["mcc"])
        self.assertIsNone(metrics["roc_auc"])
        self.assertIsNone(metrics["average_precision"])
        self.assertIsNone(metrics["pr_auc"])
        self.assertEqual(metrics["specificity"], 1.0)

    def test_calibration_helpers_use_equal_width_bins(self) -> None:
        self.assertAlmostEqual(expected_calibration_error(self.labels, self.probabilities, bins=10), 0.25)
        self.assertAlmostEqual(maximum_calibration_error(self.labels, self.probabilities, bins=10), 0.4)

    def test_threshold_selection_methods_use_supplied_validation_predictions(self) -> None:
        self.assertEqual(select_threshold(self.labels, self.probabilities, "default")["threshold"], 0.5)
        youden = select_threshold(self.labels, self.probabilities, "youden")
        self.assertGreaterEqual(youden["threshold"], 0.0)
        self.assertLessEqual(youden["threshold"], 1.0)
        self.assertEqual(select_threshold(self.labels, self.probabilities, "f1")["validation_f1"], 1.0)
        constrained = select_threshold(
            self.labels,
            self.probabilities,
            "sensitivity_constrained",
            minimum_sensitivity=1.0,
        )
        self.assertTrue(constrained["constraint_met"])
        self.assertEqual(constrained["validation_sensitivity"], 1.0)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "aligned"):
            calculate_metrics([0, 1], [0.3])
        with self.assertRaisesRegex(ValueError, "binary"):
            calculate_metrics([0, 2], [0.2, 0.8])
        with self.assertRaisesRegex(ValueError, "in \[0, 1\]"):
            calculate_metrics([0, 1], [0.2, 1.2])
        with self.assertRaisesRegex(ValueError, "minimum_sensitivity"):
            select_threshold(self.labels, self.probabilities, "sensitivity_constrained")


if __name__ == "__main__":
    unittest.main()