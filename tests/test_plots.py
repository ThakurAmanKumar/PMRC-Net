from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.plots import (
    plot_calibration_curve,
    plot_component_contributions,
    plot_confusion_matrix,
    plot_precision_recall_curves,
    plot_roc_curves,
)


class ResearchPlotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = np.array([0, 0, 1, 1])
        self.probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    def assert_exported(self, paths: list[Path]) -> None:
        self.assertEqual({path.suffix for path in paths}, {".png", ".pdf", ".svg"})
        self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in paths))

    def test_curves_and_confusion_matrix_export_from_input_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            predictions = {"PMRC-Net": (self.labels, self.probabilities)}
            self.assert_exported(plot_roc_curves(predictions, output))
            self.assert_exported(plot_precision_recall_curves(predictions, output))
            self.assert_exported(plot_calibration_curve(self.labels, self.probabilities, output))
            self.assert_exported(plot_confusion_matrix(np.array([[2, 0], [0, 2]]), output))

    def test_missing_component_logits_are_marked_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = plot_component_contributions(pd.DataFrame({"probability": [0.2]}), temporary)
            self.assert_exported(paths)


if __name__ == "__main__":
    unittest.main()