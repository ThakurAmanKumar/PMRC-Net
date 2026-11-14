from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import run_additional_baselines as baseline_runner
from src.data.dataset import load_original_dataset

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "dataset" / "cleaned_merged_heart_dataset.csv"


class AdditionalBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataframe = load_original_dataset(DATASET_PATH, baseline_runner.TARGET_COLUMN, deduplicate=True)
        cls.folds = baseline_runner.generate_seed_folds(cls.dataframe, 42)

    def test_folds_cover_every_row_exactly_once(self) -> None:
        seen = []
        for train_idx, valid_idx in self.folds:
            seen.extend(valid_idx.tolist())
            self.assertEqual(len(np.unique(valid_idx)), len(valid_idx))
            self.assertTrue(np.isin(train_idx, np.arange(len(self.dataframe))).all())
            self.assertTrue(np.isin(valid_idx, np.arange(len(self.dataframe))).all())
        self.assertEqual(len(seen), len(self.dataframe))
        self.assertEqual(len(np.unique(seen)), len(self.dataframe))

    def test_metric_calculation_matches_reference_values(self) -> None:
        y_true = np.array([0, 0, 1, 1], dtype=int)
        probabilities = np.array([0.1, 0.4, 0.6, 0.9], dtype=float)
        metrics = baseline_runner.compute_metrics(y_true, probabilities)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["brier_score"], 0.085, places=3)

    def test_probability_range_and_target_consistency(self) -> None:
        sample = pd.DataFrame({
            "sample_index": np.arange(4),
            "fold": [1, 1, 2, 2],
            "target": [0, 1, 1, 0],
            "probability": [0.1, 0.9, 0.25, 0.78],
        })
        self.assertTrue(((sample["probability"] >= 0) & (sample["probability"] <= 1)).all())
        self.assertTrue(set(sample["target"].unique()).issubset({0, 1}))

    def test_dataset_hash_matches_expected_root_file(self) -> None:
        digest = baseline_runner.ensure_dataset_hash()
        self.assertTrue(isinstance(digest, str) and len(digest) == 64)
        self.assertEqual(len(self.dataframe), 602)

    def test_seed_folds_match_preserved_stage5_oof_manifests(self) -> None:
        sample_to_fold_by_seed = {}
        for seed in baseline_runner.SEEDS:
            sample_to_fold_by_seed[seed] = {
                int(self.dataframe.iloc[index]["sample_index"]): fold_index
                for fold_index, (_, valid_idx) in enumerate(
                    baseline_runner.generate_seed_folds(self.dataframe, seed),
                    start=1,
                )
                for index in valid_idx
            }
        for seed in baseline_runner.SEEDS:
            path = (
                ROOT / "results" / "stage5_final_validation"
                / "seed_oof_predictions" / f"seed_{seed}_oof.csv"
            )
            saved_oof = pd.read_csv(path)
            self.assertEqual(len(saved_oof), len(self.dataframe))
            self.assertFalse(saved_oof["sample_index"].duplicated().any())
            expected_folds = sample_to_fold_by_seed[seed]
            for row in saved_oof.itertuples(index=False):
                sample_index = int(row.sample_index)
                self.assertEqual(int(row.fold), expected_folds[sample_index])
                expected_target = int(
                    self.dataframe.loc[
                        self.dataframe["sample_index"] == sample_index,
                        baseline_runner.TARGET_COLUMN,
                    ].iloc[0]
                )
                self.assertEqual(int(row.target), expected_target)


if __name__ == "__main__":
    unittest.main()
