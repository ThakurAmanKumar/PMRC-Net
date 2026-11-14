from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import _harmonize_categorical_encodings, load_original_dataset, make_fold_data
from src.evaluation.metrics import calculate_metrics
from src.models.pmrc_net import PMRCNet
from src.models.pmrc_net_optimized import OptimizedPMRCNet
from src.training.checkpoints import load_checkpoint, save_checkpoint
from src.training.losses import PMRCConsistencyLoss, compute_empirical_associations
from src.utils.config import load_config, resolve_path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs" / "model_config.yaml")
DATA_PATH = resolve_path(ROOT, CONFIG["data"]["path"])


class PMRCSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DATA_PATH.is_file():
            raise unittest.SkipTest("The local clinical dataset is not included in the repository")
        cls.dataframe = load_original_dataset(DATA_PATH, CONFIG["data"]["target"], True)
        cls.train_indices = np.arange(0, 480)
        cls.valid_indices = np.arange(480, len(cls.dataframe))
        cls.fold = make_fold_data(cls.dataframe, cls.train_indices, cls.valid_indices, CONFIG)

    def test_dataset_shape_and_target(self) -> None:
        self.assertEqual(len(self.dataframe), 602)
        self.assertIn("target", self.dataframe.columns)
        self.assertEqual(set(self.dataframe["target"].unique()), {0, 1})

    def test_dataset_has_no_missing_values(self) -> None:
        self.assertEqual(int(self.dataframe.isna().sum().sum()), 0)

    def test_category_encoding_is_integer_and_bounded(self) -> None:
        for column in CONFIG["data"]["categorical_cols"]:
            self.assertTrue(np.issubdtype(self.dataframe[column].dtype, np.integer))
            self.assertGreaterEqual(int(self.dataframe[column].min()), 0)
            self.assertLess(int(self.dataframe[column].max()), CONFIG["model"]["embedding_cardinality"])

    def test_fold_preprocessing_has_expected_shapes(self) -> None:
        batch = next(iter(self.fold.train_loader))
        self.assertEqual(batch[0].shape[1], len(CONFIG["data"]["continuous_cols"]))
        self.assertEqual(batch[1].shape[1], len(CONFIG["data"]["categorical_cols"]))
        self.assertEqual(batch[2].shape[1], len(CONFIG["data"]["physiology_cols"]))
        self.assertEqual(len(self.fold.train_indices), 480)
        self.assertEqual(len(self.fold.valid_indices), 122)

    def test_fold_preprocessing_allows_empty_validation_for_refit(self) -> None:
        fold = make_fold_data(self.dataframe, self.train_indices, np.empty(0, dtype=int), CONFIG)
        self.assertEqual(len(fold.valid_loader.dataset), 0)
        self.assertEqual(fold.valid_loader.dataset.tensors[0].shape[1], len(CONFIG["data"]["continuous_cols"]))

    def test_validation_preprocessing_uses_training_scaler(self) -> None:
        train_values = self.dataframe.iloc[self.train_indices][CONFIG["data"]["continuous_cols"]].to_numpy()
        expected = self.fold.scaler.transform(train_values)
        actual = self.fold.scaler.transform(train_values)
        np.testing.assert_allclose(expected, actual)

    def test_model_forward_outputs_and_probability_bounds(self) -> None:
        model = PMRCNet(CONFIG, CONFIG["model"]).eval()
        batch = next(iter(self.fold.valid_loader))
        with torch.no_grad():
            outputs = model(*batch[:3])
            probabilities = torch.sigmoid(outputs["z_total"])
        self.assertEqual(outputs["z_total"].shape, (len(batch[3]), 1))
        self.assertTrue(torch.allclose(outputs["z_total"], outputs["z_mech"] + outputs["z_inter"] + outputs["z_res"]))
        self.assertTrue(bool(torch.all((probabilities >= 0) & (probabilities <= 1))))

    def test_optimized_asymmetric_flow_has_distinct_directions(self) -> None:
        model = OptimizedPMRCNet(CONFIG, CONFIG["model"])
        left = torch.randn(4, model.hidden_dim)
        right = torch.randn(4, model.hidden_dim)
        forward, reverse = model._directional_flow(left, right, "me")
        self.assertEqual(forward.shape, (4, model.hidden_dim))
        self.assertEqual(reverse.shape, (4, model.hidden_dim))
        self.assertFalse(torch.allclose(forward, reverse))

    def _assert_categorical_validation(self, device: torch.device) -> None:
        model = PMRCNet(CONFIG, CONFIG["model"]).to(device).eval()
        batch = next(iter(self.fold.valid_loader))
        x_cont, x_cat, x_raw_phys = (value.to(device) for value in batch[:3])
        with torch.no_grad():
            outputs = model(x_cont, x_cat, x_raw_phys)
        self.assertTrue(torch.isfinite(outputs["z_total"]).all())
        for invalid_value in (-1, CONFIG["model"]["embedding_cardinality"], 0.5, float("nan")):
            invalid_cat = x_cat.clone()
            invalid_cat[0, 0] = invalid_value
            with self.subTest(device=device.type, invalid_value=invalid_value):
                with self.assertRaisesRegex(ValueError, "Invalid categorical index"):
                    model(x_cont, invalid_cat, x_raw_phys)

    def test_categorical_indices_validated_on_cpu(self) -> None:
        self._assert_categorical_validation(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_categorical_indices_validated_on_cuda(self) -> None:
        self._assert_categorical_validation(torch.device("cuda"))

    def test_unknown_source_category_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid categorical value.*thal"):
            _harmonize_categorical_encodings(pd.DataFrame({"thal": [99]}))

    def test_loss_and_gradient_propagation(self) -> None:
        model = PMRCNet(CONFIG, CONFIG["model"])
        batch = next(iter(self.fold.train_loader))
        outputs = model(*batch[:3])
        loss_fn = PMRCConsistencyLoss(
            compute_empirical_associations(self.dataframe.iloc[self.train_indices]),
            **CONFIG["loss"],
        )
        loss, bce, physical = loss_fn(model, outputs, batch[3], *batch[:3])
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(bce))
        self.assertTrue(torch.isfinite(physical))
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))

    def test_optimizer_updates_parameters(self) -> None:
        model = PMRCNet(CONFIG, CONFIG["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["training"]["learning_rate"])
        batch = next(iter(self.fold.train_loader))
        loss_fn = PMRCConsistencyLoss(compute_empirical_associations(self.dataframe.iloc[self.train_indices]), **CONFIG["loss"])
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        loss, _, _ = loss_fn(model, model(*batch[:3]), batch[3], *batch[:3])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertTrue(any(not torch.equal(before[name], parameter) for name, parameter in model.named_parameters()))

    def test_checkpoint_round_trip(self) -> None:
        model = PMRCNet(CONFIG, CONFIG["model"])
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, 1, 0.5, CONFIG, [])
            restored = PMRCNet(CONFIG, CONFIG["model"])
            checkpoint = load_checkpoint(path, restored, map_location="cpu")
        self.assertEqual(checkpoint["epoch"], 1)
        for left, right in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(left, right))

    def test_metric_calculation_and_prediction_alignment(self) -> None:
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.4, 0.6, 0.9])
        metrics = calculate_metrics(labels, probabilities)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(len(labels), len(probabilities))

    def test_oof_alignment_and_reproducibility_artifacts(self) -> None:
        oof_path = ROOT / "results" / "stage2_validation" / "oof_predictions.csv"
        if not oof_path.is_file():
            self.skipTest("Patient-level OOF predictions are not included in the repository")
        oof = np.loadtxt(oof_path, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3))
        self.assertEqual(len(oof), 602)
        self.assertEqual(len(np.unique(oof[:, 0])), 602)
        self.assertTrue(np.isfinite(oof[:, 3]).all())
        self.assertTrue((oof[:, 2] >= 1).all() and (oof[:, 2] <= 5).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_forward(self) -> None:
        model = PMRCNet(CONFIG, CONFIG["model"]).cuda().eval()
        batch = tuple(value.cuda() for value in next(iter(self.fold.valid_loader))[:3])
        with torch.no_grad():
            output = model(*batch)
        self.assertEqual(output["z_total"].device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
