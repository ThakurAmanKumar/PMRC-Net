from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedKFold

V11_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = V11_ROOT.parent
sys.path.insert(0, str(V11_ROOT))

from data.dataset import load_dataset, make_fold_data  # noqa: E402
from evaluation.metrics import calculate_metrics  # noqa: E402
from models.pmrc_net import PMRCNet  # noqa: E402
from run_nested_cv import (  # noqa: E402
    expected_outer_splits,
    make_inner_splits,
    set_seed,
    validate_run_identity,
    validate_oof_frame,
)
from training.losses import PMRCObjective  # noqa: E402


class PMRCNetV11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load((V11_ROOT / "configs" / "nested_cv.yaml").read_text(encoding="utf-8"))
        cls.data_path = (V11_ROOT / cls.config["data"]["path"]).resolve()
        cls.dataframe = load_dataset(cls.data_path, cls.config)
        cls.target = cls.dataframe[cls.config["data"]["target"]].to_numpy(dtype=int)
        splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(cls.dataframe, cls.target))
        cls.train_indices, cls.valid_indices = splits[0]
        cls.fold = make_fold_data(cls.dataframe, cls.train_indices, cls.valid_indices, cls.config, seed=7)

    def model_and_batch(self, device: torch.device, coupling: str = "asymmetric"):
        model_config = dict(self.config["model"], coupling=coupling)
        model = PMRCNet(self.config, model_config).to(device)
        batch = next(iter(self.fold.train_loader))
        return model, [value.to(device) for value in batch]

    def test_forward_backward_and_additive_logit_on_cpu(self) -> None:
        model, batch = self.model_and_batch(torch.device("cpu"))
        outputs = model(*batch[:3])
        loss = PMRCObjective()(outputs["z_total"], batch[3])
        loss.backward()
        self.assertEqual(outputs["z_total"].shape, (len(batch[3]), 1))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.allclose(outputs["z_total"], outputs["z_mech"] + outputs["z_inter"] + outputs["z_res"]))
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_forward_backward_on_cuda(self) -> None:
        model, batch = self.model_and_batch(torch.device("cuda"))
        outputs = model(*batch[:3])
        PMRCObjective()(outputs["z_total"], batch[3]).backward()
        self.assertEqual(outputs["z_total"].device.type, "cuda")
        self.assertTrue(torch.isfinite(outputs["z_total"]).all())

    def assert_categorical_rejections(self, device: torch.device) -> None:
        model, batch = self.model_and_batch(device)
        model.eval()
        continuous, categorical, physiology = batch[:3]
        with torch.inference_mode():
            valid = model(continuous, categorical, physiology)
        self.assertTrue(torch.isfinite(valid["z_total"]).all())
        for column, name in enumerate(self.config["data"]["categorical_cols"]):
            largest_allowed = max(self.config["data"]["category_values"][name])
            for value in (-1, largest_allowed + 1, 0.5, float("nan"), float("inf")):
                bad = categorical.to(dtype=torch.float32).clone()
                bad[0, column] = value
                with self.subTest(device=device.type, column=name, value=value):
                    with self.assertRaisesRegex(ValueError, f"Invalid categorical index for '{name}'"):
                        model(continuous, bad, physiology)

    def test_all_categorical_vocabularies_validate_on_cpu(self) -> None:
        self.assert_categorical_rejections(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_all_categorical_vocabularies_validate_on_cuda(self) -> None:
        self.assert_categorical_rejections(torch.device("cuda"))

    def assert_continuous_rejections(self, device: torch.device) -> None:
        model, batch = self.model_and_batch(device)
        continuous, categorical, physiology = batch[:3]
        for value in (float("nan"), float("inf"), float("-inf")):
            invalid_cont = continuous.clone()
            invalid_cont[0, 0] = value
            with self.subTest(device=device.type, source="x_cont", value=value):
                with self.assertRaisesRegex(ValueError, "Invalid continuous value for 'age'"):
                    model(invalid_cont, categorical, physiology)
            invalid_phys = physiology.clone()
            invalid_phys[0, 0] = value
            with self.subTest(device=device.type, source="x_raw_phys", value=value):
                with self.assertRaisesRegex(ValueError, "Invalid physiology value for 'age'"):
                    model(continuous, categorical, invalid_phys)

    def test_continuous_nan_and_infinity_rejected_on_cpu(self) -> None:
        self.assert_continuous_rejections(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_continuous_nan_and_infinity_rejected_on_cuda(self) -> None:
        self.assert_continuous_rejections(torch.device("cuda"))

    def test_asymmetric_reverse_gate_half_contributes_gradient(self) -> None:
        torch.manual_seed(123)
        model, batch = self.model_and_batch(torch.device("cpu"))
        model.zero_grad(set_to_none=True)
        outputs = model(*batch[:3])
        outputs["z_inter"].sum().backward()
        hidden = model.hidden_dim
        for gate_name in ("W_me", "W_ec", "W_mc"):
            gradient = getattr(model, gate_name).bias.grad
            self.assertIsNotNone(gradient)
            self.assertGreater(float(torch.linalg.vector_norm(gradient[hidden:])), 0.0, gate_name)

    def test_asymmetric_coupling_uses_distinct_ordered_flows(self) -> None:
        torch.manual_seed(456)
        model = PMRCNet(self.config).eval()
        left = torch.randn(8, model.hidden_dim)
        right = torch.randn(8, model.hidden_dim)
        mismatch = torch.randn(8, model.hidden_dim)
        gate = torch.sigmoid(model.W_me(torch.cat((left, right, mismatch), dim=-1)))
        forward_gate, reverse_gate = gate.chunk(2, dim=-1)
        forward_flow = model.trans_m(left) * model.trans_e(right)
        reverse_flow = model.trans_e(left) * model.trans_m(right)
        self.assertFalse(torch.allclose(forward_flow, reverse_flow))
        expected = forward_gate * forward_flow + reverse_gate * reverse_flow
        actual = model._couple(left, right, mismatch, model.W_me, model.trans_m, model.trans_e)
        self.assertTrue(torch.allclose(actual, expected))
        self.assertGreater(float(torch.linalg.vector_norm(reverse_gate * reverse_flow).detach()), 0.0)

    def test_symmetric_gate_retains_single_hidden_width(self) -> None:
        model, batch = self.model_and_batch(torch.device("cpu"), coupling="symmetric")
        outputs = model(*batch[:3])
        self.assertEqual(model.W_me.out_features, model.hidden_dim)
        self.assertTrue(torch.isfinite(outputs["z_total"]).all())

    def test_unknown_source_category_fails_without_remapping(self) -> None:
        for column in ("thal", "slope", "ca"):
            invalid = pd.read_csv(self.data_path).head(3).copy()
            invalid.loc[0, column] = 99
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.csv"
                invalid.to_csv(path, index=False)
                with self.subTest(column=column):
                    with self.assertRaisesRegex(ValueError, f"Invalid source category for '{column}'"):
                        load_dataset(path, self.config)

    def test_documented_source_mappings_are_preserved(self) -> None:
        source_values = {
            "thal": ([0, 1, 2, 3, 6, 7], [0, 6, 3, 7, 6, 7]),
            "slope": ([0, 1, 2, 3], [0, 1, 2, 3]),
            "ca": ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
        }
        for column, (raw_values, expected_values) in source_values.items():
            frame = pd.read_csv(self.data_path).head(len(raw_values)).copy()
            frame[column] = raw_values
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "mapped.csv"
                frame.to_csv(path, index=False)
                mapped = load_dataset(path, self.config)
            with self.subTest(column=column):
                self.assertEqual(mapped[column].tolist(), expected_values)

    def test_dataset_loader_rejects_nonfinite_continuous_values(self) -> None:
        for value in (float("nan"), float("inf")):
            invalid = pd.read_csv(self.data_path).head(3).copy()
            invalid["age"] = invalid["age"].astype(float)
            invalid.loc[0, "age"] = value
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.csv"
                invalid.to_csv(path, index=False)
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, "Non-finite continuous value for 'age'"):
                        load_dataset(path, self.config)

    def test_preprocessing_is_fit_on_training_partition_only(self) -> None:
        train = self.dataframe.iloc[self.train_indices][self.config["data"]["continuous_cols"]].to_numpy()
        np.testing.assert_allclose(self.fold.scaler.mean_, train.mean(axis=0))
        self.assertFalse(np.allclose(self.fold.scaler.mean_, self.dataframe[self.config["data"]["continuous_cols"]].to_numpy().mean(axis=0)))

    def test_nested_inner_partitions_exclude_outer_test_rows(self) -> None:
        outer_splits = expected_outer_splits(self.dataframe, self.config, [21])
        outer = outer_splits["21"][0]
        outer_train = np.asarray(outer["train_indices"], dtype=int)
        outer_valid = set(outer["valid_indices"])
        inner = make_inner_splits(outer_train, self.target, self.config, 21, 1)
        for inner_train, inner_valid in inner:
            self.assertFalse(set(inner_train) & set(inner_valid))
            self.assertFalse(set(inner_train) & outer_valid)
            self.assertFalse(set(inner_valid) & outer_valid)
            self.assertEqual(set(inner_train) | set(inner_valid), set(outer_train))

    def test_outer_split_manifest_is_deterministic_and_complete(self) -> None:
        first = expected_outer_splits(self.dataframe, self.config, [21, 42])
        second = expected_outer_splits(self.dataframe, self.config, [21, 42])
        self.assertEqual(first, second)
        for seed_folds in first.values():
            valid_indices = []
            for fold in seed_folds:
                train = set(fold["train_indices"])
                valid = set(fold["valid_indices"])
                self.assertFalse(train & valid)
                self.assertEqual(train | valid, set(range(len(self.dataframe))))
                valid_indices.extend(valid)
            self.assertEqual(sorted(valid_indices), list(range(len(self.dataframe))))

    def test_oof_validation_and_duplicate_detection(self) -> None:
        splits = expected_outer_splits(self.dataframe, self.config, [21])["21"]
        fold_ids = np.zeros(len(self.dataframe), dtype=int)
        frames = []
        for item in splits:
            fold_ids[np.asarray(item["valid_indices"], dtype=int)] = int(item["fold"])
            indices = np.asarray(item["valid_indices"], dtype=int)
            frames.append(pd.DataFrame({
                "sample_index": self.dataframe.iloc[indices]["sample_index"].to_numpy(),
                "seed": 21,
                "fold": int(item["fold"]),
                "target": self.target[indices],
                "probability": np.full(len(indices), 0.5),
            }))
        frame = pd.concat(frames, ignore_index=True)
        validate_oof_frame(frame, self.dataframe, fold_ids, 21)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_oof_frame(pd.concat([frame, frame.iloc[:1]]), self.dataframe, fold_ids, 21)
        wrong_seed = frame.copy()
        wrong_seed["seed"] = 42
        with self.assertRaisesRegex(ValueError, "seed labels"):
            validate_oof_frame(wrong_seed, self.dataframe, fold_ids, 21)

    def test_metric_recalculation_from_saved_prediction_rows(self) -> None:
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.4, 0.6, 0.9])
        metrics = calculate_metrics(labels, probabilities)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["brier_score"], 0.085)
        self.assertEqual((metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]), (2, 0, 0, 2))

    def test_deterministic_model_initialization(self) -> None:
        set_seed(918)
        first = PMRCNet(self.config)
        set_seed(918)
        second = PMRCNet(self.config)
        for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(first_parameter, second_parameter))

    def test_resume_rejects_changed_config_or_dataset_hash(self) -> None:
        metadata = {"config_sha256": "config-a", "dataset_sha256": "data-a"}
        validate_run_identity(metadata, "config-a", "data-a")
        with self.assertRaisesRegex(ValueError, "configuration hash"):
            validate_run_identity(metadata, "config-b", "data-a")
        with self.assertRaisesRegex(ValueError, "dataset hash"):
            validate_run_identity(metadata, "config-a", "data-b")

    def test_checkpoint_round_trip_preserves_parameters(self) -> None:
        model = PMRCNet(self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1_1.pt"
            torch.save({"model_state": model.state_dict(), "version": "1.1"}, path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            restored = PMRCNet(self.config)
            restored.load_state_dict(checkpoint["model_state"])
        for expected, actual in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(expected, actual))

    def test_objective_is_binary_cross_entropy_only(self) -> None:
        logits = torch.tensor([0.0, 1.0], requires_grad=True)
        targets = torch.tensor([0.0, 1.0])
        loss = PMRCObjective()(logits, targets)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        self.assertTrue(torch.equal(loss, expected))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertFalse(self.config["training"]["mixed_precision"])


if __name__ == "__main__":
    unittest.main()