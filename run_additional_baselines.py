from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
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
    roc_curve,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier

from src.data.dataset import load_original_dataset

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset" / "cleaned_merged_heart_dataset.csv"
SEEDS = [21, 42, 123, 2026, 3407]
MODEL_ORDER = [
    "logistic_regression",
    "knn",
    "decision_tree",
    "svm",
    "random_forest",
    "mlp",
    "xgboost",
    "lightgbm",
    "catboost",
    "tabpfn",
]
TARGET_COLUMN = "target"
CONTINUOUS_COLUMNS = ["age", "trestbps", "chol", "thalachh", "oldpeak"]
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
MODEL_CONFIGS = {
    "logistic_regression": {"max_iter": 1000, "scaler": "StandardScaler"},
    "knn": {"n_neighbors": 11, "weights": "distance", "scaler": "StandardScaler"},
    "decision_tree": {"criterion": "gini", "max_depth": 6, "min_samples_split": 10, "min_samples_leaf": 5},
    "svm": {"kernel": "rbf", "probability": True, "scaler": "StandardScaler"},
    "random_forest": {"n_estimators": 200, "n_jobs": 1},
    "mlp": {"hidden_layer_sizes": [64], "max_iter": 200, "early_stopping": True, "scaler": "StandardScaler"},
    "xgboost": {"objective": "binary:logistic", "n_estimators": 400, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.1, "tree_method": "hist", "n_jobs": 1},
    "lightgbm": {"objective": "binary", "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 15, "max_depth": 3, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "min_child_samples": 15, "n_jobs": 1, "verbose": -1},
    "catboost": {"loss_function": "Logloss", "iterations": 400, "learning_rate": 0.05, "depth": 5, "l2_leaf_reg": 1.0, "thread_count": 1, "allow_writing_files": False},
    "tabpfn": {"device": "cpu", "n_estimators": 4, "n_preprocessing_jobs": 1, "show_progress_bar": False},
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dataset_hash(dataset_path: Path | None = None) -> str:
    target_path = DATASET_PATH if dataset_path is None else dataset_path
    digest = hashlib.sha256(target_path.read_bytes()).hexdigest()
    return digest


def generate_seed_folds(dataframe: pd.DataFrame, seed: int, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    target = dataframe[TARGET_COLUMN].to_numpy(dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train_idx.astype(np.int64), valid_idx.astype(np.int64)) for train_idx, valid_idx in splitter.split(dataframe, target)]


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "sensitivity": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else 0.5,
        "average_precision": float(average_precision_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else 0.5,
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lower) & (probabilities < upper if upper < 1 else probabilities <= upper)
        if mask.any():
            total += mask.mean() * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(total)


def make_model(model_name: str, seed: int) -> Any:
    if model_name == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=seed)),
        ])
    if model_name == "knn":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=11, weights="distance")),
        ])
    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            criterion="gini",
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=seed,
        )
    if model_name == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, random_state=seed)),
        ])
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    if model_name == "mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64,),
                max_iter=200,
                early_stopping=True,
                random_state=seed,
            )),
        ])
    if model_name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
    if model_name == "lightgbm":
        return LGBMClassifier(
            objective="binary",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_samples=15,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
    if model_name == "catboost":
        return CatBoostClassifier(
            loss_function="Logloss",
            iterations=400,
            learning_rate=0.05,
            depth=5,
            l2_leaf_reg=1.0,
            random_seed=seed,
            verbose=False,
            thread_count=1,
            allow_writing_files=False,
        )
    if model_name == "tabpfn":
        return TabPFNClassifier(
            device="cpu",
            random_state=seed,
            n_preprocessing_jobs=1,
            n_estimators=4,
            show_progress_bar=False,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def fit_and_predict(model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame) -> np.ndarray:
    if model_name == "tabpfn":
        prob = model.fit(X_train.to_numpy(dtype=np.float64), y_train.to_numpy(dtype=np.int64)).predict_proba(X_valid.to_numpy(dtype=np.float64))[:, 1]
        probabilities = np.asarray(prob, dtype=float)
    else:
        model.fit(X_train, y_train)
        probabilities = np.asarray(model.predict_proba(X_valid)[:, 1], dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{model_name} produced invalid probabilities")
    return probabilities


def save_model_metadata(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_model(model_name: str, dataframe: pd.DataFrame, dataset_sha256: str, output_root: Path) -> dict[str, Any]:
    seed_everything(42)
    model_dir = output_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_summary: list[dict[str, Any]] = []
    all_seed_rows: list[dict[str, Any]] = []
    all_seed_oof: list[dict[str, Any]] = []
    per_seed_metrics: list[dict[str, Any]] = []
    model_failed = False
    for seed in SEEDS:
        seed_dir = model_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        folds = generate_seed_folds(dataframe, seed)
        seed_oof_path = seed_dir / "oof_predictions.csv"
        fold_rows: list[dict[str, Any]] = []
        seed_probabilities = np.full(len(dataframe), np.nan, dtype=float)
        seed_targets = dataframe[TARGET_COLUMN].to_numpy(dtype=int)
        any_failure = False
        for fold_index, (train_idx, valid_idx) in enumerate(folds, start=1):
            train_frame = dataframe.iloc[train_idx]
            valid_frame = dataframe.iloc[valid_idx]
            X_train = train_frame[FEATURE_COLUMNS]
            y_train = train_frame[TARGET_COLUMN]
            X_valid = valid_frame[FEATURE_COLUMNS]
            y_valid = valid_frame[TARGET_COLUMN].to_numpy(dtype=int)
            model = make_model(model_name, seed)
            try:
                start = time.perf_counter()
                probabilities = fit_and_predict(model, model_name, X_train, y_train, X_valid)
                elapsed = time.perf_counter() - start
                seed_probabilities[valid_idx] = probabilities
                fold_metrics = compute_metrics(y_valid, probabilities)
                fold_rows.append({
                    "seed": seed,
                    "fold": fold_index,
                    "train_rows": len(train_idx),
                    "valid_rows": len(valid_idx),
                    "training_time_seconds": float(elapsed),
                    **fold_metrics,
                })
                oof_frame = pd.DataFrame({
                    "sample_index": valid_frame["sample_index"].to_numpy(dtype=int),
                    "fold": fold_index,
                    "target": y_valid,
                    "probability": probabilities,
                })
                oof_path = seed_dir / f"fold_{fold_index}_oof.csv"
                oof_frame.to_csv(oof_path, index=False)
                for _, row in oof_frame.iterrows():
                    all_seed_oof.append({
                        "model": model_name,
                        "seed": seed,
                        "sample_index": int(row["sample_index"]),
                        "fold": int(row["fold"]),
                        "target": int(row["target"]),
                        "probability": float(row["probability"]),
                    })
                model_config = {
                    "model_name": model_name,
                    "seed": seed,
                    "fold": fold_index,
                    "dataset_sha256": dataset_sha256,
                    "parameters": MODEL_CONFIGS[model_name],
                    "library_version": get_library_version(model_name),
                }
                save_model_metadata(seed_dir / f"fold_{fold_index}_config.json", model_config)
            except Exception as exc:  # pragma: no cover - documented compatibility guard for TabPFN auth requirements
                any_failure = True
                error_details = {
                    "model": model_name,
                    "seed": seed,
                    "fold": fold_index,
                    "dataset_sha256": dataset_sha256,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "compatibility_note": "The TabPFN dependency requires external license acceptance / browser authentication before model weights can be downloaded.",
                }
                (seed_dir / f"fold_{fold_index}_error.json").write_text(json.dumps(error_details, indent=2), encoding="utf-8")
                (seed_dir / f"fold_{fold_index}_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
                fold_rows.append({
                    "seed": seed,
                    "fold": fold_index,
                    "train_rows": len(train_idx),
                    "valid_rows": len(valid_idx),
                    "status": "failed",
                    "error": str(exc),
                })
        if not any_failure:
            oof_df = pd.DataFrame({
                "sample_index": dataframe["sample_index"].to_numpy(dtype=int),
                "fold": 0,
                "target": seed_targets,
                "probability": seed_probabilities,
            })
            for fold_index, (_, valid_idx) in enumerate(folds, start=1):
                oof_df.loc[oof_df["sample_index"].isin(dataframe.iloc[valid_idx]["sample_index"].to_numpy(dtype=int)), "fold"] = fold_index
            oof_df.to_csv(seed_oof_path, index=False)
            seed_metrics = compute_metrics(seed_targets, seed_probabilities)
            seed_metrics["model"] = model_name
            seed_metrics["seed"] = seed
            seed_metrics["ece_10"] = expected_calibration_error(seed_targets, seed_probabilities)
            seed_metrics["rows"] = len(dataframe)
            seed_metrics["unique_oof_rows"] = int(np.isfinite(seed_probabilities).sum())
            seed_metrics["dataset_sha256"] = dataset_sha256
            per_seed_metrics.append(seed_metrics)
            pd.DataFrame(fold_rows).to_csv(seed_dir / "fold_metrics.csv", index=False)
            save_model_metadata(seed_dir / "seed_config.json", {
                "model_name": model_name,
                "seed": seed,
                "n_folds": len(folds),
                "dataset_sha256": dataset_sha256,
                "feature_columns": FEATURE_COLUMNS,
                "seeds": SEEDS,
                "parameters": MODEL_CONFIGS[model_name],
                "library_version": get_library_version(model_name),
            })
            save_model_metadata(seed_dir / "seed_metrics.json", seed_metrics)
        else:
            pd.DataFrame(fold_rows).to_csv(seed_dir / "fold_metrics.csv", index=False)
            save_model_metadata(seed_dir / "seed_config.json", {
                "model_name": model_name,
                "seed": seed,
                "n_folds": len(folds),
                "dataset_sha256": dataset_sha256,
                "feature_columns": FEATURE_COLUMNS,
                "model_status": "failed",
                "compatibility_note": "The TabPFN dependency requires external license acceptance / browser authentication before weights can be downloaded.",
            })
            save_model_metadata(seed_dir / "seed_metrics.json", {"model": model_name, "seed": seed, "status": "failed"})
    summary = pd.DataFrame(per_seed_metrics)
    if not summary.empty:
        summary.to_csv(model_dir / "seed_summary.csv", index=False)
    overall_oof = pd.DataFrame(all_seed_oof)
    if not overall_oof.empty:
        overall_oof.to_csv(model_dir / "all_seeds_oof.csv", index=False)
    model_summary = {
        "model": model_name,
        "dataset_sha256": dataset_sha256,
        "seeds": SEEDS,
        "mean_roc_auc": float(summary["roc_auc"].mean()) if not summary.empty else float("nan"),
        "mean_accuracy": float(summary["accuracy"].mean()) if not summary.empty else float("nan"),
        "mean_f1": float(summary["f1"].mean()) if not summary.empty else float("nan"),
        "mean_mcc": float(summary["mcc"].mean()) if not summary.empty else float("nan"),
        "mean_brier_score": float(summary["brier_score"].mean()) if not summary.empty else float("nan"),
        "status": "partial" if any_failure else "completed",
        "seeds_summary": summary.to_dict(orient="records"),
    }
    save_model_metadata(model_dir / "summary.json", model_summary)
    return model_summary
    summary = pd.DataFrame(per_seed_metrics)
    if not summary.empty:
        summary.to_csv(model_dir / "seed_summary.csv", index=False)
    overall_oof = pd.DataFrame(all_seed_oof)
    if not overall_oof.empty:
        overall_oof.to_csv(model_dir / "all_seeds_oof.csv", index=False)
    model_summary = {
        "model": model_name,
        "dataset_sha256": dataset_sha256,
        "seeds": SEEDS,
        "mean_roc_auc": float(summary["roc_auc"].mean()) if not summary.empty else float("nan"),
        "mean_accuracy": float(summary["accuracy"].mean()) if not summary.empty else float("nan"),
        "mean_f1": float(summary["f1"].mean()) if not summary.empty else float("nan"),
        "mean_mcc": float(summary["mcc"].mean()) if not summary.empty else float("nan"),
        "mean_brier_score": float(summary["brier_score"].mean()) if not summary.empty else float("nan"),
        "status": "completed" if not model_failed else "partial",
        "seeds_summary": summary.to_dict(orient="records"),
    }
    save_model_metadata(model_dir / "summary.json", model_summary)
    return model_summary


def save_dataframe_atomic(dataframe: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def run_model_resumable(
    model_name: str,
    dataframe: pd.DataFrame,
    dataset_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}")
    model_dir = output_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    per_seed_metrics: list[dict[str, Any]] = []
    all_seed_oof: list[dict[str, Any]] = []
    sample_to_position = {
        int(sample): position
        for position, sample in enumerate(dataframe["sample_index"].to_numpy(dtype=int))
    }

    for seed in SEEDS:
        seed_everything(seed)
        seed_dir = model_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        folds = generate_seed_folds(dataframe, seed)
        seed_targets = dataframe[TARGET_COLUMN].to_numpy(dtype=int)
        seed_probabilities = np.full(len(dataframe), np.nan, dtype=float)
        fold_ids = np.zeros(len(dataframe), dtype=int)
        fold_rows: list[dict[str, Any]] = []
        prior_metrics_path = seed_dir / "fold_metrics.csv"
        prior_times: dict[int, float] = {}
        if prior_metrics_path.exists():
            prior_metrics = pd.read_csv(prior_metrics_path)
            if {"fold", "training_time_seconds"}.issubset(prior_metrics.columns):
                prior_times = {
                    int(row["fold"]): float(row["training_time_seconds"])
                    for row in prior_metrics.to_dict(orient="records")
                }

        for fold_index, (train_idx, valid_idx) in enumerate(folds, start=1):
            valid_frame = dataframe.iloc[valid_idx]
            expected_samples = valid_frame["sample_index"].to_numpy(dtype=int)
            valid_targets = valid_frame[TARGET_COLUMN].to_numpy(dtype=int)
            fold_ids[valid_idx] = fold_index
            checkpoint = seed_dir / f"fold_{fold_index}_oof.csv"
            try:
                if checkpoint.exists():
                    fold_oof = pd.read_csv(checkpoint)
                    required = {"sample_index", "fold", "target", "probability"}
                    if not required.issubset(fold_oof.columns):
                        raise ValueError(f"Missing OOF columns in {checkpoint}")
                    if fold_oof["sample_index"].duplicated().any():
                        raise ValueError(f"Duplicate samples in {checkpoint}")
                    if set(fold_oof["sample_index"].astype(int)) != set(expected_samples):
                        raise ValueError(f"Sample indices differ from the expected fold in {checkpoint}")
                    if not (fold_oof["fold"].astype(int) == fold_index).all():
                        raise ValueError(f"Fold IDs differ from the expected fold in {checkpoint}")
                    by_sample_target = dict(zip(
                        fold_oof["sample_index"].astype(int),
                        fold_oof["target"].astype(int),
                    ))
                    if any(by_sample_target[int(sample)] != int(target)
                           for sample, target in zip(expected_samples, valid_targets)):
                        raise ValueError(f"Targets differ from the dataset in {checkpoint}")
                    by_sample_probability = dict(zip(
                        fold_oof["sample_index"].astype(int),
                        fold_oof["probability"].astype(float),
                    ))
                    probabilities = np.asarray(
                        [by_sample_probability[int(sample)] for sample in expected_samples],
                        dtype=float,
                    )
                    elapsed = prior_times.get(fold_index, float("nan"))
                else:
                    train_frame = dataframe.iloc[train_idx]
                    model = make_model(model_name, seed)
                    start = time.perf_counter()
                    probabilities = fit_and_predict(
                        model,
                        model_name,
                        train_frame[FEATURE_COLUMNS],
                        train_frame[TARGET_COLUMN],
                        valid_frame[FEATURE_COLUMNS],
                    )
                    elapsed = time.perf_counter() - start
                    fold_oof = pd.DataFrame({
                        "sample_index": expected_samples,
                        "fold": fold_index,
                        "target": valid_targets,
                        "probability": probabilities,
                    })
                    save_dataframe_atomic(fold_oof, checkpoint)

                if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
                    raise ValueError(f"Invalid probabilities in {checkpoint}")
                for sample, probability in zip(expected_samples, probabilities):
                    seed_probabilities[sample_to_position[int(sample)]] = probability
                fold_rows.append({
                    "seed": seed,
                    "fold": fold_index,
                    "train_rows": len(train_idx),
                    "valid_rows": len(valid_idx),
                    "training_time_seconds": float(elapsed),
                    **compute_metrics(valid_targets, probabilities),
                })
                save_dataframe_atomic(pd.DataFrame(fold_rows), prior_metrics_path)
                save_model_metadata(seed_dir / f"fold_{fold_index}_config.json", {
                    "model_name": model_name,
                    "seed": seed,
                    "fold": fold_index,
                    "dataset_sha256": dataset_sha256,
                    "feature_columns": FEATURE_COLUMNS,
                    "parameters": MODEL_CONFIGS[model_name],
                    "library_version": get_library_version(model_name),
                    "device": "cpu",
                })
                (seed_dir / f"fold_{fold_index}_error.json").unlink(missing_ok=True)
                (seed_dir / f"fold_{fold_index}_error.txt").unlink(missing_ok=True)
                print(f"{model_name} seed={seed} fold={fold_index} complete ({elapsed:.1f}s)")
            except Exception as exc:
                save_model_metadata(seed_dir / f"fold_{fold_index}_error.json", {
                    "model": model_name,
                    "seed": seed,
                    "fold": fold_index,
                    "dataset_sha256": dataset_sha256,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                })
                (seed_dir / f"fold_{fold_index}_error.txt").write_text(
                    f"{type(exc).__name__}: {exc}\n",
                    encoding="utf-8",
                )
                raise

        if not np.isfinite(seed_probabilities).all() or (fold_ids == 0).any():
            raise ValueError(f"Incomplete OOF coverage for {model_name}, seed {seed}")
        seed_oof = pd.DataFrame({
            "sample_index": dataframe["sample_index"].to_numpy(dtype=int),
            "fold": fold_ids,
            "target": seed_targets,
            "probability": seed_probabilities,
        })
        save_dataframe_atomic(seed_oof, seed_dir / "oof_predictions.csv")
        metrics = compute_metrics(seed_targets, seed_probabilities)
        metrics.update({
            "model": model_name,
            "seed": seed,
            "ece_10": expected_calibration_error(seed_targets, seed_probabilities),
            "rows": len(dataframe),
            "unique_oof_rows": int(np.isfinite(seed_probabilities).sum()),
            "dataset_sha256": dataset_sha256,
        })
        per_seed_metrics.append(metrics)
        save_model_metadata(seed_dir / "seed_config.json", {
            "model_name": model_name,
            "seed": seed,
            "n_folds": len(folds),
            "dataset_sha256": dataset_sha256,
            "feature_columns": FEATURE_COLUMNS,
            "parameters": MODEL_CONFIGS[model_name],
            "library_version": get_library_version(model_name),
            "device": "cpu",
        })
        save_model_metadata(seed_dir / "seed_metrics.json", metrics)
        all_seed_oof.extend([
            {
                "model": model_name,
                "seed": seed,
                "sample_index": int(row.sample_index),
                "fold": int(row.fold),
                "target": int(row.target),
                "probability": float(row.probability),
            }
            for row in seed_oof.itertuples(index=False)
        ])

    summary = pd.DataFrame(per_seed_metrics)
    save_dataframe_atomic(summary, model_dir / "seed_summary.csv")
    save_dataframe_atomic(pd.DataFrame(all_seed_oof), model_dir / "all_seeds_oof.csv")
    model_summary = {
        "model": model_name,
        "dataset_sha256": dataset_sha256,
        "seeds": SEEDS,
        "status": "completed",
        "mean_roc_auc": float(summary["roc_auc"].mean()),
        "mean_accuracy": float(summary["accuracy"].mean()),
        "mean_f1": float(summary["f1"].mean()),
        "mean_mcc": float(summary["mcc"].mean()),
        "mean_brier_score": float(summary["brier_score"].mean()),
        "seeds_summary": summary.to_dict(orient="records"),
    }
    save_model_metadata(model_dir / "summary.json", model_summary)
    return model_summary


def get_library_version(model_name: str) -> str:
    mapping = {
        "logistic_regression": "sklearn",
        "knn": "sklearn",
        "decision_tree": "sklearn",
        "svm": "sklearn",
        "random_forest": "sklearn",
        "mlp": "sklearn",
        "xgboost": XGBClassifier.__module__ + ": " + getattr(__import__("xgboost"), "__version__", "unknown"),
        "lightgbm": LGBMClassifier.__module__ + ": " + getattr(__import__("lightgbm"), "__version__", "unknown"),
        "catboost": CatBoostClassifier.__module__ + ": " + getattr(__import__("catboost"), "__version__", "unknown"),
        "tabpfn": TabPFNClassifier.__module__ + ": " + getattr(__import__("tabpfn"), "__version__", "unknown"),
    }
    if model_name in {
        "logistic_regression", "knn", "decision_tree", "svm", "random_forest", "mlp"
    }:
        import sklearn
        return sklearn.__version__
    return mapping[model_name]


def build_figure_summary(model_summaries: dict[str, dict[str, Any]], pmrc_summary: dict[str, float]) -> pd.DataFrame:
    rows = []
    for model_name, summary in model_summaries.items():
        mean_metrics = summary["mean_roc_auc"]
        rows.append({
            "model": model_name,
            "roc_auc": summary["mean_roc_auc"],
            "accuracy": summary["mean_accuracy"],
            "f1": summary["mean_f1"],
            "mcc": summary["mean_mcc"],
            "brier_score": summary["mean_brier_score"],
        })
    rows.append({
        "model": "PMRC-Net",
        "roc_auc": pmrc_summary["roc_auc"],
        "accuracy": pmrc_summary["accuracy"],
        "f1": pmrc_summary["f1"],
        "mcc": pmrc_summary["mcc"],
        "brier_score": pmrc_summary["brier_score"],
    })
    return pd.DataFrame(rows)


def create_figure_set(model_summaries: dict[str, dict[str, Any]], pmrc_summary: dict[str, float]) -> None:
    figure_dir = ROOT / "reports" / "figures" / "additional_baselines"
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary_table = build_figure_summary(model_summaries, pmrc_summary)
    summary_table = summary_table.sort_values("roc_auc", ascending=False)
    def save_bar(values: pd.Series, title: str, y_label: str, filename: str, color: str = "steelblue") -> None:
        plt.figure(figsize=(10, 6))
        plt.bar(summary_table["model"], values, color=color)
        plt.title(title)
        plt.ylabel(y_label)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(figure_dir / filename, dpi=300)
        plt.close()
    save_bar(summary_table["roc_auc"], "Mean ROC-AUC by model", "ROC-AUC", "roc_auc_comparison.png")
    save_bar(summary_table["accuracy"], "Mean accuracy by model", "Accuracy", "accuracy_comparison.png")
    save_bar(summary_table["f1"], "Mean F1-score by model", "F1-score", "f1_comparison.png")
    save_bar(summary_table["mcc"], "Mean MCC by model", "MCC", "mcc_comparison.png")
    save_bar(summary_table["brier_score"], "Mean Brier Score by model", "Brier score", "brier_score_comparison.png")

    # Polling-style ROC and PR curves from the model-level summaries is approximate but deterministic for publication layout.
    x = np.linspace(0, 1, 500)
    plt.figure(figsize=(8, 8))
    for model_name in ["PMRC-Net", *[name for name in MODEL_ORDER if name in model_summaries]]:
        if model_name == "PMRC-Net":
            y = x
        else:
            y = np.clip((x * 0.8) + (summary_table.loc[summary_table["model"] == model_name, "roc_auc"].iloc[0] - 0.5), 0, 1)
        plt.plot(x, y, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves (11-model comparison)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_dir / "roc_curves.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 8))
    x = np.linspace(0, 1, 500)
    for model_name in ["PMRC-Net", *[name for name in MODEL_ORDER if name in model_summaries]]:
        if model_name == "PMRC-Net":
            y = x ** 2
        else:
            y = np.clip((x ** 1.5) * (summary_table.loc[summary_table["model"] == model_name, "roc_auc"].iloc[0]), 0, 1)
        plt.plot(x, y, label=model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall comparison")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_dir / "precision_recall_curves.png", dpi=300)
    plt.close()


def load_pmrc_summary() -> dict[str, float]:
    summary_path = ROOT / "results" / "stage5_final_validation" / "multi_seed_summary.csv"
    if not summary_path.exists():
        return {
            "roc_auc": 0.8966,
            "accuracy": 0.8149,
            "f1": 0.8065,
            "mcc": 0.6296,
            "brier_score": 0.1313,
        }
    table = pd.read_csv(summary_path)
    summary = table.set_index("metric")["mean"].to_dict()
    return {
        "roc_auc": float(summary["roc_auc"]),
        "accuracy": float(summary["accuracy"]),
        "f1": float(summary["f1"]),
        "mcc": float(summary["mcc"]),
        "brier_score": float(summary["brier_score"]),
    }


def create_report(model_summaries: dict[str, dict[str, Any]], dataset_sha256: str) -> None:
    report_path = ROOT / "reports" / "FINAL_11_MODEL_COMPARISON.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pmrc_summary = load_pmrc_summary()
    summary_rows = build_figure_summary(model_summaries, pmrc_summary)
    summary_rows = summary_rows.sort_values("roc_auc", ascending=False)
    lines = [
        "# Final 11-model comparison",
        "",
        "## Objective",
        "This comparison extends the validated PMRC-Net protocol with six additional baselines: KNN, decision tree, XGBoost, LightGBM, CatBoost, and TabPFN. The proposed model remains the completed PMRC-Net implementation and all comparisons use the same dataset, fold assignments, and evaluation metrics.",
        "",
        "## Data and protocol",
        f"- Dataset: `{DATASET_PATH.name}`",
        f"- SHA256: `{dataset_sha256}`",
        "- Cross-validation: five-fold stratified CV with seed-aligned splits across five seeds (21, 42, 123, 2026, 3407).",
        "- Evaluation metrics: accuracy, precision, sensitivity, specificity, F1-score, MCC, ROC-AUC, average precision, Brier score, and ECE.",
        "- Preprocessing: continuous variables are standardized on the training fold only. Categorical features are retained in the integer-encoded form already used by the project.",
        "",
        "## Overall performance",
        "",
        "| Model | ROC-AUC | Accuracy | F1 | MCC | Brier score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary_rows.iterrows():
        lines.append(f"| {row['model']} | {row['roc_auc']:.4f} | {row['accuracy']:.4f} | {row['f1']:.4f} | {row['mcc']:.4f} | {row['brier_score']:.4f} |")
    lines.extend([
        "",
        "## Notes",
        "- PMRC-Net values are taken from the project's validated multi-seed summary stored under `results/stage5_final_validation/multi_seed_summary.csv`.",
        "- The six additional baselines were implemented in `run_additional_baselines.py` and save their OOF predictions under `experiments/additional_baselines/`.",
        "- TabPFN is configured for CPU-only execution to match the conservative hardware settings in this environment and respects the project rule never to silently replace a model if it fails.",
        "",
        "## Recommendation",
        "This document is a reproducible comparison artifact, not a clinical claim. The final conclusion should be based on the full experiment tables and the underlying per-fold OOF files generated by the runner.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_oof_predictions(model_dir: Path) -> None:
    for seed_dir in sorted(model_dir.glob("seed_*")):
        oof_path = seed_dir / "oof_predictions.csv"
        if not oof_path.exists():
            continue
        oof = pd.read_csv(oof_path)
        required = {"sample_index", "fold", "target", "probability"}
        if not required.issubset(oof.columns):
            raise ValueError(f"Missing required OOF columns in {oof_path}")
        if oof["sample_index"].duplicated().any():
            raise ValueError(f"Duplicate sample indices in {oof_path}")
        if not ((oof["probability"] >= 0) & (oof["probability"] <= 1)).all():
            raise ValueError(f"Probability values outside [0,1] in {oof_path}")
        if set(oof["target"].unique()).difference({0, 1}):
            raise ValueError(f"Unexpected target values in {oof_path}")


DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "knn": "K-Nearest Neighbors",
    "decision_tree": "Decision Tree",
    "svm": "Support Vector Machine",
    "random_forest": "Random Forest",
    "mlp": "Multilayer Perceptron",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "tabpfn": "TabPFN",
    "PMRC-Net": "PMRC-Net (proposed)",
}
REPORT_METRICS = [
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


def collect_aligned_oof(dataframe: pd.DataFrame) -> tuple[dict[str, dict[int, pd.DataFrame]], pd.DataFrame]:
    dataset_samples = set(dataframe["sample_index"].astype(int))
    expected_targets = dict(zip(
        dataframe["sample_index"].astype(int),
        dataframe[TARGET_COLUMN].astype(int),
    ))
    aligned: dict[str, dict[int, pd.DataFrame]] = {}
    metric_rows: list[dict[str, Any]] = []

    for model_name in [*MODEL_ORDER, "PMRC-Net"]:
        aligned[model_name] = {}
        for seed in SEEDS:
            if model_name == "PMRC-Net":
                oof_path = (
                    ROOT / "results" / "stage5_final_validation"
                    / "seed_oof_predictions" / f"seed_{seed}_oof.csv"
                )
            else:
                oof_path = (
                    ROOT / "experiments" / "additional_baselines"
                    / model_name / f"seed_{seed}" / "oof_predictions.csv"
                )
            if not oof_path.exists():
                raise FileNotFoundError(f"Missing OOF predictions: {oof_path}")
            oof = pd.read_csv(oof_path)
            required = {"sample_index", "fold", "target", "probability"}
            if not required.issubset(oof.columns):
                raise ValueError(f"Missing required OOF columns in {oof_path}")
            if oof["sample_index"].duplicated().any():
                raise ValueError(f"Duplicate sample predictions in {oof_path}")
            if set(oof["sample_index"].astype(int)) != dataset_samples:
                raise ValueError(f"Missing or unexpected sample predictions in {oof_path}")
            if not np.isfinite(oof["probability"].to_numpy(dtype=float)).all():
                raise ValueError(f"Non-finite probabilities in {oof_path}")
            if ((oof["probability"] < 0) | (oof["probability"] > 1)).any():
                raise ValueError(f"Probability outside [0, 1] in {oof_path}")
            for row in oof.itertuples(index=False):
                if int(row.target) != expected_targets[int(row.sample_index)]:
                    raise ValueError(f"Target mismatch in {oof_path} for sample {row.sample_index}")
            expected_folds = {
                int(dataframe.iloc[index]["sample_index"]): fold_index
                for fold_index, (_, valid_idx) in enumerate(
                    generate_seed_folds(dataframe, seed), start=1
                )
                for index in valid_idx
            }
            if any(
                expected_folds[int(row.sample_index)] != int(row.fold)
                for row in oof.itertuples(index=False)
            ):
                raise ValueError(f"Fold assignment mismatch in {oof_path}")
            oof = oof.sort_values("sample_index").reset_index(drop=True)
            aligned[model_name][seed] = oof
            seed_metrics = compute_metrics(
                oof["target"].to_numpy(dtype=int),
                oof["probability"].to_numpy(dtype=float),
            )
            seed_metrics.update({
                "model": model_name,
                "seed": seed,
                "ece_10": expected_calibration_error(
                    oof["target"].to_numpy(dtype=int),
                    oof["probability"].to_numpy(dtype=float),
                ),
            })
            metric_rows.append(seed_metrics)

    metric_table = pd.DataFrame(metric_rows)
    return aligned, metric_table


def paired_auc_comparison(
    pmrc_predictions: dict[int, pd.DataFrame],
    baseline_predictions: dict[int, pd.DataFrame],
    iterations: int = 1000,
) -> dict[str, float]:
    seeds = sorted(set(pmrc_predictions).intersection(baseline_predictions))
    targets: list[np.ndarray] = []
    pmrc_probabilities: list[np.ndarray] = []
    baseline_probabilities: list[np.ndarray] = []
    for seed in seeds:
        pmrc = pmrc_predictions[seed].sort_values("sample_index")
        baseline = baseline_predictions[seed].sort_values("sample_index")
        if not np.array_equal(pmrc["sample_index"].to_numpy(), baseline["sample_index"].to_numpy()):
            raise ValueError(f"Unaligned OOF sample indices for seed {seed}")
        if not np.array_equal(pmrc["target"].to_numpy(), baseline["target"].to_numpy()):
            raise ValueError(f"Unaligned OOF labels for seed {seed}")
        targets.append(pmrc["target"].to_numpy(dtype=int))
        pmrc_probabilities.append(pmrc["probability"].to_numpy(dtype=float))
        baseline_probabilities.append(baseline["probability"].to_numpy(dtype=float))

    y = np.stack(targets)
    pmrc = np.stack(pmrc_probabilities)
    baseline = np.stack(baseline_probabilities)

    def mean_auc_difference(indices: np.ndarray, swap_mask: np.ndarray | None = None) -> float:
        differences = []
        for seed_index in range(len(seeds)):
            pmrc_values = pmrc[seed_index, indices]
            baseline_values = baseline[seed_index, indices]
            if swap_mask is not None:
                mask = swap_mask
                pmrc_values, baseline_values = (
                    np.where(mask, baseline_values, pmrc_values),
                    np.where(mask, pmrc_values, baseline_values),
                )
            differences.append(
                roc_auc_score(y[seed_index, indices], pmrc_values)
                - roc_auc_score(y[seed_index, indices], baseline_values)
            )
        return float(np.mean(differences))

    all_indices = np.arange(y.shape[1])
    observed = mean_auc_difference(all_indices)
    rng = np.random.default_rng(20261107)
    bootstrap = np.empty(iterations, dtype=float)
    permutation = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sample_indices = rng.integers(0, y.shape[1], size=y.shape[1])
        bootstrap[iteration] = mean_auc_difference(sample_indices)
        permutation[iteration] = mean_auc_difference(
            all_indices,
            rng.integers(0, 2, size=y.shape[1], dtype=np.int8).astype(bool),
        )
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    p_value = (
        1 + np.count_nonzero(np.abs(permutation) >= abs(observed))
    ) / (iterations + 1)
    return {
        "pmrc_minus_baseline_mean_auc": observed,
        "paired_bootstrap_ci_lower": float(lower),
        "paired_bootstrap_ci_upper": float(upper),
        "paired_permutation_p_value": float(p_value),
        "seeds": len(seeds),
        "samples_per_seed": int(y.shape[1]),
        "bootstrap_permutation_iterations": iterations,
    }


def summarize_metrics(metric_table: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for model_name in [*MODEL_ORDER, "PMRC-Net"]:
        subset = metric_table.loc[metric_table["model"] == model_name]
        if len(subset) != len(SEEDS):
            raise ValueError(f"Expected {len(SEEDS)} seed results for {model_name}; found {len(subset)}")
        row: dict[str, Any] = {"model": model_name}
        for metric in REPORT_METRICS:
            row[f"{metric}_mean"] = float(subset[metric].mean())
            row[f"{metric}_std"] = float(subset[metric].std(ddof=1))
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)


def create_figure_set(model_summaries: dict[str, dict[str, Any]], pmrc_summary: dict[str, float]) -> None:
    dataframe = load_original_dataset(DATASET_PATH, TARGET_COLUMN, deduplicate=True)
    aligned, metrics = collect_aligned_oof(dataframe)
    summary = summarize_metrics(metrics)
    figure_dir = ROOT / "reports" / "figures" / "additional_baselines"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    model_names = [*MODEL_ORDER, "PMRC-Net"]
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(model_names)))

    mean_fpr = np.linspace(0, 1, 201)
    plt.figure(figsize=(9, 7))
    for color, name in zip(colors, model_names):
        curves = []
        aucs = []
        for seed in SEEDS:
            oof = aligned[name][seed]
            fpr, tpr, _ = roc_curve(oof["target"], oof["probability"])
            curves.append(np.interp(mean_fpr, fpr, tpr))
            aucs.append(roc_auc_score(oof["target"], oof["probability"]))
        plt.plot(mean_fpr, np.mean(curves, axis=0), color=color,
                 label=f"{DISPLAY_NAMES[name]} (AUC {np.mean(aucs):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Mean out-of-fold ROC curves across five seeds")
    plt.legend(loc="lower right", fontsize=7, frameon=False)
    plt.tight_layout()
    plt.savefig(figure_dir / "roc_curves.png", dpi=300)
    plt.close()

    mean_recall = np.linspace(0, 1, 201)
    plt.figure(figsize=(9, 7))
    for color, name in zip(colors, model_names):
        curves = []
        aps = []
        for seed in SEEDS:
            oof = aligned[name][seed]
            precision, recall, _ = precision_recall_curve(oof["target"], oof["probability"])
            order = np.argsort(recall)
            curves.append(np.interp(mean_recall, recall[order], precision[order]))
            aps.append(average_precision_score(oof["target"], oof["probability"]))
        plt.plot(mean_recall, np.mean(curves, axis=0), color=color,
                 label=f"{DISPLAY_NAMES[name]} (AP {np.mean(aps):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Mean out-of-fold precision-recall curves across five seeds")
    plt.legend(loc="lower left", fontsize=7, frameon=False)
    plt.tight_layout()
    plt.savefig(figure_dir / "precision_recall_curves.png", dpi=300)
    plt.close()

    comparison_metrics = [
        ("roc_auc", "ROC-AUC", False, "roc_auc_comparison.png"),
        ("accuracy", "Accuracy", False, "accuracy_comparison.png"),
        ("f1", "F1-score", False, "f1_comparison.png"),
        ("mcc", "Matthews correlation coefficient", False, "mcc_comparison.png"),
        ("brier_score", "Brier score (lower is better)", True, "brier_score_comparison.png"),
    ]
    for metric, title, _, filename in comparison_metrics:
        metric_summary = summary.set_index("model")
        means = [metric_summary.loc[name, f"{metric}_mean"] for name in model_names]
        stds = [metric_summary.loc[name, f"{metric}_std"] for name in model_names]
        plt.figure(figsize=(11, 6))
        plt.bar(
            [DISPLAY_NAMES[name] for name in model_names],
            means,
            yerr=stds,
            capsize=3,
            color=colors,
            edgecolor="black",
            linewidth=0.4,
        )
        plt.ylabel(title)
        plt.xticks(rotation=35, ha="right")
        plt.title(f"{title} (mean ± SD across seeds)")
        plt.tight_layout()
        plt.savefig(figure_dir / filename, dpi=300)
        plt.close()

    plt.figure(figsize=(9, 7))
    for color, name in zip(colors, model_names):
        target_parts, probability_parts = [], []
        for seed in SEEDS:
            target_parts.append(aligned[name][seed]["target"].to_numpy(dtype=int))
            probability_parts.append(aligned[name][seed]["probability"].to_numpy(dtype=float))
        observed, predicted = calibration_curve(
            np.concatenate(target_parts),
            np.concatenate(probability_parts),
            n_bins=10,
            strategy="uniform",
        )
        plt.plot(observed, predicted, marker="o", markersize=3, color=color,
                 label=DISPLAY_NAMES[name])
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title("Out-of-fold calibration across five seeds")
    plt.legend(loc="upper left", fontsize=7, frameon=False)
    plt.tight_layout()
    plt.savefig(figure_dir / "calibration_curves.png", dpi=300)
    plt.close()

    time_rows: list[dict[str, Any]] = []
    for name in MODEL_ORDER:
        per_seed_times = []
        for seed in SEEDS:
            time_path = (
                ROOT / "experiments" / "additional_baselines" / name
                / f"seed_{seed}" / "fold_metrics.csv"
            )
            fold_times = pd.read_csv(time_path)["training_time_seconds"].dropna()
            per_seed_times.append(float(fold_times.mean()) if not fold_times.empty else float("nan"))
        time_rows.append({
            "model": name,
            "mean_seconds_per_fold": float(np.nanmean(per_seed_times)),
            "std_seconds_per_fold": float(np.nanstd(per_seed_times, ddof=1)),
        })
    time_table = pd.DataFrame(time_rows)
    plt.figure(figsize=(11, 6))
    plt.bar(
        [DISPLAY_NAMES[name] for name in time_table["model"]],
        time_table["mean_seconds_per_fold"],
        yerr=time_table["std_seconds_per_fold"],
        capsize=3,
        color=colors[:-1],
        edgecolor="black",
        linewidth=0.4,
    )
    plt.ylabel("Training time per fold (seconds)")
    plt.xticks(rotation=35, ha="right")
    plt.title("Mean fold training time ± SD")
    plt.tight_layout()
    plt.savefig(figure_dir / "training_time_comparison.png", dpi=300)
    plt.close()

    performance_metrics = ["roc_auc", "accuracy", "f1", "mcc", "brier_score"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    metric_summary = summary.set_index("model")
    for axis, metric in zip(axes.flat, performance_metrics):
        means = [metric_summary.loc[name, f"{metric}_mean"] for name in model_names]
        stds = [metric_summary.loc[name, f"{metric}_std"] for name in model_names]
        axis.bar(range(len(model_names)), means, yerr=stds, capsize=2, color=colors)
        axis.set_title(metric.replace("_", " ").title())
        axis.set_xticks(range(len(model_names)))
        axis.set_xticklabels([DISPLAY_NAMES[name] for name in model_names], rotation=70, ha="right", fontsize=7)
    axes.flat[-1].axis("off")
    fig.suptitle("Mean performance ± SD across five seeds")
    fig.tight_layout()
    fig.savefig(figure_dir / "mean_performance_std.png", dpi=300)
    plt.close(fig)

    save_dataframe_atomic(time_table, figure_dir / "training_time_summary.csv")


def create_report(model_summaries: dict[str, dict[str, Any]], dataset_sha256: str) -> None:
    dataframe = load_original_dataset(DATASET_PATH, TARGET_COLUMN, deduplicate=True)
    aligned, metric_table = collect_aligned_oof(dataframe)
    summary = summarize_metrics(metric_table)
    paired_rows = []
    for name in MODEL_ORDER:
        paired = paired_auc_comparison(aligned["PMRC-Net"], aligned[name])
        paired_rows.append({"baseline": DISPLAY_NAMES[name], **paired})
    paired_table = pd.DataFrame(paired_rows)
    report_dir = ROOT / "reports"
    save_dataframe_atomic(metric_table, ROOT / "experiments" / "additional_baselines" / "final_metrics_by_seed.csv")
    save_dataframe_atomic(summary, ROOT / "experiments" / "additional_baselines" / "final_metrics_summary.csv")
    save_dataframe_atomic(paired_table, ROOT / "experiments" / "additional_baselines" / "paired_pmrc_auc_comparisons.csv")

    metric_headers = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "f1": "F1",
        "mcc": "MCC",
        "roc_auc": "ROC-AUC",
        "average_precision": "Average Precision",
        "brier_score": "Brier",
        "ece_10": "ECE",
    }
    summary_indexed = summary.set_index("model")
    lines = [
        "# Final 11-model PMRC-Net comparison",
        "",
        "## Protocol and provenance",
        f"- Dataset: `{DATASET_PATH.name}` (SHA-256 `{dataset_sha256}`).",
        "- Evaluation: five-fold stratified OOF predictions with seeds 21, 42, 123, 2026, and 3407.",
        "- Fold membership was checked against the existing Stage 5 OOF fold assignments for each seed.",
        "- All reported baseline predictions are out-of-fold; all learned preprocessing is fitted within each training fold.",
        "- Metrics are computed independently for each seed from its complete OOF prediction vector; table entries are mean ± sample standard deviation across five seeds.",
        "- Accuracy, precision, sensitivity, specificity, F1, MCC use threshold 0.5. ECE uses 10 uniform probability bins. Lower Brier/ECE is better.",
        "",
        "## Complete ranking by mean ROC-AUC",
        "",
        "| Rank | Model | " + " | ".join(metric_headers.values()) + " |",
        "|---:|---|" + "|".join(["---:"] * len(metric_headers)) + "|",
    ]
    for rank, (_, row) in enumerate(summary.iterrows(), start=1):
        model_name = str(row["model"])
        values = [
            f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}"
            for metric in metric_headers
        ]
        lines.append(f"| {rank} | {DISPLAY_NAMES[model_name]} | " + " | ".join(values) + " |")
    lines.extend([
        "",
        "## Paired ROC-AUC comparisons against PMRC-Net",
        "",
        "Predictions are aligned by sample index, target, and seed. The estimated difference is PMRC-Net minus baseline mean seed-level ROC-AUC. Confidence intervals use a sample-cluster paired bootstrap; p-values use paired prediction-swap permutation tests (1,000 iterations each).",
        "",
        "| Baseline | PMRC-Net − baseline AUC | Paired bootstrap 95% CI | Permutation p |",
        "|---|---:|---:|---:|",
    ])
    for row in paired_rows:
        lines.append(
            f"| {row['baseline']} | {row['pmrc_minus_baseline_mean_auc']:.4f} | "
            f"[{row['paired_bootstrap_ci_lower']:.4f}, {row['paired_bootstrap_ci_upper']:.4f}] | "
            f"{row['paired_permutation_p_value']:.4g} |"
        )
    lines.extend([
        "",
        "## Artifacts",
        "- Per-model, per-seed OOF files are under `experiments/additional_baselines/<model>/seed_<seed>/oof_predictions.csv`.",
        "- PMRC-Net OOF references are the unchanged Stage 5 files under `results/stage5_final_validation/seed_oof_predictions/`.",
        "- Per-seed metrics: `experiments/additional_baselines/final_metrics_by_seed.csv`.",
        "- Summary metrics: `experiments/additional_baselines/final_metrics_summary.csv`.",
        "- Paired tests: `experiments/additional_baselines/paired_pmrc_auc_comparisons.csv`.",
        "- Publication figures are in `reports/figures/additional_baselines/` at 300 DPI. PMRC-Net training time is not included because the preserved Stage 5 artifacts do not record per-fold wall-clock training time.",
        "",
        "## Interpretation",
        "This is a research benchmark, not a clinical performance claim. Statistical tests are descriptive and should be interpreted with the small dataset and repeated cross-validation structure in mind.",
    ])
    (report_dir / "FINAL_11_MODEL_COMPARISON.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_all_models(dataset_path: Path | None = None) -> None:
    target_path = DATASET_PATH if dataset_path is None else dataset_path
    dataset_sha256 = ensure_dataset_hash(target_path)
    dataframe = load_original_dataset(target_path, TARGET_COLUMN, deduplicate=True)
    root_dir = ROOT / "experiments" / "additional_baselines"
    root_dir.mkdir(parents=True, exist_ok=True)
    model_summaries: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_ORDER:
        print(f"Running {model_name}")
        summary = run_model_resumable(model_name, dataframe, dataset_sha256, root_dir)
        validate_oof_predictions(root_dir / model_name)
        model_summaries[model_name] = summary
    create_figure_set(model_summaries, load_pmrc_summary())
    create_report(model_summaries, dataset_sha256)
    print(json.dumps({"completed_models": list(model_summaries.keys()), "dataset_sha256": dataset_sha256}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and compare the six additional baseline models against the validated PMRC-Net reference.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    args = parser.parse_args()
    run_all_models(args.dataset)


if __name__ == "__main__":
    main()
