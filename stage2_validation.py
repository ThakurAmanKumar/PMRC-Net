from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data.dataset import load_original_dataset, make_fold_data
from src.models.pmrc_net import PMRCNet
from src.training.checkpoints import save_checkpoint
from src.training.losses import PMRCConsistencyLoss, compute_empirical_associations
from src.utils.config import load_config, resolve_path
from src.utils.runtime import configure_runtime, gpu_memory_report


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": threshold, "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "sensitivity": float(recall_score(y_true, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, predicted)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def train_pmrc(dataframe: pd.DataFrame, train_idx: np.ndarray, valid_idx: np.ndarray,
               config: dict[str, Any], device: torch.device, output: Path,
               variant: str = "complete", seed: int = 42) -> tuple[np.ndarray, dict[str, Any]]:
    seed_everything(seed)
    fold = make_fold_data(dataframe, train_idx, valid_idx, config)
    model_config = dict(config["model"])
    model_config["ablation"] = variant
    model = PMRCNet(config, model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])
    associations = compute_empirical_associations(dataframe.iloc[train_idx], config["data"]["target"])
    loss_fn = PMRCConsistencyLoss(associations, **config["loss"])
    best_loss = float("inf")
    best_epoch = 0
    patience = 0
    history: list[dict[str, Any]] = []
    best_path = output / "best.pt"
    latest_path = output / "latest.pt"
    output.mkdir(parents=True, exist_ok=True)
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        probabilities = []
        with torch.no_grad():
            for batch in fold.valid_loader:
                tensors = [item.to(device) for item in batch]
                probabilities.extend(torch.sigmoid(model(*tensors[:3])["z_total"]).squeeze(-1).cpu().tolist())
        history = checkpoint.get("history", [])
        return np.asarray(probabilities), {"variant": variant,
            "best_epoch": int(checkpoint.get("epoch", 0)),
            "best_valid_loss": float(checkpoint.get("best_value", float("nan"))),
            "history": history, "associations": associations,
            "resource": gpu_memory_report(device), "resumed": True}
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        totals = np.zeros(3, dtype=float)
        count = 0
        for batch in fold.train_loader:
            tensors = [item.to(device) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(*tensors[:3])
            loss, bce, physical = loss_fn(model, outputs, tensors[3], *tensors[:3])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"])
            optimizer.step()
            n = tensors[3].size(0)
            totals += np.array([loss.item(), bce.item(), physical.item()]) * n
            count += n
        model.eval()
        valid_loss = 0.0
        probabilities: list[float] = []
        labels: list[float] = []
        with torch.no_grad():
            for batch in fold.valid_loader:
                tensors = [item.to(device) for item in batch]
                outputs = model(*tensors[:3])
                loss, _, _ = loss_fn(model, outputs, tensors[3], *tensors[:3])
                n = tensors[3].size(0)
                valid_loss += loss.item() * n
                probabilities.extend(torch.sigmoid(outputs["z_total"]).squeeze(-1).cpu().tolist())
                labels.extend(tensors[3].cpu().tolist())
        valid_loss /= len(fold.valid_loader.dataset)
        record = {"epoch": epoch, "train_loss": float(totals[0] / count),
                  "train_bce": float(totals[1] / count),
                  "train_physical_penalty": float(totals[2] / count),
                  "valid_loss": float(valid_loss),
                  "valid_metrics": metrics(np.asarray(labels), np.asarray(probabilities))}
        history.append(record)
        save_checkpoint(latest_path, model, optimizer, epoch, best_loss, config, history)
        if valid_loss < best_loss - config["training"]["min_delta"]:
            best_loss, best_epoch, patience = valid_loss, epoch, 0
            save_checkpoint(best_path, model, optimizer, epoch, best_loss, config, history)
        else:
            patience += 1
            if patience >= config["training"]["patience"]:
                break
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    probabilities = []
    with torch.no_grad():
        for batch in fold.valid_loader:
            tensors = [item.to(device) for item in batch]
            probabilities.extend(torch.sigmoid(model(*tensors[:3])["z_total"]).squeeze(-1).cpu().tolist())
    return np.asarray(probabilities), {"variant": variant, "best_epoch": best_epoch,
        "best_valid_loss": best_loss, "history": history, "associations": associations,
        "resource": gpu_memory_report(device)}


def run_pmrc(dataframe: pd.DataFrame, config: dict[str, Any], device: torch.device,
             root: Path, folds: list[tuple[np.ndarray, np.ndarray]], variant: str = "complete") -> None:
    stage = root / "results" / "stage2_validation"
    (stage / "checkpoints").mkdir(parents=True, exist_ok=True)
    probabilities = np.full(len(dataframe), np.nan)
    fold_rows: list[dict[str, Any]] = []
    for fold_number, (train_idx, valid_idx) in enumerate(folds, start=1):
        print(f"PMRC-Net {variant}: fold {fold_number}/{len(folds)}")
        fold_dir = stage / "checkpoints" / f"{variant}_fold_{fold_number}"
        fold_prob, details = train_pmrc(dataframe, train_idx, valid_idx, config, device, fold_dir,
                                        variant=variant, seed=int(config["seed"]) + fold_number)
        probabilities[valid_idx] = fold_prob
        fold_metric = metrics(dataframe.iloc[valid_idx][config["data"]["target"]].to_numpy(), fold_prob)
        fold_rows.append({"fold": fold_number, "train_rows": len(train_idx), "valid_rows": len(valid_idx),
                          **{k: v for k, v in fold_metric.items()},
                          "best_epoch": details["best_epoch"], "best_valid_loss": details["best_valid_loss"],
                          "associations": json.dumps(details["associations"]), "resource": json.dumps(details["resource"])})
    target = dataframe[config["data"]["target"]].to_numpy(dtype=int)
    oof = dataframe[["sample_index", config["data"]["target"]]].copy()
    oof.rename(columns={config["data"]["target"]: "target"}, inplace=True)
    fold_for_row = np.zeros(len(dataframe), dtype=int)
    for fold_number, (_, valid_idx) in enumerate(folds, start=1):
        fold_for_row[valid_idx] = fold_number
    oof["fold"] = fold_for_row
    oof["probability"] = probabilities
    oof.to_csv(stage / ("oof_predictions.csv" if variant == "complete" else f"oof_predictions_{variant}.csv"), index=False)
    pd.DataFrame(fold_rows).to_csv(stage / ("fold_metrics.csv" if variant == "complete" else f"fold_metrics_{variant}.csv"), index=False)
    aggregate = metrics(target, probabilities)
    aggregate.update({"model": "PMRC-Net" if variant == "complete" else f"PMRC-Net [{variant}]",
                      "rows": len(dataframe), "folds": len(folds), "unique_oof_rows": int(np.isfinite(probabilities).sum())})
    aggregate_path = stage / ("aggregate_metrics.json" if variant == "complete" else f"aggregate_metrics_{variant}.json")
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")


def run_baselines(dataframe: pd.DataFrame, config: dict[str, Any], root: Path,
                  folds: list[tuple[np.ndarray, np.ndarray]]) -> None:
    stage = root / "results" / "stage2_validation"
    columns = config["data"]["continuous_cols"] + config["data"]["categorical_cols"]
    target = dataframe[config["data"]["target"]].to_numpy(dtype=int)
    features = dataframe[columns].to_numpy(dtype=float)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1),
        "Baseline MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=200,
                                                                      early_stopping=True, random_state=42)),
    }
    rows = []
    for name, model in models.items():
        probabilities = np.full(len(dataframe), np.nan)
        for train_idx, valid_idx in folds:
            model.fit(features[train_idx], target[train_idx])
            probabilities[valid_idx] = model.predict_proba(features[valid_idx])[:, 1]
        rows.append({"model": name, **metrics(target, probabilities)})
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                              random_state=42, n_jobs=1, tree_method="hist")
        probabilities = np.full(len(dataframe), np.nan)
        for train_idx, valid_idx in folds:
            model.fit(features[train_idx], target[train_idx])
            probabilities[valid_idx] = model.predict_proba(features[valid_idx])[:, 1]
        rows.append({"model": "XGBoost", **metrics(target, probabilities)})
    except ImportError:
        rows.append({"model": "XGBoost", "status": "not_run", "reason": "xgboost is not installed"})
    rows.append({"model": "Tabular Transformer", "status": "not_run",
                 "reason": "Only notebook-local implementation exists; no importable implementation is available"})
    (stage / "baseline_comparison.csv").parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(stage / "baseline_comparison.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential, leakage-controlled Stage 2 validation.")
    parser.add_argument("--stage", choices=["pmrc", "baselines", "ablations", "all"], default="pmrc")
    parser.add_argument("--variant", choices=["no_physiology", "no_pathways", "no_mechanistic", "no_residual"], default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = load_config(root / "configs" / "model_config.yaml")
    seed_everything(int(config["seed"]))
    device = configure_runtime(config)
    data_path = resolve_path(root, config["data"]["path"])
    dataframe = load_original_dataset(data_path, config["data"]["target"], config["data"]["deduplicate_exact_rows"])
    target = dataframe[config["data"]["target"]].to_numpy()
    splitter = StratifiedKFold(n_splits=int(config["training"]["folds"]), shuffle=True, random_state=int(config["seed"]))
    folds = [(train.astype(np.int64), valid.astype(np.int64)) for train, valid in splitter.split(dataframe, target)]
    stage = root / "results" / "stage2_validation"
    stage.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": int(config["seed"]), "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "dataset_path": str(data_path), "rows": len(dataframe), "confirmed_patient_count": None,
                "folds": [{"fold": i + 1, "train_indices": train.tolist(), "valid_indices": valid.tolist()}
                          for i, (train, valid) in enumerate(folds)]}
    (stage / "fold_assignments.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (stage / "experiment_configurations").mkdir(exist_ok=True)
    (stage / "experiment_configurations" / f"{args.stage}.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if args.stage in {"pmrc", "all"}:
        run_pmrc(dataframe, config, device, root, folds)
    if args.stage in {"baselines", "all"}:
        run_baselines(dataframe, config, root, folds)
    if args.stage in {"ablations", "all"}:
        variants = (args.variant,) if args.variant else ("no_physiology", "no_pathways", "no_mechanistic", "no_residual")
        for variant in variants:
            run_pmrc(dataframe, config, device, root, folds, variant)
    report = stage / "research_report.md"
    if not report.exists():
        report.write_text("""# Stage 2 Validation

This directory contains sequential, stratified five-fold out-of-fold evaluation.

Preprocessing is fitted within each training fold. OOF predictions are generated exactly once per deduplicated row.

The 602 rows are deduplicated observations, not 602 confirmed independent patients; no patient identifier is available.

The original notebook, controlled holdout artifacts, and PMRC-Net architecture remain unchanged. The physiological penalty is reported as zero when the training-fold association directions are negative because the original loss activates only positive-direction constraints. No clinical or causal claim follows from these experiments.
""", encoding="utf-8")
    print(json.dumps({"stage": args.stage, "device": str(device), "rows": len(dataframe), "output": str(stage)}, indent=2))


if __name__ == "__main__":
    main()