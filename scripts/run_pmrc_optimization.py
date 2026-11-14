from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "pmrc_optimization"
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_original_dataset, make_fold_data
from src.models.pmrc_net_optimized import OptimizedPMRCNet
from src.training.losses import PMRCConsistencyLoss, compute_empirical_associations
from src.utils.config import load_config, resolve_path
from src.utils.runtime import configure_runtime


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def metric_values(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "sensitivity": float(recall_score(labels, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def evaluate_on_loader(model: OptimizedPMRCNet, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    labels_all: list[int] = []
    prob_all: list[float] = []
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            tensors = [item.to(device) for item in batch]
            outputs = model(*tensors[:3])
            logits = outputs["z_total"].squeeze(-1)
            prob_all.extend(torch.sigmoid(logits).cpu().tolist())
            labels_all.extend(tensors[3].cpu().tolist())
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, tensors[3].float(), reduction="mean")
            total_loss += float(loss.item()) * tensors[3].size(0)
            total_count += tensors[3].size(0)
    return np.asarray(labels_all, dtype=int), np.asarray(prob_all, dtype=float), total_loss / max(total_count, 1)


def fit_model_on_indices(
    dataframe: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    phys_lambda: float,
    epochs: int = 30,
) -> tuple[OptimizedPMRCNet, np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    model_cfg = deepcopy(config["model"])
    model_cfg["dropout"] = dropout
    model_cfg["coupling"] = "asymmetric"
    model = OptimizedPMRCNet(config, model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    fold = make_fold_data(dataframe, train_idx, valid_idx, config)
    loss_cfg = deepcopy(config["loss"])
    loss_cfg["phys_lambda"] = phys_lambda
    loss_fn = PMRCConsistencyLoss(compute_empirical_associations(dataframe.iloc[train_idx], config["data"]["target"]), **loss_cfg)
    best_state = None
    best_valid_loss = float("inf")
    patience = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_count = 0
        for batch in fold.train_loader:
            tensors = [item.to(device) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(*tensors[:3])
            loss, _, _ = loss_fn(model, outputs, tensors[3], *tensors[:3])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item()) * tensors[3].size(0)
            epoch_count += tensors[3].size(0)
        model.eval()
        labels, probabilities, valid_loss = evaluate_on_loader(model, fold.valid_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss / max(epoch_count, 1),
            "valid_loss": valid_loss,
            "valid_metrics": metric_values(labels, probabilities),
        })
        if valid_loss < best_valid_loss - 1e-4:
            best_valid_loss = valid_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    labels, probabilities, _ = evaluate_on_loader(model, fold.valid_loader, device)
    return model, labels, probabilities, {"best_valid_loss": best_valid_loss, "history": history}, history


def select_inner_best(dataframe: pd.DataFrame, outer_train_idx: np.ndarray, config: dict[str, Any], device: torch.device) -> dict[str, float]:
    target = dataframe.iloc[outer_train_idx][config["data"]["target"]].to_numpy(dtype=int)
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    candidates = [
        {"dropout": 0.10, "learning_rate": 1e-3, "weight_decay": 1e-4, "phys_lambda": 0.10},
        {"dropout": 0.15, "learning_rate": 5e-4, "weight_decay": 5e-4, "phys_lambda": 0.15},
        {"dropout": 0.20, "learning_rate": 3e-4, "weight_decay": 1e-3, "phys_lambda": 0.20},
    ]
    best_choice = candidates[1]
    best_score = -np.inf
    for candidate in candidates:
        scores = []
        for inner_train_rel, inner_valid_rel in splitter.split(np.arange(len(outer_train_idx)), target):
            inner_train_idx = outer_train_idx[inner_train_rel]
            inner_valid_idx = outer_train_idx[inner_valid_rel]
            _, labels, probabilities, _, _ = fit_model_on_indices(
                dataframe,
                inner_train_idx,
                inner_valid_idx,
                config,
                device,
                dropout=candidate["dropout"],
                learning_rate=candidate["learning_rate"],
                weight_decay=candidate["weight_decay"],
                phys_lambda=candidate["phys_lambda"],
                epochs=30,
            )
            metrics = metric_values(labels, probabilities)
            scores.append(metrics["roc_auc"] + 0.5 * metrics["mcc"])
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_choice = candidate
    return best_choice


def run_single_seed(seed: int, dataframe: pd.DataFrame, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    target = dataframe[config["data"]["target"]].to_numpy(dtype=int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    all_probs = np.full(len(dataframe), np.nan, dtype=float)
    all_fold = np.full(len(dataframe), -1, dtype=int)
    fold_records = []
    for outer_fold, (train_idx, valid_idx) in enumerate(splitter.split(dataframe, target), start=1):
        best_choice = select_inner_best(dataframe, train_idx, config, device)
        _, labels, probabilities, details, history = fit_model_on_indices(
            dataframe,
            train_idx,
            valid_idx,
            config,
            device,
            dropout=best_choice["dropout"],
            learning_rate=best_choice["learning_rate"],
            weight_decay=best_choice["weight_decay"],
            phys_lambda=best_choice["phys_lambda"],
            epochs=30,
        )
        metrics = metric_values(labels, probabilities)
        fold_records.append({
            "seed": seed,
            "outer_fold": outer_fold,
            "best_choice": best_choice,
            **metrics,
            "best_valid_loss": details["best_valid_loss"],
            "history_epochs": len(history),
        })
        all_probs[valid_idx] = probabilities
        all_fold[valid_idx] = outer_fold
    summary = metric_values(dataframe[config["data"]["target"]].to_numpy(dtype=int), all_probs[~np.isnan(all_probs)])
    oof = pd.DataFrame({
        "sample_index": dataframe["sample_index"].to_numpy(dtype=int),
        "target": dataframe[config["data"]["target"]].to_numpy(dtype=int),
        "fold": all_fold,
        "probability": all_probs,
    })
    return {"seed": seed, "summary": summary, "fold_metrics": fold_records, "oof": oof}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled PMRC-Net optimization experiment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "configs" / "model_config.yaml")
    config["training"]["epochs"] = 30
    config["training"]["patience"] = 8
    config["training"]["num_workers"] = 0
    config["training"]["cpu_threads"] = 2
    config["loss"]["phys_lambda"] = 0.15
    config["loss"]["gate_l2_lambda"] = 1e-4
    config["loss"]["residual_l2_lambda"] = 1e-4
    config["loss"]["l2_lambda"] = 1e-2
    config["loss"]["l1_lambda"] = 1e-2
    device = configure_runtime(config)
    dataframe = load_original_dataset(resolve_path(ROOT, config["data"]["path"]), config["data"]["target"], config["data"]["deduplicate_exact_rows"])
    result = run_single_seed(args.seed, dataframe, config, device)
    result["config"] = config
    (OUTPUT / "optimization_seed42_result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    result["oof"].to_csv(OUTPUT / "optimization_seed42_oof.csv", index=False)
    pd.DataFrame(result["fold_metrics"]).to_csv(OUTPUT / "optimization_seed42_fold_metrics.csv", index=False)
    print(json.dumps({"seed": args.seed, "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
