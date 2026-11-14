from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import psutil
import torch
from scipy.optimize import minimize_scalar
from scipy.stats import wilcoxon
from scipy.stats import wilcoxon
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
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_original_dataset, make_fold_data
from src.evaluation.metrics import calculate_metrics, select_threshold
from src.evaluation.plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_precision_recall_curves,
    plot_roc_curves,
)
from src.evaluation.plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_precision_recall_curves,
    plot_roc_curves,
)
from src.models.pmrc_net_optimized import OptimizedPMRCNet
from src.training.losses import PMRCConsistencyLoss, compute_empirical_associations
from src.utils.config import load_config, resolve_path
from src.utils.runtime import configure_runtime

SEEDS = [21, 42, 123, 2026, 3407]
METRICS = [
    "accuracy", "precision", "sensitivity", "specificity", "f1", "mcc",
    "roc_auc", "average_precision", "brier_score", "ece",
]
CANDIDATES = [
    {"dropout": 0.10, "learning_rate": 1e-3, "weight_decay": 1e-4, "phys_lambda": 0.10, "gate_l2_lambda": 1e-4, "residual_l2_lambda": 1e-4},
    {"dropout": 0.15, "learning_rate": 5e-4, "weight_decay": 5e-4, "phys_lambda": 0.15, "gate_l2_lambda": 5e-4, "residual_l2_lambda": 5e-4},
    {"dropout": 0.20, "learning_rate": 3e-4, "weight_decay": 1e-3, "phys_lambda": 0.05, "gate_l2_lambda": 1e-3, "residual_l2_lambda": 0.0},
]
GPU_TELEMETRY_EVERY_EPOCHS = 5
MAX_SYSTEM_RAM_PERCENT = 95.0
MAX_GPU_MEMORY_PERCENT = 95.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def hardware_metadata(device: torch.device, config: dict[str, Any]) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    return {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if properties else None,
        "gpu_total_memory_mb": properties.total_memory / 2**20 if properties else 0,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_threads": int(config["training"]["cpu_threads"]),
        "cpu_interop_threads": int(config["training"]["cpu_interop_threads"]),
        "dataloader_workers": int(config["training"]["num_workers"]),
        "pin_memory": bool(config["hardware"]["pin_memory"]),
        "batch_size": int(config["training"]["batch_size"]),
        "mixed_precision": bool(config["training"]["mixed_precision"]),
    }


class ResourceRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: list[dict[str, Any]] = (
            pd.read_csv(path).to_dict("records") if path.exists() else []
        )
        self.process = psutil.Process()
        psutil.cpu_percent(interval=None)
        self.gpu_query_count = sum(
            pd.notna(row.get("gpu_utilization_percent")) for row in self.rows
        )
        self.process.cpu_percent(interval=None)

    def record(
        self,
        device: torch.device,
        context: dict[str, Any],
        epoch: int,
        elapsed_seconds: float,
        sample_gpu: bool,
    ) -> None:
        memory = self.process.memory_info()
        row: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **context,
            "epoch": epoch,
            "epoch_seconds": elapsed_seconds,
            "system_cpu_percent": psutil.cpu_percent(interval=None),
            "process_cpu_percent": self.process.cpu_percent(interval=None),
            "process_rss_mb": memory.rss / 2**20,
            "system_memory_percent": psutil.virtual_memory().percent,
            "system_memory_available_mb": psutil.virtual_memory().available / 2**20,
            "gpu_utilization_percent": None,
            "gpu_device_memory_used_mb": None,
            "gpu_temperature_c": None,
            "gpu_allocated_mb": 0.0,
            "gpu_reserved_mb": 0.0,
            "gpu_peak_allocated_mb": 0.0,
        }
        if device.type == "cuda":
            row["gpu_allocated_mb"] = torch.cuda.memory_allocated(device) / 2**20
            row["gpu_reserved_mb"] = torch.cuda.memory_reserved(device) / 2**20
            row["gpu_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
            if sample_gpu:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=3,
                )
                values = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
                row["gpu_utilization_percent"] = float(values[0])
                row["gpu_device_memory_used_mb"] = float(values[1])
                row["gpu_temperature_c"] = float(values[2])
                self.gpu_query_count += 1
                total_memory_mb = torch.cuda.get_device_properties(device).total_memory / 2**20
                if row["gpu_device_memory_used_mb"] >= total_memory_mb * MAX_GPU_MEMORY_PERCENT / 100:
                    raise MemoryError(
                        f"GPU memory usage reached {row['gpu_device_memory_used_mb']:.0f} MB "
                        f"of {total_memory_mb:.0f} MB"
                    )
        if row["system_memory_percent"] >= MAX_SYSTEM_RAM_PERCENT:
            raise MemoryError(
                f"System RAM usage reached {row['system_memory_percent']:.1f}%"
            )
        self.rows.append(row)
        if len(self.rows) % GPU_TELEMETRY_EVERY_EPOCHS == 0:
            self.flush()

    def flush(self) -> None:
        if self.rows:
            atomic_csv(pd.DataFrame(self.rows), self.path)

    def summary(self) -> dict[str, Any]:
        frame = pd.DataFrame(self.rows)
        if frame.empty:
            return {"samples": 0}
        numeric = [
            "system_cpu_percent", "process_cpu_percent", "process_rss_mb",
            "system_memory_percent", "system_memory_available_mb",
            "gpu_utilization_percent", "gpu_device_memory_used_mb", "gpu_temperature_c",
            "gpu_allocated_mb", "gpu_reserved_mb", "gpu_peak_allocated_mb", "epoch_seconds",
        ]
        summary: dict[str, Any] = {"samples": len(frame)}
        for column in numeric:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not values.empty:
                summary[f"mean_{column}"] = float(values.mean())
                summary[f"max_{column}"] = float(values.max())
        return summary


def expected_splits(dataframe: pd.DataFrame, config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    target = dataframe[config["data"]["target"]].to_numpy(dtype=int)
    manifest: dict[str, list[dict[str, Any]]] = {}
    for seed in SEEDS:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        manifest[str(seed)] = [
            {"fold": fold, "train_indices": train.astype(int).tolist(), "valid_indices": valid.astype(int).tolist()}
            for fold, (train, valid) in enumerate(splitter.split(dataframe, target), start=1)
        ]
    return manifest


def validate_reference_splits(manifest: dict[str, list[dict[str, Any]]], dataframe: pd.DataFrame) -> None:
    reference_dir = ROOT / "results" / "stage5_final_validation" / "seed_oof_predictions"
    for seed in SEEDS:
        path = reference_dir / f"seed_{seed}_oof.csv"
        if not path.exists():
            raise FileNotFoundError(f"Authoritative OOF split reference is missing: {path}")
        reference = pd.read_csv(path).sort_values("sample_index").reset_index(drop=True)
        expected_fold = np.full(len(dataframe), -1, dtype=int)
        for record in manifest[str(seed)]:
            expected_fold[np.asarray(record["valid_indices"], dtype=int)] = int(record["fold"])
        if not np.array_equal(reference["sample_index"].to_numpy(dtype=int), dataframe["sample_index"].to_numpy(dtype=int)):
            raise ValueError(f"Reference OOF sample alignment differs for seed {seed}")
        if not np.array_equal(reference["fold"].to_numpy(dtype=int), expected_fold):
            raise ValueError(f"Generated outer folds differ from authoritative seed-{seed} assignments")


def expected_inner_splits(train_indices: np.ndarray, labels: np.ndarray, seed: int, outer_fold: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + 1009 * outer_fold)
    relative = np.arange(len(train_indices))
    return [
        (train_indices[inner_train], train_indices[inner_valid])
        for inner_train, inner_valid in splitter.split(relative, labels[train_indices])
    ]


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1] if index < bins - 1 else probabilities <= edges[index + 1])
        if mask.any():
            result += float(mask.mean()) * abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
    return result


def metric_values(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float | None = 0.5,
    predicted_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    return calculate_metrics(
        labels,
        probabilities,
        threshold=threshold,
        predicted_labels=predicted_labels,
    )


def make_model(config: dict[str, Any], candidate: dict[str, float], directional: bool, device: torch.device) -> OptimizedPMRCNet:
    model_config = deepcopy(config["model"])
    model_config["dropout"] = candidate["dropout"]
    model_config["coupling"] = "asymmetric"
    model_config["directional_asymmetry"] = directional
    return OptimizedPMRCNet(config, model_config).to(device)


def make_loss(dataframe: pd.DataFrame, train_indices: np.ndarray, config: dict[str, Any], candidate: dict[str, float]) -> PMRCConsistencyLoss:
    target_col = config["data"]["target"]
    associations = compute_empirical_associations(dataframe.iloc[train_indices], target_col)
    loss_config = deepcopy(config["loss"])
    loss_config.update({key: candidate[key] for key in ("phys_lambda", "gate_l2_lambda", "residual_l2_lambda")})
    return PMRCConsistencyLoss(associations, **loss_config)


def cache_loader_tensors(loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[torch.Tensor, ...]:
    tensors = loader.dataset.tensors
    if device.type == "cuda":
        return tuple(tensor.pin_memory().to(device, non_blocking=True) for tensor in tensors)
    return tuple(tensor.to(device) for tensor in tensors)


def tensor_batches(
    tensors: tuple[torch.Tensor, ...], batch_size: int, shuffle: bool,
) -> Iterator[tuple[torch.Tensor, ...]]:
    count = len(tensors[-1])
    device = tensors[-1].device
    indices = torch.randperm(count, device=device) if shuffle else torch.arange(count, device=device)
    for start in range(0, count, batch_size):
        yield tuple(tensor.index_select(0, indices[start:start + batch_size]) for tensor in tensors)

    
def train_epoch(
    model: OptimizedPMRCNet,
    training_tensors: tuple[torch.Tensor, ...],
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    loss_fn: PMRCConsistencyLoss,
    device: torch.device,
    recorder: ResourceRecorder | None = None,
    context: dict[str, Any] | None = None,
    epoch: int = 0,
    max_batches: int | None = None,
    step_times: list[float] | None = None,
) -> float:
    started = time.perf_counter()
    model.train()
    total_loss = 0.0
    count = 0
    gpu_step_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for batch_number, tensors in enumerate(tensor_batches(training_tensors, batch_size, shuffle=True)):
        if max_batches is not None and batch_number >= max_batches:
            break
        if step_times is not None and device.type == "cuda":
            step_start_event = torch.cuda.Event(enable_timing=True)
            step_end_event = torch.cuda.Event(enable_timing=True)
            step_start_event.record()
        else:
            step_started = time.perf_counter() if step_times is not None else 0.0
        optimizer.zero_grad(set_to_none=True)
        outputs = model(*tensors[:3])
        loss, _, _ = loss_fn(model, outputs, tensors[3], *tensors[:3])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("Non-finite gradient norm")
        optimizer.step()
        if step_times is not None:
            if device.type == "cuda":
                step_end_event.record()
                gpu_step_events.append((step_start_event, step_end_event))
            else:
                step_times.append(time.perf_counter() - step_started)
        total_loss += float(loss.detach()) * len(tensors[3])
        count += len(tensors[3])
    if gpu_step_events:
        torch.cuda.synchronize(device)
        step_times.extend(start.elapsed_time(end) / 1000.0 for start, end in gpu_step_events)
    if recorder is not None:
        recorder.record(
            device,
            context or {},
            epoch,
            time.perf_counter() - started,
            sample_gpu=epoch == 1 or epoch % GPU_TELEMETRY_EVERY_EPOCHS == 0,
        )
    return total_loss / max(count, 1)


def infer(
    model: OptimizedPMRCNet,
    validation_tensors: tuple[torch.Tensor, ...],
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    collected: dict[str, list[float]] = {key: [] for key in ("target", "z_total", "z_mech", "z_inter", "z_res", "m_e")}
    with torch.inference_mode():
        for tensors in tensor_batches(validation_tensors, batch_size, shuffle=False):
            outputs = model(*tensors[:3])
            collected["target"].extend(tensors[3].cpu().numpy().tolist())
            for key in ("z_total", "z_mech", "z_inter", "z_res", "m_e"):
                collected[key].extend(outputs[key].reshape(-1).cpu().numpy().tolist())
    return {key: np.asarray(values, dtype=float) for key, values in collected.items()}


def fit_inner_model(
    dataframe: pd.DataFrame,
    train_indices: np.ndarray,
    fold_data: Any,
    training_tensors: tuple[torch.Tensor, ...],
    validation_tensors: tuple[torch.Tensor, ...],
    config: dict[str, Any],
    candidate: dict[str, float],
    directional: bool,
    device: torch.device,
    init_seed: int,
    recorder: ResourceRecorder,
    context: dict[str, Any],
) -> tuple[dict[str, np.ndarray], int, float]:
    set_seed(init_seed)
    model = make_model(config, candidate, directional, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate["learning_rate"], weight_decay=candidate["weight_decay"])
    loss_fn = make_loss(dataframe, train_indices, config, candidate)
    best_loss = float("inf")
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_epoch(
            model, training_tensors, int(config["training"]["batch_size"]),
            optimizer, loss_fn, device, recorder, context, epoch,
        )
        model.eval()
        valid_output = infer(model, validation_tensors, int(config["training"]["batch_size"]))
        valid_loss = float(np.mean(np.logaddexp(0.0, valid_output["z_total"]) - valid_output["target"] * valid_output["z_total"]))
        if valid_loss < best_loss - float(config["training"]["min_delta"]):
            best_loss = valid_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(config["training"]["patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    output = infer(model, validation_tensors, int(config["training"]["batch_size"]))
    return output, best_epoch, best_loss


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    def objective(log_temperature: float) -> float:
        scaled = logits / math.exp(log_temperature)
        return float(np.mean(np.logaddexp(0.0, scaled) - labels * scaled))

    result = minimize_scalar(objective, bounds=(-2.3, 2.3), method="bounded")
    return float(math.exp(result.x))


def select_inner_candidate(
    dataframe: pd.DataFrame,
    outer_train: np.ndarray,
    seed: int,
    outer_fold: int,
    config: dict[str, Any],
    directional: bool,
    device: torch.device,
    recorder: ResourceRecorder,
    arm_name: str,
) -> tuple[dict[str, float], int, float, dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    labels = dataframe[config["data"]["target"]].to_numpy(dtype=int)
    inner_splits = expected_inner_splits(outer_train, labels, seed, outer_fold)
    prepared_splits = []
    for inner_train, inner_valid in inner_splits:
        fold_data = make_fold_data(dataframe, inner_train, inner_valid, config)
        prepared_splits.append((
            inner_train,
            inner_valid,
            fold_data,
            cache_loader_tensors(fold_data.train_loader, device),
            cache_loader_tensors(fold_data.valid_loader, device),
        ))
    candidate_results: list[dict[str, Any]] = []
    candidate_outputs: list[list[tuple[int, np.ndarray, np.ndarray]]] = []
    for candidate_index, candidate in enumerate(CANDIDATES):
        fold_scores: list[float] = []
        fold_epochs: list[int] = []
        fold_outputs: list[tuple[int, np.ndarray, np.ndarray]] = []
        inner_details: list[dict[str, Any]] = []
        for inner_fold, (inner_train, inner_valid, fold_data, training_tensors, validation_tensors) in enumerate(prepared_splits, start=1):
            init_seed = seed * 100000 + outer_fold * 1000 + inner_fold
            context = {
                "arm": arm_name,
                "phase": "inner_selection",
                "seed": seed,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "candidate_index": candidate_index + 1,
            }
            output, best_epoch, best_loss = fit_inner_model(
                dataframe, inner_train, fold_data, training_tensors, validation_tensors,
                config, candidate, directional, device, init_seed, recorder, context,
            )
            metrics = metric_values(output["target"].astype(int), 1.0 / (1.0 + np.exp(-output["z_total"])))
            mcc_for_selection = metrics["mcc"] if metrics["mcc"] is not None else -1.0
            fold_scores.append(float(metrics["roc_auc"]) + 0.5 * float(mcc_for_selection))
            fold_epochs.append(best_epoch)
            fold_outputs.append((inner_fold, inner_valid, output))
            inner_details.append({"inner_fold": inner_fold, "train_rows": len(inner_train), "valid_rows": len(inner_valid), "best_epoch": best_epoch, "best_validation_bce": best_loss, "metrics": metrics})
        candidate_results.append({"candidate": candidate, "mean_selection_score": float(np.mean(fold_scores)), "inner_folds": inner_details})
        candidate_outputs.append(fold_outputs)
    best_index = int(np.argmax([item["mean_selection_score"] for item in candidate_results]))
    best_candidate = CANDIDATES[best_index]
    chosen_folds = candidate_outputs[best_index]
    selected_epochs = max(1, int(math.ceil(float(np.median([item["best_epoch"] for item in candidate_results[best_index]["inner_folds"]])))))
    inner_frames = []
    for inner_fold, indices, output in chosen_folds:
        frame = pd.DataFrame({
            "sample_index": dataframe.iloc[indices]["sample_index"].to_numpy(dtype=int),
            "inner_fold": inner_fold,
            "target": output["target"].astype(int),
            "logit": output["z_total"],
            "z_mech": output["z_mech"],
            "z_inter": output["z_inter"],
            "z_res": output["z_res"],
            "m_e": output["m_e"],
        })
        inner_frames.append(frame)
    inner_oof = pd.concat(inner_frames, ignore_index=True).sort_values("sample_index").reset_index(drop=True)
    temperature = fit_temperature(inner_oof["logit"].to_numpy(), inner_oof["target"].to_numpy(dtype=float))
    inner_oof["probability_calibrated"] = 1.0 / (1.0 + np.exp(-inner_oof["logit"].to_numpy() / temperature))
    threshold_selection = select_threshold(
        inner_oof["target"].to_numpy(dtype=int),
        inner_oof["probability_calibrated"].to_numpy(dtype=float),
        method="youden",
    )
    return best_candidate, selected_epochs, temperature, threshold_selection, candidate_results, inner_oof


def refit_outer(
    dataframe: pd.DataFrame,
    train_indices: np.ndarray,
    config: dict[str, Any],
    candidate: dict[str, float],
    directional: bool,
    device: torch.device,
    seed: int,
    outer_fold: int,
    epochs: int,
    recorder: ResourceRecorder,
    arm_name: str,
) -> tuple[OptimizedPMRCNet, Any]:
    set_seed(seed * 100000 + outer_fold * 1000 + 999)
    fold_data = make_fold_data(dataframe, train_indices, np.empty(0, dtype=np.int64), config)
    training_tensors = cache_loader_tensors(fold_data.train_loader, device)
    model = make_model(config, candidate, directional, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate["learning_rate"], weight_decay=candidate["weight_decay"])
    loss_fn = make_loss(dataframe, train_indices, config, candidate)
    for epoch in range(1, epochs + 1):
        train_epoch(
            model, training_tensors, int(config["training"]["batch_size"]),
            optimizer, loss_fn, device, recorder,
            {"arm": arm_name, "phase": "outer_refit", "seed": seed, "outer_fold": outer_fold}, epoch,
        )
    return model, fold_data


def summarize_seed_oof(frame: pd.DataFrame) -> dict[str, Any]:
    predicted_labels = frame["predicted_label"].to_numpy(dtype=int) if "predicted_label" in frame else None
    return metric_values(
        frame["target"].to_numpy(dtype=int),
        frame["probability"].to_numpy(dtype=float),
        threshold=None if predicted_labels is not None else 0.5,
        predicted_labels=predicted_labels,
    )


def fold_summary_record(arm_name: str, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": report["seed"],
        "fold": report["outer_fold"],
        "arm": arm_name,
        **{f"raw_{key}": value for key, value in report["raw_metrics"].items()},
        **{f"calibrated_{key}": value for key, value in report["calibrated_metrics"].items()},
        "selected_epochs": report["selected_epochs"],
        "temperature": report["calibration_temperature_inner_only"],
        "threshold": report["threshold_selection_inner_only"]["threshold"],
        "threshold_method": report["threshold_selection_inner_only"]["method"],
        "selected_candidate": json.dumps(report["selected_candidate"], sort_keys=True),
    }


def write_arm_outputs(arm_dir: Path, fold_records: list[dict[str, Any]], seed_frames: dict[int, pd.DataFrame], dataframe: pd.DataFrame) -> None:
    if set(seed_frames) != set(SEEDS) or len(fold_records) != len(SEEDS) * 5:
        raise RuntimeError(f"Incomplete nested-CV output for {arm_dir.name}")
    fold_frame = pd.DataFrame(fold_records)
    atomic_csv(fold_frame, arm_dir / "fold_metrics.csv")
    seed_metrics: list[dict[str, Any]] = []
    all_errors: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    expected_indices = dataframe["sample_index"].to_numpy(dtype=int)
    expected_targets = dataframe["target"].to_numpy(dtype=int)
    for seed, frame in seed_frames.items():
        ordered = frame.sort_values("sample_index").reset_index(drop=True)
        if (
            not np.array_equal(ordered["sample_index"].to_numpy(dtype=int), expected_indices)
            or not np.array_equal(ordered["target"].to_numpy(dtype=int), expected_targets)
        ):
            raise RuntimeError(f"Incomplete or misaligned OOF predictions for {arm_dir.name}, seed {seed}")
        atomic_csv(ordered, arm_dir / f"seed_{seed}_oof.csv")
        all_predictions.append(ordered)
        row = {"seed": seed, **summarize_seed_oof(ordered)}
        seed_metrics.append(row)
        predicted = (
            ordered["predicted_label"].to_numpy(dtype=int)
            if "predicted_label" in ordered
            else (ordered["probability"].to_numpy() >= 0.5).astype(int)
        )
        errors = ordered.loc[predicted != ordered["target"].to_numpy()].copy()
        errors["error_type"] = np.where(errors["target"].to_numpy(dtype=int) == 1, "false_negative", "false_positive")
        raw_features = dataframe.set_index("sample_index")
        feature_cols = [col for col in ("age", "sex", "trestbps", "chol", "thalachh", "oldpeak", "exang", "ca", "cp", "thal") if col in raw_features]
        errors = errors.merge(raw_features[feature_cols], left_on="sample_index", right_index=True, how="left")
        if "seed" not in errors:
            errors.insert(1, "seed", seed)
        all_errors.append(errors)
    seed_frame = pd.DataFrame(seed_metrics)
    atomic_csv(seed_frame, arm_dir / "seed_metrics.csv")
    atomic_csv(pd.concat(all_predictions, ignore_index=True), arm_dir / "oof_predictions.csv")
    summary = []
    for metric in METRICS:
        summary.append({"metric": metric, "mean": float(seed_frame[metric].mean()), "std": float(seed_frame[metric].std(ddof=1))})
    atomic_csv(pd.DataFrame(summary), arm_dir / "mean_std_summary.csv")
    if all_errors:
        atomic_csv(pd.concat(all_errors, ignore_index=True), arm_dir / "false_positive_false_negative_cases.csv")


def run_arm(
    arm_name: str,
    directional: bool,
    dataframe: pd.DataFrame,
    config: dict[str, Any],
    manifest: dict[str, list[dict[str, Any]]],
    device: torch.device,
    root_dir: Path,
    resume: bool,
    dataset_hash: str,
    recorder: ResourceRecorder,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    arm_dir = root_dir / arm_name
    if arm_dir.exists() and not resume:
        raise FileExistsError(f"Experiment arm already exists; use --resume or a new run ID: {arm_dir}")
    arm_dir.mkdir(parents=True, exist_ok=True)
    arm_config = {
        "model_variant": arm_name,
        "directional_asymmetry": directional,
        "base_config": config,
        "candidates": CANDIDATES,
        "selection_metric": "mean inner-fold ROC-AUC + 0.5 * MCC",
        "epoch_selection": "inner-fold validation BCE; ceiling of median best epoch",
        "calibration": "temperature scaling fitted on selected candidate inner OOF logits only",
        "threshold": 0.5,
        "outer_test_used_for_selection": False,
    }
    config_hash = sha256_json(arm_config)
    metadata_path = arm_dir / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["config_sha256"] != config_hash or metadata["dataset_sha256"] != dataset_hash:
            raise ValueError(f"Cannot resume {arm_name}: run configuration or dataset hash changed")
    else:
        atomic_json(arm_dir / "config.json", arm_config)
        atomic_json(arm_dir / "split_manifest.json", {"outer_splits_by_seed": manifest, "inner_split_rule": "StratifiedKFold(4, shuffle=True, random_state=seed + 1009 * outer_fold)"})
        atomic_json(metadata_path, {"arm": arm_name, "created_utc": datetime.now(timezone.utc).isoformat(), "dataset_sha256": dataset_hash, "config_sha256": config_hash, "status": "RUNNING", "device": str(device)})

    fold_records: list[dict[str, Any]] = []
    seed_frames: dict[int, pd.DataFrame] = {}
    for seed in SEEDS:
        seed_dir = arm_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        fold_frames: list[pd.DataFrame] = []
        target = dataframe[config["data"]["target"]].to_numpy(dtype=int)
        for split in manifest[str(seed)]:
            outer_fold = int(split["fold"])
            train_indices = np.asarray(split["train_indices"], dtype=np.int64)
            valid_indices = np.asarray(split["valid_indices"], dtype=np.int64)
            fold_dir = seed_dir / f"fold_{outer_fold}"
            prediction_path = fold_dir / "outer_oof.csv"
            report_path = fold_dir / "fold.json"
            checkpoint_path = fold_dir / "checkpoint.pt"
            if prediction_path.exists() and report_path.exists() and checkpoint_path.exists():
                saved = pd.read_csv(prediction_path)
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                expected_ids = dataframe.iloc[valid_indices]["sample_index"].to_numpy(dtype=int)
                expected_targets = dataframe.iloc[valid_indices][config["data"]["target"]].to_numpy(dtype=int)
                if (
                    not np.array_equal(saved["sample_index"].to_numpy(dtype=int), expected_ids)
                    or not np.array_equal(saved["target"].to_numpy(dtype=int), expected_targets)
                    or not np.array_equal(saved["seed"].to_numpy(dtype=int), np.full(len(expected_ids), seed))
                    or not np.array_equal(saved["fold"].to_numpy(dtype=int), np.full(len(expected_ids), outer_fold))
                    or checkpoint.get("config_sha256") != config_hash
                    or checkpoint.get("dataset_sha256") != dataset_hash
                ):
                    raise ValueError(f"Saved checkpoint or OOF identity does not match seed {seed}, fold {outer_fold}")
                fold_frames.append(saved)
                fold_records.append(fold_summary_record(arm_name, json.loads(report_path.read_text(encoding="utf-8"))))
                continue

            print(f"{arm_name}: seed {seed}, outer fold {outer_fold}/5", flush=True)
            candidate, selected_epochs, temperature, threshold_selection, selection_details, inner_oof = select_inner_candidate(
                dataframe, train_indices, seed, outer_fold, config, directional, device, recorder, arm_name,
            )
            atomic_csv(inner_oof, fold_dir / "inner_validation_oof.csv")
            model, fold_data = refit_outer(
                dataframe, train_indices, config, candidate, directional, device,
                seed, outer_fold, selected_epochs, recorder, arm_name,
            )
            valid_dataset = make_fold_data(dataframe, train_indices, valid_indices, config)
            valid_tensors = cache_loader_tensors(valid_dataset.valid_loader, device)
            valid_output = infer(model, valid_tensors, int(config["training"]["batch_size"]))
            raw_probabilities = 1.0 / (1.0 + np.exp(-valid_output["z_total"]))
            calibrated_probabilities = 1.0 / (1.0 + np.exp(-valid_output["z_total"] / temperature))
            selected_threshold = float(threshold_selection["threshold"])
            predicted_labels = (calibrated_probabilities >= selected_threshold).astype(int)
            sample_indices = dataframe.iloc[valid_indices]["sample_index"].to_numpy(dtype=int)
            prediction = pd.DataFrame({
                "sample_index": sample_indices,
                "seed": seed,
                "fold": outer_fold,
                "target": valid_output["target"].astype(int),
                "logit": valid_output["z_total"],
                "z_mech": valid_output["z_mech"],
                "z_inter": valid_output["z_inter"],
                "z_res": valid_output["z_res"],
                "m_e": valid_output["m_e"],
                "probability_raw": raw_probabilities,
                "probability_calibrated": calibrated_probabilities,
                "probability": calibrated_probabilities,
                "predicted_label": predicted_labels,
                "temperature_inner_only": temperature,
                "selected_threshold_inner_only": selected_threshold,
            })
            raw_metrics = metric_values(prediction["target"].to_numpy(), raw_probabilities)
            calibrated_metrics = metric_values(
                prediction["target"].to_numpy(),
                calibrated_probabilities,
                threshold=selected_threshold,
            )
            report = {
                "seed": seed,
                "outer_fold": outer_fold,
                "train_rows": len(train_indices),
                "outer_test_rows": len(valid_indices),
                "selected_candidate": candidate,
                "selected_epochs": selected_epochs,
                "calibration_temperature_inner_only": temperature,
                "threshold_selection_inner_only": threshold_selection,
                "candidate_inner_results": selection_details,
                "raw_metrics": raw_metrics,
                "calibrated_metrics": calibrated_metrics,
                "outer_test_used_for_selection": False,
            }
            fold_dir.mkdir(parents=True, exist_ok=True)
            atomic_csv(prediction, prediction_path)
            atomic_json(report_path, report)
            atomic_torch_save({
                "state_dict": model.state_dict(),
                "seed": seed,
                "outer_fold": outer_fold,
                "selected_candidate": candidate,
                "selected_epochs": selected_epochs,
                "calibration_temperature_inner_only": temperature,
                "threshold_selection_inner_only": threshold_selection,
                "directional_asymmetry": directional,
                "scaler_mean": fold_data.scaler.mean_.tolist(),
                "scaler_scale": fold_data.scaler.scale_.tolist(),
                "config_sha256": config_hash,
                "dataset_sha256": dataset_hash,
            }, checkpoint_path)
            fold_frames.append(prediction)
            fold_records.append(fold_summary_record(arm_name, report))
            atomic_csv(pd.DataFrame(fold_records), arm_dir / "fold_metrics.csv")
            del model, fold_data, valid_dataset, valid_tensors, valid_output
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        seed_frames[seed] = pd.concat(fold_frames, ignore_index=True).sort_values("sample_index").reset_index(drop=True)
        atomic_csv(seed_frames[seed], arm_dir / f"seed_{seed}_oof.csv")

    if not fold_records:
        for seed in SEEDS:
            for split in manifest[str(seed)]:
                report_path = arm_dir / f"seed_{seed}" / f"fold_{split['fold']}" / "fold.json"
                report = json.loads(report_path.read_text(encoding="utf-8"))
                fold_records.append(fold_summary_record(arm_name, report))
    write_arm_outputs(arm_dir, fold_records, seed_frames, dataframe)
    atomic_json(metadata_path, {"arm": arm_name, "created_utc": json.loads(metadata_path.read_text(encoding="utf-8"))["created_utc"], "completed_utc": datetime.now(timezone.utc).isoformat(), "dataset_sha256": dataset_hash, "config_sha256": config_hash, "status": "COMPLETE", "device": str(device)})
    return (
        pd.read_csv(arm_dir / "seed_metrics.csv"),
        pd.read_csv(arm_dir / "mean_std_summary.csv"),
        seed_frames,
    )


def baseline_predictions(dataframe: pd.DataFrame) -> dict[int, pd.DataFrame]:
    reference_dir = ROOT / "results" / "stage5_final_validation" / "seed_oof_predictions"
    expected_indices = dataframe["sample_index"].to_numpy(dtype=int)
    expected_targets = dataframe["target"].to_numpy(dtype=int)
    predictions: dict[int, pd.DataFrame] = {}
    for seed in SEEDS:
        frame = pd.read_csv(reference_dir / f"seed_{seed}_oof.csv").sort_values("sample_index").reset_index(drop=True)
        if (
            not np.array_equal(frame["sample_index"].to_numpy(dtype=int), expected_indices)
            or not np.array_equal(frame["target"].to_numpy(dtype=int), expected_targets)
        ):
            raise ValueError(f"Baseline OOF predictions do not match the run dataset for seed {seed}")
        frame.insert(1, "seed", seed)
        predictions[seed] = frame
    return predictions


def write_paired_statistics(
    root_dir: Path,
    arm_predictions: dict[str, dict[int, pd.DataFrame]],
    baseline_frames: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    metrics_by_arm = {
        arm: {seed: summarize_seed_oof(frame) for seed, frame in frames.items()}
        for arm, frames in arm_predictions.items()
    }
    metrics_by_arm["authoritative_baseline_oof"] = {
        seed: summarize_seed_oof(frame) for seed, frame in baseline_frames.items()
    }
    pairs = (
        ("directional_coupling", "authoritative_baseline_oof"),
        ("legacy_coupling_control", "authoritative_baseline_oof"),
        ("directional_coupling", "legacy_coupling_control"),
    )
    rows = []
    for model_a, model_b in pairs:
        for metric in METRICS:
            values_a = np.asarray([metrics_by_arm[model_a][seed][metric] for seed in SEEDS], dtype=float)
            values_b = np.asarray([metrics_by_arm[model_b][seed][metric] for seed in SEEDS], dtype=float)
            differences = values_a - values_b
            if np.allclose(differences, 0.0, rtol=0.0, atol=1e-12):
                statistic, p_value = 0.0, 1.0
            else:
                test = wilcoxon(values_a, values_b, alternative="two-sided", method="auto")
                statistic, p_value = float(test.statistic), float(test.pvalue)
            rows.append({
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric,
                "paired_seed_count": len(SEEDS),
                "mean_paired_difference_a_minus_b": float(differences.mean()),
                "median_paired_difference_a_minus_b": float(np.median(differences)),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_value": p_value,
                "test": "two-sided exact Wilcoxon signed-rank on five paired seed-level metrics",
            })
    result = pd.DataFrame(rows)
    atomic_csv(result, root_dir / "statistical_comparison.csv")
    return result


def write_diagnostic_plots(
    root_dir: Path,
    arm_predictions: dict[str, dict[int, pd.DataFrame]],
    baseline_frames: dict[int, pd.DataFrame],
) -> None:
    diagnostic_root = root_dir / "diagnostics"
    plot_sets: list[tuple[str, dict[str, pd.DataFrame]]] = []
    for seed in SEEDS:
        plot_sets.append((
            f"seed_{seed}",
            {
                "authoritative_baseline": baseline_frames[seed],
                **{arm: frames[seed] for arm, frames in arm_predictions.items()},
            },
        ))
    plot_sets.append((
        "all_seed_oof",
        {
            "authoritative_baseline": pd.concat(list(baseline_frames.values()), ignore_index=True),
            **{
                arm: pd.concat(list(frames.values()), ignore_index=True)
                for arm, frames in arm_predictions.items()
            },
        },
    ))
    for label, named_frames in plot_sets:
        baseline_frame = named_frames["authoritative_baseline"]
        labels = baseline_frame["target"].to_numpy(dtype=int)
        if any(not np.array_equal(frame["target"].to_numpy(dtype=int), labels) for frame in named_frames.values()):
            raise ValueError(f"Prediction targets are not paired for {label}")
        plot_dir = diagnostic_root / label
        curve_predictions = {
            name: (
                frame["target"].to_numpy(dtype=int),
                frame["probability"].to_numpy(dtype=float)
                if "probability" in frame
                else frame["probability_calibrated"].to_numpy(dtype=float),
            )
            for name, frame in named_frames.items()
        }
        plot_roc_curves(curve_predictions, plot_dir, "roc_curves")
        plot_precision_recall_curves(curve_predictions, plot_dir, "pr_curves")
        for name, frame in named_frames.items():
            model_dir = plot_dir / name
            probabilities = (
                frame["probability"].to_numpy(dtype=float)
                if "probability" in frame
                else frame["probability_calibrated"].to_numpy(dtype=float)
            )
            predicted = (
                frame["predicted_label"].to_numpy(dtype=int)
                if "predicted_label" in frame
                else (probabilities >= 0.5).astype(int)
            )
            plot_calibration_curve(labels, probabilities, model_dir, name="calibration_curve")
            matrix = confusion_matrix(labels, predicted, labels=[0, 1])
            normalized = matrix.astype(float) / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
            plot_confusion_matrix(matrix, model_dir, "confusion_matrix")
            plot_confusion_matrix(normalized, model_dir, "confusion_matrix_normalized", normalized=True)
            count_frame = pd.DataFrame(matrix, index=["actual_0", "actual_1"], columns=["predicted_0", "predicted_1"])
            normalized_frame = pd.DataFrame(normalized, index=["actual_0", "actual_1"], columns=["predicted_0", "predicted_1"])
            atomic_csv(count_frame.rename_axis("actual").reset_index(), model_dir / "confusion_matrix.csv")
            atomic_csv(normalized_frame.rename_axis("actual").reset_index(), model_dir / "confusion_matrix_normalized.csv")


def write_comparison(
    root_dir: Path,
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]],
    baseline_frames: dict[int, pd.DataFrame],
    run_config: dict[str, Any],
    elapsed_seconds: float,
) -> None:
    archived_baseline = {
        "accuracy": (0.8150, 0.0085),
        "sensitivity": (0.7952, 0.0122),
        "specificity": (0.8335, 0.0122),
        "f1": (0.8065, 0.0090),
        "mcc": (0.6296, 0.0171),
        "roc_auc": (0.8966, 0.0050),
        "average_precision": (0.8923, 0.0058),
        "brier_score": (0.1313, 0.0025),
    }
    baseline_seed_metrics = pd.DataFrame([
        {"seed": seed, **summarize_seed_oof(frame)}
        for seed, frame in baseline_frames.items()
    ])
    atomic_csv(baseline_seed_metrics, root_dir / "authoritative_baseline_seed_metrics.csv")
    baseline_summary = pd.DataFrame([
        {
            "metric": metric,
            "mean": float(baseline_seed_metrics[metric].mean()),
            "std": float(baseline_seed_metrics[metric].std(ddof=1)),
        }
        for metric in METRICS
    ])
    atomic_csv(baseline_summary, root_dir / "authoritative_baseline_mean_std.csv")
    rows = [
        {"arm": "archived_published_baseline_summary", "metric": metric, "mean": mean, "std": std}
        for metric, (mean, std) in archived_baseline.items()
    ]
    for _, row in baseline_summary.iterrows():
        rows.append({
            "arm": "authoritative_baseline_oof_recomputed",
            "metric": row["metric"],
            "mean": row["mean"],
            "std": row["std"],
        })
    for arm, (_, summary, _) in results.items():
        for _, row in summary.iterrows():
            rows.append({"arm": arm, "metric": row["metric"], "mean": row["mean"], "std": row["std"]})
    atomic_csv(pd.DataFrame(rows), root_dir / "comparison_mean_std.csv")
    atomic_csv(pd.DataFrame([
        {
            "seed": seed,
            "metric": metric,
            "directional_coupling": summarize_seed_oof(results["directional_coupling"][2][seed])[metric],
            "legacy_coupling_control": summarize_seed_oof(results["legacy_coupling_control"][2][seed])[metric],
            "directional_minus_legacy": (
                summarize_seed_oof(results["directional_coupling"][2][seed])[metric]
                - summarize_seed_oof(results["legacy_coupling_control"][2][seed])[metric]
            ),
        }
        for seed in SEEDS for metric in METRICS
    ]), root_dir / "ablation_comparison.csv")
    arm_predictions = {arm: result[2] for arm, result in results.items()}
    write_paired_statistics(root_dir, arm_predictions, baseline_frames)
    write_diagnostic_plots(root_dir, arm_predictions, baseline_frames)

    display = ["accuracy", "sensitivity", "specificity", "f1", "mcc", "roc_auc", "average_precision", "brier_score"]
    lines = [
        "# PMRC-Net final nested-CV research evaluation",
        "",
        "## Protocol and reproducibility",
        "",
        f"- Dataset: `{run_config['base_config']['data']['path']}`",
        f"- Dataset SHA-256: `{run_config['dataset_sha256']}`",
        f"- Loaded rows after configured exact-row deduplication: `{len(baseline_frames[SEEDS[0]])}` per seed.",
        f"- Seeds: `{SEEDS}`; five outer folds and four inner folds per outer training partition.",
        f"- Batch size: `{run_config['base_config']['training']['batch_size']}`; CPU threads/inter-op: `{run_config['base_config']['training']['cpu_threads']}/{run_config['base_config']['training']['cpu_interop_threads']}`; workers: `{run_config['base_config']['training']['num_workers']}`; precision: FP32.",
        f"- Hardware: `{run_config['hardware']}`",
        f"- Training and evaluation wall time: `{elapsed_seconds:.1f}` seconds.",
        "- Outer assignments were checked against preserved five-seed OOF references. Candidate and epoch selection, temperature scaling, and threshold selection used inner folds only. Each outer test fold was evaluated once.",
        "",
        "## Five-seed OOF results",
        "",
        "| Arm | Accuracy | Sensitivity | Specificity | F1 | MCC | ROC-AUC | Average precision | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in [
        ("authoritative_baseline_oof_recomputed", baseline_summary),
        *[(arm, summary) for arm, (_, summary, _) in results.items()],
    ]:
        values = {row["metric"]: row for _, row in metrics.iterrows()}
        cells = [f"{values[key]['mean']:.4f} +/- {values[key]['std']:.4f}" for key in display]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "Mean +/- sample standard deviation is across five per-seed pooled OOF metrics. Raw/calibrated fold metrics are in each arm's `fold_metrics.csv`; per-seed metrics and complete OOF predictions are also saved per arm. The historically published baseline summary is labeled separately from metrics recomputed from its split-aligned OOF predictions.",
        "",
        "## Paired statistical comparison",
        "",
        "`statistical_comparison.csv` reports two-sided Wilcoxon signed-rank tests on five paired seed-level metrics for the directional model versus the historical baseline OOF, the legacy-coupling control versus that baseline, and directional versus control. These are exploratory (n=5), low-power comparisons; folds/seeds are not independent patients, no multiplicity correction is applied, and p-values alone do not establish improvement. Consider paired effect sizes and uncertainty.",
        "",
        "## Ablation and limitations",
        "",
        "`ablation_comparison.csv` reports per-seed metric differences between directional coupling and the legacy-coupling control, isolating the configured directionality change within the optimized backbone; it is not a full-architecture comparison against an external model.",
        "",
        "This is internal cross-validation, not external clinical validation. Patient-level independence and cohort provenance are unverified because no patient identifier is present. The baseline comparison uses preserved, split-aligned OOF predictions. Five seeds provide limited statistical power.",
        "",
        "## Artifacts",
        "",
        "Per-fold and per-seed metrics, five-seed mean/std, complete OOF predictions, confusion matrices, ROC/PR/calibration plots, ablation and paired statistical comparisons, resource usage, run configuration and dataset hash, fold reports/checkpoints, and this report are included under this run directory.",
    ])
    (root_dir / "FINAL_RESEARCH_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_preflight_smoke(
    config: dict[str, Any],
    dataframe: pd.DataFrame,
    device: torch.device,
) -> dict[str, Any]:
    seed = 42
    labels = dataframe[config["data"]["target"]].to_numpy(dtype=int)
    outer_train = np.asarray(expected_splits(dataframe, config)[str(seed)][0]["train_indices"], dtype=np.int64)
    inner_train, inner_valid = expected_inner_splits(outer_train, labels, seed, 1)[0]
    fold_data = make_fold_data(dataframe, inner_train, inner_valid, config)
    training_tensors = cache_loader_tensors(fold_data.train_loader, device)
    validation_tensors = cache_loader_tensors(fold_data.valid_loader, device)
    candidate = CANDIDATES[0]
    batch_size = int(config["training"]["batch_size"])
    set_seed(4242)
    model = make_model(config, candidate, True, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate["learning_rate"], weight_decay=candidate["weight_decay"])
    loss_fn = make_loss(dataframe, inner_train, config, candidate)
    output_dir = ROOT / "results" / "PMRC_NET_FINAL_RESEARCH"
    output_dir.mkdir(parents=True, exist_ok=True)
    resources_path = output_dir / "preflight_resources.csv"
    recorder = ResourceRecorder(resources_path)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loss = train_epoch(
        model, training_tensors, batch_size, optimizer, loss_fn, device, recorder,
        {"arm": "preflight", "phase": "single_batch_smoke", "seed": seed, "outer_fold": 1},
        epoch=1, max_batches=1,
    )
    model.eval()
    with torch.inference_mode():
        expected_logits = infer(model, validation_tensors, batch_size)["z_total"]
    with tempfile.TemporaryDirectory(prefix="pmrc_checkpoint_smoke_") as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "smoke_checkpoint.pt"
        atomic_torch_save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint_path)
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        restored = make_model(config, candidate, True, device).eval()
        restored.load_state_dict(saved["state_dict"])
        with torch.inference_mode():
            restored_logits = infer(restored, validation_tensors, batch_size)["z_total"]
        np.testing.assert_allclose(expected_logits, restored_logits, rtol=0.0, atol=1e-7)
        optimizer.load_state_dict(saved["optimizer"])

    peak_allocated_mb = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    total_vram_mb = torch.cuda.get_device_properties(device).total_memory / 2**20 if device.type == "cuda" else 0.0
    process = psutil.Process()
    report = {
        "status": "passed" if math.isfinite(loss) else "failed",
        "hardware": hardware_metadata(device, config),
        "dataset_path": config["data"]["path"],
        "dataset_sha256": sha256_file(resolve_path(ROOT, config["data"]["path"])),
        "batch_size_single_step_loss": loss,
        "checkpoint_save_reload_logits_match": True,
        "batch_size": batch_size,
        "batch_size_peak_allocated_mb": peak_allocated_mb,
        "batch_size_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0,
        "gpu_total_memory_mb": total_vram_mb,
        "mixed_precision_enabled": False,
        "mixed_precision_note": "FP16 remains disabled until full metric consistency is established.",
        "process_rss_mb": process.memory_info().rss / 2**20,
        "system_memory_percent": psutil.virtual_memory().percent,
        "resource_log": str(resources_path.relative_to(ROOT)),
    }
    recorder.flush()
    report_path = output_dir / "preflight.json"
    atomic_json(report_path, report)
    del model, optimizer, fold_data, training_tensors, validation_tensors
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "passed":
        raise RuntimeError(f"Preflight smoke failed; see {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nested, resumable PMRC-Net directional-coupling optimization")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(ROOT / "configs" / "model_config.yaml")
    config["data"]["path"] = "dataset/cleaned_merged_heart_dataset.csv"
    config["training"].update({
        "epochs": 30,
        "patience": 8,
        "min_delta": 1e-4,
        "num_workers": 0,
        "cpu_threads": 2,
        "cpu_interop_threads": 1,
        "batch_size": 16,
        "mixed_precision": False,
    })
    config["loss"].update({"gate_l2_lambda": 1e-4, "residual_l2_lambda": 1e-4})
    device = configure_runtime(config)
    config["hardware"]["pin_memory"] = device.type == "cuda"
    data_path = resolve_path(ROOT, config["data"]["path"])
    dataframe = load_original_dataset(data_path, config["data"]["target"], config["data"]["deduplicate_exact_rows"])
    dataset_hash = sha256_file(data_path)
    manifest = expected_splits(dataframe, config)
    validate_reference_splits(manifest, dataframe)
    if args.smoke:
        run_preflight_smoke(config, dataframe, device)
        return
    root_dir = ROOT / "results" / "PMRC_NET_FINAL_RESEARCH"
    if root_dir.exists() and not args.resume and any(
        (root_dir / name).exists()
        for name in ("run_config.json", "run_metadata.json", "legacy_coupling_control", "directional_coupling")
    ):
        raise FileExistsError(f"Run already exists; use --resume: {root_dir}")
    root_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "run_directory": str(root_dir.relative_to(ROOT)),
        "dataset_path": config["data"]["path"],
        "base_config": config,
        "seeds": SEEDS,
        "inner_folds": 4,
        "outer_folds": 5,
        "candidates": CANDIDATES,
        "selection_metric": "mean inner-fold ROC-AUC + 0.5 * MCC",
        "threshold_selection": "Youden J selected on calibrated selected-candidate inner OOF probabilities only",
        "dataset_sha256": dataset_hash,
        "hardware": hardware_metadata(device, config),
        "resource_sampling": {
            "per_epoch": [
                "CPU utilization", "process CPU", "process RSS", "system RAM percent/available",
                "CUDA allocated/reserved/peak", "epoch duration",
            ],
            "nvidia_smi_every_epochs": GPU_TELEMETRY_EVERY_EPOCHS,
            "stop_thresholds": {
                "system_ram_percent": MAX_SYSTEM_RAM_PERCENT,
                "gpu_vram_percent": MAX_GPU_MEMORY_PERCENT,
            },
        },
    }
    run_config_hash = sha256_json(run_config)
    run_config_path = root_dir / "run_config.json"
    run_metadata_path = root_dir / "run_metadata.json"
    if run_config_path.exists():
        existing = json.loads(run_config_path.read_text(encoding="utf-8"))
        if sha256_json(existing) != run_config_hash:
            raise ValueError("Run configuration changed; use a new run ID rather than mixing artifacts")
    elif args.resume:
        raise FileNotFoundError(f"Cannot resume without saved run configuration: {run_config_path}")
    else:
        atomic_json(run_config_path, run_config)
        atomic_json(run_metadata_path, {
            "run_id": root_dir.name,
            "status": "RUNNING",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": run_config_hash,
            "dataset_sha256": dataset_hash,
            "hardware": run_config["hardware"],
        })
    if args.resume:
        metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        metadata["status"] = "RUNNING"
        metadata["resumed_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(run_metadata_path, metadata)
    recorder = ResourceRecorder(root_dir / "resource_metrics.csv")
    results = {}
    status = "COMPLETE"
    started = time.perf_counter()
    try:
        baseline_frames = baseline_predictions(dataframe)
        for arm_name, directional in (("legacy_coupling_control", False), ("directional_coupling", True)):
            seed_metrics, summary, seed_frames = run_arm(
                arm_name, directional, dataframe, config, manifest, device,
                root_dir, args.resume, dataset_hash, recorder,
            )
            results[arm_name] = (seed_metrics, summary, seed_frames)
        initial_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        total_wall_seconds = (
            datetime.now(timezone.utc)
            - datetime.fromisoformat(initial_metadata["created_utc"])
        ).total_seconds()
        write_comparison(
            root_dir, results, baseline_frames, run_config, total_wall_seconds,
        )
        print(f"Completed nested comparison: {root_dir}", flush=True)
    except KeyboardInterrupt:
        status = "STOPPED_BY_USER"
        print("Nested CV stopped cleanly; completed fold checkpoints are preserved.", flush=True)
    except torch.cuda.OutOfMemoryError:
        status = "STOPPED_CUDA_OOM"
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise
    except Exception as error:
        status = "FAILED"
        atomic_json(root_dir / "failure.json", {
            "error_type": type(error).__name__,
            "error": str(error),
            "failed_utc": datetime.now(timezone.utc).isoformat(),
        })
        raise
    finally:
        recorder.flush()
        metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        updated_utc = datetime.now(timezone.utc)
        total_wall_seconds = (
            updated_utc - datetime.fromisoformat(metadata["created_utc"])
        ).total_seconds()
        atomic_json(root_dir / "resource_usage_summary.json", {
            "status": status,
            "hardware": run_config["hardware"],
            "elapsed_seconds": total_wall_seconds,
            "current_attempt_elapsed_seconds": time.perf_counter() - started,
            **recorder.summary(),
        })
        metadata["status"] = status
        metadata["updated_utc"] = updated_utc.isoformat()
        metadata["elapsed_seconds"] = total_wall_seconds
        metadata["current_attempt_elapsed_seconds"] = time.perf_counter() - started
        atomic_json(run_metadata_path, metadata)
        failure_path = root_dir / "failure.json"
        if status == "COMPLETE" and failure_path.exists():
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            failure["resolved_by_resume"] = True
            failure["resolved_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(failure_path, failure)


if __name__ == "__main__":
    main()