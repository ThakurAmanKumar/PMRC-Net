from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedKFold

V11_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = V11_ROOT.parent
sys.path.insert(0, str(V11_ROOT))

from data.dataset import FoldData, load_dataset, make_fold_data  # noqa: E402
from evaluation.metrics import calculate_metrics  # noqa: E402
from models.pmrc_net import PMRCNet  # noqa: E402
from training.losses import PMRCObjective  # noqa: E402


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
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def configure_runtime(config: dict[str, Any]) -> torch.device:
    torch.set_num_threads(int(config["training"]["cpu_threads"]))
    try:
        torch.set_num_interop_threads(int(config["training"]["cpu_threads"]))
    except RuntimeError:
        pass
    requested = config["hardware"].get("device", "auto")
    if requested == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def training_step(
    model: PMRCNet,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    objective: PMRCObjective,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    count = 0
    for batch in loader:
        tensors = [item.to(device) for item in batch]
        optimizer.zero_grad(set_to_none=True)
        outputs = model(*tensors[:3])
        loss = objective(outputs["z_total"], tensors[3])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("Non-finite gradient norm")
        optimizer.step()
        batch_count = tensors[3].shape[0]
        total_loss += float(loss.detach()) * batch_count
        count += batch_count
    return total_loss / count


def validation_loss(
    model: PMRCNet,
    loader: torch.utils.data.DataLoader,
    objective: PMRCObjective,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.inference_mode():
        for batch in loader:
            tensors = [item.to(device) for item in batch]
            outputs = model(*tensors[:3])
            loss = objective(outputs["z_total"], tensors[3])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite inner-validation loss")
            batch_count = tensors[3].shape[0]
            total_loss += float(loss) * batch_count
            count += batch_count
    return total_loss / count


def make_inner_splits(
    outer_train_indices: np.ndarray,
    target: np.ndarray,
    config: dict[str, Any],
    seed: int,
    outer_fold: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    outer_train_indices = np.asarray(outer_train_indices, dtype=np.int64)
    outer_target = np.asarray(target, dtype=int)[outer_train_indices]
    splitter = StratifiedKFold(
        n_splits=int(config["training"]["inner_folds"]),
        shuffle=True,
        random_state=seed + 1009 * outer_fold,
    )
    return [
        (outer_train_indices[train], outer_train_indices[valid])
        for train, valid in splitter.split(outer_train_indices, outer_target)
    ]


def select_epoch_count(
    dataframe: pd.DataFrame,
    outer_train_indices: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    outer_fold: int,
) -> dict[str, Any]:
    target = dataframe[config["data"]["target"]].to_numpy(dtype=int)
    best_epochs: list[int] = []
    inner_records: list[dict[str, Any]] = []
    inner_splits = make_inner_splits(outer_train_indices, target, config, seed, outer_fold)
    for inner_fold, (inner_train, inner_valid) in enumerate(inner_splits, start=1):
        inner_seed = seed * 100000 + outer_fold * 1000 + inner_fold
        set_seed(inner_seed)
        fold_data = make_fold_data(dataframe, inner_train, inner_valid, config, inner_seed)
        model = PMRCNet(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        objective = PMRCObjective()
        best_loss = float("inf")
        best_epoch = 0
        patience = 0
        for epoch in range(1, int(config["training"]["max_epochs"]) + 1):
            training_step(
                model, fold_data.train_loader, optimizer, objective, device,
                float(config["training"]["gradient_clip_norm"]),
            )
            current_loss = validation_loss(model, fold_data.valid_loader, objective, device)
            if current_loss < best_loss - float(config["training"]["min_delta"]):
                best_loss = current_loss
                best_epoch = epoch
                patience = 0
            else:
                patience += 1
                if patience >= int(config["training"]["patience"]):
                    break
        best_epoch = max(best_epoch, 1)
        best_epochs.append(best_epoch)
        inner_records.append({
            "inner_fold": inner_fold,
            "train_rows": len(inner_train),
            "valid_rows": len(inner_valid),
            "best_epoch": best_epoch,
            "best_validation_bce": best_loss,
        })
        del model, optimizer, objective
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selected = max(1, int(math.ceil(float(np.median(best_epochs)))))
    return {
        "selection_metric": "inner_validation_binary_cross_entropy",
        "aggregation": "ceiling of median inner-fold best epoch",
        "inner_best_epochs": best_epochs,
        "selected_epochs": selected,
        "inner_folds": inner_records,
    }


def refit_outer_training_partition(
    dataframe: pd.DataFrame,
    outer_train_indices: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    outer_fold: int,
    selected_epochs: int,
) -> tuple[PMRCNet, FoldData, float]:
    refit_seed = seed * 100000 + outer_fold * 1000 + 999
    set_seed(refit_seed)
    fold_data = make_fold_data(
        dataframe, outer_train_indices, np.empty(0, dtype=np.int64), config, refit_seed
    )
    model = PMRCNet(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    objective = PMRCObjective()
    last_train_loss = float("nan")
    for _ in range(selected_epochs):
        last_train_loss = training_step(
            model, fold_data.train_loader, optimizer, objective, device,
            float(config["training"]["gradient_clip_norm"]),
        )
    del optimizer, objective
    return model, fold_data, last_train_loss


def expected_outer_splits(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
    seeds: list[int],
) -> dict[str, list[dict[str, Any]]]:
    target = dataframe[config["data"]["target"]].to_numpy(dtype=int)
    manifest: dict[str, list[dict[str, Any]]] = {}
    for seed in seeds:
        splitter = StratifiedKFold(
            n_splits=int(config["training"]["outer_folds"]),
            shuffle=True,
            random_state=int(seed),
        )
        manifest[str(seed)] = [
            {
                "fold": fold_number,
                "train_indices": train.astype(int).tolist(),
                "valid_indices": valid.astype(int).tolist(),
            }
            for fold_number, (train, valid) in enumerate(splitter.split(dataframe, target), start=1)
        ]
    return manifest


def validate_oof_frame(
    frame: pd.DataFrame,
    dataframe: pd.DataFrame,
    expected_fold_ids: np.ndarray,
    seed: int,
) -> None:
    required = {"sample_index", "seed", "fold", "target", "probability"}
    if not required.issubset(frame.columns):
        raise ValueError(f"OOF predictions are missing columns: {sorted(required - set(frame.columns))}")
    ordered = frame.sort_values("sample_index").reset_index(drop=True)
    if not np.array_equal(ordered["seed"].to_numpy(dtype=int), np.full(len(ordered), seed, dtype=int)):
        raise ValueError(f"OOF seed labels are misaligned for seed {seed}")
    expected = dataframe[["sample_index", "target"]].sort_values("sample_index").reset_index(drop=True)
    if len(ordered) != len(expected) or ordered["sample_index"].duplicated().any():
        raise ValueError(f"OOF predictions do not cover every sample exactly once for seed {seed}")
    if not np.array_equal(ordered["sample_index"].to_numpy(dtype=int), expected["sample_index"].to_numpy(dtype=int)):
        raise ValueError(f"OOF sample IDs are misaligned for seed {seed}")
    if not np.array_equal(ordered["target"].to_numpy(dtype=int), expected["target"].to_numpy(dtype=int)):
        raise ValueError(f"OOF target labels are misaligned for seed {seed}")
    if not np.array_equal(ordered["fold"].to_numpy(dtype=int), expected_fold_ids):
        raise ValueError(f"OOF fold assignments are misaligned for seed {seed}")
    probabilities = ordered["probability"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"OOF probabilities are invalid for seed {seed}")


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_run_identity(metadata: dict[str, Any], config_hash: str, dataset_hash: str) -> None:
    if metadata["config_sha256"] != config_hash:
        raise ValueError("Cannot resume: configuration hash differs from the original run")
    if metadata["dataset_sha256"] != dataset_hash:
        raise ValueError("Cannot resume: dataset hash differs from the original run")


def run_nested_cv(run_id: str, resume: bool = False) -> Path:
    config_path = V11_ROOT / "configs" / "nested_cv.yaml"
    config = load_config(config_path)
    if config["training"]["mixed_precision"] or config["training"]["precision"].lower() != "fp32":
        raise ValueError("v1.1 currently requires FP32; mixed precision is not implemented")
    dataset_path = (V11_ROOT / config["data"]["path"]).resolve()
    dataframe = load_dataset(dataset_path, config)
    dataset_hash = sha256_file(dataset_path)
    config_hash = sha256_json(config)
    seeds = [int(seed) for seed in config["seeds"]]
    run_directory = V11_ROOT / "experiments" / run_id
    checkpoint_directory = V11_ROOT / "checkpoints" / run_id
    prediction_directory = V11_ROOT / "predictions" / run_id
    metadata_path = run_directory / "run_metadata.json"
    if run_directory.exists():
        if not resume:
            raise FileExistsError(f"Run already exists; choose a new run ID or pass --resume: {run_directory}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        validate_run_identity(metadata, config_hash, dataset_hash)
    else:
        if resume:
            raise FileNotFoundError(f"Cannot resume a run that does not exist: {run_directory}")
        run_directory.mkdir(parents=True)
        metadata = {
            "run_id": run_id,
            "status": "RUNNING",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_hash,
            "config_path": "v1.1/configs/nested_cv.yaml",
            "dataset_sha256": dataset_hash,
            "dataset_path": str(dataset_path),
            "rows": len(dataframe),
            "seeds": seeds,
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_build": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        }
        atomic_json(metadata_path, metadata)
        atomic_json(run_directory / "config.json", config)
        manifest = {
            "dataset_sha256": dataset_hash,
            "rows": len(dataframe),
            "outer_folds": int(config["training"]["outer_folds"]),
            "inner_folds": int(config["training"]["inner_folds"]),
            "split_strategy": "StratifiedKFold(shuffle=True, random_state=seed)",
            "outer_splits_by_seed": expected_outer_splits(dataframe, config, seeds),
        }
        atomic_json(run_directory / "split_manifest.json", manifest)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    prediction_directory.mkdir(parents=True, exist_ok=True)
    device = configure_runtime(config)
    if device.type == "cuda":
        print(f"Device: {device} ({torch.cuda.get_device_name(device)})", flush=True)
    outer_manifest = json.loads((run_directory / "split_manifest.json").read_text(encoding="utf-8"))
    fold_records: list[dict[str, Any]] = []
    for seed in seeds:
        for fold_record in outer_manifest["outer_splits_by_seed"][str(seed)]:
            outer_fold = int(fold_record["fold"])
            train_indices = np.asarray(fold_record["train_indices"], dtype=np.int64)
            valid_indices = np.asarray(fold_record["valid_indices"], dtype=np.int64)
            checkpoint_path = checkpoint_directory / f"seed_{seed}" / f"fold_{outer_fold}.pt"
            prediction_path = prediction_directory / f"seed_{seed}" / f"fold_{outer_fold}_oof.csv"
            fold_report_path = run_directory / f"seed_{seed}" / f"fold_{outer_fold}.json"
            if checkpoint_path.exists() and prediction_path.exists() and fold_report_path.exists():
                saved = pd.read_csv(prediction_path)
                expected_ids = dataframe.iloc[valid_indices]["sample_index"].to_numpy(dtype=int)
                if not np.array_equal(saved["sample_index"].to_numpy(dtype=int), expected_ids):
                    raise ValueError(f"Saved prediction alignment mismatch for seed {seed}, fold {outer_fold}")
                fold_records.append(json.loads(fold_report_path.read_text(encoding="utf-8")))
                continue
            print(f"v1.1 seed {seed}: outer fold {outer_fold}/{config['training']['outer_folds']}", flush=True)
            started = time.perf_counter()
            selection = select_epoch_count(
                dataframe, train_indices, config, device, seed, outer_fold
            )
            model, outer_data, final_train_loss = refit_outer_training_partition(
                dataframe, train_indices, config, device, seed, outer_fold,
                int(selection["selected_epochs"]),
            )
            raw = dataframe.iloc[valid_indices]
            continuous = config["data"]["continuous_cols"]
            categorical = config["data"]["categorical_cols"]
            physiology = config["data"]["physiology_cols"]
            cont = outer_data.scaler.transform(raw[continuous].to_numpy(dtype=np.float64)).astype(np.float32)
            cat = raw[categorical].to_numpy(dtype=np.int64)
            phys = raw[physiology].to_numpy(dtype=np.float32)
            valid_y = raw[config["data"]["target"]].to_numpy(dtype=np.int64)
            sample_ids = raw["sample_index"].to_numpy(dtype=np.int64)
            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(cont), torch.from_numpy(cat), torch.from_numpy(phys),
                torch.from_numpy(valid_y.astype(np.float32)), torch.from_numpy(sample_ids),
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False,
                num_workers=int(config["training"]["num_workers"]),
            )
            probabilities: list[float] = []
            model.eval()
            with torch.inference_mode():
                for batch in loader:
                    tensors = [item.to(device) for item in batch]
                    output = model(*tensors[:3])
                    probabilities.extend(torch.sigmoid(output["z_total"]).squeeze(-1).cpu().tolist())
            fold_predictions = pd.DataFrame({
                "sample_index": sample_ids,
                "seed": seed,
                "fold": outer_fold,
                "target": valid_y,
                "probability": probabilities,
            })
            metrics = calculate_metrics(
                valid_y, np.asarray(probabilities), ece_bins=int(config["statistics"]["ece_bins"])
            )
            checkpoint_payload = {
                "version": "1.1",
                "run_id": run_id,
                "seed": seed,
                "outer_fold": outer_fold,
                "selected_epochs": int(selection["selected_epochs"]),
                "selection": selection,
                "dataset_sha256": dataset_hash,
                "config_sha256": config_hash,
                "train_sample_indices": dataframe.iloc[train_indices]["sample_index"].to_numpy(dtype=int).tolist(),
                "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "scaler_mean": outer_data.scaler.mean_.tolist(),
                "scaler_scale": outer_data.scaler.scale_.tolist(),
            }
            atomic_torch_save(checkpoint_payload, checkpoint_path)
            atomic_csv(fold_predictions, prediction_path)
            fold_result = {
                "seed": seed,
                "fold": outer_fold,
                "train_rows": len(train_indices),
                "outer_test_rows": len(valid_indices),
                "selected_epochs": int(selection["selected_epochs"]),
                "selection": selection,
                "outer_train_loss_at_selected_epoch": final_train_loss,
                "outer_metrics": metrics,
                "checkpoint": str(checkpoint_path.relative_to(V11_ROOT)),
                "prediction_file": str(prediction_path.relative_to(V11_ROOT)),
                "elapsed_seconds": time.perf_counter() - started,
                "outer_test_used_for_selection": False,
            }
            atomic_json(fold_report_path, fold_result)
            fold_records.append(fold_result)
            del model, outer_data, loader, dataset
            if device.type == "cuda":
                torch.cuda.empty_cache()
            atomic_csv(pd.DataFrame(fold_records), run_directory / "fold_metrics.csv")
    predictions_by_seed: list[pd.DataFrame] = []
    for seed in seeds:
        frames = [
            pd.read_csv(prediction_directory / f"seed_{seed}" / f"fold_{fold}_oof.csv")
            for fold in range(1, int(config["training"]["outer_folds"]) + 1)
        ]
        combined = pd.concat(frames, ignore_index=True).sort_values("sample_index").reset_index(drop=True)
        if len(combined) != len(dataframe) or combined["sample_index"].duplicated().any():
            raise ValueError(f"Incomplete or duplicate OOF predictions for seed {seed}")
        expected_fold_ids = np.zeros(len(dataframe), dtype=np.int64)
        for item in outer_manifest["outer_splits_by_seed"][str(seed)]:
            expected_fold_ids[np.asarray(item["valid_indices"], dtype=np.int64)] = int(item["fold"])
        validate_oof_frame(combined, dataframe, expected_fold_ids, seed)
        expected = dataframe[["sample_index", config["data"]["target"]]].rename(
            columns={config["data"]["target"]: "target"}
        ).sort_values("sample_index").reset_index(drop=True)
        atomic_csv(combined, prediction_directory / f"seed_{seed}_oof.csv")
        seed_metrics = calculate_metrics(
            expected["target"].to_numpy(), combined["probability"].to_numpy(),
            ece_bins=int(config["statistics"]["ece_bins"]),
        )
        predictions_by_seed.append(combined)
        atomic_json(run_directory / f"seed_{seed}_metrics.json", seed_metrics)
    all_predictions = pd.concat(predictions_by_seed, ignore_index=True)
    atomic_csv(all_predictions, prediction_directory / "all_seeds_oof.csv")
    metadata["status"] = "COMPLETED"
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["device"] = str(device)
    metadata["folds_completed"] = len(fold_records)
    atomic_json(metadata_path, metadata)
    return run_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nested PMRC-Net v1.1 validation without touching v1.0 artifacts")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("nested_%Y%m%dT%H%M%SZ")
    result = run_nested_cv(run_id, resume=args.resume)
    print(json.dumps({"status": "COMPLETED", "run_directory": str(result)}, indent=2))


if __name__ == "__main__":
    main()