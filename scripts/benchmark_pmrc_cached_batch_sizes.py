from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import run_pmrc_nested_optimization as optimization  # noqa: E402
from src.data.dataset import load_original_dataset, make_fold_data  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.runtime import configure_runtime  # noqa: E402

OUTPUT = ROOT / "experiments" / "batch_size_comparison_20261113"
REPETITIONS = 3
WARMUP_EPOCHS = 1
MEASURED_EPOCHS = 3
MIN_FREE_RAM_GB = 1.5
BASE_SEED = 20261113


def sample_gpu(device: torch.device) -> dict[str, float | None]:
    values: dict[str, float | None] = {
        "gpu_utilization_percent": None,
        "gpu_device_memory_used_mb": None,
        "gpu_temperature_c": None,
    }
    if device.type != "cuda":
        return values
    try:
        result = optimization.subprocess.run(
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
        fields = [field.strip() for field in result.stdout.splitlines()[0].split(",")]
        values["gpu_utilization_percent"] = float(fields[0])
        values["gpu_device_memory_used_mb"] = float(fields[1])
        values["gpu_temperature_c"] = float(fields[2])
    except (OSError, optimization.subprocess.SubprocessError, IndexError, ValueError):
        pass
    return values


def process_ram_mb() -> float:
    process = psutil.Process()
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            continue
    return total / 2**20


def validation_loss(
    model: optimization.OptimizedPMRCNet,
    validation_tensors: tuple[torch.Tensor, ...],
    batch_size: int,
) -> float:
    output = optimization.infer(model, validation_tensors, batch_size)
    logits = output["z_total"]
    labels = output["target"]
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def run_epoch(
    model: optimization.OptimizedPMRCNet,
    train_tensors: tuple[torch.Tensor, ...],
    valid_tensors: tuple[torch.Tensor, ...],
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    record_steps: bool,
) -> dict[str, Any]:
    step_times: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    epoch_started = time.perf_counter()
    train_loss = optimization.train_epoch(
        model,
        train_tensors,
        batch_size,
        optimizer,
        loss_fn,
        device,
        step_times=step_times if record_steps else None,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    epoch_seconds = time.perf_counter() - epoch_started
    valid_loss = validation_loss(model, valid_tensors, batch_size)
    gpu_stats = {
        "gpu_allocated_mb": torch.cuda.memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
        "gpu_reserved_mb": torch.cuda.memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0,
        "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
        "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0,
    }
    return {
        "epoch_seconds": epoch_seconds,
        "training_loss": train_loss,
        "validation_loss": valid_loss,
        "step_times_seconds": step_times,
        "mean_step_seconds": float(np.mean(step_times)) if step_times else None,
        "median_step_seconds": float(np.median(step_times)) if step_times else None,
        "step_count": len(step_times),
        "process_tree_rss_mb": process_ram_mb(),
        "system_available_ram_gb": psutil.virtual_memory().available / 2**30,
        **gpu_stats,
        **sample_gpu(device),
    }


def new_model_and_optimizer(
    config: dict[str, Any],
    candidate: dict[str, float],
    initial_state: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
) -> tuple[optimization.OptimizedPMRCNet, torch.optim.Optimizer]:
    optimization.set_seed(seed)
    model = optimization.make_model(config, candidate, True, device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=candidate["learning_rate"],
        weight_decay=candidate["weight_decay"],
    )
    return model, optimizer


def benchmark_configuration(
    batch_size: int,
    config: dict[str, Any],
    dataframe: pd.DataFrame,
    inner_train: np.ndarray,
    train_tensors: tuple[torch.Tensor, ...],
    valid_tensors: tuple[torch.Tensor, ...],
    loss_fn: torch.nn.Module,
    initial_state: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate = optimization.CANDIDATES[0]
    epochs: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []

    warm_model, warm_optimizer = new_model_and_optimizer(
        config, candidate, initial_state, device, BASE_SEED,
    )
    run_epoch(warm_model, train_tensors, valid_tensors, batch_size, warm_optimizer, loss_fn, device, record_steps=False)
    del warm_model, warm_optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    for repetition in range(1, REPETITIONS + 1):
        model, optimizer = new_model_and_optimizer(
            config, candidate, initial_state, device, BASE_SEED,
        )
        for epoch in range(1, MEASURED_EPOCHS + 1):
            record = run_epoch(
                model, train_tensors, valid_tensors, batch_size,
                optimizer, loss_fn, device, record_steps=True,
            )
            epoch_record = {
                "batch_size": batch_size,
                "repetition": repetition,
                "epoch": epoch,
                **{key: value for key, value in record.items() if key != "step_times_seconds"},
            }
            epochs.append(epoch_record)
            steps.extend({
                "batch_size": batch_size,
                "repetition": repetition,
                "epoch": epoch,
                "step": step,
                "step_seconds": seconds,
            } for step, seconds in enumerate(record["step_times_seconds"], start=1))
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return epochs, steps


def summarize(batch_size: int, epochs: pd.DataFrame, steps: pd.DataFrame) -> dict[str, Any]:
    selected = epochs[epochs["batch_size"] == batch_size]
    selected_steps = steps[steps["batch_size"] == batch_size]
    final_val_by_rep = selected[selected["epoch"] == MEASURED_EPOCHS]["validation_loss"].to_numpy(dtype=float)
    train_loss_by_rep = selected[selected["epoch"] == MEASURED_EPOCHS]["training_loss"].to_numpy(dtype=float)
    val_loss_by_rep = selected[selected["epoch"] == MEASURED_EPOCHS]["validation_loss"].to_numpy(dtype=float)
    return {
        "batch_size": batch_size,
        "mean_epoch_seconds": float(selected["epoch_seconds"].mean()),
        "median_epoch_seconds": float(selected["epoch_seconds"].median()),
        "mean_step_seconds": float(selected_steps["step_seconds"].mean()),
        "median_step_seconds": float(selected_steps["step_seconds"].median()),
        "mean_final_training_loss": float(np.mean(train_loss_by_rep)),
        "mean_final_validation_loss": float(np.mean(val_loss_by_rep)),
        "final_validation_loss_std_across_repetitions": float(np.std(final_val_by_rep, ddof=1)),
        "max_process_tree_rss_mb": float(selected["process_tree_rss_mb"].max()),
        "mean_system_available_ram_gb": float(selected["system_available_ram_gb"].mean()),
        "peak_cuda_allocated_mb": float(selected["gpu_peak_allocated_mb"].max()),
        "peak_cuda_reserved_mb": float(selected["gpu_peak_reserved_mb"].max()),
        "mean_gpu_utilization_percent": float(selected["gpu_utilization_percent"].dropna().mean()) if selected["gpu_utilization_percent"].notna().any() else None,
        "max_gpu_temperature_c": float(selected["gpu_temperature_c"].dropna().max()) if selected["gpu_temperature_c"].notna().any() else None,
        "finite_losses": bool(np.isfinite(selected[["training_loss", "validation_loss"]].to_numpy()).all()),
        "cuda_oom": False,
        "reproducible_validation_trajectory": bool(np.max(np.ptp(selected.pivot(index="epoch", columns="repetition", values="validation_loss").to_numpy(), axis=1)) < 1e-6),
        "repetition_final_validation_losses": final_val_by_rep.tolist(),
    }


def make_report(
    config: dict[str, Any],
    hardware: dict[str, Any],
    dataset_hash: str,
    split_identity: dict[str, Any],
    summaries: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> str:
    by_batch = {int(item["batch_size"]): item for item in summaries}
    batch8 = by_batch.get(8)
    batch16 = by_batch.get(16)
    recommendation = "Retain batch size 8."
    justification = "The benchmark was incomplete or batch 16 failed stability checks."
    if batch8 and batch16 and batch8["finite_losses"] and batch16["finite_losses"] and batch8["reproducible_validation_trajectory"] and batch16["reproducible_validation_trajectory"]:
        speedup = batch8["median_epoch_seconds"] / batch16["median_epoch_seconds"]
        validation_delta = batch16["mean_final_validation_loss"] - batch8["mean_final_validation_loss"]
        enough_headroom = batch16["peak_cuda_reserved_mb"] < 0.75 * hardware["gpu_total_memory_mb"]
        if speedup >= 1.10 and validation_delta <= 0.02 and enough_headroom and not batch16["cuda_oom"]:
            recommendation = "Batch size 16 is a viable speed candidate, but do not change the full experiment without approval and a config-separated paired evaluation."
            justification = f"Median epoch speedup was {speedup:.2f}x, final validation-loss delta was {validation_delta:+.5f}, losses were finite/reproducible, and reserved VRAM remained below 75% of total."
        else:
            justification = f"Batch 16 did not meet all guards: median speedup {speedup:.2f}x (need >=1.10x), final validation-loss delta {validation_delta:+.5f} (allow <=0.02), headroom={enough_headroom}."

    lines = [
        "# PMRC-Net Actual-Path Batch-Size Comparison",
        "",
        f"Dataset: `{config['data']['path']}`; SHA-256 `{dataset_hash}`.",
        f"Hardware: {hardware['gpu_name']}; dedicated VRAM {hardware['gpu_total_memory_mb']:.0f} MiB; system RAM {hardware['system_ram_total_gb']:.2f} GiB; PyTorch {hardware['torch_version']}; CUDA {hardware['cuda_version']}.",
        "",
        f"Data split: {split_identity['description']}. No outer-test rows were loaded into training, validation, timing selection, or batch-size recommendation. Preprocessing was fitted on inner-training rows only.",
        "",
        f"Conditions: identical initial weights, loss, optimizer, learning rate, candidate hyperparameters, seed ({BASE_SEED}), inner train/validation tensors, and FP32 precision. One warm-up epoch per batch size, then {REPETITIONS} fresh repetitions of {MEASURED_EPOCHS} epochs each. CPU threads=2, inter-op=1, workers=0, pinned host staging followed by one-time CUDA caching.",
        "",
        "Training loss is the PMRC composite training objective; validation loss is binary cross-entropy on the inner-validation partition.",
        "",
        "| Batch | Mean epoch (s) | Median epoch (s) | Mean step (s) | Median step (s) | Final train loss | Final validation loss | Max RAM (MiB) | Peak CUDA allocated (MiB) | Peak CUDA reserved (MiB) | Mean GPU util | Max temp (C) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['batch_size']} | {item['mean_epoch_seconds']:.4f} | {item['median_epoch_seconds']:.4f} | "
            f"{item['mean_step_seconds']:.5f} | {item['median_step_seconds']:.5f} | "
            f"{item['mean_final_training_loss']:.5f} | {item['mean_final_validation_loss']:.5f} | "
            f"{item['max_process_tree_rss_mb']:.1f} | {item['peak_cuda_allocated_mb']:.2f} | "
            f"{item['peak_cuda_reserved_mb']:.2f} | {item['mean_gpu_utilization_percent']} | {item['max_gpu_temperature_c']} |"
        )
    lines.extend([
        "",
        f"Recommendation: {recommendation}",
        justification,
        "",
        "This is a resource/training-path benchmark, not nested-CV performance evaluation. Batch-size choice has not been validated on outer folds; no model-performance improvement is claimed. The full five-seed nested CV was not started or modified.",
        "",
        f"System RAM available before the benchmark: {hardware['system_ram_available_initial_gb']:.2f} GiB. See `resource_telemetry.csv` for per-epoch RAM and GPU samples.",
    ])
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append(f"- Batch {failure['batch_size']}: {failure['error_type']}: {failure['error']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark batch size 8 vs 16 using PMRC-Net's cached-CUDA training path")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark directory: {output}")
    output.mkdir(parents=True)

    config = load_config(ROOT / "configs" / "model_config.yaml")
    config["data"]["path"] = "dataset/cleaned_merged_heart_dataset(4).csv"
    config["training"].update({
        "cpu_threads": 2,
        "cpu_interop_threads": 1,
        "num_workers": 0,
        "batch_size": 8,
        "mixed_precision": False,
    })
    device = configure_runtime(config)
    if device.type != "cuda":
        raise RuntimeError("This actual-path benchmark requires CUDA; no CPU fallback will be silently compared.")
    config["hardware"]["pin_memory"] = True

    dataset_path = resolve_path(ROOT, config["data"]["path"])
    dataframe = load_original_dataset(dataset_path, config["data"]["target"], config["data"]["deduplicate_exact_rows"])
    manifest = optimization.expected_splits(dataframe, config)
    optimization.validate_reference_splits(manifest, dataframe)
    seed, outer_fold = 42, 1
    outer_train = np.asarray(manifest[str(seed)][outer_fold - 1]["train_indices"], dtype=np.int64)
    labels = dataframe[config["data"]["target"]].to_numpy(dtype=int)
    inner_train, inner_valid = optimization.expected_inner_splits(outer_train, labels, seed, outer_fold)[0]
    if np.intersect1d(inner_train, inner_valid).size or np.setdiff1d(np.concatenate([inner_train, inner_valid]), outer_train).size:
        raise ValueError("Benchmark inner split is not isolated within the outer-training partition")

    available_ram_gb = psutil.virtual_memory().available / 2**30
    if available_ram_gb < MIN_FREE_RAM_GB:
        raise MemoryError(f"Only {available_ram_gb:.2f} GiB system RAM is available; need at least {MIN_FREE_RAM_GB:.2f} GiB free")

    fold_data = make_fold_data(dataframe, inner_train, inner_valid, config)
    train_tensors = optimization.cache_loader_tensors(fold_data.train_loader, device)
    validation_tensors = optimization.cache_loader_tensors(fold_data.valid_loader, device)
    candidate = optimization.CANDIDATES[0]
    loss_fn = optimization.make_loss(dataframe, inner_train, config, candidate)
    optimization.set_seed(BASE_SEED)
    reference_model = optimization.make_model(config, candidate, True, device)
    initial_state = {key: value.detach().cpu().clone() for key, value in reference_model.state_dict().items()}
    del reference_model
    torch.cuda.empty_cache()

    hardware = optimization.hardware_metadata(device, config)
    hardware["cpu_model"] = platform.processor() or platform.machine()
    hardware["system_ram_total_gb"] = psutil.virtual_memory().total / 2**30
    hardware["system_ram_available_initial_gb"] = available_ram_gb
    hardware["runtime_configuration"] = {
        "intra_op_threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
        "num_workers": 0,
        "pin_memory": True,
        "device": str(device),
        "precision": "FP32",
    }
    dataset_hash = optimization.sha256_file(dataset_path)
    split_identity = {
        "seed": seed,
        "outer_fold": outer_fold,
        "inner_fold": 1,
        "description": f"seed {seed}, outer fold {outer_fold} training partition; inner fold 1 validation from the configured deterministic four-fold splitter",
        "inner_train_rows": int(len(inner_train)),
        "inner_validation_rows": int(len(inner_valid)),
        "outer_test_rows_used": 0,
        "outer_test_indices_loaded": False,
    }
    benchmark_config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": config["data"]["path"],
        "dataset_sha256": dataset_hash,
        "hardware": hardware,
        "model_variant": "OptimizedPMRCNet with directional_coupling=True; architecture unchanged",
        "candidate": candidate,
        "split_identity": split_identity,
        "conditions": {"batch_sizes": [8, 16], "repetitions": REPETITIONS, "warmup_epochs": WARMUP_EPOCHS, "measured_epochs_per_repetition": MEASURED_EPOCHS, "base_seed": BASE_SEED, "precision": "FP32", "workers": 0},
    }
    optimization.atomic_json(output / "benchmark_config.json", benchmark_config)

    all_epochs: list[dict[str, Any]] = []
    all_steps: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for batch_size in (8, 16):
        print(f"Starting cached-CUDA batch-size benchmark: batch={batch_size}", flush=True)
        try:
            epoch_rows, step_rows = benchmark_configuration(
                batch_size, config, dataframe, inner_train,
                train_tensors, validation_tensors, loss_fn, initial_state, device,
            )
            all_epochs.extend(epoch_rows)
            all_steps.extend(step_rows)
            epoch_frame = pd.DataFrame(epoch_rows)
            step_frame = pd.DataFrame(step_rows)
            summary = summarize(batch_size, epoch_frame, step_frame)
            summaries.append(summary)
            optimization.atomic_csv(pd.DataFrame(all_epochs), output / "benchmark_results.csv")
            optimization.atomic_csv(pd.DataFrame(all_steps), output / "step_timing.csv")
            optimization.atomic_json(output / "progress.json", {"completed_batch_sizes": [item["batch_size"] for item in summaries], "failures": failures})
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            failures.append({"batch_size": batch_size, "error_type": type(error).__name__, "error": str(error)})
            optimization.atomic_json(output / "progress.json", {"completed_batch_sizes": [item["batch_size"] for item in summaries], "failures": failures})
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    epoch_frame = pd.DataFrame(all_epochs)
    step_frame = pd.DataFrame(all_steps)
    if not epoch_frame.empty:
        optimization.atomic_csv(epoch_frame, output / "benchmark_results.csv")
        optimization.atomic_csv(step_frame, output / "step_timing.csv")
        telemetry_columns = [
            "batch_size", "repetition", "epoch", "process_tree_rss_mb", "system_available_ram_gb",
            "gpu_allocated_mb", "gpu_reserved_mb", "gpu_peak_allocated_mb", "gpu_peak_reserved_mb",
            "gpu_utilization_percent", "gpu_device_memory_used_mb", "gpu_temperature_c",
        ]
        optimization.atomic_csv(epoch_frame[telemetry_columns], output / "resource_telemetry.csv")
    report = make_report(config, hardware, dataset_hash, split_identity, summaries, failures)
    (output / "comparison_report.md").write_text(report, encoding="utf-8")
    optimization.atomic_json(output / "benchmark_summary.json", {
        "summaries": summaries,
        "failures": failures,
        "report": "comparison_report.md",
    })
    print(f"Benchmark complete: {output}", flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()