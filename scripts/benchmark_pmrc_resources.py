from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import run_pmrc_nested_optimization as optimization  # noqa: E402
from src.data.dataset import load_original_dataset, make_fold_data  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.runtime import configure_runtime  # noqa: E402

RUN_ID = "resource_benchmark_20261113"
OUTPUT = ROOT / "experiments" / "pmrc_optimization" / RUN_ID
BENCHMARKS = [
    {"name": "A", "batch_size": 8, "cpu_threads": 2, "num_workers": 0},
    {"name": "B", "batch_size": 8, "cpu_threads": 2, "num_workers": 1},
    {"name": "C", "batch_size": 8, "cpu_threads": 2, "num_workers": 2},
    {"name": "D", "batch_size": 16, "cpu_threads": 2, "num_workers": 0},
]
EPOCHS = 2
BENCHMARK_ROWS = 128
MIN_FREE_RAM_GB = 1.0
EXPECTED_WORKER_RAM_GB = 0.75


def gpu_telemetry() -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "gpu_utilization_percent": None,
        "gpu_device_memory_used_mb": None,
        "gpu_temperature_c": None,
    }
    try:
        completed = optimization.subprocess.run(
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
        values = [value.strip() for value in completed.stdout.splitlines()[0].split(",")]
        result["gpu_utilization_percent"] = float(values[0])
        result["gpu_device_memory_used_mb"] = float(values[1])
        result["gpu_temperature_c"] = float(values[2])
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        pass
    return result


def process_tree_usage(process: psutil.Process) -> tuple[float, float]:
    children = process.children(recursive=True)
    rss = process.memory_info().rss
    cpu_percent = process.cpu_percent(interval=None)
    for child in children:
        try:
            rss += child.memory_info().rss
            cpu_percent += child.cpu_percent(interval=None)
        except psutil.Error:
            continue
    return cpu_percent, rss / 2**20


def prepare_streaming_loader(
    dataset: TensorDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )


def run_configuration(
    benchmark: dict[str, Any],
    dataset: TensorDataset,
    dataframe: pd.DataFrame,
    config: dict[str, Any],
    device: torch.device,
    resources: list[dict[str, Any]],
) -> dict[str, Any]:
    available_gb = psutil.virtual_memory().available / 2**30
    required_gb = MIN_FREE_RAM_GB + EXPECTED_WORKER_RAM_GB * int(benchmark["num_workers"])
    base_result: dict[str, Any] = {
        **benchmark,
        "status": "not_started",
        "available_ram_before_gb": available_gb,
        "required_ram_headroom_gb": required_gb,
        "cuda_oom": False,
    }
    if available_gb < required_gb:
        base_result.update({
            "status": "skipped_low_system_ram",
            "reason": f"Available RAM {available_gb:.2f} GB was below the conservative {required_gb:.2f} GB guard.",
        })
        return base_result

    run_config = dict(config)
    run_config["training"] = dict(config["training"], batch_size=benchmark["batch_size"], num_workers=benchmark["num_workers"])
    model_config = dict(run_config["model"], dropout=optimization.CANDIDATES[0]["dropout"], coupling="asymmetric", directional_asymmetry=True)
    optimization.set_seed(4242)
    model = optimization.OptimizedPMRCNet(run_config, model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimization.CANDIDATES[0]["learning_rate"],
        weight_decay=optimization.CANDIDATES[0]["weight_decay"],
    )
    train_indices = np.arange(len(dataframe), dtype=np.int64)
    loss_fn = optimization.make_loss(dataframe, train_indices, run_config, optimization.CANDIDATES[0])
    loader = prepare_streaming_loader(
        dataset,
        benchmark["batch_size"],
        benchmark["num_workers"],
        device.type == "cuda",
    )
    process = psutil.Process()
    process.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    step_times: list[float] = []
    epoch_times: list[float] = []
    epoch_rows: list[dict[str, Any]] = []
    base_result["status"] = "completed"
    try:
        for epoch in range(1, EPOCHS + 1):
            epoch_started = time.perf_counter()
            epoch_step_times: list[float] = []
            model.train()
            for batch in loader:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step_started = time.perf_counter()
                tensors = [item.to(device, non_blocking=device.type == "cuda") for item in batch]
                optimizer.zero_grad(set_to_none=True)
                outputs = model(*tensors[:3])
                loss, _, _ = loss_fn(model, outputs, tensors[3], *tensors[:3])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Non-finite loss in benchmark {benchmark['name']}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step_seconds = time.perf_counter() - step_started
                epoch_step_times.append(step_seconds)
                step_times.append(step_seconds)
            epoch_seconds = time.perf_counter() - epoch_started
            epoch_times.append(epoch_seconds)
            process_cpu, process_rss = process_tree_usage(process)
            memory = {
                "gpu_allocated_mb": torch.cuda.memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
                "gpu_reserved_mb": torch.cuda.memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0,
                "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
                "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0,
            }
            row = {
                "config": benchmark["name"],
                "batch_size": benchmark["batch_size"],
                "cpu_threads": benchmark["cpu_threads"],
                "num_workers": benchmark["num_workers"],
                "epoch": epoch,
                "step_count": len(epoch_step_times),
                "mean_step_seconds": float(np.mean(epoch_step_times)),
                "epoch_seconds": epoch_seconds,
                "system_cpu_percent": psutil.cpu_percent(interval=None),
                "process_tree_cpu_percent": process_cpu,
                "process_tree_rss_mb": process_rss,
                **memory,
                **gpu_telemetry(),
            }
            resources.append(row)
            epoch_rows.append(row)
    except torch.cuda.OutOfMemoryError as error:
        base_result.update({"status": "failed_cuda_oom", "cuda_oom": True, "reason": str(error)})
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except (FloatingPointError, MemoryError, RuntimeError) as error:
        if isinstance(error, RuntimeError) and "out of memory" not in str(error).lower():
            raise
        base_result.update({"status": "failed", "reason": f"{type(error).__name__}: {error}"})
        if device.type == "cuda":
            torch.cuda.empty_cache()
    finally:
        base_result["mean_step_seconds"] = float(np.mean(step_times)) if step_times else None
        base_result["mean_epoch_seconds"] = float(np.mean(epoch_times)) if epoch_times else None
        base_result["max_process_tree_rss_mb"] = max((row["process_tree_rss_mb"] for row in epoch_rows), default=None)
        base_result["peak_gpu_allocated_mb"] = max((row["gpu_peak_allocated_mb"] for row in epoch_rows), default=0.0)
        base_result["peak_gpu_reserved_mb"] = max((row["gpu_peak_reserved_mb"] for row in epoch_rows), default=0.0)
        base_result["mean_gpu_utilization_percent"] = float(np.nanmean([row["gpu_utilization_percent"] for row in epoch_rows])) if any(row["gpu_utilization_percent"] is not None for row in epoch_rows) else None
        base_result["max_gpu_temperature_c"] = max((row["gpu_temperature_c"] for row in epoch_rows if row["gpu_temperature_c"] is not None), default=None)
        del loader, model, optimizer, loss_fn
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return base_result


def write_report(configurations: list[dict[str, Any]], hardware: dict[str, Any], dataset_hash: str, initial_available_ram_gb: float) -> None:
    successful = [item for item in configurations if item["status"] == "completed"]
    selected = None
    reason = "No configuration completed within the system-RAM safety guards."
    if successful:
        batch_8 = [item for item in successful if item["batch_size"] == 8 and item["num_workers"] == 0]
        default = batch_8[0] if batch_8 else successful[0]
        fastest = min(successful, key=lambda item: item["mean_epoch_seconds"] or float("inf"))
        if fastest is not default and fastest["mean_epoch_seconds"] < 0.95 * default["mean_epoch_seconds"]:
            selected = fastest
            reason = "The alternative reduced mean epoch time by at least 5% without OOM; this recommendation applies only to the streaming benchmark path."
        else:
            selected = default
            reason = "No tested alternative was at least 5% faster than the stable batch-8, zero-worker reference; retain that conservative configuration."

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": dataset_hash,
        "benchmark_rows": BENCHMARK_ROWS,
        "epochs_per_configuration": EPOCHS,
        "initial_available_ram_gb": initial_available_ram_gb,
        "ram_headroom_policy": {
            "minimum_free_ram_gb": MIN_FREE_RAM_GB,
            "expected_ram_per_worker_gb": EXPECTED_WORKER_RAM_GB,
            "note": "Worker configurations are skipped rather than risk system memory pressure when the available-RAM guard is not met.",
        },
        "hardware": hardware,
        "configurations": configurations,
        "selected_configuration": selected,
        "selection_reason": reason,
        "method_note": "This is a small streaming-DataLoader throughput benchmark, not nested CV. The full experiment currently caches prepared fold tensors on CUDA and does not iterate its DataLoaders during model training; worker results therefore do not alter its configuration automatically.",
    }
    optimization.atomic_json(OUTPUT / "benchmark_report.json", summary)
    optimization.atomic_csv(pd.DataFrame(configurations), OUTPUT / "benchmark_summary.csv")
    lines = [
        "# PMRC-Net resource benchmark",
        "",
        f"Dataset SHA-256: `{dataset_hash}`. Training-only sample: {BENCHMARK_ROWS} rows from seed 42 outer-fold 1's training partition. This benchmark did not score or select on the outer-test fold.",
        "",
        f"Hardware: {hardware['gpu_name'] or 'CPU'}; {hardware['gpu_total_memory_mb']:.0f} MiB VRAM; {hardware.get('system_ram_total_gb', 0):.2f} GiB system RAM; {hardware.get('cpu_model') or hardware['cpu_count']} logical CPUs; PyTorch {hardware['torch_version']}; CUDA {hardware['cuda_version']}; initial available system RAM {initial_available_ram_gb:.2f} GiB.",
        "",
        "| Config | Batch | CPU threads | Workers | Status | Mean step (s) | Mean epoch (s) | Max process-tree RAM (MiB) | Peak allocated (MiB) | Peak reserved (MiB) | Mean GPU util | Max temp (C) |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    notes = []
    for item in configurations:
        fmt = lambda value: "n/a" if value is None else f"{value:.3f}"
        lines.append(
            f"| {item['name']} | {item['batch_size']} | {item['cpu_threads']} | {item['num_workers']} | {item['status']} | "
            f"{fmt(item.get('mean_step_seconds'))} | {fmt(item.get('mean_epoch_seconds'))} | "
            f"{fmt(item.get('max_process_tree_rss_mb'))} | {fmt(item.get('peak_gpu_allocated_mb'))} | "
            f"{fmt(item.get('peak_gpu_reserved_mb'))} | {fmt(item.get('mean_gpu_utilization_percent'))} | "
            f"{fmt(item.get('max_gpu_temperature_c'))} |"
        )
        if item.get("reason"):
            notes.append(f"Config {item['name']} note: {item['reason']}")
    selected_name = selected["name"] if selected else "none"
    lines.extend([
        "",
        *notes,
        "",
        f"Selected benchmark configuration: {selected_name}. {reason}",
        "",
        summary["method_note"],
        "",
        "No change was made to the nested-CV configuration or its saved run directory. FP32 was used; no AMP comparison was performed.",
    ])
    (OUTPUT / "RESOURCE_BENCHMARK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark small PMRC-Net training resource configurations")
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    global OUTPUT
    OUTPUT = ROOT / "experiments" / "pmrc_optimization" / args.run_id
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Benchmark output already exists; choose a new --run-id: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "configs" / "model_config.yaml")
    config["data"]["path"] = "dataset/cleaned_merged_heart_dataset(4).csv"
    config["training"].update({"cpu_threads": 2, "cpu_interop_threads": 1, "num_workers": 0, "batch_size": 8, "mixed_precision": False})
    device = configure_runtime(config)
    config["hardware"]["pin_memory"] = device.type == "cuda"
    dataset_path = resolve_path(ROOT, config["data"]["path"])
    dataframe = load_original_dataset(dataset_path, config["data"]["target"], config["data"]["deduplicate_exact_rows"])
    outer_train = optimization.expected_splits(dataframe, config)["42"][0]["train_indices"]
    benchmark_frame = dataframe.iloc[np.asarray(outer_train[:BENCHMARK_ROWS], dtype=np.int64)].reset_index(drop=True)
    local_indices = np.arange(len(benchmark_frame), dtype=np.int64)
    fold_data = make_fold_data(benchmark_frame, local_indices, np.empty(0, dtype=np.int64), config)
    dataset = fold_data.train_loader.dataset
    hardware = optimization.hardware_metadata(device, config)
    hardware["cpu_model"] = platform.processor() or platform.machine()
    hardware["system_ram_total_gb"] = psutil.virtual_memory().total / 2**30
    initial_available_ram_gb = psutil.virtual_memory().available / 2**30
    optimization.atomic_json(OUTPUT / "benchmark_config.json", {
        "hardware": hardware,
        "dataset_path": config["data"]["path"],
        "dataset_sha256": optimization.sha256_file(dataset_path),
        "rows": BENCHMARK_ROWS,
        "epochs": EPOCHS,
        "configurations": BENCHMARKS,
        "fp32": True,
    })
    resources: list[dict[str, Any]] = []
    results = []
    for benchmark in BENCHMARKS:
        print(f"Benchmark {benchmark['name']}: batch={benchmark['batch_size']}, workers={benchmark['num_workers']}", flush=True)
        result = run_configuration(benchmark, dataset, benchmark_frame, config, device, resources)
        results.append(result)
        optimization.atomic_csv(pd.DataFrame(resources), OUTPUT / "benchmark_epoch_resources.csv")
        optimization.atomic_json(OUTPUT / "benchmark_progress.json", {"completed_configurations": results, "next_config": len(results) + 1})
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    write_report(results, hardware, optimization.sha256_file(dataset_path), initial_available_ram_gb)
    print(f"Resource benchmark complete: {OUTPUT}", flush=True)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()