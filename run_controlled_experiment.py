from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
from ctypes import wintypes
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.data.dataset import load_original_dataset, make_fold_data
from src.evaluation.metrics import calculate_metrics
from src.models.pmrc_net import PMRCNet
from src.training.checkpoints import save_checkpoint
from src.training.losses import PMRCConsistencyLoss, compute_empirical_associations
from src.utils.config import load_config, resolve_path
from src.utils.runtime import configure_runtime, gpu_memory_report


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_psapi = ctypes.WinDLL("psapi", use_last_error=True)
_kernel32.GetCurrentProcess.restype = wintypes.HANDLE
_psapi.GetProcessMemoryInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(ProcessMemoryCounters),
    wintypes.DWORD,
]
_psapi.GetProcessMemoryInfo.restype = wintypes.BOOL


def cpu_memory_report() -> dict[str, float]:
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not _psapi.GetProcessMemoryInfo(
        _kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise OSError("GetProcessMemoryInfo failed")
    return {
        "rss_gb": counters.WorkingSetSize / 2**30,
        "peak_rss_gb": counters.PeakWorkingSetSize / 2**30,
    }


def resource_report(device: torch.device) -> dict[str, float | str]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {**cpu_memory_report(), **gpu_memory_report(device)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> None:
    root = Path(__file__).resolve().parent
    config = load_config(root / "configs" / "model_config.yaml")
    seed = int(config["seed"])
    set_seed(seed)
    device = configure_runtime(config)

    required_settings = {
        "batch_size": 8,
        "num_workers": 0,
        "cpu_threads": 2,
    }
    for key, expected in required_settings.items():
        actual = config["training"][key]
        if actual != expected:
            raise ValueError(f"Expected training.{key}={expected}, found {actual}")
    if not config["hardware"]["sequential_only"]:
        raise ValueError("Sequential execution is required for this experiment")

    experiment_name = f"controlled_pmrc_seed{seed}"
    experiment_dir = root / "experiments" / experiment_name
    checkpoint_dir = root / "checkpoints" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    data_path = resolve_path(root, config["data"]["path"])
    dataframe = load_original_dataset(
        data_path,
        config["data"]["target"],
        config["data"]["deduplicate_exact_rows"],
    )
    indices = np.arange(len(dataframe), dtype=np.int64)
    targets = dataframe[config["data"]["target"]].to_numpy()
    train_indices, valid_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=targets,
        random_state=seed,
    )
    train_indices = np.asarray(train_indices, dtype=np.int64)
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    fold = make_fold_data(dataframe, train_indices, valid_indices, config)

    dataset_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    software = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "scikit_learn": package_version("scikit-learn"),
        "matplotlib": package_version("matplotlib"),
        "pyyaml": package_version("PyYAML"),
    }
    (experiment_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (experiment_dir / "software_versions.json").write_text(
        json.dumps(software, indent=2), encoding="utf-8"
    )
    (experiment_dir / "split_indices.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "strategy": "stratified_train_test_split",
                "test_size": 0.2,
                "train_indices": train_indices.tolist(),
                "valid_indices": valid_indices.tolist(),
                "dataset_sha256": dataset_hash,
                "dataset_path": str(data_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    model = PMRCNet(config, config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    associations = compute_empirical_associations(
        dataframe.iloc[train_indices], config["data"]["target"]
    )
    loss_fn = PMRCConsistencyLoss(associations, **config["loss"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history: list[dict] = []
    best_valid_loss = float("inf")
    best_epoch = 0
    patience_count = 0

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        train_loss_total = 0.0
        train_bce_total = 0.0
        train_phys_total = 0.0
        train_count = 0
        for batch in fold.train_loader:
            tensors = [item.to(device) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(*tensors[:3])
            loss, loss_bce, loss_phys = loss_fn(
                model, outputs, tensors[3], *tensors[:3]
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["training"]["gradient_clip_norm"]
            )
            optimizer.step()
            count = tensors[3].size(0)
            train_count += count
            train_loss_total += loss.item() * count
            train_bce_total += loss_bce.item() * count
            train_phys_total += loss_phys.item() * count

        model.eval()
        valid_loss_total = 0.0
        valid_probabilities: list[float] = []
        valid_labels: list[float] = []
        with torch.no_grad():
            for batch in fold.valid_loader:
                tensors = [item.to(device) for item in batch]
                outputs = model(*tensors[:3])
                loss, _, _ = loss_fn(model, outputs, tensors[3], *tensors[:3])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite validation loss at epoch {epoch}")
                count = tensors[3].size(0)
                valid_loss_total += loss.item() * count
                valid_probabilities.extend(
                    torch.sigmoid(outputs["z_total"]).squeeze(-1).cpu().tolist()
                )
                valid_labels.extend(tensors[3].cpu().tolist())

        train_loss = train_loss_total / train_count
        valid_loss = valid_loss_total / len(fold.valid_loader.dataset)
        valid_metrics = calculate_metrics(valid_labels, valid_probabilities)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_bce": train_bce_total / train_count,
            "train_physical_penalty": train_phys_total / train_count,
            "valid_loss": valid_loss,
            "valid_metrics": valid_metrics,
            "resources": resource_report(device),
        }
        history.append(epoch_record)
        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model,
            optimizer,
            epoch,
            best_valid_loss,
            config,
            history,
        )
        if valid_loss < best_valid_loss - config["training"]["min_delta"]:
            best_valid_loss = valid_loss
            best_epoch = epoch
            patience_count = 0
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_valid_loss,
                config,
                history,
            )
        else:
            patience_count += 1
            if patience_count >= config["training"]["patience"]:
                break

    (experiment_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    best_record = next(item for item in history if item["epoch"] == best_epoch)
    report = {
        "experiment": experiment_name,
        "status": "completed",
        "seed": seed,
        "device": str(device),
        "precision": "fp32",
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "best_epoch_valid_metrics": best_record["valid_metrics"],
        "final_epoch": history[-1],
        "associations_from_training_split": associations,
        "train_rows": len(train_indices),
        "validation_rows": len(valid_indices),
        "checkpoint": str(checkpoint_dir / "best.pt"),
        "loss_curve": str(experiment_dir / "loss_curve.png"),
        "warnings": [
            "This is one stratified holdout experiment and does not establish generalization or clinical validity.",
            "The YAML mixed_precision flag is not consumed by the validated trainer; this run intentionally used FP32.",
        ],
        "software_versions": software,
    }
    (experiment_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (experiment_dir / "report.md").write_text(
        "\n".join(
            [
                f"# Controlled PMRC-Net Experiment: `{experiment_name}`",
                "",
                f"- Status: **{report['status']}**",
                f"- Seed: `{seed}`",
                f"- Device: `{device}`",
                f"- Precision: `{report['precision']}`",
                f"- Rows: `{report['train_rows']}` train / `{report['validation_rows']}` validation",
                f"- Epochs completed: `{report['epochs_completed']}`",
                f"- Best epoch: `{report['best_epoch']}`",
                f"- Best validation loss: `{report['best_valid_loss']:.6f}`",
                "",
                "## Validation Metrics at Best Epoch",
                "",
                "```json",
                json.dumps(report["best_epoch_valid_metrics"], indent=2),
                "```",
                "",
                "## Limitations",
                "",
                "- One stratified holdout does not establish generalization or clinical validity.",
                "- No nested cross-validation, tuning, or parallel experiment was run.",
                "- Training used FP32; mixed precision remains disabled in the validated path.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    epochs = [item["epoch"] for item in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [item["train_loss"] for item in history], label="Training loss")
    plt.plot(epochs, [item["valid_loss"] for item in history], label="Validation loss")
    plt.axvline(best_epoch, color="tab:green", linestyle="--", label="Best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Controlled PMRC-Net Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(experiment_dir / "loss_curve.png", dpi=150)
    plt.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
