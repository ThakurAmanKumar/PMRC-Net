from __future__ import annotations

import logging
import os
import random
from typing import Any

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def configure_runtime(config: dict[str, Any]) -> torch.device:
    threads = int(config["training"]["cpu_threads"])
    torch.set_num_threads(threads)
    interop_threads = int(config["training"].get("cpu_interop_threads", threads))
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        LOGGER.debug("PyTorch inter-op thread count was already initialized")
    seed = int(config["seed"])
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    requested = config["hardware"].get("device", "auto")
    if requested == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    LOGGER.info("Using device: %s", device)
    if device.type == "cuda":
        LOGGER.info("GPU: %s; VRAM: %.2f GB", torch.cuda.get_device_name(device), torch.cuda.get_device_properties(device).total_memory / 2**30)
    return device


def gpu_memory_report(device: torch.device) -> dict[str, float | str]:
    if device.type != "cuda":
        return {"device": str(device), "allocated_gb": 0.0, "reserved_gb": 0.0, "peak_allocated_gb": 0.0}
    return {
        "device": str(device),
        "allocated_gb": torch.cuda.memory_allocated(device) / 2**30,
        "reserved_gb": torch.cuda.memory_reserved(device) / 2**30,
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
    }
