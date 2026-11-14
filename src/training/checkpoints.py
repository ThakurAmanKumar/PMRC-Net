from __future__ import annotations

from pathlib import Path
import torch


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, best_value: float, config: dict, history: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best_value": best_value,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "history": history,
    }, path)


def load_checkpoint(path: str | Path, model, optimizer=None, map_location="cpu") -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint
