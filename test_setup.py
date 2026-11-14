from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from src.data.dataset import load_original_dataset, make_fold_data
from src.models.pmrc_net import PMRCNet
from src.training.checkpoints import load_checkpoint, save_checkpoint
from src.training.losses import PMRCConsistencyLoss, compute_empirical_associations
from src.utils.config import load_config, resolve_path
from src.utils.runtime import configure_runtime, gpu_memory_report


def main() -> None:
    root = Path(__file__).resolve().parent
    config = load_config(root / "configs" / "model_config.yaml")
    device = configure_runtime(config)
    dataframe = load_original_dataset(resolve_path(root, config["data"]["path"]), "target", True)
    assert len(dataframe) == 602
    indices = torch.arange(len(dataframe)).numpy()
    fold = make_fold_data(dataframe, indices[:480], indices[480:], config)
    model = PMRCNet(config, config["model"]).to(device)
    batch = next(iter(fold.train_loader))
    tensors = [item.to(device) for item in batch]
    outputs = model(*tensors[:3])
    assert outputs["z_total"].shape == (len(tensors[3]), 1)
    assert torch.allclose(outputs["z_total"], outputs["z_mech"] + outputs["z_inter"] + outputs["z_res"])
    loss_fn = PMRCConsistencyLoss(compute_empirical_associations(dataframe.iloc[:480]), **config["loss"])
    loss, _, _ = loss_fn(model, outputs, tensors[3], *tensors[:3])
    loss.backward()
    with tempfile.TemporaryDirectory() as temp_dir:
        optimizer = torch.optim.AdamW(model.parameters())
        path = Path(temp_dir) / "checkpoint.pt"
        save_checkpoint(path, model, optimizer, 1, float(loss.item()), config, [])
        restored = PMRCNet(config, config["model"]).to(device)
        load_checkpoint(path, restored, map_location=device)
    print("PASS: imports, dataset, model forward, one batch/backward, checkpoint round-trip")
    print(f"PASS: device={device}")
    print(f"GPU memory: {gpu_memory_report(device)}")
    print("PASS: laptop-friendly settings: batch=8, workers=0, CPU threads=2, sequential folds")


if __name__ == "__main__":
    main()
