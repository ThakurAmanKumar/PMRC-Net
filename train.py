from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from src.data.dataset import load_original_dataset, make_fold_data
from src.models.pmrc_net import PMRCNet
from src.training.checkpoints import save_checkpoint
from src.training.losses import PMRCConsistencyLoss, compute_empirical_associations
from src.utils.config import load_config, resolve_path
from src.utils.runtime import configure_runtime, gpu_memory_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manually train PMRC-Net sequentially.")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = load_config(root / args.config)
    device = configure_runtime(config)
    data_path = resolve_path(root, config["data"]["path"])
    dataframe = load_original_dataset(data_path, config["data"]["target"], config["data"]["deduplicate_exact_rows"])
    experiment_id = time.strftime("pmrc_%Y%m%d_%H%M%S")
    experiment_dir = root / "experiments" / experiment_id
    checkpoint_dir = root / "checkpoints" / experiment_id
    experiment_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    dataset_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    (experiment_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (experiment_dir / "dataset.json").write_text(json.dumps({"path": str(data_path), "sha256": dataset_hash, "rows_used": len(dataframe)}, indent=2), encoding="utf-8")
    target = dataframe[config["data"]["target"]].to_numpy()
    splitter = StratifiedKFold(n_splits=int(config["training"]["folds"]), shuffle=True, random_state=int(config["seed"]))
    folds = []
    for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(dataframe, target), start=1):
        logging.info("Starting fold %d/%d sequentially", fold_id, config["training"]["folds"])
        fold = make_fold_data(dataframe, train_idx, valid_idx, config)
        model = PMRCNet(config, config["model"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
        associations = compute_empirical_associations(dataframe.iloc[train_idx], config["data"]["target"])
        loss_fn = PMRCConsistencyLoss(associations, **config["loss"])
        best_loss = float("inf")
        history = []
        patience_count = 0
        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            model.train()
            total = 0.0
            for batch in fold.train_loader:
                tensors = [item.to(device) for item in batch]
                optimizer.zero_grad(set_to_none=True)
                outputs = model(*tensors[:3])
                loss, _, _ = loss_fn(model, outputs, tensors[3], *tensors[:3])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at fold {fold_id}, epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"])
                optimizer.step()
                total += loss.item() * tensors[3].size(0)
            model.eval()
            valid_total = 0.0
            with torch.no_grad():
                for batch in fold.valid_loader:
                    tensors = [item.to(device) for item in batch]
                    outputs = model(*tensors[:3])
                    valid_loss, _, _ = loss_fn(model, outputs, tensors[3], *tensors[:3])
                    valid_total += valid_loss.item() * tensors[3].size(0)
            train_loss = total / len(fold.train_loader.dataset)
            valid_loss_value = valid_total / len(fold.valid_loader.dataset)
            history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss_value})
            save_checkpoint(checkpoint_dir / f"fold_{fold_id}_latest.pt", model, optimizer, epoch, best_loss, config, history)
            if valid_loss_value < best_loss - config["training"]["min_delta"]:
                best_loss = valid_loss_value
                patience_count = 0
                save_checkpoint(checkpoint_dir / f"fold_{fold_id}_best.pt", model, optimizer, epoch, best_loss, config, history)
            else:
                patience_count += 1
                if patience_count >= config["training"]["patience"]:
                    break
        folds.append({"fold": fold_id, "train_indices": train_idx.tolist(), "valid_indices": valid_idx.tolist(), "best_valid_loss": best_loss})
        del model, optimizer, loss_fn
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (experiment_dir / "fold_assignments.json").write_text(json.dumps(folds), encoding="utf-8")
    (experiment_dir / "runtime.json").write_text(json.dumps(gpu_memory_report(device), indent=2), encoding="utf-8")
    shutil.copytree(checkpoint_dir, experiment_dir / "checkpoints")
    logging.info("Experiment saved at %s", experiment_dir)


if __name__ == "__main__":
    main()
