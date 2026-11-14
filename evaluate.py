from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.data.dataset import load_original_dataset, make_fold_data
from src.evaluation.metrics import calculate_metrics
from src.models.pmrc_net import PMRCNet
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config, resolve_path
from src.utils.runtime import configure_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved PMRC-Net checkpoints.")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = load_config(root / args.config)
    device = configure_runtime(config)
    dataframe = load_original_dataset(resolve_path(root, config["data"]["path"]), config["data"]["target"], config["data"]["deduplicate_exact_rows"])
    target = dataframe[config["data"]["target"]].to_numpy()
    from sklearn.model_selection import StratifiedKFold
    folds = list(StratifiedKFold(n_splits=int(config["training"]["folds"]), shuffle=True, random_state=int(config["seed"])).split(dataframe, target))
    train_idx, valid_idx = folds[args.fold - 1]
    fold = make_fold_data(dataframe, train_idx, valid_idx, config)
    model = PMRCNet(config, config["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for batch in fold.valid_loader:
            tensors = [item.to(device) for item in batch]
            probabilities.extend(torch.sigmoid(model(*tensors[:3])["z_total"]).squeeze(-1).cpu().numpy())
            labels.extend(tensors[3].cpu().numpy())
    metrics = calculate_metrics(labels, probabilities, args.threshold)
    output_dir = Path(args.output) if args.output else root / "reports" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    ConfusionMatrixDisplay(confusion_matrix(labels, (np.asarray(probabilities) >= args.threshold).astype(int))).plot()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
