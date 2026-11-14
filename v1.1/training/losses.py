from __future__ import annotations

import torch
import torch.nn as nn


class PMRCObjective(nn.Module):
    """Binary cross entropy; optimizer weight decay is configured separately."""

    def __init__(self) -> None:
        super().__init__()
        self.binary_cross_entropy = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.binary_cross_entropy(logits.squeeze(-1), targets.float())