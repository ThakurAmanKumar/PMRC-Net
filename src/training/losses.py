from __future__ import annotations

import torch
import torch.nn as nn


def compute_empirical_associations(dataframe, target_col: str = "target") -> dict[str, int]:
    associations = {}
    for column in ("oldpeak", "exang", "ca"):
        if column in dataframe.columns:
            correlation = dataframe[column].corr(dataframe[target_col])
            associations[column] = 1 if correlation >= 0 else -1
    return associations


class PMRCConsistencyLoss(nn.Module):
    def __init__(self, associations: dict[str, int], l1_lambda: float = 0.01, l2_lambda: float = 0.01, phys_lambda: float = 0.1, gate_l2_lambda: float = 0.0, residual_l2_lambda: float = 0.0) -> None:
        super().__init__()
        self.associations = associations
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.phys_lambda = phys_lambda
        self.gate_l2_lambda = gate_l2_lambda
        self.residual_l2_lambda = residual_l2_lambda
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, model, outputs, targets, x_cont, x_cat, x_raw_phys):
        loss_bce = self.bce(outputs["z_total"].squeeze(-1), targets.float())
        loss_phys = torch.zeros((), device=targets.device)
        if self.associations.get("oldpeak") == 1:
            perturbed = x_raw_phys.clone()
            perturbed[:, 2] = torch.clamp(perturbed[:, 2] + 0.5, max=6.0)
            with torch.no_grad():
                changed = model(x_cont, x_cat, perturbed)
            loss_phys += torch.mean(torch.clamp(outputs["z_total"] - changed["z_total"], min=0.0))
        if self.associations.get("exang") == 1:
            perturbed = x_raw_phys.clone()
            perturbed[:, 3] = 1.0
            with torch.no_grad():
                changed = model(x_cont, x_cat, perturbed)
            loss_phys += torch.mean(torch.clamp(outputs["z_total"] - changed["z_total"], min=0.0))
        loss_coupling = torch.zeros((), device=targets.device)
        if model.coupling_mode != "none":
            for name, parameter in model.named_parameters():
                if "W_" in name:
                    loss_coupling += torch.sum(parameter ** 2)
        gate_regularization = torch.zeros((), device=targets.device)
        if hasattr(model, "gate_scale"):
            gate_regularization = gate_regularization + model.gate_scale.pow(2)
        if hasattr(model, "_gate_regularization"):
            gate_regularization = gate_regularization + model._gate_regularization()
        residual_regularization = torch.zeros((), device=targets.device)
        if hasattr(model, "residual_scale"):
            residual_regularization = residual_regularization + model.residual_scale.pow(2)
        total = loss_bce + self.phys_lambda * loss_phys + self.l2_lambda * loss_coupling + self.gate_l2_lambda * gate_regularization + self.residual_l2_lambda * residual_regularization
        return total, loss_bce, loss_phys
