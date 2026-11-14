from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pmrc_net import PMRCNet


class OptimizedPMRCNet(PMRCNet):
    """A constrained optimization variant that keeps the original PMRC-Net backbone but adds
    pathway normalization, stable residual scaling, and gate regularization.
    """

    def __init__(self, config: dict, model_config: dict) -> None:
        super().__init__(config, model_config)
        self.pathway_norm = nn.ModuleDict({
            "metabolic": nn.LayerNorm(self.hidden_dim),
            "exercise": nn.LayerNorm(self.hidden_dim),
            "coronary": nn.LayerNorm(self.hidden_dim),
            "mismatch": nn.LayerNorm(self.hidden_dim),
        })
        self.directional_asymmetry = bool(model_config.get("directional_asymmetry", True))
        self.gate_scale = nn.Parameter(torch.tensor(1.0))
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def _normalize_hidden(self, hidden: torch.Tensor, name: str) -> torch.Tensor:
        return self.pathway_norm[name](hidden)

    def _gate_regularization(self) -> torch.Tensor:
        total = torch.zeros((), device=self.out_res.weight.device)
        for name in ("W_me", "W_ec", "W_mc"):
            weight = getattr(self, name)
            total = total + torch.mean(weight.weight.pow(2))
        return total

    def _directional_flow(self, left: torch.Tensor, right: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        if name == "me":
            left_transform, right_transform = self.trans_m, self.trans_e
        elif name == "ec":
            left_transform, right_transform = self.trans_e, self.trans_c
        else:
            left_transform, right_transform = self.trans_m, self.trans_c
        forward = left_transform(left) * right_transform(right)
        reverse = right_transform(left) * left_transform(right)
        return forward, reverse

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor, x_raw_phys: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_categorical_indices(x_cat)
        age, max_hr, oldpeak, exang = (x_raw_phys[:, i:i + 1] for i in range(4))
        m_e, _, _ = self.phys_module(age, max_hr, oldpeak, exang)
        if self.ablation == "no_physiology":
            m_e = torch.zeros_like(m_e)
        m_e_feat = self.dropout(F.relu(self.proj_me(m_e)))
        m_e_feat = self._normalize_hidden(m_e_feat, "mismatch")

        h_m = self.dropout(F.relu(self.proj_metabolic(self._assemble(x_cont, x_cat, self.metabolic_cont_indices, self.metabolic_cat_indices, self.categorical_cols))))
        h_e = self.dropout(F.relu(self.proj_exercise(self._assemble(x_cont, x_cat, self.exercise_cont_indices, self.exercise_cat_indices, self.categorical_cols))))
        h_c = self.dropout(F.relu(self.proj_coronary(self._assemble(x_cont, x_cat, self.coronary_cont_indices, self.coronary_cat_indices, self.categorical_cols))))

        h_m = self._normalize_hidden(h_m, "metabolic")
        h_e = self._normalize_hidden(h_e, "exercise")
        h_c = self._normalize_hidden(h_c, "coronary")

        c_me, c_ec, c_mc = torch.zeros_like(h_m), torch.zeros_like(h_e), torch.zeros_like(h_c)
        if self.coupling_mode != "none":
            pairs = ((h_m, h_e, self.W_me, "me"), (h_e, h_c, self.W_ec, "ec"), (h_m, h_c, self.W_mc, "mc"))
            for left, right, gate_layer, name in pairs:
                gate = torch.sigmoid(self.gate_scale * gate_layer(torch.cat([left, right, m_e_feat], dim=-1)))
                if self.coupling_mode == "asymmetric":
                    forward, reverse = self._directional_flow(left, right, name)
                    if self.directional_asymmetry:
                        forward_gate, reverse_gate = gate.chunk(2, dim=-1)
                        value = forward_gate * forward + reverse_gate * reverse
                    else:
                        value = gate[:, :self.hidden_dim] * forward
                else:
                    transforms = (self.trans_m(left), self.trans_e(right)) if name == "me" else (
                        (self.trans_e(left), self.trans_c(right)) if name == "ec" else (self.trans_m(left), self.trans_c(right))
                    )
                    value = gate * (transforms[0] * transforms[1])
                if name == "me":
                    c_me = value
                elif name == "ec":
                    c_ec = value
                else:
                    c_mc = value

        z_mech = self.out_mech(m_e_feat + h_c)
        z_inter = self.out_inter(c_me + c_ec + c_mc)
        z_res = self.out_res(self.residual_scale * (h_m + h_e + h_c))
        if self.ablation == "no_mechanistic":
            z_mech = torch.zeros_like(z_mech)
        elif self.ablation == "no_residual":
            z_res = torch.zeros_like(z_res)
        elif self.ablation == "no_pathways":
            z_mech = self.out_mech(m_e_feat)
            z_inter = torch.zeros_like(z_inter)
            z_res = torch.zeros_like(z_res)
        z_total = z_mech + z_inter + z_res
        return {"z_total": z_total, "z_mech": z_mech, "z_inter": z_inter, "z_res": z_res, "m_e": m_e}
