from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClinicalPhysiologyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a1 = nn.Parameter(torch.ones(1) * 0.5)
        self.a2 = nn.Parameter(torch.ones(1) * 0.5)
        self.a3 = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, age_raw: torch.Tensor, max_hr_raw: torch.Tensor, oldpeak_raw: torch.Tensor, exang_raw: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hr_pred = 208.0 - 0.7 * age_raw
        hr_util = max_hr_raw / torch.clamp(hr_pred, min=1.0)
        hr_deficit = torch.clamp(1.0 - hr_util, min=0.0)
        a1 = torch.clamp(self.a1, min=0.0)
        a2 = torch.clamp(self.a2, min=0.0)
        a3 = torch.clamp(self.a3, min=0.0)
        ischemia = F.softplus(a1 * oldpeak_raw + a2 * exang_raw + a3 * oldpeak_raw * exang_raw)
        return ischemia * hr_deficit, hr_deficit, ischemia


class PMRCNet(nn.Module):
    """The PMRC-Net architecture extracted from the original notebook."""

    def __init__(self, config: dict, model_config: dict) -> None:
        super().__init__()
        self.config = config
        self.model_config = model_config
        self.hidden_dim = model_config.get("hidden_dim", 16)
        self.dropout_rate = model_config.get("dropout", 0.1)
        self.coupling_mode = model_config.get("coupling", "asymmetric")
        self.ablation = model_config.get("ablation", "complete")
        pathways = config["data"]["pathway_map"]
        continuous = config["data"]["continuous_cols"]
        categorical = config["data"]["categorical_cols"]
        self.categorical_cols = list(categorical)
        self.metabolic_cols, self.exercise_cols, self.coronary_cols = (pathways[k] for k in ("metabolic", "exercise", "coronary"))
        self.metabolic_cont_indices = [continuous.index(c) for c in self.metabolic_cols if c in continuous]
        self.metabolic_cat_indices = [categorical.index(c) for c in self.metabolic_cols if c in categorical]
        self.exercise_cont_indices = [continuous.index(c) for c in self.exercise_cols if c in continuous]
        self.exercise_cat_indices = [categorical.index(c) for c in self.exercise_cols if c in categorical]
        self.coronary_cont_indices = [continuous.index(c) for c in self.coronary_cols if c in continuous]
        self.coronary_cat_indices = [categorical.index(c) for c in self.coronary_cols if c in categorical]
        cardinality = int(model_config.get("embedding_cardinality", 10))
        self.embedding_cardinality = cardinality
        self.embeddings = nn.ModuleDict({col: nn.Embedding(cardinality, self.hidden_dim) for col in categorical})
        self.proj_metabolic = nn.Linear(len(self.metabolic_cont_indices) + len(self.metabolic_cat_indices) * self.hidden_dim, self.hidden_dim)
        self.proj_exercise = nn.Linear(len(self.exercise_cont_indices) + len(self.exercise_cat_indices) * self.hidden_dim, self.hidden_dim)
        self.proj_coronary = nn.Linear(len(self.coronary_cont_indices) + len(self.coronary_cat_indices) * self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.phys_module = ClinicalPhysiologyModule()
        self.proj_me = nn.Linear(1, self.hidden_dim)
        if self.coupling_mode in {"asymmetric", "symmetric"}:
            output_dim = self.hidden_dim * (2 if self.coupling_mode == "asymmetric" else 1)
            self.W_me = nn.Linear(self.hidden_dim * 3, output_dim)
            self.W_ec = nn.Linear(self.hidden_dim * 3, output_dim)
            self.W_mc = nn.Linear(self.hidden_dim * 3, output_dim)
        self.trans_m = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.trans_e = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.trans_c = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_mech = nn.Linear(self.hidden_dim, 1)
        self.out_inter = nn.Linear(self.hidden_dim, 1)
        self.out_res = nn.Linear(self.hidden_dim, 1)

    def _assemble(self, x_cont: torch.Tensor, x_cat: torch.Tensor, cont_indices: list[int], cat_indices: list[int], cat_cols: list[str]) -> torch.Tensor:
        features = [x_cont[:, cont_indices]] if cont_indices else []
        for idx in cat_indices:
            features.append(self.embeddings[cat_cols[idx]](x_cat[:, idx].long()))
        return torch.cat(features, dim=-1)

    def _validate_categorical_indices(self, x_cat: torch.Tensor) -> None:
        if x_cat.ndim != 2 or x_cat.shape[1] != len(self.categorical_cols):
            raise ValueError(f"x_cat must have shape (batch, {len(self.categorical_cols)})")
        if x_cat.is_complex():
            raise ValueError("Categorical indices must be real integers")
        invalid = (x_cat < 0) | (x_cat >= self.embedding_cardinality)
        if x_cat.is_floating_point():
            invalid |= ~torch.isfinite(x_cat) | (x_cat != torch.trunc(x_cat))
        if bool(invalid.any()):
            row, column = torch.nonzero(invalid, as_tuple=False)[0].tolist()
            value = x_cat[row, column].item()
            raise ValueError(
                f"Invalid categorical index for '{self.categorical_cols[column]}' at batch row {row}: "
                f"{value!r}; expected a finite integer in [0, {self.embedding_cardinality})"
            )

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor, x_raw_phys: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_categorical_indices(x_cat)
        age, max_hr, oldpeak, exang = (x_raw_phys[:, i:i + 1] for i in range(4))
        m_e, _, _ = self.phys_module(age, max_hr, oldpeak, exang)
        if self.ablation == "no_physiology":
            m_e = torch.zeros_like(m_e)
        m_e_feat = self.dropout(F.relu(self.proj_me(m_e)))
        h_m = self.dropout(F.relu(self.proj_metabolic(self._assemble(x_cont, x_cat, self.metabolic_cont_indices, self.metabolic_cat_indices, self.categorical_cols))))
        h_e = self.dropout(F.relu(self.proj_exercise(self._assemble(x_cont, x_cat, self.exercise_cont_indices, self.exercise_cat_indices, self.categorical_cols))))
        h_c = self.dropout(F.relu(self.proj_coronary(self._assemble(x_cont, x_cat, self.coronary_cont_indices, self.coronary_cat_indices, self.categorical_cols))))
        c_me, c_ec, c_mc = torch.zeros_like(h_m), torch.zeros_like(h_e), torch.zeros_like(h_c)
        if self.coupling_mode != "none":
            pairs = ((h_m, h_e, self.W_me, "me"), (h_e, h_c, self.W_ec, "ec"), (h_m, h_c, self.W_mc, "mc"))
            for left, right, gate_layer, name in pairs:
                gate = torch.sigmoid(gate_layer(torch.cat([left, right, m_e_feat], dim=-1)))
                if self.coupling_mode == "asymmetric":
                    flow = torch.cat([
                        self.trans_m(left) * self.trans_e(right),
                        self.trans_e(right) * self.trans_m(left),
                    ], dim=-1) if name == "me" else (
                        torch.cat([self.trans_e(left) * self.trans_c(right), self.trans_c(right) * self.trans_e(left)], dim=-1)
                        if name == "ec" else
                        torch.cat([self.trans_m(left) * self.trans_c(right), self.trans_c(right) * self.trans_m(left)], dim=-1)
                    )
                    value = (gate * flow).chunk(2, dim=-1)[0]
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
        z_res = self.out_res(h_m + h_e + h_c)
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
