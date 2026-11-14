from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClinicalPhysiologyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a1 = nn.Parameter(torch.full((1,), 0.5))
        self.a2 = nn.Parameter(torch.full((1,), 0.5))
        self.a3 = nn.Parameter(torch.full((1,), 0.5))

    def forward(
        self,
        age_raw: torch.Tensor,
        max_hr_raw: torch.Tensor,
        oldpeak_raw: torch.Tensor,
        exang_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted_hr = 208.0 - 0.7 * age_raw
        utilization = max_hr_raw / torch.clamp(predicted_hr, min=1.0)
        heart_rate_deficit = torch.clamp(1.0 - utilization, min=0.0)
        ischemia = F.softplus(
            torch.clamp(self.a1, min=0.0) * oldpeak_raw
            + torch.clamp(self.a2, min=0.0) * exang_raw
            + torch.clamp(self.a3, min=0.0) * oldpeak_raw * exang_raw
        )
        return ischemia * heart_rate_deficit, heart_rate_deficit, ischemia


class PMRCNet(nn.Module):
    """PMRC-Net v1.1 with ordered asymmetric pairwise interactions."""

    def __init__(self, config: dict, model_config: dict | None = None) -> None:
        super().__init__()
        self.config = config
        self.model_config = model_config or config["model"]
        self.hidden_dim = int(self.model_config.get("hidden_dim", 16))
        self.dropout_rate = float(self.model_config.get("dropout", 0.1))
        self.coupling_mode = str(self.model_config.get("coupling", "asymmetric"))
        if self.coupling_mode not in {"asymmetric", "symmetric", "none"}:
            raise ValueError(f"Unsupported coupling mode: {self.coupling_mode}")
        data_config = config["data"]
        continuous = data_config["continuous_cols"]
        categorical = data_config["categorical_cols"]
        self.continuous_cols = list(continuous)
        self.categorical_cols = list(categorical)
        self.category_values = {
            name: tuple(int(value) for value in values)
            for name, values in data_config["category_values"].items()
        }
        if set(self.category_values) != set(self.categorical_cols):
            raise ValueError("category_values must define every categorical column exactly once")
        pathways = data_config["pathway_map"]
        self.metabolic_cols, self.exercise_cols, self.coronary_cols = (
            pathways[name] for name in ("metabolic", "exercise", "coronary")
        )
        self.metabolic_cont_indices = [continuous.index(c) for c in self.metabolic_cols if c in continuous]
        self.metabolic_cat_indices = [categorical.index(c) for c in self.metabolic_cols if c in categorical]
        self.exercise_cont_indices = [continuous.index(c) for c in self.exercise_cols if c in continuous]
        self.exercise_cat_indices = [categorical.index(c) for c in self.exercise_cols if c in categorical]
        self.coronary_cont_indices = [continuous.index(c) for c in self.coronary_cols if c in continuous]
        self.coronary_cat_indices = [categorical.index(c) for c in self.coronary_cols if c in categorical]
        self.embedding_cardinality = int(self.model_config.get("embedding_cardinality", 10))
        self.embeddings = nn.ModuleDict({
            column: nn.Embedding(self.embedding_cardinality, self.hidden_dim)
            for column in categorical
        })
        self.proj_metabolic = nn.Linear(
            len(self.metabolic_cont_indices) + len(self.metabolic_cat_indices) * self.hidden_dim,
            self.hidden_dim,
        )
        self.proj_exercise = nn.Linear(
            len(self.exercise_cont_indices) + len(self.exercise_cat_indices) * self.hidden_dim,
            self.hidden_dim,
        )
        self.proj_coronary = nn.Linear(
            len(self.coronary_cont_indices) + len(self.coronary_cat_indices) * self.hidden_dim,
            self.hidden_dim,
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.phys_module = ClinicalPhysiologyModule()
        self.proj_me = nn.Linear(1, self.hidden_dim)
        if self.coupling_mode in {"asymmetric", "symmetric"}:
            gate_dim = self.hidden_dim * (2 if self.coupling_mode == "asymmetric" else 1)
            self.W_me = nn.Linear(self.hidden_dim * 3, gate_dim)
            self.W_ec = nn.Linear(self.hidden_dim * 3, gate_dim)
            self.W_mc = nn.Linear(self.hidden_dim * 3, gate_dim)
        self.trans_m = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.trans_e = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.trans_c = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_mech = nn.Linear(self.hidden_dim, 1)
        self.out_inter = nn.Linear(self.hidden_dim, 1)
        self.out_res = nn.Linear(self.hidden_dim, 1)

    @staticmethod
    def _first_bad(mask: torch.Tensor) -> tuple[int, ...]:
        return tuple(int(index) for index in torch.nonzero(mask, as_tuple=False)[0].tolist())

    def _validate_inputs(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
        x_raw_phys: torch.Tensor,
    ) -> None:
        expected_shapes = (
            (x_cont, len(self.continuous_cols), "x_cont"),
            (x_cat, len(self.categorical_cols), "x_cat"),
            (x_raw_phys, 4, "x_raw_phys"),
        )
        batch_sizes = []
        for tensor, width, name in expected_shapes:
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape[1] != width:
                raise ValueError(f"{name} must have shape (batch, {width})")
            batch_sizes.append(tensor.shape[0])
        if len(set(batch_sizes)) != 1:
            raise ValueError("x_cont, x_cat, and x_raw_phys must have matching batch sizes")
        if not x_cont.is_floating_point() or x_cont.is_complex():
            raise ValueError("Continuous inputs must be real floating-point values")
        invalid_cont = ~torch.isfinite(x_cont)
        if bool(invalid_cont.any()):
            row, column = self._first_bad(invalid_cont)
            raise ValueError(
                f"Invalid continuous value for '{self.continuous_cols[column]}' at batch row {row}: "
                f"{x_cont[row, column].item()!r}; expected a finite value"
            )
        if not x_raw_phys.is_floating_point() or x_raw_phys.is_complex():
            raise ValueError("Raw physiology inputs must be real floating-point values")
        invalid_phys = ~torch.isfinite(x_raw_phys)
        if bool(invalid_phys.any()):
            row, column = self._first_bad(invalid_phys)
            phys_cols = self.config["data"]["physiology_cols"]
            raise ValueError(
                f"Invalid physiology value for '{phys_cols[column]}' at batch row {row}: "
                f"{x_raw_phys[row, column].item()!r}; expected a finite value"
            )
        if x_cat.is_complex() or x_cat.dtype == torch.bool:
            raise ValueError("Categorical indices must be real integer values")
        invalid_cat = torch.zeros_like(x_cat, dtype=torch.bool)
        for column, name in enumerate(self.categorical_cols):
            values = x_cat[:, column]
            valid = torch.zeros_like(values, dtype=torch.bool)
            for code in self.category_values[name]:
                valid |= values == code
            invalid = ~valid
            if values.is_floating_point():
                invalid |= ~torch.isfinite(values) | (values != torch.trunc(values))
            invalid_cat[:, column] = invalid
        if bool(invalid_cat.any()):
            row, column = self._first_bad(invalid_cat)
            value = x_cat[row, column].item()
            allowed = list(self.category_values[self.categorical_cols[column]])
            raise ValueError(
                f"Invalid categorical index for '{self.categorical_cols[column]}' at batch row {row}: "
                f"{value!r}; expected one of {allowed}"
            )
        if any(max(values) >= self.embedding_cardinality for values in self.category_values.values()):
            raise ValueError("embedding_cardinality does not cover the configured categorical vocabularies")

    def _assemble(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
        cont_indices: list[int],
        cat_indices: list[int],
    ) -> torch.Tensor:
        features = [x_cont[:, cont_indices]] if cont_indices else []
        for index in cat_indices:
            column = self.categorical_cols[index]
            features.append(self.embeddings[column](x_cat[:, index].long()))
        return torch.cat(features, dim=-1)

    def _couple(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        mismatch: torch.Tensor,
        gate_layer: nn.Linear,
        left_transform: nn.Linear,
        right_transform: nn.Linear,
    ) -> torch.Tensor:
        gate_logits = gate_layer(torch.cat((left, right, mismatch), dim=-1))
        forward_flow = left_transform(left) * right_transform(right)
        if self.coupling_mode == "symmetric":
            return torch.sigmoid(gate_logits) * forward_flow
        forward_gate, reverse_gate = torch.sigmoid(gate_logits).chunk(2, dim=-1)
        reverse_flow = right_transform(left) * left_transform(right)
        return forward_gate * forward_flow + reverse_gate * reverse_flow

    def forward(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
        x_raw_phys: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._validate_inputs(x_cont, x_cat, x_raw_phys)
        age, max_hr, oldpeak, exang = (x_raw_phys[:, i:i + 1] for i in range(4))
        mismatch, _, _ = self.phys_module(age, max_hr, oldpeak, exang)
        mismatch_feature = self.dropout(F.relu(self.proj_me(mismatch)))
        h_m = self.dropout(F.relu(self.proj_metabolic(self._assemble(
            x_cont, x_cat, self.metabolic_cont_indices, self.metabolic_cat_indices
        ))))
        h_e = self.dropout(F.relu(self.proj_exercise(self._assemble(
            x_cont, x_cat, self.exercise_cont_indices, self.exercise_cat_indices
        ))))
        h_c = self.dropout(F.relu(self.proj_coronary(self._assemble(
            x_cont, x_cat, self.coronary_cont_indices, self.coronary_cat_indices
        ))))
        c_me = torch.zeros_like(h_m)
        c_ec = torch.zeros_like(h_e)
        c_mc = torch.zeros_like(h_c)
        if self.coupling_mode != "none":
            c_me = self._couple(h_m, h_e, mismatch_feature, self.W_me, self.trans_m, self.trans_e)
            c_ec = self._couple(h_e, h_c, mismatch_feature, self.W_ec, self.trans_e, self.trans_c)
            c_mc = self._couple(h_m, h_c, mismatch_feature, self.W_mc, self.trans_m, self.trans_c)
        z_mech = self.out_mech(mismatch_feature + h_c)
        z_inter = self.out_inter(c_me + c_ec + c_mc)
        z_res = self.out_res(h_m + h_e + h_c)
        z_total = z_mech + z_inter + z_res
        return {
            "z_total": z_total,
            "z_mech": z_mech,
            "z_inter": z_inter,
            "z_res": z_res,
            "m_e": mismatch,
        }