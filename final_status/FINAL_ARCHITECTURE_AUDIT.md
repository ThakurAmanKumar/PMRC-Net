# Final Architecture Audit

The implemented PMRC-Net has metabolic, exercise-response, and coronary/electrical pathway projections; categorical embeddings; a raw-scale physiology mismatch module; asymmetric or symmetric coupling gates; mechanistic, interaction, and residual output heads; and an additive total logit `z_total = z_mech + z_inter + z_res`.

The gradient report records finite gradients and optimizer updates for inspected groups in both coupling diagnostics. Ablation results show a large pathway-removal degradation and smaller changes when mechanistic or residual heads are removed. These establish computational behavior, not causal or physiological validity.

No architecture change was made.

---

## Authoritative current status

- Overall project status: **INCONCLUSIVE**.
- Patient-level independence: **INCONCLUSIVE**.
- Independent external validation: **BLOCKED**.
- Clinical readiness: **NOT ESTABLISHED**.
