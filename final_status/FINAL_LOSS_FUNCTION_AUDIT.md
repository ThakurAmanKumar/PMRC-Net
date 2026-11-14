# Final Loss Function Audit

The implemented loss is BCE plus a conditional physiological hinge weighted by `phys_lambda` and coupling-gate L2 regularization weighted by `l2_lambda`. `l1_lambda` is declared but unused. The physiological hinge is activated only for positive empirical directions; observed associations were negative, so the recorded physiological penalty was zero.

The zero penalty is an implementation consequence, not evidence that the physiological assumptions are validated. No loss component was artificially activated and no loss or architecture change was made.

---

## Authoritative current status

- Overall project status: **INCONCLUSIVE**.
- Patient-level independence: **INCONCLUSIVE**.
- Independent external validation: **BLOCKED**.
- Clinical readiness: **NOT ESTABLISHED**.
