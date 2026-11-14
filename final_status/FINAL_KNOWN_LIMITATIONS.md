# Final Known Limitations

- 602 rows are exact-unique observations, not confirmed independent patients.
- The original 1888-row dataset has unresolved source and patient-level provenance.
- Row-level bootstrap intervals cannot model unknown patient clustering.
- The positive-direction physiological hinge was inactive in observed folds.
- `l1_lambda` is declared but unused.
- The configured mixed-precision flag is not implemented in the maintained training path.
- UCI candidates may overlap the local development source; external validation is blocked.
- No clinical utility, diagnostic readiness, causality, or clinical superiority is established.

---

## Authoritative current status

- Overall project status: **INCONCLUSIVE**.
- Patient-level independence: **INCONCLUSIVE**.
- Independent external validation: **BLOCKED**.
- Clinical readiness: **NOT ESTABLISHED**.
