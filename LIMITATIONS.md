# Limitations and Threats to Validity

- Exact deduplication reduces repeated records but does not establish patient independence.
- The source data contain mixed clinical coding conventions and no patient-level provenance.
- The sample is small for comparing flexible models; confidence intervals and external validation remain necessary.
- Hyperparameter selection and threshold selection must remain inside inner training folds for confirmatory claims.
- The physiological penalty is inactive under the original data-driven direction rule in this dataset. Activating it by forcing signs from target correlations would violate the stated scientific rule.
- Attribution and perturbation results describe model behavior, not causality or clinical safety.
- Missing XGBoost and notebook-only Tabular Transformer implementations are recorded as unavailable, not replaced silently.