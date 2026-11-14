# PMRC-Net Architecture

PMRC-Net is preserved from the original notebook and migrated implementation. Continuous variables are standardized within each training fold. Categorical variables use learned embeddings. Three pathway encoders represent metabolic, exercise, and coronary feature groups.

The clinical physiology module computes an age-adjusted heart-rate deficit and an ischemia score from `oldpeak`, `exang`, and their interaction. These are combined into a mechanistic feature. Asymmetric gated coupling forms pairwise pathway interactions, and the final logit is decomposed into mechanistic, interaction, and residual terms:

`z_total = z_mech + z_inter + z_res`.

The `stage2_validation.py` ablation switches expose explicit component removals for sensitivity analysis. They do not alter the original notebook or the validated holdout experiment.