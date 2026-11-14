# Author Confirmation Checklist — Required Before Submission

- [ ] Confirm every author name, order, affiliation, corresponding author, and contribution statement.
- [ ] Identify the original dataset source, collection sites, target definition, license, redistribution rights, and provenance of the merged file.
- [ ] Determine whether rows represent unique patients; document participant overlap, repeat visits, and any source-level identifiers.
- [ ] Confirm ethics review/approval or exemption, consent requirements, and applicable institutional/jurisdictional rules.
- [ ] Confirm funding, conflicts of interest, data availability, and code/data sharing statements.
- [ ] Confirm that retaining `ca=4` is correct for the intended source coding; it was preserved by the actual loader, not imputed.
- [ ] Confirm whether the paper should report the historical baseline at all. It is selection-contaminated and cannot support comparative inference; the current package labels it descriptive only.
- [ ] Do not state that PMRC-Net outperforms the historical baseline. Its outer validation fold was used for epoch/checkpoint selection.
- [ ] Confirm the target label semantics and whether "heart-disease classification" accurately describes the dataset target.
- [ ] Confirm the venue's current template, word/page limits, reference style, and supplementary-file rules; this package uses a generic two-column LaTeX layout.
- [ ] Review all metrics against the source CSVs and retain the dataset hash and software versions.
- [ ] Treat the five-seed Wilcoxon results as exploratory; no multiplicity correction was applied and seeds are not independent patients.
- [ ] Do not describe the engineered score as a validated physiological mechanism, causal pathway, diagnosis, or treatment recommendation.
- [ ] Add external validation and a protocol-matched baseline before making generalization or superiority claims.
