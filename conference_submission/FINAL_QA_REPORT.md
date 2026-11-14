# Package Quality Assurance

## Automated checks

- Required core files: PASS (13 checked).
- Figures: PASS (8 figures; PNG and PDF versions present).
- Tables: PASS (5 CSV tables, 5 LaTeX fragments, 5 captions).
- Citation links: PASS (15 distinct cited keys resolve to both the manuscript bibliography and BibTeX entries).
- DOCX archive integrity: PASS.
- PDF presence/signature/size: PASS.
- PDF rendering/text extraction: PASS (10 pages; 25899 extracted characters).
- Editable DOCX rendering: PASS (opened in Microsoft Word; 11 pages and approximately 3,043 words).
- Nested experiment artifacts: PASS (run status `COMPLETE`; complete run configuration and dataset hash included).
- Submission privacy: PASS (no row-level OOF predictions, raw dataset, or model checkpoints copied into the package).

## Scientific and reproducibility checks

- Historical-baseline protocol mismatch is stated in the abstract, results, audit, supplement, figure captions, reviewer critique, and checklist.
- Baseline paired tests are retained only as nominal historical values and explicitly invalidated as controlled inference.
- No superiority or clinical-readiness claim is made.
- The `ca=4` mapping, unused `l1_lambda`, unused `ca` hinge association, and limited activation of the physiology hinge are disclosed.
- Five-seed uncertainty, shared rows/overlapping folds, no external validation, and unknown patient-level independence are disclosed.
- All performance values originate in saved experiment outputs. No metric is generated from target labels in a way that substitutes for the run's scored predictions.

## Items requiring author completion

Ethics/consent, dataset source and license, target meaning, patient independence, affiliations, author contributions, funding/conflicts, intended venue template, and external validation remain unresolved. These are not inferred by automated QA.

## Summary

Package verification result: **PASS**. This generic-format package should not be submitted until author confirmations and venue-format checks are complete.
