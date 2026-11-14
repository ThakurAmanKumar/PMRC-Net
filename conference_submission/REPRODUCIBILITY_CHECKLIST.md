# Reproducibility Checklist

- [x] Dataset path and SHA-256 recorded.
- [x] Exact duplicate-row handling and fold-local continuous standardization described.
- [x] Seeds, outer/inner fold counts, candidate grid, selection score, calibration and threshold rules recorded.
- [x] Hardware, key software versions, and resource summary included.
- [x] All 25 outer folds per nested arm completed; aggregate fold and seed metrics included.
- [x] Eight figures, five table CSVs, and matching LaTeX table fragments generated from saved experiment outputs.
- [x] Editable manuscript sources provided as Markdown, LaTeX, and DOCX.
- [x] Historical-baseline protocol mismatch and invalidity of nominal baseline tests disclosed.
- [x] Row-level OOF predictions excluded from the submission package.
- [ ] Author verification of dataset provenance, ethics, rights, target meaning, patient independence, and declarations.
- [ ] Venue-specific template/page-limit check.
- [ ] External validation and nested protocol-matched baseline are not part of this experiment.

Run the generator from the repository root with `.venv\Scripts\python.exe scripts\generate_conference_submission.py`. The paper-generation additions require `python-docx`; `pypdf` is needed for `--require-pdf` validation. Versions used here are listed in `supplementary/requirements-paper.txt`. Compile the PDF from `conference_submission\manuscript` with a current Tectonic or LaTeX installation. The generator refuses to replace an existing package unless `--force` is provided and the package's own generation manifest exists.
