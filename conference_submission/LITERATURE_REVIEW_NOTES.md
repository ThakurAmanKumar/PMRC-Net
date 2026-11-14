# Literature Review Notes and Source Verification

## Evidence framing

This review uses literature to position the work, not to claim architectural novelty or clinical validity. PMRC-Net is evaluated on one small, merged, deduplicated tabular dataset with unknown patient-level independence and no external validation. The named feature groups are an inductive-bias choice; they are not validated biological pathways, causal structure, or a transparent explanation of an individual prediction.

Tabular neural-network studies motivate careful model selection and strong, well-matched comparisons rather than assuming deep architectures outperform other model families. Clinical prediction and reporting references motivate transparent description of data provenance, calibration, validation, and limitations. Cross-validation references support isolating model/epoch selection from outer test data. Calibration references justify reporting proper scoring/calibration diagnostics separately from discrimination. The present five-seed comparison is exploratory and does not inherit the statistical power of evaluations across independent datasets.

## Verified references

- **Arik, S. O., and Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning.** Proceedings of the AAAI Conference on Artificial Intelligence, 35(8), 6679-6687. https://doi.org/10.1609/aaai.v35i8.16826 — A representative attentive tabular neural architecture; useful context, not a direct benchmark.
- **Cawley, G. C., and Talbot, N. L. C. (2010). On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation.** Journal of Machine Learning Research, 11, 2079-2107. https://www.jmlr.org/papers/v11/cawley10a.html — Model selection can bias reported performance; supports the nested-protocol requirement.
- **Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets.** Journal of Machine Learning Research, 7, 1-30. https://www.jmlr.org/papers/v7/demsar06a.html — Statistical comparison guidance; its multi-dataset setting does not make five repeated seeds independent.
- **Gorishniy, Y., Rubachev, I., Khrulkov, V., and Babenko, A. (2021). Revisiting Deep Learning Models for Tabular Data.** Advances in Neural Information Processing Systems, 34. https://arxiv.org/abs/2106.11959 — Tabular neural-network benchmarking and the importance of appropriate baselines.
- **Grinsztajn, L., Oyallon, E., and Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on tabular data?.** arXiv preprint arXiv:2207.08815. https://arxiv.org/abs/2207.08815 — Empirical limitations of deep learning on small/medium tabular benchmarks; cited as an arXiv preprint.
- **Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks.** Proceedings of the 34th International Conference on Machine Learning, PMLR 70, 1321-1330. https://proceedings.mlr.press/v70/guo17a.html — Post-hoc temperature scaling for neural-network calibration.
- **Rajkomar, A., Oren, E., Chen, K., et al. (2018). Scalable and accurate deep learning with electronic health records.** npj Digital Medicine, 1, 18. https://doi.org/10.1038/s41746-018-0029-1 — Clinical EHR deep-learning context; not directly comparable to this small merged tabular dataset.
- **Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead.** Nature Machine Intelligence, 1, 206-215. https://doi.org/10.1038/s42256-019-0048-x — Interpretability cautions; named feature groups do not make PMRC-Net inherently interpretable.
- **Saito, T., and Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.** PLOS ONE, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432 — Precision-recall metrics can add information when evaluating imbalanced binary classifiers.
- **Tanaka, H., Monahan, K. D., and Seals, D. R. (2001). Age-predicted maximal heart rate revisited.** Journal of the American College of Cardiology, 37(1), 153-156. https://doi.org/10.1016/S0735-1097(00)01054-8 — Source for the age-predicted maximal-heart-rate equation used as one engineered term.
- **Collins, G. S., Reitsma, J. B., Altman, D. G., and Moons, K. G. M. (2015). Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD): The TRIPOD Statement.** Annals of Internal Medicine, 162(1), 55-63. https://doi.org/10.7326/M14-0697 — Reporting guidance for prediction-model studies.
- **Varma, S., and Simon, R. (2006). Bias in error estimation when using cross-validation for model selection.** BMC Bioinformatics, 7, 91. https://doi.org/10.1186/1471-2105-7-91 — Cross-validation selection bias and the need for nested selection.
- **Varoquaux, G. (2018). Cross-validation failure: Small sample sizes lead to large error bars.** NeuroImage, 180, 68-77. https://doi.org/10.1016/j.neuroimage.2017.06.061 — Uncertainty from small sample sizes and repeated cross-validation.
- **Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L., and Steyerberg, E. W. (2019). Calibration: the Achilles heel of predictive analytics.** BMC Medicine, 17, 230. https://doi.org/10.1186/s12916-019-1466-7 — Calibration is distinct from discrimination and should be assessed explicitly.
- **Weng, S. F., Reps, J., Kai, J., Garibaldi, J. M., and Qureshi, N. (2017). Can machine-learning improve cardiovascular risk prediction using routine clinical data?.** PLOS ONE, 12(4), e0174944. https://doi.org/10.1371/journal.pone.0174944 — Cardiovascular prediction context using routine clinical data; population and protocol differ.

## Source-level verification record

- Publisher/proceedings records or DOI/Crossref metadata were checked for the journal and conference sources listed above.
- The TabNet article was verified against its AAAI proceedings record; the calibration article against PMLR; the JMLR records against the official JMLR pages; the heart-rate equation against Crossref/Elsevier DOI metadata; and the PLOS/clinical sources against DOI metadata.
- `Grinsztajn et al.` is intentionally cited as the arXiv preprint `arXiv:2207.08815`; no unverified venue claim is made.
- The bibliography is kept in `references.bib`; article text makes no long verbatim quotations.
