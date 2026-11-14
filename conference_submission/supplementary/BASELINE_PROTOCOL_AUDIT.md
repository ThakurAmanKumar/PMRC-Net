# Historical Baseline Protocol Audit

The historical Stage-5 baseline shares the deduplicated row ordering, targets, seeds, and outer fold assignments with the nested run. This alignment is useful for describing historical outputs, but it does not establish an unbiased head-to-head comparison.

`results/stage5_final_validation/stage5_validation.py` selects its best model/checkpoint using the fold-specific validation loader. That loader represents the same fold subsequently treated as the outer test fold in the saved OOF predictions. Consequently, outer-fold labels influenced baseline training/model selection. The baseline also uses the original `PMRCNet`, a fixed 0.5 threshold, and no inner-OOF temperature calibration or threshold selection.

Therefore:

- Show historical metrics only with a clear protocol-mismatch label.
- Do not claim the nested arms outperform, underperform, or are equivalent to this baseline.
- Do not interpret baseline paired p-values.
- A controlled baseline requires rerunning a baseline architecture under the same nested outer/inner splits, inner-only selection, calibration/threshold fitting, and one-time outer test evaluation.
