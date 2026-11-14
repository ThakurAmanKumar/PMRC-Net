# Experimental Methodology

The analysis uses 602 exact-deduplicated observations. They are not described as 602 confirmed independent patients because the dataset has no patient identifier.

Five stratified outer folds are generated once with seed 42 and saved in `results/stage2_validation/fold_assignments.json`. Every model uses those assignments. Scaling is fitted only on each outer training partition. PMRC-Net association estimates are also computed only on that partition.

OOF probabilities are pooled before calculating aggregate metrics. Average precision is reported as scikit-learn average precision, which is distinct from trapezoidal PR-AUC. Threshold 0.5 is fixed for the primary classification metrics; threshold sensitivity is reported separately.

Training is sequential, FP32, batch size 8, two CPU threads, zero workers, and one active process. Fold checkpoints contain model state, optimizer state, epoch, configuration, and training history.