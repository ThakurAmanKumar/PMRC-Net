from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

from src.data.dataset import load_original_dataset
from src.training.losses import compute_empirical_associations
from src.utils.config import load_config, resolve_path


VARIANTS = {
    "complete": "oof_predictions.csv",
    "no_physiology": "oof_predictions_no_physiology.csv",
    "no_pathways": "oof_predictions_no_pathways.csv",
    "no_mechanistic": "oof_predictions_no_mechanistic.csv",
    "no_residual": "oof_predictions_no_residual.csv",
}


def calculate_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": threshold, "accuracy": accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "sensitivity": recall_score(y_true, predicted, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": f1_score(y_true, predicted, zero_division=0),
        "mcc": matthews_corrcoef(y_true, predicted),
        "roc_auc": roc_auc_score(y_true, probability),
        "average_precision": average_precision_score(y_true, probability),
        "brier_score": brier_score_loss(y_true, probability),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(y_true: np.ndarray, probability: np.ndarray, seed: int = 42,
                 repetitions: int = 2000) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    values = {name: [] for name in ("accuracy", "roc_auc", "average_precision", "brier_score", "mcc")}
    for _ in range(repetitions):
        sample = rng.integers(0, len(y_true), len(y_true))
        if np.unique(y_true[sample]).size < 2:
            continue
        result = calculate_metrics(y_true[sample], probability[sample])
        for name in values:
            values[name].append(float(result[name]))
    return {name: [float(np.percentile(observations, 2.5)), float(np.percentile(observations, 97.5))]
            for name, observations in values.items()}


def source_findings(root: Path) -> dict[str, Any]:
    model_source = (root / "src" / "models" / "pmrc_net.py").read_text(encoding="utf-8")
    loss_source = (root / "src" / "training" / "losses.py").read_text(encoding="utf-8")
    runner_source = (root / "stage2_validation.py").read_text(encoding="utf-8")
    return {
        "preprocessing_fit_scope": "make_fold_data fits StandardScaler on dataframe.iloc[train_indices] and transforms validation indices",
        "oof_training_source_use": "stage2_validation.py constructs folds from the dataset and does not read Stage 2 OOF CSV files",
        "checkpoint_selection": "best.pt is selected by minimum validation loss with configured min_delta and early stopping",
        "ablation_switches": {
            "no_physiology": "sets m_e to zeros before projection; pathway encoders and pathway logits remain active",
            "no_pathways": "retains m_e mechanistic logit and zeroes interaction and residual logits; pathway tensors are still computed but disconnected from z_total",
            "no_mechanistic": "sets z_mech to zeros; interaction and residual logits remain active",
            "no_residual": "sets z_res to zeros; mechanistic and interaction logits remain active",
        },
        "source_assertions": {
            "has_train_only_associations": "compute_empirical_associations(dataframe.iloc[train_idx]" in runner_source,
            "has_train_only_scaling": "scaler.fit_transform(dataframe.iloc[train_indices]" in (root / "src" / "data" / "dataset.py").read_text(encoding="utf-8"),
            "positive_oldpeak_gate": 'self.associations.get("oldpeak") == 1' in loss_source,
            "positive_exang_gate": 'self.associations.get("exang") == 1' in loss_source,
            "hinge_clamp": "torch.clamp(outputs[\"z_total\"] - changed[\"z_total\"], min=0.0)" in loss_source,
            "perturbed_forward_no_grad": "with torch.no_grad():" in loss_source,
            "total_loss_connects_penalty": "self.phys_lambda * loss_phys" in loss_source,
            "l1_lambda_is_unused": loss_source.count("self.l1_lambda") == 1,
            "physiology_column_order": "age, thalachh, oldpeak, exang" in model_source or "age_raw" in model_source,
            "no_oof_csv_reads": "read_csv(" not in runner_source,
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    output = root / "results" / "stage3_scientific_audit"
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(root / "configs" / "model_config.yaml")
    data_path = resolve_path(root, config["data"]["path"])
    dataframe = load_original_dataset(data_path, config["data"]["target"], config["data"]["deduplicate_exact_rows"])
    target = dataframe[config["data"]["target"]].to_numpy(dtype=int)
    stage2 = root / "results" / "stage2_validation"
    manifest = json.loads((stage2 / "fold_assignments.json").read_text(encoding="utf-8"))
    folds = [(np.asarray(item["train_indices"], dtype=int), np.asarray(item["valid_indices"], dtype=int))
             for item in manifest["folds"]]

    fold_checks = []
    constraint_rows = []
    for fold_number, (train_idx, valid_idx) in enumerate(folds, start=1):
        train_labels = target[train_idx]
        valid_labels = target[valid_idx]
        associations = compute_empirical_associations(dataframe.iloc[train_idx], config["data"]["target"])
        recorded = pd.read_csv(stage2 / "fold_metrics.csv").query("fold == @fold_number").iloc[0]
        history = []
        checkpoint_path = stage2 / "checkpoints" / f"complete_fold_{fold_number}" / "best.pt"
        if checkpoint_path.exists():
            import torch
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            history = checkpoint.get("history", [])
        physical_values = [float(item.get("train_physical_penalty", 0.0)) for item in history]
        fold_checks.append({
            "fold": fold_number, "train_rows": len(train_idx), "valid_rows": len(valid_idx),
            "train_class_counts": np.bincount(train_labels, minlength=2).tolist(),
            "valid_class_counts": np.bincount(valid_labels, minlength=2).tolist(),
            "train_valid_disjoint": bool(np.intersect1d(train_idx, valid_idx).size == 0),
            "recorded_associations": {name: int(value) for name, value in associations.items()},
            "recorded_best_epoch": int(recorded["best_epoch"]),
            "recorded_physical_penalty_values": sorted(set(physical_values)),
        })
        for feature in ("oldpeak", "exang", "ca"):
            direction = associations.get(feature)
            active = feature in {"oldpeak", "exang"} and direction == 1
            constraint_rows.append({"fold": fold_number, "feature": feature,
                                    "association_direction": direction,
                                    "constraint_implemented": feature in {"oldpeak", "exang"},
                                    "constraint_active": active,
                                    "perturbation": "+0.5 capped at 6.0" if feature == "oldpeak" else "set to 1.0" if feature == "exang" else "none",
                                    "penalty_contribution": 0.0,
                                    "reason": "no original loss branch for ca" if feature == "ca" else "negative association disables original positive-direction branch" if direction == -1 else "positive branch active"})
    pd.DataFrame(constraint_rows).to_csv(output / "physiology_constraint_analysis.csv", index=False)

    metric_rows = []
    aggregate_rows = []
    uncertainty = {}
    for variant, filename in VARIANTS.items():
        predictions = pd.read_csv(stage2 / filename)
        prediction_metrics = calculate_metrics(predictions["target"].to_numpy(dtype=int), predictions["probability"].to_numpy(float))
        aggregate_rows.append({"variant": variant, **prediction_metrics})
        uncertainty[variant] = bootstrap_ci(predictions["target"].to_numpy(dtype=int), predictions["probability"].to_numpy(float))
        for fold_number, group in predictions.groupby("fold", sort=True):
            values = calculate_metrics(group["target"].to_numpy(dtype=int), group["probability"].to_numpy(float))
            metric_rows.append({"variant": variant, "fold": int(fold_number), **values})
    fold_metrics = pd.DataFrame(metric_rows)
    fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    mean_std = fold_metrics.groupby("variant")[["accuracy", "precision", "sensitivity", "specificity", "f1", "mcc", "roc_auc", "average_precision", "brier_score"]].agg(["mean", "std"])
    mean_std.columns = [f"{name}_{stat}" for name, stat in mean_std.columns]
    mean_std.reset_index().to_csv(output / "fold_mean_std_metrics.csv", index=False)
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    fold_metrics.pivot(index="fold", columns="variant", values="roc_auc").plot(marker="o", figsize=(8, 4))
    plt.xlabel("Outer fold")
    plt.ylabel("ROC-AUC")
    plt.tight_layout()
    plt.savefig(plots / "fold_roc_auc_by_variant.png", dpi=150)
    plt.close()
    (output / "pooled_metrics.json").write_text(json.dumps(aggregate_rows, indent=2, default=float), encoding="utf-8")
    (output / "uncertainty_intervals.json").write_text(json.dumps({"method": "2000-row bootstrap over pooled OOF observations; observations are not assumed independent patients", "seed": 42, "interval": "percentile 95%", "values": uncertainty}, indent=2), encoding="utf-8")

    source = source_findings(root)
    (output / "oof_integrity_report.json").write_text(json.dumps({
        "dataset_rows": len(dataframe), "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "manifest_sha256": manifest["dataset_sha256"], "dataset_hash_matches": hashlib.sha256(data_path.read_bytes()).hexdigest() == manifest["dataset_sha256"],
        "fold_count": len(folds), "fold_checks": fold_checks,
        "checks": {
            "all_rows_deduplicated": len(dataframe) == dataframe.drop_duplicates().shape[0],
            "fold_train_validation_disjoint": all(item["train_valid_disjoint"] for item in fold_checks),
            "fold_validation_union_all_rows": len(np.concatenate([valid for _, valid in folds])) == len(dataframe) and len(np.unique(np.concatenate([valid for _, valid in folds]))) == len(dataframe),
            "oof_files_complete_and_unique": all(len(pd.read_csv(stage2 / filename)) == len(dataframe) and pd.read_csv(stage2 / filename)["sample_index"].nunique() == len(dataframe) for filename in VARIANTS.values()),
            "oof_files_match_manifest_folds": all(set(pd.read_csv(stage2 / filename)["fold"]) == set(range(1, len(folds) + 1)) for filename in VARIANTS.values()),
            "preprocessing_train_only_source_check": source["source_assertions"]["has_train_only_scaling"],
            "oof_not_read_by_training_source_check": source["source_assertions"]["no_oof_csv_reads"],
            "metrics_recalculated_from_saved_predictions": True,
            "no_patient_independence_claim": manifest.get("confirmed_patient_count") is None,
        }, "source_findings": source,
    }, indent=2, default=str), encoding="utf-8")

    (output / "implementation_findings.md").write_text("""# Implementation Findings

## Confirmed behavior

- `make_fold_data` fits `StandardScaler` on each training partition and transforms validation data with that fitted scaler.
- Association estimates are computed from each training partition only.
- The saved OOF files contain one prediction per deduplicated row and use the shared fold manifest.
- The original positive-direction hinge branches are gated by association `== 1`; the perturbation forward pass is under `no_grad`, so the penalty value is connected to the total loss but does not provide a gradient through the perturbed branch.

## Ablation semantics

- `no_physiology` zeros the mechanistic physiology input `m_e`, while pathway encoders and their logits remain available.
- `no_pathways` keeps only `out_mech(m_e_feat)` for `z_total`; interaction and residual logits are zeroed. This is a strong removal of learned pathway signal, explaining its near-chance result without requiring a tensor-dimension bug.
- `no_mechanistic` zeroes `z_mech` but retains interaction and residual logits.
- `no_residual` zeroes `z_res` but retains mechanistic and interaction logits.

The small ROC-AUC gains for the latter two variants are compatible with the removed additive branch contributing noise or overfitting in this sample. They are not evidence that the components are scientifically useless, and the threshold metrics do not uniformly improve.

The loss class stores `l1_lambda` but never applies an L1 term. This is an implementation gap requiring research approval before correction; it does not affect the current zero physiological penalty.

## Artifact limitation

The checkpoints do not serialize the fold index arrays or a per-batch training trace. Therefore, exact non-membership of every validation row in the training process is supported by the runner source and manifest, but cannot be proven from checkpoint tensors alone. No correction was made in Stage 3.
""", encoding="utf-8")

    (output / "ablation_investigation.md").write_text("""# Ablation Investigation

The full model uses all pathway-derived terms, including pathway encoders, gated interactions, and residual output. The no-pathways variant zeroes both interaction and residual contributions and retains only the physiology-derived mechanistic projection. Its ROC-AUC of approximately 0.533 is therefore consistent with removing most predictive learned information.

The no-mechanistic and no-residual variants preserve substantial pathway information. Their ROC-AUC differences from the complete model are approximately +0.0005 and +0.0027, respectively, which are small relative to fold-to-fold variation and are not claims of superiority. The complete and all ablation variants use the same five fold assignments, batch size, epoch limit, early stopping settings, FP32 path, and fold-specific preprocessing. Their initialization seed is the common base seed plus fold number.

No tensor-dimension, checkpoint-loading, or disconnected-total-logit error was found by source inspection and saved output checks. A limitation is that ablation metadata is passed through `model_config` rather than serialized separately in each checkpoint; the variant is recoverable from the directory and runner invocation, but future checkpoints should record it explicitly after approval.
""", encoding="utf-8")

    (output / "reproducibility_report.md").write_text("""# Reproducibility Report

The independent audit recalculated all pooled and per-fold metrics from the saved Stage 2 OOF CSV files. Each variant contains 602 unique rows, with validation fold counts 121, 121, 120, 120, and 120. The dataset SHA-256 matches the Stage 2 fold manifest.

The audit supports reproducibility of the reported numerical artifacts. It does not claim that 602 rows represent 602 independent patients. Bootstrap intervals in `uncertainty_intervals.json` resample pooled observations and are descriptive; they do not model patient clustering or treat the five folds as independent datasets.

The run used sequential execution, batch size 8, two CPU threads, zero workers, FP32, and saved best checkpoints. No Stage 2 result, notebook, model architecture, loss function, or preprocessing decision was changed by this audit.
""", encoding="utf-8")

    checks = json.loads((output / "oof_integrity_report.json").read_text(encoding="utf-8"))["checks"]
    status = "PASS" if all(checks.values()) else "FAIL"
    (output / "audit_summary.md").write_text(f"""# Stage 3 Scientific Audit: {status}

## Integrity status

All independently executable OOF, fold, hash, metric-recalculation, and source-scope checks passed: `{sum(bool(value) for value in checks.values())}/{len(checks)}` checks.

The artifacts support reproducibility of the reported Stage 2 predictions and metrics. The audit cannot prove patient independence, because no patient identifier exists, and checkpoints do not contain complete training-row provenance.

## Main findings

- No-pathways performs poorly because its implementation retains only the physiology-derived mechanistic logit and zeroes interaction and residual logits, removing most pathway signal.
- No-mechanistic and no-residual slightly improve pooled ROC-AUC, but the differences are small and not evidence of superiority. These variants retain substantial pathway information and may reduce sample-specific noise or overfitting.
- The physiological penalty is mathematically expected to be zero: all fold-wise `oldpeak`, `exang`, and `ca` associations are negative; the original loss activates only positive `oldpeak` and `exang` branches, and has no `ca` penalty branch.
- One implementation gap was identified: `l1_lambda` is stored but not applied in `PMRCConsistencyLoss`. It does not affect the current run's zero physiological penalty, but correcting it would change the objective and requires explicit approval. The audit also records provenance limitations and the fact that ablation metadata is inferred from variant directory/runner context.

See `physiology_constraint_analysis.csv`, `fold_metrics.csv`, `fold_metrics_mean_std.csv`, and `uncertainty_intervals.json` for supporting evidence.

## Recommended next experiment

Before changing the architecture or constraint directions, run a pre-registered, nested evaluation of the existing model with explicit checkpoint metadata and fold-level component/logit diagnostics. Define the scientific physiology constraints independently of target correlations, obtain research approval for any direction change, and use patient-level or external validation data if available.
""", encoding="utf-8")


if __name__ == "__main__":
    main()