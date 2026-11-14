# PMRC-Net Research

This local project preserves the PMRC-Net architecture extracted from the original Colab notebook. The original notebook, dataset, and existing `results/` artifacts are unchanged.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the PyTorch wheel recommended by the [official PyTorch selector](https://pytorch.org/get-started/locally/) for the installed NVIDIA driver if the generic requirement does not provide CUDA support.

## Setup test (does not train)

```powershell
python test_setup.py
```

This checks imports, original-data loading and exact deduplication, model initialization, forward propagation, one backward batch, checkpoint round-trip, CUDA detection, and the conservative runtime settings.

## Training

Training is manual and never starts when VS Code opens:

```powershell
python train.py
```

The default configuration uses batch size 8, CPU thread count 2, zero DataLoader workers, sequential five-fold execution, checkpointing, early stopping, and CPU fallback when CUDA is unavailable. To resume, pass a specific `fold_N_latest.pt` checkpoint to a future resume workflow; no automatic training or background process is created.

## Evaluation

```powershell
python evaluate.py --checkpoint checkpoints/<experiment-id>/fold_1_best.pt
```

Evaluation reports threshold-0.5 metrics by default and writes JSON plus a confusion matrix under `reports/evaluation`. A validation-selected threshold must be supplied explicitly with `--threshold`; it is not mixed with the fixed 0.5 result.

## Scientific safeguards

The original CSV is deduplicated to 602 exact unique records before cross-validation because the audit reports 1,286 exact duplicates. Scaling is fitted only on each training fold. Fold assignments and sample indices are saved with each experiment. The dataset has no patient-level identifier, so 602 rows are not described as 602 confirmed independent patients.

## Final research package

The manuscript is at `paper/PMRC_Net_Research_Paper.md`. Audits, figures, tables, external-dataset screening, and completion status are indexed in `reports/FINAL_DELIVERABLES_INDEX.md`. Five-seed validation is summarized in `results/stage5_final_validation/multi_seed_summary.csv`.

Independent external validation is not complete: UCI candidates may overlap the merged development source and were not evaluated. The project is suitable for internal research review, not clinical use or diagnostic decisions. See `ENVIRONMENT.md` for the local environment record.
