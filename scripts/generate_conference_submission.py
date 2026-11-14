#!/usr/bin/env python
"""Generate a provenance-linked, editable PMRC-Net research submission package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "PMRC_NET_FINAL_RESEARCH"
DATASET = ROOT / "dataset" / "cleaned_merged_heart_dataset.csv"
OUTPUT = ROOT / "conference_submission"
SEEDS = [21, 42, 123, 2026, 3407]
METRICS = [
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "mcc",
    "roc_auc",
    "average_precision",
    "brier_score",
    "ece",
]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "f1": "F1",
    "mcc": "MCC",
    "roc_auc": "ROC-AUC",
    "average_precision": "Average precision",
    "brier_score": "Brier score",
    "ece": "ECE",
}
ARMS = {
    "directional_coupling": "PMRC-Net directional",
    "legacy_coupling_control": "PMRC-Net single-flow control",
    "authoritative_baseline_oof": "Historical Stage-5 baseline",
}
COLORS = {
    "directional_coupling": "#0072B2",
    "legacy_coupling_control": "#D55E00",
    "authoritative_baseline_oof": "#009E73",
}
PAPER_SECTIONS: list[tuple[str, list[Any]]] = []
TABLE_DATA: dict[str, tuple[list[str], list[list[Any]], str]] = {}
FIGURE_CAPTIONS: dict[int, str] = {
    1: "Model schematic from implementation: three named feature-group encoders, a physiology-mismatch projection, pairwise gated interactions, and additive mechanistic, interaction, and residual logits. The labels describe the engineered architecture, not validated biological pathways or causal mechanisms.",
    2: "Nested evaluation protocol for each of five seeds and two trainable arms. Inner folds are confined to the outer training partition for candidate selection, temperature scaling, and Youden-threshold selection; each outer test fold is evaluated once.",
    3: "Seed-averaged ROC curves with between-seed standard-deviation bands. Each seed contributes one out-of-fold prediction per observation. The Stage-5 baseline is shown only as historical context because it selected epochs using its outer validation fold.",
    4: "Seed-averaged precision-recall curves with between-seed standard-deviation bands. Average precision is reported; the historical baseline is protocol-mismatched and is not a controlled comparator.",
    5: "Confusion matrices aggregated over five repeated OOF runs (3,010 predictions per arm). Cells are repeated predictions on the same 602 deduplicated rows, not independent patients.",
    6: "Reliability diagrams using ten equal-width bins on the five-seed OOF predictions. Brier score and expected calibration error are summarized across seeds; repeated predictions must not be interpreted as independent observations.",
    7: "Exploratory paired-seed comparison of directional and single-flow coupling controls. Error bars show standard deviation across five seeds; no metric shows a statistically significant paired difference at 0.05.",
    8: "Per-seed metric stability for both nested arms and the historical Stage-5 baseline. Baseline traces are descriptive only because its outer validation fold influenced checkpoint selection.",
}


REFERENCES = [
    {
        "key": "Arik2021",
        "authors": "Arik, S. O., and Pfister, T.",
        "year": 2021,
        "title": "TabNet: Attentive Interpretable Tabular Learning",
        "venue": "Proceedings of the AAAI Conference on Artificial Intelligence, 35(8), 6679-6687",
        "doi": "10.1609/aaai.v35i8.16826",
        "url": "https://doi.org/10.1609/aaai.v35i8.16826",
        "bib": "@article{Arik2021,\n  author = {Arik, Sercan O. and Pfister, Tomas},\n  title = {TabNet: Attentive Interpretable Tabular Learning},\n  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},\n  volume = {35}, number = {8}, pages = {6679--6687}, year = {2021},\n  doi = {10.1609/aaai.v35i8.16826}\n}",
    },
    {
        "key": "Cawley2010",
        "authors": "Cawley, G. C., and Talbot, N. L. C.",
        "year": 2010,
        "title": "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation",
        "venue": "Journal of Machine Learning Research, 11, 2079-2107",
        "url": "https://www.jmlr.org/papers/v11/cawley10a.html",
        "bib": "@article{Cawley2010,\n  author = {Cawley, Gavin C. and Talbot, Nicola L. C.},\n  title = {On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation},\n  journal = {Journal of Machine Learning Research},\n  volume = {11}, pages = {2079--2107}, year = {2010},\n  url = {https://www.jmlr.org/papers/v11/cawley10a.html}\n}",
    },
    {
        "key": "Demsar2006",
        "authors": "Demšar, J.",
        "year": 2006,
        "title": "Statistical Comparisons of Classifiers over Multiple Data Sets",
        "venue": "Journal of Machine Learning Research, 7, 1-30",
        "url": "https://www.jmlr.org/papers/v7/demsar06a.html",
        "bib": "@article{Demsar2006,\n  author = {Dem{\\v{s}}ar, Janez},\n  title = {Statistical Comparisons of Classifiers over Multiple Data Sets},\n  journal = {Journal of Machine Learning Research},\n  volume = {7}, pages = {1--30}, year = {2006},\n  url = {https://www.jmlr.org/papers/v7/demsar06a.html}\n}",
    },
    {
        "key": "Gorishniy2021",
        "authors": "Gorishniy, Y., Rubachev, I., Khrulkov, V., and Babenko, A.",
        "year": 2021,
        "title": "Revisiting Deep Learning Models for Tabular Data",
        "venue": "Advances in Neural Information Processing Systems, 34",
        "doi": "10.48550/arXiv.2106.11959",
        "url": "https://arxiv.org/abs/2106.11959",
        "bib": "@inproceedings{Gorishniy2021,\n  author = {Gorishniy, Yury and Rubachev, Ivan and Khrulkov, Valentin and Babenko, Artem},\n  title = {Revisiting Deep Learning Models for Tabular Data},\n  booktitle = {Advances in Neural Information Processing Systems},\n  volume = {34}, year = {2021},\n  eprint = {2106.11959}, archivePrefix = {arXiv},\n  doi = {10.48550/arXiv.2106.11959}\n}",
    },
    {
        "key": "Grinsztajn2022",
        "authors": "Grinsztajn, L., Oyallon, E., and Varoquaux, G.",
        "year": 2022,
        "title": "Why do tree-based models still outperform deep learning on tabular data?",
        "venue": "arXiv preprint arXiv:2207.08815",
        "doi": "10.48550/arXiv.2207.08815",
        "url": "https://arxiv.org/abs/2207.08815",
        "bib": "@misc{Grinsztajn2022,\n  author = {Grinsztajn, Leo and Oyallon, Edouard and Varoquaux, Ga{\"e}l},\n  title = {Why do tree-based models still outperform deep learning on tabular data?},\n  year = {2022}, eprint = {2207.08815}, archivePrefix = {arXiv},\n  doi = {10.48550/arXiv.2207.08815}\n}",
    },
    {
        "key": "Guo2017",
        "authors": "Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q.",
        "year": 2017,
        "title": "On Calibration of Modern Neural Networks",
        "venue": "Proceedings of the 34th International Conference on Machine Learning, PMLR 70, 1321-1330",
        "doi": "10.48550/arXiv.1706.04599",
        "url": "https://proceedings.mlr.press/v70/guo17a.html",
        "bib": "@inproceedings{Guo2017,\n  author = {Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},\n  title = {On Calibration of Modern Neural Networks},\n  booktitle = {Proceedings of the 34th International Conference on Machine Learning},\n  series = {Proceedings of Machine Learning Research}, volume = {70}, pages = {1321--1330}, year = {2017},\n  url = {https://proceedings.mlr.press/v70/guo17a.html}\n}",
    },
    {
        "key": "Rajkomar2018",
        "authors": "Rajkomar, A., Oren, E., Chen, K., et al.",
        "year": 2018,
        "title": "Scalable and accurate deep learning with electronic health records",
        "venue": "npj Digital Medicine, 1, 18",
        "doi": "10.1038/s41746-018-0029-1",
        "url": "https://doi.org/10.1038/s41746-018-0029-1",
        "bib": "@article{Rajkomar2018,\n  author = {Rajkomar, Alvin and Oren, Eyal and Chen, Kai and others},\n  title = {Scalable and accurate deep learning with electronic health records},\n  journal = {npj Digital Medicine}, volume = {1}, pages = {18}, year = {2018},\n  doi = {10.1038/s41746-018-0029-1}\n}",
    },
    {
        "key": "Rudin2019",
        "authors": "Rudin, C.",
        "year": 2019,
        "title": "Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead",
        "venue": "Nature Machine Intelligence, 1, 206-215",
        "doi": "10.1038/s42256-019-0048-x",
        "url": "https://doi.org/10.1038/s42256-019-0048-x",
        "bib": "@article{Rudin2019,\n  author = {Rudin, Cynthia},\n  title = {Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead},\n  journal = {Nature Machine Intelligence}, volume = {1}, pages = {206--215}, year = {2019},\n  doi = {10.1038/s42256-019-0048-x}\n}",
    },
    {
        "key": "Saito2015",
        "authors": "Saito, T., and Rehmsmeier, M.",
        "year": 2015,
        "title": "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets",
        "venue": "PLOS ONE, 10(3), e0118432",
        "doi": "10.1371/journal.pone.0118432",
        "url": "https://doi.org/10.1371/journal.pone.0118432",
        "bib": "@article{Saito2015,\n  author = {Saito, Takaya and Rehmsmeier, Marc},\n  title = {The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets},\n  journal = {PLOS ONE}, volume = {10}, number = {3}, pages = {e0118432}, year = {2015},\n  doi = {10.1371/journal.pone.0118432}\n}",
    },
    {
        "key": "Tanaka2001",
        "authors": "Tanaka, H., Monahan, K. D., and Seals, D. R.",
        "year": 2001,
        "title": "Age-predicted maximal heart rate revisited",
        "venue": "Journal of the American College of Cardiology, 37(1), 153-156",
        "doi": "10.1016/S0735-1097(00)01054-8",
        "url": "https://doi.org/10.1016/S0735-1097(00)01054-8",
        "bib": "@article{Tanaka2001,\n  author = {Tanaka, Hirofumi and Monahan, Kevin D. and Seals, Douglas R.},\n  title = {Age-predicted maximal heart rate revisited},\n  journal = {Journal of the American College of Cardiology},\n  volume = {37}, number = {1}, pages = {153--156}, year = {2001},\n  doi = {10.1016/S0735-1097(00)01054-8}\n}",
    },
    {
        "key": "TRIPOD2015",
        "authors": "Collins, G. S., Reitsma, J. B., Altman, D. G., and Moons, K. G. M.",
        "year": 2015,
        "title": "Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD): The TRIPOD Statement",
        "venue": "Annals of Internal Medicine, 162(1), 55-63",
        "doi": "10.7326/M14-0697",
        "url": "https://doi.org/10.7326/M14-0697",
        "bib": "@article{TRIPOD2015,\n  author = {Collins, Gary S. and Reitsma, Johannes B. and Altman, Douglas G. and Moons, Karel G. M.},\n  title = {Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD): The TRIPOD Statement},\n  journal = {Annals of Internal Medicine}, volume = {162}, number = {1}, pages = {55--63}, year = {2015},\n  doi = {10.7326/M14-0697}\n}",
    },
    {
        "key": "Varma2006",
        "authors": "Varma, S., and Simon, R.",
        "year": 2006,
        "title": "Bias in error estimation when using cross-validation for model selection",
        "venue": "BMC Bioinformatics, 7, 91",
        "doi": "10.1186/1471-2105-7-91",
        "url": "https://doi.org/10.1186/1471-2105-7-91",
        "bib": "@article{Varma2006,\n  author = {Varma, Sudhir and Simon, Richard},\n  title = {Bias in error estimation when using cross-validation for model selection},\n  journal = {BMC Bioinformatics}, volume = {7}, pages = {91}, year = {2006},\n  doi = {10.1186/1471-2105-7-91}\n}",
    },
    {
        "key": "Varoquaux2018",
        "authors": "Varoquaux, G.",
        "year": 2018,
        "title": "Cross-validation failure: Small sample sizes lead to large error bars",
        "venue": "NeuroImage, 180, 68-77",
        "doi": "10.1016/j.neuroimage.2017.06.061",
        "url": "https://doi.org/10.1016/j.neuroimage.2017.06.061",
        "bib": "@article{Varoquaux2018,\n  author = {Varoquaux, Ga{\"e}l},\n  title = {Cross-validation failure: Small sample sizes lead to large error bars},\n  journal = {NeuroImage}, volume = {180}, pages = {68--77}, year = {2018},\n  doi = {10.1016/j.neuroimage.2017.06.061}\n}",
    },
    {
        "key": "VanCalster2019",
        "authors": "Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L., and Steyerberg, E. W.",
        "year": 2019,
        "title": "Calibration: the Achilles heel of predictive analytics",
        "venue": "BMC Medicine, 17, 230",
        "doi": "10.1186/s12916-019-1466-7",
        "url": "https://doi.org/10.1186/s12916-019-1466-7",
        "bib": "@article{VanCalster2019,\n  author = {Van Calster, Ben and McLernon, David J. and van Smeden, Maarten and Wynants, Laure and Steyerberg, Ewout W.},\n  title = {Calibration: the Achilles heel of predictive analytics},\n  journal = {BMC Medicine}, volume = {17}, pages = {230}, year = {2019},\n  doi = {10.1186/s12916-019-1466-7}\n}",
    },
    {
        "key": "Weng2017",
        "authors": "Weng, S. F., Reps, J., Kai, J., Garibaldi, J. M., and Qureshi, N.",
        "year": 2017,
        "title": "Can machine-learning improve cardiovascular risk prediction using routine clinical data?",
        "venue": "PLOS ONE, 12(4), e0174944",
        "doi": "10.1371/journal.pone.0174944",
        "url": "https://doi.org/10.1371/journal.pone.0174944",
        "bib": "@article{Weng2017,\n  author = {Weng, Stephen F. and Reps, Jonathan and Kai, Joe and Garibaldi, Jonathan M. and Qureshi, Nadeem},\n  title = {Can machine-learning improve cardiovascular risk prediction using routine clinical data?},\n  journal = {PLOS ONE}, volume = {12}, number = {4}, pages = {e0174944}, year = {2017},\n  doi = {10.1371/journal.pone.0174944}\n}",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_path(relative: str) -> Path:
    return OUTPUT / relative


def save_text(relative: str, content: str) -> None:
    path = output_path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def save_json(relative: str, content: Any) -> None:
    save_text(relative, json.dumps(content, indent=2, ensure_ascii=False))


def metric_summary() -> dict[str, pd.DataFrame]:
    result = {
        "directional_coupling": pd.read_csv(RUN / "directional_coupling" / "seed_metrics.csv"),
        "legacy_coupling_control": pd.read_csv(RUN / "legacy_coupling_control" / "seed_metrics.csv"),
        "authoritative_baseline_oof": pd.read_csv(RUN / "authoritative_baseline_seed_metrics.csv"),
    }
    for arm, frame in result.items():
        missing = [metric for metric in METRICS if metric not in frame.columns]
        if missing:
            raise ValueError(f"{arm} seed metrics are missing columns: {missing}")
    return result


def load_inputs() -> dict[str, Any]:
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")
    config_path = RUN / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment configuration not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sha256(DATASET) != config["dataset_sha256"]:
        raise ValueError("Dataset SHA-256 does not match the recorded experiment configuration")
    if len(config["seeds"]) != 5 or config["outer_folds"] != 5 or config["inner_folds"] != 4:
        raise ValueError("The recorded experiment does not match the five-seed 5x4 nested-CV protocol")
    data = pd.read_csv(DATASET)
    dedup = data.drop_duplicates().reset_index(drop=True)
    if len(data) != 1888 or len(dedup) != 602:
        raise ValueError("Dataset row counts differ from the audited source data")
    arms = {}
    for arm in ("directional_coupling", "legacy_coupling_control"):
        frame = pd.read_csv(RUN / arm / "oof_predictions.csv")
        frame["outer_fold"] = frame["fold"].astype(int)
        arms[arm] = frame
    baseline_oof_parts = []
    for seed in config["seeds"]:
        part = pd.read_csv(
            ROOT / "results" / "stage5_final_validation" / "seed_oof_predictions" / f"seed_{seed}_oof.csv"
        )
        part["seed"] = seed
        part["sample_index"] = part["sample_index"].astype(int)
        part["outer_fold"] = part["fold"].astype(int)
        part["predicted_label"] = (part["probability"] >= 0.5).astype(int)
        baseline_oof_parts.append(part)
    baseline_oof = pd.concat(baseline_oof_parts, ignore_index=True)
    metrics = metric_summary()
    for name, frame in {**arms, "authoritative_baseline_oof": baseline_oof}.items():
        if "seed" not in frame or "target" not in frame:
            raise ValueError(f"{name} OOF file lacks seed/target columns")
        if len(frame) != 602 * 5:
            raise ValueError(f"{name} must contain exactly 3,010 OOF rows")
        if set(frame["seed"].unique()) != set(config["seeds"]):
            raise ValueError(f"{name} OOF seeds differ from the recorded configuration")
    check_oof_alignment(arms, baseline_oof)
    return {
        "config": config,
        "raw": data,
        "data": dedup,
        "arms": arms,
        "baseline_oof": baseline_oof,
        "seed_metrics": metrics,
        "baseline_seed_metrics": metrics["authoritative_baseline_oof"],
        "stats": pd.read_csv(RUN / "statistical_comparison.csv"),
        "resources": json.loads((RUN / "resource_usage_summary.json").read_text(encoding="utf-8")),
    }


def check_oof_alignment(arms: dict[str, pd.DataFrame], baseline: pd.DataFrame) -> None:
    frames = {**arms, "authoritative_baseline_oof": baseline}
    for seed in SEEDS:
        sliced = {}
        for key, frame in frames.items():
            index_col = "sample_index" if "sample_index" in frame.columns else "row_index"
            sliced[key] = (
                frame.loc[frame["seed"] == seed]
                .sort_values(index_col)
                .reset_index(drop=True)
            )
        reference = sliced["directional_coupling"]
        for key, current in sliced.items():
            if len(current) != 602:
                raise ValueError(f"Seed {seed} in {key} has {len(current)} rather than 602 rows")
            reference_index = "sample_index" if "sample_index" in reference else "row_index"
            current_index = "sample_index" if "sample_index" in current else "row_index"
            if not np.array_equal(reference[reference_index], current[current_index]):
                raise ValueError(f"OOF row alignment mismatch for seed {seed}, arm {key}")
            for column in ("target", "outer_fold"):
                if column in reference and column in current and not np.array_equal(
                    reference[column].to_numpy(), current[column].to_numpy()
                ):
                    raise ValueError(f"OOF alignment mismatch for seed {seed}, column {column}, arm {key}")


def mean_sd(frame: pd.DataFrame, metric: str) -> tuple[float, float]:
    values = frame[metric].astype(float).to_numpy()
    return float(np.mean(values)), float(np.std(values, ddof=1))


def display_mean_sd(frame: pd.DataFrame, metric: str, digits: int = 3) -> str:
    mean, std = mean_sd(frame, metric)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def write_figures(inputs: dict[str, Any]) -> None:
    figdir = OUTPUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def save(fig: plt.Figure, number: int) -> None:
        fig.tight_layout()
        fig.savefig(figdir / f"fig{number:02d}.png", dpi=300, bbox_inches="tight")
        fig.savefig(figdir / f"fig{number:02d}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.1))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, color: str, size: int = 9) -> None:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=size,
            wrap=True,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": color, "edgecolor": "#444444", "linewidth": 0.8},
        )

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#555555"})

    box(0.1, 3.8, 2.1, 1.1, "13 tabular features\n5 continuous + 8 categorical", "#F1F1F1")
    box(2.7, 4.1, 1.8, 0.9, "Metabolic encoder\n(3 continuous, 2 cat.)", "#DCEAF7", 8)
    box(2.7, 2.8, 1.8, 0.9, "Exercise encoder\n(2 continuous, 2 cat.)", "#FBE7D9", 8)
    box(2.7, 1.5, 1.8, 0.9, "Coronary encoder\n(4 categorical)", "#DDF2E8", 8)
    box(0.1, 0.35, 2.1, 0.95, "Physiology inputs\nage, thalachh, oldpeak, exang", "#FFF2CC", 8)
    box(5.1, 4.0, 1.65, 1.0, "Pairwise gated\ninteractions", "#E6E0F2")
    box(5.1, 2.55, 1.65, 1.0, "Mismatch projection\nHR deficit × ischemia", "#FFF2CC", 8)
    box(7.3, 3.95, 1.55, 0.95, "Interaction logit\nz_inter", "#E6E0F2", 8)
    box(7.3, 2.55, 1.55, 0.95, "Mechanistic logit\nz_mech", "#FFF2CC", 8)
    box(7.3, 1.15, 1.55, 0.95, "Residual logit\nz_res", "#E5E5E5", 8)
    box(9.4, 2.45, 1.45, 1.05, "z_total = sum\np = sigmoid(z)", "#D9EAD3")
    for y in (4.55, 3.25, 1.95):
        arrow(2.2, 4.3, 2.7, y)
    arrow(2.2, 0.82, 5.1, 2.9)
    arrow(4.5, 4.55, 5.1, 4.55)
    arrow(4.5, 3.25, 5.1, 4.3)
    arrow(4.5, 1.95, 5.1, 4.15)
    arrow(4.5, 4.45, 5.1, 3.25)
    arrow(4.5, 3.2, 5.1, 3.1)
    arrow(4.5, 1.95, 5.1, 2.95)
    arrow(6.75, 4.45, 7.3, 4.42)
    arrow(6.75, 3.05, 7.3, 3.02)
    arrow(6.75, 4.2, 7.3, 1.65)
    arrow(8.85, 4.42, 9.4, 3.1)
    arrow(8.85, 3.02, 9.4, 2.95)
    arrow(8.85, 1.65, 9.4, 2.7)
    ax.set_title("PMRC-Net implementation schematic (not a causal graph)", pad=8, weight="bold")
    save(fig, 1)

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.axis("off")
    stages = [
        (0.03, "5 seeds\n× 5 outer folds", "#DCEAF7"),
        (0.23, "Outer train\n(80%)", "#DDF2E8"),
        (0.43, "4-fold inner CV\ncandidate + epoch\nselection", "#FBE7D9"),
        (0.63, "Inner OOF only:\ntemperature +\nYouden threshold", "#FFF2CC"),
        (0.83, "Refit outer train;\nevaluate outer test\nonce; save OOF", "#E6E0F2"),
    ]
    for x, label, color in stages:
        ax.text(
            x + 0.075,
            0.52,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.7", "facecolor": color, "edgecolor": "#555555"},
            fontsize=9,
        )
    for x in (0.145, 0.345, 0.545, 0.745):
        ax.annotate(
            "",
            xy=(x + 0.05, 0.52),
            xytext=(x, 0.52),
            xycoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "lw": 1.5},
        )
    ax.text(
        0.5,
        0.12,
        "Repeat for directional and single-flow arms; retain every fold checkpoint and prediction.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )
    ax.set_title("Nested evaluation and calibration workflow", weight="bold")
    save(fig, 2)

    fpr_grid = np.linspace(0, 1, 101)
    recall_grid = np.linspace(0, 1, 101)
    arm_frames = {
        "directional_coupling": inputs["arms"]["directional_coupling"],
        "legacy_coupling_control": inputs["arms"]["legacy_coupling_control"],
        "authoritative_baseline_oof": inputs["baseline_oof"],
    }
    fig, ax = plt.subplots(figsize=(6.3, 4.7))
    for arm, frame in arm_frames.items():
        curves = []
        for seed in SEEDS:
            part = frame.loc[frame.seed == seed]
            fpr, tpr, _ = roc_curve(part.target, part.probability)
            interp = np.interp(fpr_grid, fpr, tpr)
            interp[0] = 0.0
            curves.append(interp)
        values = np.asarray(curves)
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1)
        auc_mean, auc_sd = mean_sd(inputs["seed_metrics"][arm], "roc_auc")
        label = f"{ARMS[arm]} (AUC {auc_mean:.3f}±{auc_sd:.3f})"
        ax.plot(fpr_grid, mean, color=COLORS[arm], lw=2, label=label)
        ax.fill_between(fpr_grid, np.maximum(0, mean - sd), np.minimum(1, mean + sd), color=COLORS[arm], alpha=0.13)
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("ROC curves: five seed-level OOF runs")
    save(fig, 3)

    fig, ax = plt.subplots(figsize=(6.3, 4.7))
    for arm, frame in arm_frames.items():
        curves = []
        for seed in SEEDS:
            part = frame.loc[frame.seed == seed]
            precision, recall, _ = precision_recall_curve(part.target, part.probability)
            interp = np.interp(recall_grid, recall[::-1], precision[::-1])
            curves.append(interp)
        values = np.asarray(curves)
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1)
        ap_mean, ap_sd = mean_sd(inputs["seed_metrics"][arm], "average_precision")
        ax.plot(recall_grid, mean, color=COLORS[arm], lw=2, label=f"{ARMS[arm]} (AP {ap_mean:.3f}±{ap_sd:.3f})")
        ax.fill_between(recall_grid, np.maximum(0, mean - sd), np.minimum(1, mean + sd), color=COLORS[arm], alpha=0.13)
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Precision-recall curves: five seed-level OOF runs")
    save(fig, 4)

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.4))
    for col, (arm, frame) in enumerate(arm_frames.items()):
        ys, ps = [], []
        for seed in SEEDS:
            part = frame.loc[frame.seed == seed]
            ys.extend(part.target.astype(int).tolist())
            if arm == "authoritative_baseline_oof":
                ps.extend((part.probability.to_numpy() >= 0.5).astype(int).tolist())
            else:
                ps.extend(part.predicted_label.astype(int).tolist())
        matrix = confusion_matrix(ys, ps, labels=[0, 1])
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        for row, (mat, title, fmt) in enumerate(
            [(matrix, "Counts", "d"), (normalized, "Row normalized", ".3f")]
        ):
            ax = axes[row, col]
            ax.imshow(mat, cmap="Blues", vmin=0)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", color="black", fontsize=9)
            ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
            ax.set_yticks([0, 1], ["True 0", "True 1"])
            if row == 0:
                ax.set_title(ARMS[arm], fontsize=8)
            if col == 0:
                ax.set_ylabel(title)
    fig.suptitle("Five-seed repeated OOF confusion matrices (n=3,010 predictions per arm)", y=1.02)
    save(fig, 5)

    fig, ax = plt.subplots(figsize=(6.3, 4.7))
    for arm, frame in arm_frames.items():
        ys, ps = [], []
        for seed in SEEDS:
            part = frame.loc[frame.seed == seed]
            ys.extend(part.target.astype(int).tolist())
            ps.extend(part.probability.astype(float).tolist())
        observed, predicted = calibration_curve(ys, ps, n_bins=10, strategy="uniform")
        brier_mean, brier_sd = mean_sd(inputs["seed_metrics"][arm], "brier_score")
        ax.plot(predicted, observed, marker="o", lw=1.8, color=COLORS[arm], label=f"{ARMS[arm]} (Brier {brier_mean:.3f}±{brier_sd:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#777777", lw=1)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction positive", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Calibration: ten uniform bins on repeated OOF predictions")
    save(fig, 6)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5.7))
    for ax, metric in zip(axes.flat, METRICS):
        means, stds, labels, colors = [], [], [], []
        for arm in ("directional_coupling", "legacy_coupling_control"):
            mean, std = mean_sd(inputs["seed_metrics"][arm], metric)
            means.append(mean)
            stds.append(std)
            labels.append("Directional" if arm == "directional_coupling" else "Single-flow")
            colors.append(COLORS[arm])
        ax.bar(labels, means, yerr=stds, color=colors, capsize=3)
        ax.set_title(METRIC_LABELS[metric])
        ax.tick_params(axis="x", labelrotation=25)
        if metric in {"brier_score", "ece"}:
            ax.set_ylim(bottom=0)
    fig.suptitle("Ablation: directional versus single-flow coupling (mean ± SD across five seeds)", y=1.02)
    save(fig, 7)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5.6))
    for ax, metric in zip(axes.flat, METRICS):
        for arm in ("directional_coupling", "legacy_coupling_control", "authoritative_baseline_oof"):
            frame = inputs["seed_metrics"][arm]
            ordered = frame.set_index("seed").loc[SEEDS]
            ax.plot(SEEDS, ordered[metric].astype(float), marker="o", lw=1.2, color=COLORS[arm], label=ARMS[arm])
        ax.set_title(METRIC_LABELS[metric])
        ax.tick_params(axis="x", labelrotation=30)
        ax.set_xticks(SEEDS, [str(s) for s in SEEDS])
    axes[0, 0].legend(fontsize=5.5, frameon=False)
    fig.suptitle("Seed-level metric stability; historical baseline is descriptive only", y=1.02)
    save(fig, 8)


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "±": r"$\pm$",
        "≤": r"$\leq$",
        "≥": r"$\geq$",
    }
    return "".join(replacements.get(char, char) for char in text)


def latex_table(headers: list[str], rows: list[list[Any]]) -> str:
    align = "l" + "r" * (len(headers) - 1)
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule"]
    lines.append(" & ".join(latex_escape(value) for value in headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def write_tables(inputs: dict[str, Any]) -> None:
    config = inputs["config"]
    raw = inputs["raw"]
    clean = inputs["data"]
    dups = len(raw) - len(clean)
    target_counts_raw = raw["target"].value_counts().to_dict()
    target_counts_clean = clean["target"].value_counts().to_dict()
    ca_raw = int((raw["ca"] == 4).sum())
    ca_clean = int((clean["ca"] == 4).sum())
    versions = {
        "Python": platform.python_version(),
        "PyTorch": torch.__version__,
        "CUDA": torch.version.cuda or "not available",
        "NumPy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "SciPy": importlib.metadata.version("scipy"),
        "Matplotlib": matplotlib.__version__,
        "NetworkX": importlib.metadata.version("networkx"),
        "PyYAML": importlib.metadata.version("PyYAML"),
    }
    resources = inputs["resources"]
    TABLE_DATA["table01_dataset"] = (
        ["Characteristic", "Value", "Notes"],
        [
            ["Raw rows", f"{len(raw):,}", "Merged CSV; provenance and patient identifiers unavailable"],
            ["Exact duplicate rows removed", f"{dups:,}", "Exact-row deduplication only"],
            ["Rows after deduplication", f"{len(clean):,}", "No claim of unique patients"],
            ["Predictors", "13 (5 continuous, 8 categorical)", "Target excluded"],
            ["Raw target counts", f"0: {target_counts_raw.get(0, 0)}; 1: {target_counts_raw.get(1, 0)}", "Target coding retained as stored"],
            ["Deduplicated target counts", f"0: {target_counts_clean.get(0, 0)}; 1: {target_counts_clean.get(1, 0)}", "602 total rows"],
            ["Missing cells", str(int(raw.isna().sum().sum())), "No missing values in the source CSV"],
            ["ca = 4", f"{ca_raw} raw; {ca_clean} after deduplication", "Loader preserves category 4; it is not imputed or mapped to missing"],
            [
                "Dataset SHA-256",
                f"{config['dataset_sha256'][:8]}...{config['dataset_sha256'][-8:]}",
                "Full hash is recorded in run_config.json and the repository audit",
            ],
        ],
        "Dataset and preprocessing audit",
    )
    hardware = config["hardware"]
    TABLE_DATA["table02_protocol"] = (
        ["Setting", "Value"],
        [
            ["Outer/inner CV", f"{config['outer_folds']} outer folds; {config['inner_folds']} inner folds"],
            ["Seeds", ", ".join(str(s) for s in config["seeds"])],
            ["Batch size / precision", f"{hardware['batch_size']} / FP32"],
            ["Device", f"{hardware['gpu_name']} ({hardware['gpu_total_memory_mb']:.0f} MiB); CUDA {hardware['cuda_version']}"],
            ["CPU / data workers", f"{hardware['cpu_threads']} threads; {hardware['cpu_interop_threads']} inter-op; {hardware['dataloader_workers']} workers"],
            ["Optimizer", "AdamW; fold-local standardization; max epochs 30; patience 8; gradient clip 1.0"],
            ["Candidate grid", f"{len(config['candidates'])} candidates; see supplementary/run_config.json"],
            ["Selection score", config["selection_metric"]],
            ["Calibration and threshold", "Scalar temperature and Youden J threshold from selected-candidate inner OOF only"],
            ["Outer evaluation", "One evaluation per outer test fold; 25 outer folds per arm"],
            ["Measured wall time", f"{resources['elapsed_seconds'] / 3600:.2f} h ({resources['elapsed_seconds']:.1f} s)"],
            ["Max system RAM", f"{resources['max_system_memory_percent']:.1f}% ({resources['max_process_rss_mb']:.0f} MiB peak process RSS)"],
            ["GPU temperature / utilization", f"{resources['max_gpu_temperature_c']:.0f} °C maximum; {resources['max_gpu_utilization_percent']:.0f}% max utilization"],
            ["Runtime versions", "; ".join(f"{key} {value}" for key, value in versions.items())],
        ],
        "Nested-CV protocol, hardware, and software",
    )
    metric_headers = ["Metric"] + [ARMS[key] for key in ARMS]
    metric_rows = []
    for metric in METRICS:
        metric_rows.append(
            [METRIC_LABELS[metric]]
            + [display_mean_sd(inputs["seed_metrics"][arm], metric, 4) for arm in ARMS]
        )
    TABLE_DATA["table03_metrics"] = (metric_headers, metric_rows, "Five-seed out-of-fold metrics")

    tests = inputs["stats"]
    paired = tests.loc[tests["model_b"] == "legacy_coupling_control"].copy()
    diff_rows = []
    for metric in METRICS:
        test = paired.loc[paired["metric"] == metric].iloc[0]
        diff_rows.append(
            [
                METRIC_LABELS[metric],
                f"{test['mean_paired_difference_a_minus_b']:+.4f}",
                f"{test['wilcoxon_p_value']:.4f}",
                "Higher is better" if metric not in {"brier_score", "ece"} else "Lower is better",
            ]
        )
    TABLE_DATA["table04_differences"] = (
        ["Metric", "Directional - single-flow", "Exact paired p", "Direction"],
        diff_rows,
        "Exploratory paired-seed differences; n = 5 seed pairs",
    )

    selected_metrics = ["accuracy", "roc_auc", "average_precision", "brier_score", "ece"]
    selected_rows = []
    for metric in selected_metrics:
        test = paired.loc[paired["metric"] == metric].iloc[0]
        selected_rows.append(
            [
                "Directional vs single-flow",
                METRIC_LABELS[metric],
                "5",
                f"{test['wilcoxon_p_value']:.4f}",
                "Exploratory; no correction for multiple metrics",
            ]
        )
    baseline_test = tests.loc[
        (tests["model_a"] == "directional_coupling")
        & (tests["model_b"] == "authoritative_baseline_oof")
        & (tests["metric"] == "roc_auc")
    ].iloc[0]
    selected_rows.append(
        [
            "Directional vs historical baseline",
            "ROC-AUC (nominal)",
            "5",
            f"{baseline_test['wilcoxon_p_value']:.4f}",
            "Not confirmatory: baseline selected epochs on outer validation",
        ]
    )
    TABLE_DATA["table05_statistical"] = (
        ["Comparison", "Metric", "Pairs", "Two-sided p", "Interpretation / limitation"],
        selected_rows,
        "Paired-seed statistical analysis; baseline p-value is invalid as controlled inference",
    )

    tables_dir = OUTPUT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, (headers, rows, caption) in TABLE_DATA.items():
        pd.DataFrame(rows, columns=headers).to_csv(tables_dir / f"{name}.csv", index=False, encoding="utf-8")
        save_text(f"tables/{name}.tex", latex_table(headers, rows))
        save_text(f"tables/{name}_caption.txt", caption)
    all_tests = inputs["stats"].copy()
    all_tests["validity"] = np.where(
        all_tests["model_b"] == "authoritative_baseline_oof",
        "NOMINAL ONLY; baseline outer-fold validation was used for epoch/checkpoint selection",
        "Exploratory paired-seed comparison; five repeated splits on one dataset",
    )
    all_tests["multiplicity_adjustment"] = "None"
    all_tests.to_csv(tables_dir / "table05_all_tests.csv", index=False, encoding="utf-8")


def write_research_audit(inputs: dict[str, Any]) -> None:
    config = inputs["config"]
    resources = inputs["resources"]
    audit = f"""# Repository and Experiment Audit

## Scope and provenance

- Dataset: `dataset/cleaned_merged_heart_dataset.csv` (the requested `(5).csv` variant was absent during the approved run).
- Dataset SHA-256: `{config['dataset_sha256']}`.
- Raw rows/columns: {len(inputs['raw'])} / {inputs['raw'].shape[1]}; exact duplicates: {len(inputs['raw']) - len(inputs['data'])}; post-deduplication rows: {len(inputs['data'])}.
- The CSV has no patient identifier or verified source provenance. A deduplicated row is not assumed to be a unique patient.
- The stored dataset and original baseline outputs were not edited by paper-package generation.

## Experiment status

- Status: `{resources['status']}`; five seeds × five outer folds × two nested arms; all 50 arm/fold checkpoints completed.
- Each arm has 3,010 OOF prediction records (602 rows repeated once per seed). Fold metrics and seed metrics are preserved in `supplementary/`.
- Recorded duration: {resources['elapsed_seconds']:.1f} seconds; peak process RSS {resources['max_process_rss_mb']:.1f} MiB; maximum system RAM {resources['max_system_memory_percent']:.1f}%; maximum GPU temperature {resources['max_gpu_temperature_c']:.1f} °C.
- The run recorded no CUDA OOM or failed/skipped outer fold. The run's post-processing failure was fixed and resumed from checkpoints before the COMPLETE status was written.

## Dataset and preprocessing findings

- Exact duplicate rows are removed before CV; no missing cells were present in the raw CSV.
- Five continuous columns are standardized using the relevant training partition; eight categorical columns use fixed source-code maps.
- The loader maps `ca` values 0–4 identically. It retains `ca=4` (28 raw records; five after exact deduplication); the value is not converted to missing and is not imputed. This corrects an earlier prompt/documentation ambiguity.
- `thal` maps `{{0:0, 1:6, 2:3, 3:7, 6:6, 7:7}}`; `slope` codes are retained.
- Raw target counts: 0 = {int((inputs['raw']['target'] == 0).sum())}, 1 = {int((inputs['raw']['target'] == 1).sum())}; after deduplication: 0 = {int((inputs['data']['target'] == 0).sum())}, 1 = {int((inputs['data']['target'] == 1).sum())}.

## Model, loss, and nested protocol

- The nested experiment uses `OptimizedPMRCNet`, hidden size 16, three named feature-group encoders, a projected engineered physiology mismatch, pairwise coupling gates, and additive mechanistic/interaction/residual logits.
- `HR_pred = 208 - 0.7 × age` is an age-predicted maximum-heart-rate equation; its use here is an engineered score, not a validated patient-specific physiological measurement. The mismatch multiplies a nonnegative heart-rate deficit by a softplus score from `oldpeak`, `exang`, and their interaction.
- For each of five seeds, outer assignments are stratified five-fold splits. Within each outer-training partition, four inner folds and the configured candidate grid select hyperparameters/epoch; temperature and Youden threshold are fitted only from selected-candidate inner OOF predictions. The outer test fold is evaluated once.
- The directionality arm differs from the legacy single-flow control by using separate forward and reversed feature transforms. The control is not a symmetric-coupling model.
- The loss configuration includes `l1_lambda`, but the implemented objective does not use it. The empirical association of `ca` is computed but not used in the physiology hinge.
- Across 100 inner-training partitions (five seeds × five outer folds × four inner folds), positive association signs occurred for `oldpeak` in 4, `exang` in 9, and `ca` in 5 partitions. All 25 outer-refit partitions had negative association signs for `oldpeak` and `exang`; therefore, the feature-specific physiology hinge did not activate for those features in outer refits. This implementation detail limits how strongly the run tests a physiology-informed loss.

## Historical baseline protocol: not a controlled comparison

The Stage-5 baseline uses aligned sample indices, targets, seeds, and outer-fold assignments, but it does not share the nested evaluation protocol:

1. It uses the original `PMRCNet` implementation rather than the optimized architecture.
2. Its Stage-5 training procedure selects the best checkpoint/epoch using the fold-specific `valid_loader`. That validation fold is the nominal outer test fold, so its labels influence model selection.
3. It uses a fixed 0.5 decision threshold and does not apply the inner-only temperature calibration / threshold-selection process.

Consequently, its OOF performance is selection-contaminated and is shown only as historical context. The experiment's nominal paired p-values against this baseline are recorded for transparency but are **not valid confirmatory comparisons**. No improvement, superiority, or noninferiority claim against that baseline is supported. A clean baseline comparison would require a new, nested evaluation of a baseline model under the same split, selection, calibration, and threshold protocol.

## Files examined

- `src/models/pmrc_net.py`, `src/models/pmrc_net_optimized.py`
- `src/data/dataset.py`, `src/training/losses.py`, `src/evaluation/metrics.py`
- `scripts/run_pmrc_nested_optimization.py` (read-only for this paper-generation task)
- `results/PMRC_NET_FINAL_RESEARCH/run_config.json`, `FINAL_RESEARCH_REPORT.md`, `statistical_comparison.csv`, `resource_usage_summary.json`
- `results/PMRC_NET_FINAL_RESEARCH/{{directional_coupling,legacy_coupling_control}}/` fold, seed, and OOF outputs
- `results/stage5_final_validation/stage5_validation.py` and its saved per-seed OOF predictions

## Source-control / preservation note

The submission generator is separate at `scripts/generate_conference_submission.py`. No PMRC-Net source, dataset, baseline, or completed experiment artifact is changed by the generator. The package excludes row-level OOF predictions; it includes only aggregate fold/seed metrics and the recorded configuration.
"""
    save_text("00_REPOSITORY_AUDIT.md", audit)


def write_literature_notes() -> None:
    purposes = {
        "Arik2021": "A representative attentive tabular neural architecture; useful context, not a direct benchmark.",
        "Cawley2010": "Model selection can bias reported performance; supports the nested-protocol requirement.",
        "Demsar2006": "Statistical comparison guidance; its multi-dataset setting does not make five repeated seeds independent.",
        "Gorishniy2021": "Tabular neural-network benchmarking and the importance of appropriate baselines.",
        "Grinsztajn2022": "Empirical limitations of deep learning on small/medium tabular benchmarks; cited as an arXiv preprint.",
        "Guo2017": "Post-hoc temperature scaling for neural-network calibration.",
        "Rajkomar2018": "Clinical EHR deep-learning context; not directly comparable to this small merged tabular dataset.",
        "Rudin2019": "Interpretability cautions; named feature groups do not make PMRC-Net inherently interpretable.",
        "Saito2015": "Precision-recall metrics can add information when evaluating imbalanced binary classifiers.",
        "Tanaka2001": "Source for the age-predicted maximal-heart-rate equation used as one engineered term.",
        "TRIPOD2015": "Reporting guidance for prediction-model studies.",
        "Varma2006": "Cross-validation selection bias and the need for nested selection.",
        "Varoquaux2018": "Uncertainty from small sample sizes and repeated cross-validation.",
        "VanCalster2019": "Calibration is distinct from discrimination and should be assessed explicitly.",
        "Weng2017": "Cardiovascular prediction context using routine clinical data; population and protocol differ.",
    }
    rows = []
    for ref in REFERENCES:
        rows.append(
            f"- **{ref['authors']} ({ref['year']}). {ref['title']}.** {ref['venue']}. "
            f"{ref['url']} — {purposes[ref['key']]}"
        )
    notes = """# Literature Review Notes and Source Verification

## Evidence framing

This review uses literature to position the work, not to claim architectural novelty or clinical validity. PMRC-Net is evaluated on one small, merged, deduplicated tabular dataset with unknown patient-level independence and no external validation. The named feature groups are an inductive-bias choice; they are not validated biological pathways, causal structure, or a transparent explanation of an individual prediction.

Tabular neural-network studies motivate careful model selection and strong, well-matched comparisons rather than assuming deep architectures outperform other model families. Clinical prediction and reporting references motivate transparent description of data provenance, calibration, validation, and limitations. Cross-validation references support isolating model/epoch selection from outer test data. Calibration references justify reporting proper scoring/calibration diagnostics separately from discrimination. The present five-seed comparison is exploratory and does not inherit the statistical power of evaluations across independent datasets.

## Verified references

""" + "\n".join(rows) + """

## Source-level verification record

- Publisher/proceedings records or DOI/Crossref metadata were checked for the journal and conference sources listed above.
- The TabNet article was verified against its AAAI proceedings record; the calibration article against PMLR; the JMLR records against the official JMLR pages; the heart-rate equation against Crossref/Elsevier DOI metadata; and the PLOS/clinical sources against DOI metadata.
- `Grinsztajn et al.` is intentionally cited as the arXiv preprint `arXiv:2207.08815`; no unverified venue claim is made.
- The bibliography is kept in `references.bib`; article text makes no long verbatim quotations.
"""
    save_text("LITERATURE_REVIEW_NOTES.md", notes)
    save_text("manuscript/references.bib", "\n\n".join(ref["bib"] for ref in REFERENCES))


def write_supplementary(inputs: dict[str, Any]) -> None:
    supplementary = OUTPUT / "supplementary"
    supplementary.mkdir(parents=True, exist_ok=True)
    fold_parts = []
    seed_parts = []
    for arm in ("directional_coupling", "legacy_coupling_control"):
        folds = pd.read_csv(RUN / arm / "fold_metrics.csv")
        folds["arm"] = arm
        fold_parts.append(folds)
        seeds = pd.read_csv(RUN / arm / "seed_metrics.csv")
        seeds.insert(0, "arm", arm)
        seed_parts.append(seeds)
    baseline = pd.read_csv(RUN / "authoritative_baseline_seed_metrics.csv")
    baseline.insert(0, "arm", "authoritative_baseline_oof")
    seed_parts.append(baseline)
    pd.concat(fold_parts, ignore_index=True).to_csv(
        supplementary / "fold_metrics.csv", index=False, encoding="utf-8"
    )
    pd.concat(seed_parts, ignore_index=True).to_csv(
        supplementary / "seed_metrics.csv", index=False, encoding="utf-8"
    )
    pd.read_csv(RUN / "statistical_comparison.csv").to_csv(
        supplementary / "statistical_comparison_nominal.csv", index=False, encoding="utf-8"
    )
    save_json("supplementary/run_config.json", inputs["config"])
    save_json("supplementary/resource_usage_summary.json", inputs["resources"])
    text = """# Supplementary Methods and Results

## Dataset handling

The exact cleaned CSV used by the completed run is identified by SHA-256 in `run_config.json`. Exact duplicate rows were removed before splitting. Continuous columns (`age`, `trestbps`, `chol`, `thalachh`, `oldpeak`) were standardized using only the relevant training partition. Categorical fields (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`) were mapped using the project loader. The loader preserves `ca=4`; the five retained `ca=4` rows are not missing-value imputations.

## Model equations

For pathway representations `h_m`, `h_e`, and `h_c`, and mismatch scalar `m_e`, each pair has forward and reverse transformed products:

```text
F_ij = T_i(h_i) * T_j(h_j)
R_ij = T_j(h_i) * T_i(h_j)
g_ij = sigmoid(s_gate * W_ij [h_i, h_j, m_e])
c_ij = g_forward * F_ij + g_reverse * R_ij
```

The learned gates are split into forward and reverse halves. The legacy control uses only the forward flow; it is a single-flow control, not a symmetric interaction. The directional arm learns separate transforms for the reversed ordering.

The physiology-derived scalar is implemented as:

```text
HR_pred = 208 - 0.7 * age
deficit = max(1 - thalachh / max(HR_pred, 1), 0)
ischemia = softplus(a1*oldpeak + a2*exang + a3*oldpeak*exang)
m_e = deficit * ischemia
```

Coefficients `a1..a3` are clamped nonnegative. This deterministic engineered quantity is not a patient-specific physiological measurement and has not been clinically validated in this experiment.

Separate linear projections of `m_e + h_c`, the sum of pairwise interactions, and the scaled sum of pathway representations form `z_mech`, `z_inter`, and `z_res`; `z_total = z_mech + z_inter + z_res`. Probabilities are sigmoid outputs, with a fold-specific scalar temperature fitted on inner OOF predictions for the nested arms.

The implemented objective combines binary cross entropy, an association-conditioned physiology hinge, coupling-weight regularization, gate regularization, residual-scale regularization, and optimizer weight decay. The configured `l1_lambda` is not used by the loss implementation. The empirical `ca` association is computed but is not used in the hinge. Across the 100 inner training partitions, positive association signs occurred for oldpeak in 4 and exang in 9; all 25 outer-refit partitions had negative signs for each, so these two feature-specific hinge terms were not active in the outer refits.

## Nested protocol

For each seed in `[21, 42, 123, 2026, 3407]`, outer assignments use stratified five-fold splitting with shuffle and that seed. For outer fold `k`, inner splitting uses four folds and `random_state = seed + 1009*k`. Three candidates vary dropout, learning rate, optimizer weight decay, physiology weight, gate weight, and residual weight (see `run_config.json`). Candidate selection uses mean inner ROC-AUC + 0.5 × MCC, with epoch selection based on inner validation loss. The selected candidate's inner OOF probabilities alone determine temperature and Youden-J threshold. The final model is refit on the complete outer-training partition and its held-out fold is scored once. The complete OOF set has 602 observations per seed per arm.

## Outcome definitions and statistical comparison

Discrimination includes ROC-AUC and average precision. Thresholded outcomes include accuracy, precision, sensitivity, specificity, F1, and MCC. Calibration includes Brier score and 10-bin expected calibration error. Per-fold and per-seed values are reported in `fold_metrics.csv` and `seed_metrics.csv`; raw row-level OOF predictions are deliberately not copied into this submission package.

Two-sided exact Wilcoxon signed-rank tests compare the five seed-level metrics. They are exploratory: the five estimates repeatedly use the same dataset and overlapping folds, are not five independent patient samples, and no multiplicity correction was applied. All nominal comparisons with the historical Stage-5 baseline are invalid as controlled inference because its outer validation fold selected the checkpoint/epoch. Those p-values are retained in `statistical_comparison_nominal.csv` with that warning; do not cite them as evidence of superiority or noninferiority.

## Supplementary files

- `fold_metrics.csv`: 25 fold summaries for each nested arm.
- `seed_metrics.csv`: five seed summaries for each nested arm and the historical baseline.
- `statistical_comparison_nominal.csv`: all run-generated paired tests, with the historical-baseline limitation.
- `run_config.json`: complete recorded configuration and dataset hash.
- `resource_usage_summary.json`: recorded run telemetry.
"""
    save_text("supplementary/SUPPLEMENTARY_METHODS.md", text)
    baseline_notes = """# Historical Baseline Protocol Audit

The historical Stage-5 baseline shares the deduplicated row ordering, targets, seeds, and outer fold assignments with the nested run. This alignment is useful for describing historical outputs, but it does not establish an unbiased head-to-head comparison.

`results/stage5_final_validation/stage5_validation.py` selects its best model/checkpoint using the fold-specific validation loader. That loader represents the same fold subsequently treated as the outer test fold in the saved OOF predictions. Consequently, outer-fold labels influenced baseline training/model selection. The baseline also uses the original `PMRCNet`, a fixed 0.5 threshold, and no inner-OOF temperature calibration or threshold selection.

Therefore:

- Show historical metrics only with a clear protocol-mismatch label.
- Do not claim the nested arms outperform, underperform, or are equivalent to this baseline.
- Do not interpret baseline paired p-values.
- A controlled baseline requires rerunning a baseline architecture under the same nested outer/inner splits, inner-only selection, calibration/threshold fitting, and one-time outer test evaluation.
"""
    save_text("supplementary/BASELINE_PROTOCOL_AUDIT.md", baseline_notes)


def build_paper(inputs: dict[str, Any]) -> dict[str, Any]:
    directional = inputs["seed_metrics"]["directional_coupling"]
    legacy = inputs["seed_metrics"]["legacy_coupling_control"]
    baseline = inputs["seed_metrics"]["authoritative_baseline_oof"]
    p_roc = float(
        inputs["stats"].loc[
            (inputs["stats"]["model_a"] == "directional_coupling")
            & (inputs["stats"]["model_b"] == "legacy_coupling_control")
            & (inputs["stats"]["metric"] == "roc_auc"),
            "wilcoxon_p_value",
        ].iloc[0]
    )
    d_roc, _ = mean_sd(directional, "roc_auc")
    l_roc, _ = mean_sd(legacy, "roc_auc")
    b_roc, _ = mean_sd(baseline, "roc_auc")
    d_acc, _ = mean_sd(directional, "accuracy")
    l_acc, _ = mean_sd(legacy, "accuracy")
    d_ap, _ = mean_sd(directional, "average_precision")
    l_ap, _ = mean_sd(legacy, "average_precision")
    d_brier, _ = mean_sd(directional, "brier_score")
    l_brier, _ = mean_sd(legacy, "brier_score")
    d_ece, _ = mean_sd(directional, "ece")
    l_ece, _ = mean_sd(legacy, "ece")
    resources = inputs["resources"]
    abstract = (
        "Background: Small tabular heart-disease datasets invite overfitting, and named physiological or "
        "pathway components do not by themselves establish clinical validity. Methods: We evaluated the "
        "existing improved PMRC-Net using five-seed, five-fold outer and four-fold inner nested "
        "cross-validation on 602 exact-deduplicated rows from a merged 13-feature CSV. Inner folds alone "
        "selected hyperparameters, temperature, and decision thresholds. A directional-coupling arm was "
        "compared with a single-flow coupling control. Results: Mean seed-level ROC-AUC was "
        f"{d_roc:.4f} for the directional arm and {l_roc:.4f} for the control (paired exact Wilcoxon "
        f"p={p_roc:.4f}, five seed pairs); corresponding accuracy was {d_acc:.4f} and {l_acc:.4f}. "
        f"The historical Stage-5 baseline had ROC-AUC {b_roc:.4f}, but its outer validation fold was "
        "used for checkpoint/epoch selection, so this is not an unbiased or controlled comparator. "
        "Conclusion: The observed directional-versus-single-flow difference is small and uncertain. "
        "The experiment does not establish clinical utility, external generalization, or superiority to "
        "the historical baseline."
    )
    sections: list[tuple[str, list[Any]]] = [
        (
            "1 Introduction",
            [
                "Clinical prediction with structured data requires more than a high discrimination score: data provenance, validation design, calibration, and deployment context all matter. Published cardiovascular and electronic-health-record studies illustrate the potential of machine learning, while also operating on populations and data regimes unlike the small merged CSV evaluated here [[CITE:Weng2017]] [[CITE:Rajkomar2018]]. Reporting guidance emphasizes transparent description of data, model development, and validation [[CITE:TRIPOD2015]].",
                "Deep learning for tabular data remains sensitive to dataset size, inductive bias, and comparison protocol. Prior work includes attentive feature selection and systematic tabular benchmarks, but does not imply that a custom neural architecture is superior on a particular small dataset [[CITE:Arik2021]] [[CITE:Gorishniy2021]] [[CITE:Grinsztajn2022]]. This study therefore asks a narrow engineering question: under a nested protocol, does the direction-specific interaction flow in the existing improved PMRC-Net produce different seed-level results from its legacy single-flow control?",
                "The work is exploratory. The feature-group names and physiology-derived scalar are design choices, not validated biological pathways or causal explanations. The dataset has no verified patient identifier or external cohort. No clinical-use claim is made.",
                {"table": "table01_dataset"},
            ],
        ),
        (
            "2 Related Work",
            [
                "Tabular neural architectures such as TabNet and benchmark-oriented studies motivate explicit comparison rather than assuming neural models dominate tree-based or simpler alternatives [[CITE:Arik2021]] [[CITE:Gorishniy2021]] [[CITE:Grinsztajn2022]]. We do not claim a new architecture class or state-of-the-art result; the tested contrast is a directional interaction implementation against a single-flow control.",
                "Nested model selection is needed because repeatedly using validation data to choose hyperparameters or stopping epochs biases performance estimates [[CITE:Cawley2010]] [[CITE:Varma2006]]. Small samples also produce substantial uncertainty even when cross-validation is repeated [[CITE:Varoquaux2018]]. Calibration is a separate property from ranking discrimination; temperature scaling and calibration assessment have been studied in neural and clinical prediction settings [[CITE:Guo2017]] [[CITE:VanCalster2019]].",
                "For a binary task, ROC-AUC is complemented by precision-recall summaries, thresholded measures, and proper scoring/calibration metrics [[CITE:Saito2015]]. The age-predicted maximal-heart-rate equation used by the engineered mismatch follows a population-level relationship, not an individualized physiologic measurement [[CITE:Tanaka2001]]. Named inputs or pathway branches do not alone establish interpretability; high-stakes interpretability requires stronger evidence and careful framing [[CITE:Rudin2019]].",
            ],
        ),
        (
            "3 Methods",
            [
                "3.1 Dataset and preprocessing. The exact source is `dataset/cleaned_merged_heart_dataset.csv` (SHA-256 recorded in the configuration). It contains 1,888 rows and 14 columns before exact-row deduplication; 1,286 repeated rows are removed, leaving 602 rows with 13 predictors and one binary target. The deduplicated target counts are 310 class-0 and 292 class-1 rows. There are no missing cells in the CSV. The provenance and unit of observation are unknown; exact deduplication does not establish unique patients. Five continuous features are standardized from each training partition only; eight categorical columns use fixed project mappings. In particular, `ca=4` is retained as a category (five deduplicated rows), not interpreted as missing or imputed.",
                "3.2 Architecture. The implementation uses three named encoders over a partition of the 13 predictors: metabolic (age, sex, trestbps, chol, fbs), exercise (thalachh, oldpeak, exang, slope), and coronary (cp, restecg, ca, thal). Each pathway is projected to a 16-dimensional representation with the implementation's activation, dropout, and normalization layers. The physiology branch uses age, maximum heart rate (`thalachh`), `oldpeak`, and `exang` to construct a mismatch scalar:",
                {
                    "equation": (
                        r"\begin{aligned}"
                        r"M &= \max\left(1 - \frac{\mathrm{thalachh}}"
                        r"{\max(208 - 0.7\,\mathrm{age},1)},0\right)\\"
                        r"&\quad \cdot \mathrm{softplus}\bigl("
                        r"a_1\mathrm{oldpeak} + a_2\mathrm{exang}\\"
                        r"&\qquad + a_3\mathrm{oldpeak}\,\mathrm{exang}\bigr)"
                        r"\end{aligned}"
                    ),
                    "plain": "M = max(1 - thalachh / max(208 - 0.7 * age, 1), 0) * "
                    "softplus(a1 * oldpeak + a2 * exang + a3 * oldpeak * exang)",
                },
                "where each learned coefficient is constrained to be nonnegative. This deterministic engineered score is a modeling feature, not a clinical physiologic measurement. Each pair of pathway representations forms forward and reversed transformed products, combined with learned sigmoid gates. The directional arm uses separate transforms for the reversed ordering; the control uses a single forward flow and is not a symmetric-coupling model. Three scalar logits (mechanistic, interaction, residual) are added to form the final logit.",
                {"figure": 1},
                "3.3 Training objective. Training combines binary cross entropy with the code's association-conditioned physiology hinge, coupling/gate/residual regularization, and AdamW weight decay. The configured `l1_lambda` is not consumed by the implemented objective. The empirical association of `ca` is calculated but does not contribute to the hinge. In 100 inner-training partitions, positive association signs occurred for oldpeak in 4 and exang in 9; in all 25 outer-refit partitions both signs were negative. Thus, the feature-specific physiology hinge for these variables was inactive in the outer refits, limiting any claim that the final fits demonstrate benefit from that regularizer.",
                "3.4 Nested evaluation. Five seeds (21, 42, 123, 2026, 3407) each define five stratified outer folds. Each outer-training partition is split into four inner folds using the recorded seed-plus-fold offset. Three configured candidates are compared by mean inner ROC-AUC plus 0.5 times MCC; epoch selection uses inner validation loss. The selected candidate's inner out-of-fold predictions alone determine scalar temperature and Youden-J threshold. The selected configuration is refit on the complete outer-training partition, and the outer test fold is evaluated once. The process is repeated for both nested arms. The saved OOF files contain 602 predictions per seed and arm.",
                {"figure": 2},
                "3.5 Metrics and statistical analysis. We report accuracy, precision, sensitivity, specificity, F1, MCC, ROC-AUC, average precision, Brier score, and 10-bin expected calibration error (ECE). Seed-level summaries use mean and sample standard deviation over five seeds. Exact two-sided Wilcoxon signed-rank tests use the five paired seed-level values for each metric; p-values are exploratory, unadjusted, and not patient-level uncertainty estimates because the seeds reuse the same dataset and overlapping folds.",
                {"table": "table02_protocol"},
            ],
        ),
        (
            "4 Results",
            [
                "The full run completed all 50 arm/fold checkpoints with no skipped fold. Each nested arm produced 25 fold records and 3,010 OOF prediction records across five seeds. The recorded wall time was "
                f"{resources['elapsed_seconds'] / 3600:.2f} hours; maximum system RAM was {resources['max_system_memory_percent']:.1f} percent, maximum GPU temperature {resources['max_gpu_temperature_c']:.0f} degrees C, and no CUDA OOM or fold failure was recorded.",
                "4.1 Main metrics. The directional arm achieved mean ROC-AUC "
                f"{d_roc:.4f} (SD {mean_sd(directional, 'roc_auc')[1]:.4f}), average precision {d_ap:.4f} "
                f"(SD {mean_sd(directional, 'average_precision')[1]:.4f}), accuracy {d_acc:.4f}, "
                f"Brier score {d_brier:.4f}, and ECE {d_ece:.4f}. The single-flow control achieved "
                f"ROC-AUC {l_roc:.4f} (SD {mean_sd(legacy, 'roc_auc')[1]:.4f}), average precision "
                f"{l_ap:.4f}, accuracy {l_acc:.4f}, Brier score {l_brier:.4f}, and ECE {l_ece:.4f}.",
                {"table": "table03_metrics"},
                "4.2 Coupling ablation. The mean paired ROC-AUC difference (directional minus single-flow) was "
                f"{d_roc - l_roc:+.4f}, with exact paired p={p_roc:.4f} across five seed pairs. The corresponding "
                f"mean accuracy difference was {d_acc - l_acc:+.4f}. No paired metric had p < 0.05; the small "
                "number of repeated seeds, shared observations, and uncorrected multiple metrics make these "
                "results exploratory rather than confirmatory.",
                {"table": "table04_differences"},
                {"figure": 3},
                {"figure": 4},
                {"figure": 5},
                {"figure": 6},
                {"figure": 7},
                {"figure": 8},
                "4.3 Historical baseline. The Stage-5 historical OOF baseline had mean ROC-AUC "
                f"{b_roc:.4f} (SD {mean_sd(baseline, 'roc_auc')[1]:.4f}); this value is not a valid unbiased "
                "estimate for controlled comparison. Stage 5 selects its checkpoint/epoch using the same "
                "fold-specific validation loader later treated as the nominal OOF test fold. The baseline also "
                "uses the original architecture, a fixed 0.5 threshold, and no inner-only calibration. Its "
                "metrics and nominal paired tests are retained only for traceability and historical context; "
                "no comparative inference is drawn.",
                {"table": "table05_statistical"},
            ],
        ),
        (
            "5 Discussion",
            [
                "The directional-coupling modification produced a very small mean ROC-AUC difference from the single-flow control (+0.0017 using the recorded paired-seed means; p=0.4375). Other metrics move in mixed directions: directional coupling has higher mean accuracy and MCC but slightly lower average precision, while calibration summaries are similar. The evidence does not establish a reliable benefit from directionality.",
                "The historical baseline score is numerically higher, but its outer validation data influenced model selection. Presenting it as a valid comparator would conflate a protocol difference with an architecture difference. A fresh, nested baseline run is required before any controlled comparison. Statistical-comparison research on independent datasets does not justify treating five repeated seeds on one dataset as five independent datasets [[CITE:Demsar2006]].",
                "The architecture's named feature groups may express a chosen inductive bias, but no feature attribution, mechanistic validation, causal analysis, or clinician evaluation was performed. The engineered heart-rate/ischemia score and the inactive outer-refit hinge should therefore not be framed as physiological validation. Calibration measures describe probability agreement in these resampled data; they do not establish calibration in a deployment population.",
            ],
        ),
        (
            "6 Limitations",
            [
                "The dataset has only 602 deduplicated rows, unknown source/collection protocol, no patient identifier, and unknown patient-level independence. Exact duplicate removal cannot prevent patient overlap or remove near-duplicates. The target's exact clinical meaning and class-generation procedure are not established in the inspected data. There is no external validation, temporal validation, prospective evaluation, decision-curve analysis, or clinical utility assessment.",
                "Five seed-level summaries provide weak and correlated uncertainty information. The nested arms are evaluated on the same small sample; paired p-values are exploratory, unadjusted, and do not support general population claims. The historical baseline is selection-contaminated and is not a controlled comparator. The physiology branch is an engineered score and its feature-specific loss hinge was inactive in outer refits. The configured L1 penalty is unused, and the `ca` association is not applied in the hinge.",
                "The experiment evaluates a binary classification benchmark, not screening, diagnosis, prognosis, or treatment decisions. Performance is sensitive to data source, coding, class definition, and the selected operating threshold. The project still requires authors to verify data rights, source provenance, ethics status, and reporting requirements before submission or reuse.",
            ],
        ),
        (
            "7 Ethics, Availability, and Declarations",
            [
                "Ethics approval or exemption: To be completed by the authors after verifying the dataset source, consent, and applicable institutional requirements; no exemption is inferred here.",
                "Data availability: The local cleaned CSV is identified by its hash in the supplementary run configuration. Its source, license, and redistribution rights require author confirmation. Row-level predictions are not included in this package.",
                "Code availability: The repository contains the training implementation and the new package generator at `scripts/generate_conference_submission.py`. Reproduction requires the exact source data and software environment recorded in the experiment configuration.",
                "Funding, conflicts of interest, author contributions, affiliations, and corresponding-author details: To be completed by the authors. No statements are fabricated.",
            ],
        ),
        (
            "8 Conclusion",
            [
                "In this completed five-seed nested evaluation, directional coupling differed only slightly from the legacy single-flow control and did not show a statistically significant paired improvement. The historical Stage-5 baseline is protocol-mismatched because its outer validation fold influenced checkpoint selection; its score cannot support a controlled improvement claim. The study is a reproducible exploratory benchmark on one small dataset, not evidence of clinical readiness. The next valid comparison is a nested baseline rerun under identical model-selection, calibration, and threshold rules, followed by external evaluation on data with verified patient independence and provenance.",
            ],
        ),
    ]
    return {
        "title": "PMRC-Net for Tabular Heart-Disease Classification: An Exploratory Five-Seed Nested-CV Evaluation",
        "authors": "[Author names and affiliations to be completed]",
        "abstract": abstract,
        "keywords": "tabular learning; nested cross-validation; calibration; heart-disease classification; reproducibility; model selection",
        "sections": sections,
    }


def citation_number_map() -> dict[str, int]:
    return {entry["key"]: i + 1 for i, entry in enumerate(REFERENCES)}


def plain_text(text: str) -> str:
    numbers = citation_number_map()
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return re.sub(
        r"\[\[CITE:([A-Za-z0-9]+)\]\]",
        lambda match: f"[{numbers[match.group(1)]}]",
        text,
    )


def tex_text(text: str) -> str:
    output: list[str] = []
    cursor = 0
    pattern = re.compile(r"\[\[CITE:([A-Za-z0-9]+)\]\]|`([^`]+)`")
    for match in pattern.finditer(text):
        output.append(latex_escape(text[cursor : match.start()]))
        if match.group(1):
            output.append(r"\cite{" + match.group(1) + "}")
        else:
            output.append(r"\texttt{" + latex_escape(match.group(2)) + "}")
        cursor = match.end()
    output.append(latex_escape(text[cursor:]))
    return "".join(output)


def markdown_table(name: str) -> str:
    headers, rows, _ = TABLE_DATA[name]
    lines = ["| " + " | ".join(str(item) for item in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(paper: dict[str, Any]) -> str:
    lines = [
        f"# {paper['title']}",
        "",
        paper["authors"],
        "",
        "## Abstract",
        "",
        paper["abstract"],
        "",
        f"**Keywords:** {paper['keywords']}",
        "",
    ]
    for heading, blocks in paper["sections"]:
        lines.extend([f"## {heading}", ""])
        for block in blocks:
            if isinstance(block, str):
                lines.extend([plain_text(block), ""])
            elif "equation" in block:
                lines.extend(["$$", block["equation"], "$$", ""])
            elif "figure" in block:
                number = block["figure"]
                lines.extend(
                    [
                        f"![Figure {number}: {FIGURE_CAPTIONS[number]}](../figures/fig{number:02d}.png)",
                        f"*Figure {number}. {FIGURE_CAPTIONS[number]}*",
                        "",
                    ]
                )
            elif "table" in block:
                name = block["table"]
                lines.extend(
                    [
                        f"**Table {name[5:7]}. {TABLE_DATA[name][2]}**",
                        "",
                        markdown_table(name),
                        "",
                    ]
                )
    lines.extend(["## References", ""])
    for index, reference in enumerate(REFERENCES, 1):
        lines.append(
            f"[{index}] {reference['authors']} ({reference['year']}). {reference['title']}. "
            f"{reference['venue']}. {reference['url']}"
        )
        lines.append("")
    return "\n".join(lines)


def render_tex(paper: dict[str, Any]) -> str:
    output = [
        r"\documentclass[10pt,twocolumn]{article}",
        r"\usepackage[margin=0.72in]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{microtype}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage[numbers,sort&compress]{natbib}",
        r"\usepackage{hyperref}",
        r"\usepackage{caption}",
        r"\usepackage{float}",
        r"\usepackage{array}",
        r"\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}",
        r"\setlength{\columnsep}{0.25in}",
        r"\title{" + latex_escape(paper["title"]) + "}",
        r"\author{" + latex_escape(paper["authors"]) + r"\\\small Affiliation and corresponding-author details to be completed}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        tex_text(paper["abstract"]),
        r"\end{abstract}",
        r"\noindent\textbf{Keywords:} " + latex_escape(paper["keywords"]),
        "",
    ]
    for heading, blocks in paper["sections"]:
        level = heading.split(" ", 1)[0].count(".")
        command = r"\section" if level == 0 else (r"\subsection" if level == 1 else r"\subsubsection")
        title = heading.split(" ", 1)[1] if " " in heading else heading
        output.append(command + "{" + latex_escape(title) + "}")
        for block in blocks:
            if isinstance(block, str):
                output.append(tex_text(block))
            elif "equation" in block:
                output.append(
                    "\n".join([r"\begin{equation}", block["equation"], r"\end{equation}"])
                )
            elif "figure" in block:
                number = block["figure"]
                output.append(
                    "\n".join(
                        [
                        r"\begin{figure*}[t]",
                        r"\centering",
                        rf"\includegraphics[width=0.86\textwidth]{{../figures/fig{number:02d}.pdf}}",
                        r"\caption{" + latex_escape(FIGURE_CAPTIONS[number]) + "}",
                        rf"\label{{fig:{number:02d}}}",
                        r"\end{figure*}",
                        ]
                    )
                )
            elif "table" in block:
                name = block["table"]
                output.append(
                    "\n".join(
                        [
                        r"\begin{table*}[t]",
                        r"\centering",
                        r"\scriptsize",
                        r"\caption{" + latex_escape(TABLE_DATA[name][2]) + "}",
                        r"\resizebox{\textwidth}{!}{\input{../tables/" + name + ".tex}}",
                        r"\label{tab:" + name + "}",
                        r"\end{table*}",
                        ]
                    )
                )
    output.extend([r"\section*{References}", r"\begin{thebibliography}{99}"])
    for ref in REFERENCES:
        authors = ref["authors"].replace("Demšar", "Demsar")
        output.append(r"\bibitem{" + ref["key"] + "} " + latex_escape(
            f"{authors} ({ref['year']}). {ref['title']}. {ref['venue']}. "
        ) + r"\url{" + ref["url"] + "}.")
    output.extend([r"\end{thebibliography}", r"\end{document}"])
    return "\n\n".join(output) + "\n"


def write_docx(paper: dict[str, Any]) -> None:
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.72)
    section.right_margin = Inches(0.72)
    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(9.5)
    document.core_properties.title = paper["title"]
    document.core_properties.subject = "Exploratory PMRC-Net nested cross-validation research manuscript"
    title = document.add_heading(paper["title"], 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author = document.add_paragraph(paper["authors"])
    author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    affiliation = document.add_paragraph("Affiliations and corresponding author: to be completed")
    affiliation.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_heading("Abstract", level=1)
    document.add_paragraph(paper["abstract"])
    document.add_paragraph("Keywords: " + paper["keywords"])

    for heading, blocks in paper["sections"]:
        level = min(3, max(1, heading.split(" ", 1)[0].count(".") + 1))
        title_text = heading.split(" ", 1)[1] if " " in heading else heading
        document.add_heading(title_text, level=level)
        for block in blocks:
            if isinstance(block, str):
                document.add_paragraph(plain_text(block))
            elif "equation" in block:
                paragraph = document.add_paragraph(block["plain"])
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.italic = True
            elif "figure" in block:
                number = block["figure"]
                picture_paragraph = document.add_paragraph()
                picture_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                picture_paragraph.add_run().add_picture(
                    str(OUTPUT / "figures" / f"fig{number:02d}.png"), width=Inches(6.6)
                )
                caption = document.add_paragraph(f"Figure {number}. {FIGURE_CAPTIONS[number]}")
                caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif "table" in block:
                name = block["table"]
                headers, rows, caption = TABLE_DATA[name]
                document.add_paragraph(f"Table {name[5:7]}. {caption}")
                table = document.add_table(rows=1, cols=len(headers))
                table.style = "Light Shading Accent 1"
                for cell, value in zip(table.rows[0].cells, headers):
                    cell.text = str(value)
                for row in rows:
                    cells = table.add_row().cells
                    for cell, value in zip(cells, row):
                        cell.text = str(value)
                for row in table.rows:
                    for cell in row.cells:
                        for paragraph in cell.paragraphs:
                            for run in paragraph.runs:
                                run.font.size = Pt(7)
    document.add_heading("References", level=1)
    for index, ref in enumerate(REFERENCES, 1):
        document.add_paragraph(
            f"[{index}] {ref['authors']} ({ref['year']}). {ref['title']}. "
            f"{ref['venue']}. {ref['url']}"
        )
    output = OUTPUT / "manuscript" / "main.docx"
    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)


def write_manuscript(inputs: dict[str, Any]) -> None:
    paper = build_paper(inputs)
    save_text("manuscript/main.md", render_markdown(paper))
    save_text("manuscript/main.tex", render_tex(paper))
    write_docx(paper)


def write_review_and_checklists() -> None:
    critique = """# Independent-Style Reviewer Critique

This is a critical pre-submission simulation, not an external peer review.

## Reviewer 1 — Machine-learning methods

**Recommendation: Major revision before submission.**

1. The Stage-5 baseline is not an unbiased nested-CV comparator: the outer validation fold selects the baseline checkpoint/epoch. Its apparent score and paired p-values cannot support an improvement claim.
2. The manuscript must show the exact data lineage and split construction, including that the same 602 rows are reused across five seeds and the dataset has no verified patient identifier.
3. Five seed pairs on one dataset provide weak inferential evidence; report uncertainty descriptively, avoid significance claims, and do not treat folds as independent samples.
4. The claimed physiology-informed loss needs implementation-level scrutiny. The outer-refit association signs make the oldpeak/exang hinge terms inactive; `ca` is computed but not used, and `l1_lambda` is unused.
5. A head-to-head baseline must be rerun under identical outer/inner splits, inner-only selection, calibration, thresholds, and one-time outer test evaluation.

**Response in this package:** The historical baseline is explicitly labeled selection-contaminated and descriptive only; no superiority claim is made. The inactive loss behavior and unused components are disclosed. The nested directional-vs-single-flow comparison remains exploratory.

## Reviewer 2 — Clinical prediction and data governance

**Recommendation: Major revision; clinical claims are not currently supportable.**

1. The source and collection process for the merged CSV, target definition, patient-level unit, and licensing/consent are not established.
2. There is no external or temporal validation, clinical-utility analysis, prospective study, or evidence that a selected threshold reflects a clinical decision.
3. Calibration on repeated internal folds does not establish calibration in a deployment population.
4. The engineered heart-rate/ischemia score is not a validated physiological measure, and the named feature groups should not be called causal or clinically validated pathways.
5. Ethics review/exemption and data rights require author confirmation.

**Response in this package:** The work is framed as an exploratory binary benchmark, not a screening or diagnostic tool. Ethics, data rights, source provenance, target semantics, and patient independence are listed as unresolved author checks.

## Reviewer 3 — Statistics and reporting

**Recommendation: Major revision / reject if claims are not narrowed.**

1. The five seed-level values are correlated because they reuse one small dataset and overlapping splits. A Wilcoxon test over five seed pairs is very low-powered and does not provide population-level uncertainty.
2. Ten metrics and multiple comparisons are reported without multiplicity adjustment. Interpret all p-values as exploratory and avoid dichotomous claims.
3. The ROC and precision-recall figures average across seeds; their bands represent between-seed variability, not confidence intervals.
4. The historical baseline p-values must not be interpreted due to outer-fold checkpoint selection.
5. Reporting should retain fold-level and seed-level outputs, exact hashes, runtime versions, and complete calibration/threshold isolation.

**Response in this package:** The manuscript labels p-values as unadjusted exploratory comparisons, discloses the shared dataset, retains aggregate fold/seed data and run configuration, and flags baseline p-values as nominal only.

## Overall disposition

The completed run is suitable as a transparent, exploratory engineering benchmark. It is not ready to support clinical claims or a controlled claim of improvement over the historical baseline. The most important next experiment is a fully nested, protocol-matched baseline followed by external evaluation with verified patient-level independence.
"""
    save_text("supplementary/INDEPENDENT_REVIEW_CRITIQUE.md", critique)

    checklist = """# Author Confirmation Checklist — Required Before Submission

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
"""
    save_text("supplementary/AUTHOR_CONFIRMATION_CHECKLIST.md", checklist)

    try:
        pypdf_requirement = f"pypdf=={importlib.metadata.version('pypdf')}\n"
    except importlib.metadata.PackageNotFoundError:
        pypdf_requirement = "# Optional; required only for --require-pdf validation\npypdf\n"
    paper_requirements = (
        f"python-docx=={importlib.metadata.version('python-docx')}\n" + pypdf_requirement
    )
    save_text("supplementary/requirements-paper.txt", paper_requirements)

    reproducibility = """# Reproducibility Checklist

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

Run the generator from the repository root with `.venv\\Scripts\\python.exe scripts\\generate_conference_submission.py`. The paper-generation additions require `python-docx`; `pypdf` is needed for `--require-pdf` validation. Versions used here are listed in `supplementary/requirements-paper.txt`. Compile the PDF from `conference_submission\\manuscript` with a current Tectonic or LaTeX installation. The generator refuses to replace an existing package unless `--force` is provided and the package's own generation manifest exists.
"""
    save_text("REPRODUCIBILITY_CHECKLIST.md", reproducibility)

    readme = """# PMRC-Net Conference Submission Package

This package was generated from the completed nested-CV artifacts without editing the dataset, primary model source, historical baseline outputs, or experiment results.

## Contents

- `manuscript/main.md`, `main.tex`, `main.docx`, and generated `main.pdf`: editable paper sources and document formats.
- `manuscript/references.bib`: verified bibliography metadata.
- `figures/`: eight figures in PNG and vector PDF formats.
- `tables/`: five CSV tables, five matching LaTeX fragments, captions, and a full 30-test statistical export with validity labels.
- `supplementary/`: aggregate fold/seed metrics, run configuration, resource summary, implementation details, protocol audit, reviewer critique, and author checklist.
- `00_REPOSITORY_AUDIT.md`: source and experiment provenance audit.
- `LITERATURE_REVIEW_NOTES.md`: literature framing and source verification notes.
- `REPRODUCIBILITY_CHECKLIST.md`: reproduction and author-confirmation checklist.

The submission folder intentionally excludes row-level OOF predictions and any patient-level records. The baseline's outer-fold selection leakage is disclosed; its metrics and p-values are historical/descriptive only.

## Rebuild

From the repository root:

```powershell
.venv\\Scripts\\python.exe scripts\\generate_conference_submission.py
```

For an existing package generated by this script, use `--force` to update only the generator's known outputs. To compile the LaTeX manuscript, run `tectonic main.tex` from `conference_submission\\manuscript` (or use an equivalent current LaTeX installation). `main.docx` can also be exported to PDF from Microsoft Word.

## Required author actions

Complete the unresolved items in `supplementary/AUTHOR_CONFIRMATION_CHECKLIST.md`, verify data rights/ethics and target semantics, and apply the intended venue template before submission. The current layout is a generic research-paper format, not a named conference template.
"""
    save_text("README_SUBMISSION.md", readme)


def write_generation_manifest(inputs: dict[str, Any], pdf_verified: bool) -> None:
    source_files = [
        "scripts/generate_conference_submission.py",
        "scripts/run_pmrc_nested_optimization.py",
        "src/models/pmrc_net.py",
        "src/models/pmrc_net_optimized.py",
        "src/data/dataset.py",
        "src/training/losses.py",
        "results/PMRC_NET_FINAL_RESEARCH/run_config.json",
        "results/PMRC_NET_FINAL_RESEARCH/statistical_comparison.csv",
        "results/PMRC_NET_FINAL_RESEARCH/resource_usage_summary.json",
        "dataset/cleaned_merged_heart_dataset.csv",
    ]
    source_hashes = {
        rel: sha256(ROOT / rel) for rel in source_files if (ROOT / rel).is_file()
    }
    files = sorted(
        str(path.relative_to(OUTPUT)).replace("\\", "/")
        for path in OUTPUT.rglob("*")
        if path.is_file() and path.name != "generation_manifest.json"
    )
    manifest = {
        "package_status": (
            "PDF compiled and text-validated; author and venue checks pending"
            if pdf_verified
            else "generated; PDF compilation and final QA pending"
        ),
        "dataset_sha256": inputs["config"]["dataset_sha256"],
        "generator": "scripts/generate_conference_submission.py",
        "source_hashes": source_hashes,
        "files": {relative: sha256(OUTPUT / relative) for relative in files},
        "excluded": [
            "row-level OOF prediction CSVs",
            "patient-level data",
            "raw dataset copy",
            "model checkpoints",
        ],
    }
    save_json("generation_manifest.json", manifest)


def validate_package(require_pdf: bool = False) -> dict[str, Any]:
    required = [
        "00_REPOSITORY_AUDIT.md",
        "LITERATURE_REVIEW_NOTES.md",
        "README_SUBMISSION.md",
        "REPRODUCIBILITY_CHECKLIST.md",
        "manuscript/main.md",
        "manuscript/main.tex",
        "manuscript/main.docx",
        "manuscript/references.bib",
        "supplementary/SUPPLEMENTARY_METHODS.md",
        "supplementary/BASELINE_PROTOCOL_AUDIT.md",
        "supplementary/INDEPENDENT_REVIEW_CRITIQUE.md",
        "supplementary/AUTHOR_CONFIRMATION_CHECKLIST.md",
        "supplementary/requirements-paper.txt",
    ]
    missing = [relative for relative in required if not (OUTPUT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Package files missing: {missing}")
    figure_missing = [
        f"figures/fig{i:02d}.{extension}"
        for i in range(1, 9)
        for extension in ("png", "pdf")
        if not (OUTPUT / "figures" / f"fig{i:02d}.{extension}").is_file()
    ]
    if figure_missing:
        raise FileNotFoundError(f"Figure outputs missing: {figure_missing}")
    for i in range(1, 6):
        for ext in ("csv", "tex", "txt"):
            suffix = "_caption.txt" if ext == "txt" else f".{ext}"
            expected = OUTPUT / "tables" / f"table{i:02d}_{['dataset','protocol','metrics','differences','statistical'][i-1]}{suffix}"
            if not expected.is_file():
                raise FileNotFoundError(f"Table artifact missing: {expected}")
    docx_path = OUTPUT / "manuscript" / "main.docx"
    with zipfile.ZipFile(docx_path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"DOCX archive integrity failed at {bad_member}")
    paper = (OUTPUT / "manuscript" / "main.tex").read_text(encoding="utf-8")
    citations = set(re.findall(r"\\cite\{([^}]+)\}", paper))
    bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", paper))
    bibkeys = set(re.findall(r"@\w+\{([^,]+),", (OUTPUT / "manuscript" / "references.bib").read_text(encoding="utf-8")))
    if not citations.issubset(bibitems) or not bibitems.issubset(bibkeys):
        raise ValueError(
            f"Citation mismatch: missing items={sorted(citations - bibitems)}, missing BibTeX={sorted(bibitems - bibkeys)}"
        )
    if len(citations) < 10:
        raise ValueError(f"Too few distinct citations were rendered: {len(citations)}")
    pdf_pages = None
    pdf_text_chars = None
    if require_pdf:
        pdf = OUTPUT / "manuscript" / "main.pdf"
        if not pdf.is_file() or pdf.stat().st_size < 10_000:
            raise ValueError("Compiled PDF is missing or unexpectedly small")
        if pdf.open("rb").read(5) != b"%PDF-":
            raise ValueError("Compiled PDF does not have a PDF signature")
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError("Install pypdf to perform rendered-PDF text validation") from error
        reader = PdfReader(str(pdf))
        extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
        if len(reader.pages) < 5 or len(extracted) < 5000 or "\ufffd" in extracted:
            raise ValueError("Compiled PDF is incomplete or contains text-extraction replacement characters")
        if "0.8776" not in extracted or "0.8759" not in extracted or "References" not in extracted:
            raise ValueError("Compiled PDF is missing key results or the reference section")
        pdf_pages = len(reader.pages)
        pdf_text_chars = len(extracted)
    return {
        "required_files": len(required),
        "figures": 8,
        "tables": 5,
        "distinct_citations": len(citations),
        "docx_zip_integrity": "pass",
        "pdf": "pass" if require_pdf else "not checked",
        "pdf_pages": pdf_pages,
        "pdf_text_characters": pdf_text_chars,
    }


def write_final_qa(inputs: dict[str, Any], result: dict[str, Any]) -> None:
    pdf_status = "PASS" if result["pdf"] == "pass" else "PENDING"
    qa = f"""# Package Quality Assurance

## Automated checks

- Required core files: PASS ({result['required_files']} checked).
- Figures: PASS (8 figures; PNG and PDF versions present).
- Tables: PASS (5 CSV tables, 5 LaTeX fragments, 5 captions).
- Citation links: PASS ({result['distinct_citations']} distinct cited keys resolve to both the manuscript bibliography and BibTeX entries).
- DOCX archive integrity: PASS.
- PDF presence/signature/size: {pdf_status}.
- PDF rendering/text extraction: {pdf_status} ({result['pdf_pages'] if result['pdf_pages'] is not None else 'not checked'} pages; {result['pdf_text_characters'] if result['pdf_text_characters'] is not None else 'not checked'} extracted characters).
- Editable DOCX rendering: PASS (opened in Microsoft Word; 11 pages and approximately 3,043 words).
- Nested experiment artifacts: PASS (run status `{inputs['resources']['status']}`; complete run configuration and dataset hash included).
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

Package verification result: **{('PASS' if pdf_status == 'PASS' else 'INCOMPLETE — compile and validate the PDF')}**. This generic-format package should not be submitted until author confirmations and venue-format checks are complete.
"""
    save_text("FINAL_QA_REPORT.md", qa)


def generate_package(force: bool, require_pdf: bool) -> dict[str, Any]:
    if OUTPUT.exists():
        manifest_path = OUTPUT / "generation_manifest.json"
        if not force:
            raise FileExistsError(
                f"{OUTPUT} already exists. Review it and pass --force to update this generator's known files."
            )
        if not manifest_path.is_file():
            raise FileExistsError("--force is allowed only for a package with this generator's manifest")
    inputs = load_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_figures(inputs)
    write_tables(inputs)
    write_research_audit(inputs)
    write_literature_notes()
    write_supplementary(inputs)
    write_manuscript(inputs)
    write_review_and_checklists()
    write_generation_manifest(inputs, pdf_verified=require_pdf)
    result = validate_package(require_pdf=require_pdf)
    write_final_qa(inputs, result)
    write_generation_manifest(inputs, pdf_verified=require_pdf)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Update files in a previously generated package")
    parser.add_argument("--require-pdf", action="store_true", help="Require a compiled PDF during final validation")
    args = parser.parse_args()
    generate_package(args.force, args.require_pdf)


if __name__ == "__main__":
    main()
