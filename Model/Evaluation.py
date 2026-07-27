"""Reusable metrics and visual evidence for Milestone 2.

All functions in this module evaluate out-of-fold predictions for real
projects.  Synthetic training rows must never be passed as test evidence.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

CLASS_NAMES = ("Low ROI", "Medium ROI", "High ROI")


def summarize_predictions(
    model_name: str,
    cross_validation_result: dict[str, Any],
    class_names: Iterable[str] = CLASS_NAMES,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    """Create overall, per-class, and confusion-matrix results."""

    class_names = tuple(class_names)
    y_true = np.asarray(cross_validation_result["y_true"], dtype=int)
    y_pred = np.asarray(cross_validation_result["y_pred"], dtype=int)
    labels = np.arange(len(class_names))

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size == 0:
        raise ValueError("At least one prediction is required")
    if not np.isin(y_true, labels).all() or not np.isin(y_pred, labels).all():
        raise ValueError("Predictions contain a class outside class_names")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    absolute_class_error = np.abs(y_pred - y_true)
    severe_error_count = int(np.sum(absolute_class_error == labels[-1]))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    summary = {
        "model": model_name,
        "real_test_samples": int(y_true.size),
        "folds": int(cross_validation_result["folds"]),
        "synthetic_samples_per_fold": int(
            cross_validation_result["synthetic_samples_per_fold"]
        ),
        "random_state": cross_validation_result["random_state"],
        "feature_columns": list(cross_validation_result.get("feature_columns", [])),
        "model_parameters": cross_validation_result.get("model_parameters", {}),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "fold_test_accuracy_mean": float(cross_validation_result["test_accuracy"]),
        "fold_test_accuracy_std": float(cross_validation_result["test_accuracy_std"]),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "fold_test_macro_f1_mean": float(cross_validation_result["test_macro_f1"]),
        "fold_test_macro_f1_std": float(cross_validation_result["test_macro_f1_std"]),
        "weighted_f1": float(weighted_f1),
        "training_accuracy_mean": float(cross_validation_result["training_accuracy"]),
        "training_macro_f1_mean": float(cross_validation_result["training_macro_f1"]),
        "accuracy_generalization_gap": float(
            cross_validation_result["training_accuracy"]
            - cross_validation_result["test_accuracy"]
        ),
        "ordinal_mean_absolute_error": float(absolute_class_error.mean()),
        "severe_low_high_errors": severe_error_count,
        "severe_low_high_error_rate": float(severe_error_count / y_true.size),
    }

    per_class = pd.DataFrame(
        {
            "class": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )

    return summary, per_class, matrix


def save_evaluation_artifacts(
    model_name: str,
    cross_validation_result: dict[str, Any],
    output_dir: str | Path,
    class_names: Iterable[str] = CLASS_NAMES,
) -> dict[str, Any]:
    """Save report-ready metrics, fold results, and two confusion matrices."""

    class_names = tuple(class_names)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(model_name)

    summary, per_class, matrix = summarize_predictions(
        model_name,
        cross_validation_result,
        class_names,
    )
    normalized_matrix = _normalize_rows(matrix)

    metrics_path = output_dir / f"{slug}_metrics.json"
    per_class_path = output_dir / f"{slug}_per_class_metrics.csv"
    fold_metrics_path = output_dir / f"{slug}_fold_metrics.csv"
    matrix_path = output_dir / f"{slug}_confusion_matrix.csv"
    normalized_matrix_path = output_dir / f"{slug}_confusion_matrix_normalized.csv"
    counts_plot_path = output_dir / f"{slug}_confusion_matrix_counts.png"
    normalized_plot_path = output_dir / f"{slug}_confusion_matrix_normalized.png"

    metrics_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    per_class.to_csv(per_class_path, index=False)
    pd.DataFrame(cross_validation_result["fold_metrics"]).to_csv(
        fold_metrics_path,
        index=False,
    )
    _matrix_frame(matrix, class_names).to_csv(matrix_path)
    _matrix_frame(normalized_matrix, class_names).to_csv(normalized_matrix_path)

    _save_confusion_matrix_plot(
        matrix,
        class_names,
        f"{model_name}: held-out real projects (counts)",
        counts_plot_path,
        values_format="d",
    )
    _save_confusion_matrix_plot(
        normalized_matrix,
        class_names,
        f"{model_name}: recall by actual class",
        normalized_plot_path,
        values_format=".1%",
    )

    return {
        "summary": summary,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "paths": {
            "metrics": metrics_path,
            "per_class_metrics": per_class_path,
            "fold_metrics": fold_metrics_path,
            "confusion_matrix": matrix_path,
            "normalized_confusion_matrix": normalized_matrix_path,
            "confusion_matrix_plot": counts_plot_path,
            "normalized_confusion_matrix_plot": normalized_plot_path,
        },
    }


def save_model_comparison(
    evaluations: Iterable[dict[str, Any]],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Save a compact accuracy/macro-F1 comparison table and chart."""

    evaluations = list(evaluations)
    if not evaluations:
        raise ValueError("At least one evaluation is required")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for evaluation in evaluations:
        summary = evaluation["summary"]
        rows.append(
            {
                "model": summary["model"],
                "pooled_test_accuracy": summary["test_accuracy"],
                "fold_accuracy_mean": summary["fold_test_accuracy_mean"],
                "fold_accuracy_std": summary["fold_test_accuracy_std"],
                "pooled_macro_f1": summary["macro_f1"],
                "fold_macro_f1_mean": summary["fold_test_macro_f1_mean"],
                "fold_macro_f1_std": summary["fold_test_macro_f1_std"],
                "balanced_accuracy": summary["balanced_accuracy"],
                "ordinal_mae": summary["ordinal_mean_absolute_error"],
                "severe_error_rate": summary["severe_low_high_error_rate"],
            }
        )

    comparison = pd.DataFrame(rows)
    table_path = output_dir / "model_comparison.csv"
    plot_path = output_dir / "model_comparison.png"
    comparison.to_csv(table_path, index=False)

    positions = np.arange(len(comparison))
    width = 0.36
    fig, axis = plt.subplots(figsize=(max(8, len(comparison) * 2.4), 5.5))
    accuracy_bars = axis.bar(
        positions - width / 2,
        comparison["fold_accuracy_mean"],
        width,
        yerr=comparison["fold_accuracy_std"],
        capsize=4,
        label="Accuracy",
        color="#2878B5",
    )
    f1_bars = axis.bar(
        positions + width / 2,
        comparison["fold_macro_f1_mean"],
        width,
        yerr=comparison["fold_macro_f1_std"],
        capsize=4,
        label="Macro-F1",
        color="#E07B39",
    )

    axis.set_title("5-fold performance on held-out real projects")
    axis.set_ylabel("Score (higher is better)")
    axis.set_ylim(0, 1)
    axis.set_xticks(positions, comparison["model"], rotation=12, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.bar_label(accuracy_bars, fmt="%.3f", padding=3, fontsize=9)
    axis.bar_label(f1_bars, fmt="%.3f", padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return table_path, plot_path


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )


def _matrix_frame(
    matrix: np.ndarray,
    class_names: tuple[str, ...],
) -> pd.DataFrame:
    return pd.DataFrame(
        matrix,
        index=[f"actual_{name}" for name in class_names],
        columns=[f"predicted_{name}" for name in class_names],
    )


def _save_confusion_matrix_plot(
    matrix: np.ndarray,
    class_names: tuple[str, ...],
    title: str,
    output_path: Path,
    values_format: str,
) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )
    display.plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
        values_format=values_format,
    )
    axis.set_title(title)
    axis.set_xlabel("Predicted ROI class")
    axis.set_ylabel("Actual ROI class")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
