"""Cross-validation helpers for the ROI classifiers.

Synthetic samples are generated from each training fold only.  Every reported
test prediction is therefore made for a real project that was not used to fit
the model or the synthetic-data distribution.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Data.Code import SyntheticData as sd


def syntheticKFoldCrossValidation(
    data: pd.DataFrame,
    feature_cols: list[str],
    synthetic_info_cols: list[str],
    model: Any,
    k: int,
    n: int,
    random_state: int | None = 310,
    return_details: bool = False,
    verbose: bool = True,
    preprocess: bool = False,
    asymmetric = False
):
    """Evaluate a classifier with stratified K-fold cross-validation.

    Parameters
    ----------
    data:
        Real project data.  The final column in ``synthetic_info_cols`` must be
        the integer class label.
    feature_cols:
        Columns available to the classifier at prediction time.
    synthetic_info_cols:
        Columns needed by the synthetic generator, ending with the class label.
    model:
        Any classifier implementing ``fit(X, y)`` and ``predict(X)``.
    k:
        Number of folds.
    n:
        Number of synthetic training rows generated per fold.  Use ``0`` for a
        real-data-only experiment.
    random_state:
        Seed used for fold assignment and synthetic generation.
    return_details:
        When true, return fold scores and out-of-fold predictions in a
        dictionary.  The default preserves the original five-value return.
    verbose:
        Print the real/synthetic train and real test sizes for each fold.
    preprocess:
        preprocess the data if True.
    asymmetric:
        when running an ensemble, if some models needs synthetic data when some do not

    Returns
    -------
    tuple or dict
        By default: confusion matrix, mean training accuracy, mean training
        macro-F1, mean test accuracy, and mean test macro-F1.  Confusion-matrix
        rows are actual classes and columns are predicted classes.
    """

    if k < 2:
        raise ValueError("k must be at least 2")
    if n < 0:
        raise ValueError("n must be non-negative")
    if not synthetic_info_cols:
        raise ValueError("synthetic_info_cols must include a target column")

    required_columns = set(feature_cols) | set(synthetic_info_cols)
    if n:
        required_columns.add("ROI")
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    target_col = synthetic_info_cols[-1]
    selected_columns = list(synthetic_info_cols)
    if n:
        selected_columns.append("ROI")
    real_data = data[selected_columns].copy()
    real_data[target_col] = real_data[target_col].astype(int)
    y_all = real_data[target_col].to_numpy()
    class_labels = np.sort(np.unique(y_all))

    if not np.array_equal(class_labels, np.arange(len(class_labels))):
        raise ValueError("Class labels must be consecutive integers starting at 0")

    class_counts = pd.Series(y_all).value_counts()
    if class_counts.min() < k:
        raise ValueError("Each class must contain at least k real examples")

    splitter = StratifiedKFold(
        n_splits=k,
        shuffle=True,
        random_state=random_state,
    )

    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=int)
    fold_metrics: list[dict[str, float | int]] = []
    all_actuals: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    for fold_number, (train_indices, test_indices) in enumerate(
        splitter.split(real_data, y_all),
        start=1,
    ):
        train_real = real_data.iloc[train_indices].reset_index(drop=True)
        test_real = real_data.iloc[test_indices].reset_index(drop=True)

        X_test = test_real[feature_cols].to_numpy()
        y_test = test_real[target_col].to_numpy()

        if n:
            thresholds = [
                train_real.loc[
                    train_real[target_col] == category,
                    "ROI",
                ].max()
                for category in class_labels[:-1]
            ]
            fold_seed = None if random_state is None else random_state + fold_number
            synth_data = sd.multivariateLognormalDistributionGeneration(
                train_real,
                synthetic_info_cols,
                thresholds,
                n,
                random_state=fold_seed,
            )
            training_data = pd.concat(
                [train_real, synth_data],
                ignore_index=True,
            )
        else:
            training_data = train_real

        X_train = training_data[feature_cols].to_numpy()
        y_train = training_data[target_col].astype(int).to_numpy()

        if preprocess:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        if not asymmetric:
            model.fit(X_train, y_train)
        else:
            leny = len(y_train)
            model.fit(X_train[0:leny],y_train[0:leny],X_train[leny:],y_train[leny:])
        training_predictions = np.asarray(model.predict(X_train))
        test_predictions = np.asarray(model.predict(X_test))

        train_accuracy = accuracy(training_predictions, y_train)
        train_macro_f1 = macroF1Score(
            training_predictions,
            y_train,
            len(class_labels),
        )
        test_accuracy = accuracy(test_predictions, y_test)
        test_macro_f1 = macroF1Score(
            test_predictions,
            y_test,
            len(class_labels),
        )

        fold_metrics.append(
            {
                "fold": fold_number,
                "real_train_size": len(train_real),
                "synthetic_train_size": len(training_data) - len(train_real),
                "real_test_size": len(test_real),
                "train_accuracy": train_accuracy,
                "train_macro_f1": train_macro_f1,
                "test_accuracy": test_accuracy,
                "test_macro_f1": test_macro_f1,
            }
        )
        confusion += getConfusionMatrix(
            test_predictions,
            y_test,
            len(class_labels),
        )
        all_actuals.append(y_test)
        all_predictions.append(test_predictions)

        if verbose:
            print(
                f"Fold {fold_number}/{k}: "
                f"{len(train_real)} real + "
                f"{len(training_data) - len(train_real)} synthetic train, "
                f"{len(test_real)} real test"
            )

    train_accuracies = np.array(
        [row["train_accuracy"] for row in fold_metrics],
        dtype=float,
    )
    train_f1_scores = np.array(
        [row["train_macro_f1"] for row in fold_metrics],
        dtype=float,
    )
    test_accuracies = np.array(
        [row["test_accuracy"] for row in fold_metrics],
        dtype=float,
    )
    test_f1_scores = np.array(
        [row["test_macro_f1"] for row in fold_metrics],
        dtype=float,
    )

    if return_details:
        return {
            "confusion_matrix": confusion,
            "training_accuracy": float(train_accuracies.mean()),
            "training_macro_f1": float(train_f1_scores.mean()),
            "test_accuracy": float(test_accuracies.mean()),
            "test_macro_f1": float(test_f1_scores.mean()),
            "test_accuracy_std": float(test_accuracies.std(ddof=1)),
            "test_macro_f1_std": float(test_f1_scores.std(ddof=1)),
            "fold_metrics": fold_metrics,
            "y_true": np.concatenate(all_actuals),
            "y_pred": np.concatenate(all_predictions),
            "class_labels": class_labels,
            "folds": k,
            "synthetic_samples_per_fold": n,
            "random_state": random_state,
        }

    return (
        confusion,
        float(train_accuracies.mean()),
        float(train_f1_scores.mean()),
        float(test_accuracies.mean()),
        float(test_f1_scores.mean()),
    )


def getConfusionMatrix(predictions, actuals, n):
    """Return an ``n`` by ``n`` matrix with actual rows and predicted columns."""

    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)
    if predictions.shape != actuals.shape:
        raise ValueError("predictions and actuals must have the same shape")

    return np.array(
        [
            [
                np.sum((actuals == actual_class) & (predictions == predicted_class))
                for predicted_class in range(n)
            ]
            for actual_class in range(n)
        ],
        dtype=int,
    )


def accuracy(predictions, actuals):
    """Return the fraction of exactly correct predictions."""

    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)
    if predictions.shape != actuals.shape:
        raise ValueError("predictions and actuals must have the same shape")
    if len(actuals) == 0:
        raise ValueError("actuals must not be empty")

    return float(np.mean(predictions == actuals))


def macroF1Score(predictions, actuals, num_classes):
    """Return unweighted mean F1, assigning zero to an undefined class F1."""

    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)
    if predictions.shape != actuals.shape:
        raise ValueError("predictions and actuals must have the same shape")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")

    class_f1_scores = []

    for class_label in range(num_classes):
        true_positives = np.sum((predictions == class_label) & (actuals == class_label))
        false_positives = np.sum(
            (predictions == class_label) & (actuals != class_label)
        )
        false_negatives = np.sum(
            (predictions != class_label) & (actuals == class_label)
        )

        precision_denominator = true_positives + false_positives
        recall_denominator = true_positives + false_negatives
        precision = (
            true_positives / precision_denominator if precision_denominator else 0.0
        )
        recall = true_positives / recall_denominator if recall_denominator else 0.0
        f1_denominator = precision + recall
        class_f1_scores.append(
            2 * precision * recall / f1_denominator if f1_denominator else 0.0
        )

    return float(np.mean(class_f1_scores))
