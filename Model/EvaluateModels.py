"""Generate reproducible Milestone 2 model-evaluation evidence.

Example
-------
python3 -m Model.EvaluateModels

The default run compares a majority-class baseline with scikit-learn random
forests trained on real data alone and on real plus synthetic data.  Add
``--models dummy sklearn custom`` to include the custom implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import itertools

import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from Model import CrossValidation as cv
from Model.Evaluation import (
    CLASS_NAMES,
    save_evaluation_artifacts,
    save_model_comparison,
)
from Model.RandomForest import RandomForest
from Model.OrdinalLogisticRegression import Regressor
from Model.Voter import Voter

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = (
    REPOSITORY_ROOT / "Data" / "CSVs" / "Sorted" / "TH_DATA_BY_PROJECT_FINAL.csv"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "Presentation Artifacts" 

FEATURE_COLUMNS = [
    "Initial Assessment",
    "Income",
    "Distance To Downtown",
    "Distance To Nearest Transit Stop",
    "Cost Per Unit",
    "Distance to Nearest Park",
    "Distance to Nearest Public School",
]
SYNTHETIC_COLUMNS = [
    "EstProjectCost",
    "Initial Assessment",
    "Final Assessment",
    "Income",
    "Distance To Downtown",
    "Distance To Nearest Transit Stop",
    "Cost Per Unit",
    "Distance to Nearest Park",
    "Distance to Nearest Public School",
    "Classification",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ROI classifiers on held-out real projects and save "
            "Milestone 2 metrics/plots."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to TH_DATA_BY_PROJECT_FINAL.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV, JSON, and PNG evidence."
    )

    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=310)
    parser.add_argument("--trees", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=120)
    parser.add_argument(
            "--synthetic-samples-rf",
            type=int,
            nargs="+",
            default=[0, 1200],
            help=(
                "Synthetic rows per training fold for the random forest. Defaults to a real-only and a "
                "900-row augmentation experiment."
            ),
        )
    parser.add_argument(
        "--synthetic-samples-olr",
        type=int,
        nargs="+",
        default=[0, 1200],
        help=(
            "Synthetic rows per training fold for ordinial logistic regression. Defaults to a real-only and a "
            "900-row augmentation experiment."
        ),
    )
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("dummy", "sklearn-rf", "custom-rf","custom-olr","custom-ensemble"),
        default=("dummy", "custom-ensemble"),
        help="Models to compare.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_state = {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_worktree_dirty": bool(_git_output("status", "--porcelain")),
    }
    data = pd.read_csv(args.data)
    _validate_dataset(data)

    class_counts = (
        data["Classification"].value_counts().sort_index().astype(int).to_dict()
    )
    print(f"Loaded {len(data)} real projects from {args.data}")
    print(
        "Classes:",
        ", ".join(
            f"{CLASS_NAMES[label]}={count}" for label, count in class_counts.items()
        ),
    )
    print(
        "Leakage control: every test fold contains real held-out projects; "
        "synthetic rows are generated from its training fold only."
    )

    evaluations = []

    if "dummy" in args.models:
        dummy = DummyClassifier(strategy="most_frequent")
        evaluations.append(
            _run_experiment(
                "Majority baseline (real only)",
                dummy,
                data,
                args,
                synthetic_samples=0,
                asymmetric = False
            )
        )

    for synthetic_samples in dict.fromkeys(args.synthetic_samples_rf):
        estimated_real_train_size = int(len(data) * (args.folds - 1) / args.folds)
        bootstrap_samples = min(
            args.bootstrap_samples,
            estimated_real_train_size + synthetic_samples,
        )
        training_label = (
            "real only" if synthetic_samples == 0 else f"+{synthetic_samples} synthetic"
        )

        if "sklearn" in args.models:
            sklearn_forest = RandomForestClassifier(
                n_estimators=args.trees,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples,
                max_features=3,
                bootstrap=True,
                max_samples=bootstrap_samples,
                random_state=args.seed,
                n_jobs=-1,
            )
            evaluations.append(
                _run_experiment(
                    f"scikit RF ({training_label})",
                    sklearn_forest,
                    data,
                    args,
                    synthetic_samples,
                    asymmetric = False
                )
            )

        if "custom-rf" in args.models:
            custom_forest = RandomForest(
                num_trees=args.trees,
                num_splitting_features=3,
                bootstrap_sample_size=bootstrap_samples,
                max_depth=args.max_depth,
                min_samples=args.min_samples,
                min_information=0,
                num_classifications=len(CLASS_NAMES),
                with_replacement=True,
                random_state=args.seed,
            )
            evaluations.append(
                _run_experiment(
                    f"Custom RF ({training_label})",
                    custom_forest,
                    data,
                    args,
                    synthetic_samples,
                    asymmetric = False
                )
            )

    for synthetic_samples in dict.fromkeys(args.synthetic_samples_olr):
            estimated_real_train_size = int(len(data) * (args.folds - 1) / args.folds)
            training_label = (
                "real only" if synthetic_samples == 0 else f"+{synthetic_samples} synthetic"
            )

            if "custom-olr" in args.models:
                custom_regressor = Regressor(
                    max_iter = args.max_iter,
                    learning_rate = args.learning_rate,
                    num_classes = len(CLASS_NAMES),
                    batch_size = args.batch_size,
                    seed=args.seed,
                )
                evaluations.append(
                    _run_experiment(
                        f"Custom OLR ({training_label})",
                        custom_regressor,
                        data,
                        args,
                        synthetic_samples,
                        asymmetric = False
                    )
                )

    for synthetic_samples_olr in dict.fromkeys(args.synthetic_samples_olr):
        for synthetic_samples_rf in  dict.fromkeys(args.synthetic_samples_rf):
                estimated_real_train_size = int(len(data) * (args.folds - 1) / args.folds)
                bootstrap_samples = min(
                    args.bootstrap_samples,
                    estimated_real_train_size + synthetic_samples_rf,
                )
                if synthetic_samples_olr == 0 and synthetic_samples_rf == 0:
                    training_label = "real only"
                elif synthetic_samples_olr == 0 and synthetic_samples_rf > 0:
                    training_label = f"+{synthetic_samples_rf} synthetic rf, olr real only"
                elif synthetic_samples_olr > 0 and synthetic_samples_rf == 0:
                    training_label = f"+{synthetic_samples_olr} synthetic olr, rf real only"
                else:
                    training_label = f"+{synthetic_samples_olr} synthetic olr, +{synthetic_samples_rf} synthetic rf"
    
                if "custom-ensemble" in args.models:
                    custom_regressor = Regressor(
                        max_iter = args.max_iter,
                        learning_rate = args.learning_rate,
                        num_classes = len(CLASS_NAMES),
                        batch_size = args.batch_size,
                        seed=args.seed,
                    )
                    custom_forest = RandomForest(
                        num_trees=args.trees,
                        num_splitting_features=3,
                        bootstrap_sample_size=bootstrap_samples,
                        max_depth=args.max_depth,
                        min_samples=args.min_samples,
                        min_information=0,
                        num_classifications=len(CLASS_NAMES),
                        with_replacement=True,
                        random_state=args.seed,
                    )
                    ensemble = Voter(
                        models = [custom_forest, custom_regressor],
                        num_tree_synth = synthetic_samples_rf,
                        num_olr_synth = synthetic_samples_olr
                    )
                    evaluations.append(
                        _run_experiment(
                            f"Ensemble ({training_label})",
                            ensemble,
                            data,
                            args,
                            max(synthetic_samples_olr, synthetic_samples_rf),
                            asymmetric = True
                        )
                    )

    table_path, plot_path = save_model_comparison(
        evaluations,
        args.output_dir,
    )
    manifest_path = _save_run_manifest(
        args,
        data,
        class_counts,
        evaluations,
        table_path,
        plot_path,
        source_state,
    )

    print("\nSummary (all test rows are real and out-of-fold)")
    for evaluation in evaluations:
        summary = evaluation["summary"]
        print(
            f"- {summary['model']}: "
            f"accuracy={summary['fold_test_accuracy_mean']:.3f}"
            f"±{summary['fold_test_accuracy_std']:.3f}, "
            f"macro-F1={summary['fold_test_macro_f1_mean']:.3f}"
            f"±{summary['fold_test_macro_f1_std']:.3f}, "
            f"ordinal-MAE={summary['ordinal_mean_absolute_error']:.3f}, "
            f"severe errors={summary['severe_low_high_errors']}"
        )
    print(f"\nComparison table: {table_path}")
    print(f"Comparison plot:  {plot_path}")
    print(f"Run manifest:     {manifest_path}")
    print(f"All evidence:     {args.output_dir}")


def _run_experiment(
    model_name: str,
    model,
    data: pd.DataFrame,
    args: argparse.Namespace,
    synthetic_samples: int,
    asymmetric: bool
):
    print(f"\nEvaluating {model_name}")
    result = cv.syntheticKFoldCrossValidation(
        data,
        FEATURE_COLUMNS,
        SYNTHETIC_COLUMNS,
        model,
        args.folds,
        synthetic_samples,
        random_state=args.seed,
        return_details=True,
        verbose=True,
        asymmetric = asymmetric
    )
    result["feature_columns"] = FEATURE_COLUMNS
    result["model_parameters"] = _model_parameters(model)
    return save_evaluation_artifacts(
        model_name,
        result,
        args.output_dir,
        CLASS_NAMES,
    )


def _validate_dataset(data: pd.DataFrame) -> None:
    required_columns = set(FEATURE_COLUMNS) | set(SYNTHETIC_COLUMNS)
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    if data[list(required_columns)].isna().any().any():
        raise ValueError("Modeling columns contain missing values")

    labels = sorted(data["Classification"].unique().tolist())
    expected_labels = list(range(len(CLASS_NAMES)))
    if labels != expected_labels:
        raise ValueError(f"Expected class labels {expected_labels}, found {labels}")

    leaked_features = {"Final Assessment", "ROI", "Classification"} & set(
        FEATURE_COLUMNS
    )
    if leaked_features:
        raise ValueError(
            f"Outcome columns must not be model inputs: {sorted(leaked_features)}"
        )


def _model_parameters(model) -> dict:
    if hasattr(model, "get_params"):
        return {
            key: _json_safe(value)
            for key, value in model.get_params(deep=False).items()
        }

    parameter_names = (
        "num_trees",
        "num_splitting_features",
        "bootstrapping_sample_size",
        "max_depth",
        "min_samples",
        "min_information",
        "num_classifications",
        "with_replacement",
        "max_iter",
        "learning_rate",
        "num_classes",
        "batch_size"
    )
    return {
        name: _json_safe(getattr(model, name))
        for name in parameter_names
        if hasattr(model, name)
    }


def _save_run_manifest(
    args: argparse.Namespace,
    data: pd.DataFrame,
    class_counts: dict,
    evaluations: list[dict],
    table_path: Path,
    plot_path: Path,
    source_state: dict,
) -> Path:
    manifest_path = args.output_dir / "run_manifest.json"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(
            [sys.executable, "-m", "Model.EvaluateModels", *sys.argv[1:]]
        ),
        **source_state,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "dataset": {
            "path": str(args.data.resolve()),
            "sha256": _sha256(args.data),
            "real_rows": len(data),
            "class_counts": {
                CLASS_NAMES[label]: count for label, count in class_counts.items()
            },
        },
        "configuration": {
            "folds": args.folds,
            "seed": args.seed,
            "trees": args.trees,
            "max_depth": args.max_depth,
            "min_samples": args.min_samples,
            "bootstrap_samples": args.bootstrap_samples,
            "requested_synthetic_samples_rf": args.synthetic_samples_rf,
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "requested_synthetic_samples_olr": args.synthetic_samples_olr,
            "models": args.models,
            "feature_columns": FEATURE_COLUMNS,
        },
        "experiments": [
            {
                "model": evaluation["summary"]["model"],
                "synthetic_samples_per_fold": evaluation["summary"][
                    "synthetic_samples_per_fold"
                ],
                "model_parameters": evaluation["summary"]["model_parameters"],
            }
            for evaluation in evaluations
        ],
        "comparison_table": str(table_path.resolve()),
        "comparison_plot": str(plot_path.resolve()),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as data_file:
        for chunk in iter(lambda: data_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


if __name__ == "__main__":
    main()
