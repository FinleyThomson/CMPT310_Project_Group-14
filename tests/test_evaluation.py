import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from Data.Code import SyntheticData
from Model import CrossValidation
from Model.DecisionTree import DecisionTree, TreeNode
from Model.Evaluation import summarize_predictions


class CrossValidationMetricTests(unittest.TestCase):
    def test_confusion_matrix_uses_actual_rows_and_predicted_columns(self):
        actual = np.array([0, 0, 1, 2])
        predicted = np.array([0, 1, 1, 0])

        result = CrossValidation.getConfusionMatrix(predicted, actual, 3)

        expected = np.array(
            [
                [1, 1, 0],
                [0, 1, 0],
                [1, 0, 0],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_macro_f1_matches_sklearn_when_a_class_is_never_predicted(self):
        actual = np.array([0, 0, 1, 1, 2, 2])
        predicted = np.array([0, 0, 1, 1, 1, 1])

        result = CrossValidation.macroF1Score(predicted, actual, 3)
        expected = f1_score(
            actual,
            predicted,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )

        self.assertAlmostEqual(result, expected)


class EvaluationSummaryTests(unittest.TestCase):
    def test_summary_counts_ordinal_and_severe_errors(self):
        result = {
            "y_true": np.array([0, 1, 2, 0]),
            "y_pred": np.array([2, 1, 1, 0]),
            "folds": 2,
            "synthetic_samples_per_fold": 0,
            "random_state": 310,
            "test_accuracy": 0.5,
            "test_accuracy_std": 0.0,
            "test_macro_f1": 0.5,
            "test_macro_f1_std": 0.0,
            "training_accuracy": 0.75,
            "training_macro_f1": 0.75,
            "fold_metrics": [],
        }

        summary, per_class, matrix = summarize_predictions("test", result)

        self.assertEqual(summary["severe_low_high_errors"], 1)
        self.assertAlmostEqual(
            summary["ordinal_mean_absolute_error"],
            0.75,
        )
        self.assertEqual(int(matrix.sum()), 4)
        self.assertEqual(
            per_class["class"].tolist(),
            [
                "Low ROI",
                "Medium ROI",
                "High ROI",
            ],
        )


class DecisionTreeBoundaryTests(unittest.TestCase):
    def test_prediction_routes_threshold_value_to_left_child(self):
        tree = DecisionTree(1, 1, 0, 3, random_state=310)
        root = TreeNode(
            data=np.empty((0, 2)),
            feature_index=0,
            feature_val=1.0,
            prediction_probs=[0.0, 0.0, 1.0],
            information_gain=1.0,
        )
        root.left = TreeNode(
            data=np.empty((0, 2)),
            feature_index=None,
            feature_val=None,
            prediction_probs=[1.0, 0.0, 0.0],
            information_gain=0.0,
        )
        root.right = TreeNode(
            data=np.empty((0, 2)),
            feature_index=None,
            feature_val=None,
            prediction_probs=[0.0, 1.0, 0.0],
            information_gain=0.0,
        )
        tree.tree = root

        prediction = tree.prediction(np.array([[1.0]]))

        np.testing.assert_array_equal(prediction, np.array([0]))


class SyntheticDataTests(unittest.TestCase):
    def test_seeded_generator_is_reproducible_and_returns_requested_size(self):
        synthetic_columns = [
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
        rows = []
        for label in range(3):
            for offset in range(3):
                project_cost = 100 + 10 * label + offset
                initial_assessment = 200 + 10 * label + offset
                roi = (0.15, 0.45, 0.85)[label]
                row = {
                    column: float(10 + label + offset)
                    for column in synthetic_columns
                }
                row["EstProjectCost"] = project_cost
                row["Initial Assessment"] = initial_assessment
                row["Final Assessment"] = (
                    project_cost + initial_assessment
                ) * (1 + roi)
                row["Classification"] = label
                rows.append(row)
        source = pd.DataFrame(rows)
        thresholds = [0.3, 0.65]

        first = SyntheticData.multivariateLognormalDistributionGeneration(
            source,
            synthetic_columns,
            thresholds,
            10,
            random_state=310,
        )
        second = SyntheticData.multivariateLognormalDistributionGeneration(
            source,
            synthetic_columns,
            thresholds,
            10,
            random_state=310,
        )

        self.assertEqual(len(first), 10)
        self.assertTrue(set(first["Classification"]).issubset({0, 1, 2}))
        self.assertIn("ROI", first.columns)
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
