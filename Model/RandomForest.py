import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

try:
    from . import CrossValidation as cv
    from . import DecisionTree as dt
except ImportError:
    import CrossValidation as cv
    import DecisionTree as dt


# ------------
# Constants
# ------------

NUM_CLASSIFICATIONS = 3

# ------------------
# Hyperparameters
# ------------------

NUM_TREES = 100
NUM_SPLITTING_FEATURES = 3
BOOTSTRAP_SAMPLE_SIZE = 1500
MAX_DEPTH = 7
MIN_SAMPLES = 5
MIN_INFORMATION = 0
WITH_REPLACEMENT = True
RANDOM_STATE = 310


# ----------------------
# Random Forest Class
# ----------------------

class RandomForest():

    def __init__(
        self,
        num_trees,
        num_splitting_features,
        bootstrap_sample_size,
        max_depth,
        min_samples,
        min_information,
        num_classifications,
        with_replacement,
        random_state=None,
    ):
        """Creat Random Forest, takes self, num_trees, num_splitting_features, bootstrap_sample_size, max_depth, min_samples, min_information, num_classifications"""
        self.num_trees = num_trees
        self.num_splitting_features = num_splitting_features
        self.bootstrapping_sample_size = bootstrap_sample_size
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_information = min_information
        self.num_classifications = num_classifications
        self.with_replacement = with_replacement
        self.rng = np.random.default_rng(random_state)


    def bootstrappingSample(self, data):
        """Takes in  data (classifications included) and outputs the indices of a random sample with replacement"""

        high = len(data)
        indices = self.rng.choice(
            high,
            self.bootstrapping_sample_size,
            replace=self.with_replacement,
        )

        return indices


    def createForest(self, X_values, y_values):
        """Creates the forest"""

        forest = []

        for i in range(self.num_trees):
            bootstrap_indices = self.bootstrappingSample(y_values)
            bootstrap_X = X_values[bootstrap_indices]
            bootstrap_Y = y_values[bootstrap_indices]
            
            tree_seed = int(self.rng.integers(0, np.iinfo(np.int32).max))
            tree = dt.DecisionTree(
                self.max_depth,
                self.min_samples,
                self.min_information,
                self.num_classifications,
                num_splitting_features=self.num_splitting_features,
                random_state=tree_seed,
            )
            tree.train(bootstrap_X, bootstrap_Y)

            forest.append(tree)

        return np.array(forest, dtype=object)


        
    def fit(self, X_values, y_values):
    
        self.forest = self.createForest(X_values, y_values)


    def treePreditcion(self, tree: dt.DecisionTree, data):
         """Returns Prediction for a single tree on a set of data returns the array of predictions."""

         prediction_probs = tree.predictionProbs(data)

         return prediction_probs #2D Array


    def proba(self, data):
        """Returns the prediction for the whole forest of a set of data."""

        prediction_probs = []

        for i in range(self.num_trees):
            prediction_probs.append(self.treePreditcion(self.forest[i], data)) #3D array, contains the probailites for predictions for each data point by every tree

        averaged_probs = np.mean(prediction_probs, axis = 0) #2D Array with averages for each classification probability for each data point
        
        return averaged_probs


    def predict(self, data):

        preds = np.argmax(self.proba(data), axis = 1)

        return preds

    def printForestSize(self):

        print(np.size(self.forest))



# ---------------------------------
# Testing the forest with the data
# ---------------------------------

def main():
    """Run the original custom-versus-scikit random-forest comparison."""

    k_folds = 5
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(
        parent_dir,
        "Data",
        "CSVs",
        "Sorted",
        "TH_DATA_BY_PROJECT_FINAL.csv",
    )
    df = pd.read_csv(path)

    feature_columns = [
        "Initial Assessment",
        "Income",
        "Distance To Downtown",
        "Distance To Nearest Transit Stop",
        "Cost Per Unit",
        "Distance to Nearest Park",
        "Distance to Nearest Public School",
    ]
    synth_columns = [
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

    my_forest = RandomForest(
        NUM_TREES,
        NUM_SPLITTING_FEATURES,
        BOOTSTRAP_SAMPLE_SIZE,
        MAX_DEPTH,
        MIN_SAMPLES,
        MIN_INFORMATION,
        NUM_CLASSIFICATIONS,
        WITH_REPLACEMENT,
        random_state=RANDOM_STATE,
    )
    custom_results = cv.syntheticKFoldCrossValidation(
        df,
        feature_columns,
        synth_columns,
        my_forest,
        k_folds,
        3000,
        random_state=RANDOM_STATE,
    )

    print("Custom random forest")
    print("Confusion matrix (actual rows, predicted columns):")
    print(custom_results[0])
    print("Train accuracy:", custom_results[1])
    print("Train macro-F1:", custom_results[2])
    print("Test accuracy:", custom_results[3])
    print("Test macro-F1:", custom_results[4])

    sk_forest = RandomForestClassifier(
        n_estimators=NUM_TREES,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES,
        max_features=NUM_SPLITTING_FEATURES,
        bootstrap=True,
        max_samples=BOOTSTRAP_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    sklearn_results = cv.syntheticKFoldCrossValidation(
        df,
        feature_columns,
        synth_columns,
        sk_forest,
        k_folds,
        3000,
        random_state=RANDOM_STATE,
    )

    print("\nscikit-learn random forest")
    print("Confusion matrix (actual rows, predicted columns):")
    print(sklearn_results[0])
    print("Train accuracy:", sklearn_results[1])
    print("Train macro-F1:", sklearn_results[2])
    print("Test accuracy:", sklearn_results[3])
    print("Test macro-F1:", sklearn_results[4])


if __name__ == "__main__":
    main()
