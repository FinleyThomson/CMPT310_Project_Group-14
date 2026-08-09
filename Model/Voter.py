from Model import RandomForest as rf
from Model import OrdinalLogisticRegression as ol
import os
import pandas as pd
from Model import CrossValidation as cv
import numpy as np



class Voter:
    
    def __init__(self, models, num_tree_synth):
        """Parameters
           ----------
           models: (list) list of models with 'proba' function which returns probabilities

        """
        self.models = models
        self.num_tree_synth = num_tree_synth

    def proba(self, X):
        self.results = []
        for model in self.models:
            results = model.proba(X)
            self.results.append(results)
        final_results = np.mean(self.results,axis=0)
        return final_results

    def fit(self, X_real, y_real, X_synth, y_synth):

        if len(y_synth) > 0:
            X_full = np.vstack((X_real, X_synth))
            y_full = np.hstack((y_real, y_synth))
            len_real = len(y_real)
            idx = len_real + self.num_tree_synth + 1
            self.models[0].fit(X_full[:idx], y_real[:idx])
            self.models[1].fit(X_full, y_full)
        else:
            for model in self.models:
                model.fit(X_real, y_real)

         
    def predict(self, X):
        preds = np.argmax(self.proba(X), axis = 1)
        return preds



def main():

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
    

    feature_columns_n = [
            "Initial Assessment",
            "Income",
            "Distance To Downtown",
            "Distance To Nearest Transit Stop",
            "Cost Per Unit",
            "Distance to Nearest Park",
            "Distance to Nearest Public School",
        ]

    synth_columns_n = [
            "EstProjectCost",
            "Initial Assessment",
            "Final Assessment",
            "Income",
            "Distance To Downtown",
            "Distance To Nearest Transit Stop",
            "Cost Per Unit",
            "Distance to Nearest Park",
            "Distance to Nearest Public School",
            "Classification"
        ]

    regressor = ol.Regressor(5000, 0.001, 3, 72)
    
    ol_results = cv.syntheticKFoldCrossValidation(
        df,
        feature_columns_n,
        synth_columns_n,
        regressor,
        k_folds,
        900,
        random_state=310,
        preprocess = True
    )

    forest = rf.RandomForest(60,3,120,10,7,0,3,True)

    rf_results = cv.syntheticKFoldCrossValidation(
            df,
            feature_columns_n,
            synth_columns_n,
            forest,
            k_folds,
            0,
            random_state=310,
            preprocess = True
        )

    models = [forest, regressor]

    voter = Voter(models, 0)

    voted_results =  cv.syntheticKFoldCrossValidation(
            df,
            feature_columns_n,
            synth_columns_n,
            voter,
            k_folds,
            900,
            random_state=310,
            preprocess = True,
            asymmetric = True
        )

    print("Custom ordinal regressor")
    print("Confusion matrix (actual rows, predicted columns):")
    print(ol_results[0])
    print("Train accuracy:", ol_results[1])
    print("Train macro-F1:", ol_results[2])
    print("Test accuracy:", ol_results[3])
    print("Test macro-F1:", ol_results[4])

    print("Custom random forest")
    print("Confusion matrix (actual rows, predicted columns):")
    print(rf_results[0])
    print("Train accuracy:", rf_results[1])
    print("Train macro-F1:", rf_results[2])
    print("Test accuracy:", rf_results[3])
    print("Test macro-F1:", rf_results[4])

    print("Voted")
    print("Confusion matrix (actual rows, predicted columns):")
    print(voted_results[0])
    print("Train accuracy:", voted_results[1])
    print("Train macro-F1:", voted_results[2])
    print("Test accuracy:", voted_results[3])
    print("Test macro-F1:", voted_results[4])


if __name__ == "__main__":
    main()
