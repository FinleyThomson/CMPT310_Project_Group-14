from Model import RandomForest as rf
from Model import OrdinalLogisticRegression as ol
import os
import pandas as pd
from Model import CrossValidation as cv
import numpy as np
import random



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



# def main():

#     k_folds = 5
#     parent_dir = os.path.dirname(os.path.dirname(__file__))
#     path = os.path.join(
#         parent_dir,
#         "Data",
#         "CSVs",
#         "Sorted",
#         "TH_DATA_BY_PROJECT_FINAL.csv",
#     )
#     df = pd.read_csv(path)
    

#     feature_columns_n = [
#             "Initial Assessment",
#             "Income",
#             "Distance To Downtown",
#             "Distance To Nearest Transit Stop",
#             "Cost Per Unit",
#             "Distance to Nearest Park",
#             "Distance to Nearest Public School",
#         ]

#     synth_columns_n = [
#             "EstProjectCost",
#             "Initial Assessment",
#             "Final Assessment",
#             "Income",
#             "Distance To Downtown",
#             "Distance To Nearest Transit Stop",
#             "Cost Per Unit",
#             "Distance to Nearest Park",
#             "Distance to Nearest Public School",
#             "Classification"
#         ]

#     regressor = ol.Regressor(
#         max_iter=5000, 
#         learning_rate=0.001, 
#         num_classes=3, 
#         batch_size=72
#     )
    
#     # ol_results = cv.syntheticKFoldCrossValidation(
#     #     df,
#     #     feature_columns_n,
#     #     synth_columns_n,
#     #     regressor,
#     #     k_folds,
#     #     900,
#     #     random_state=310,
#     #     preprocess = True
#     # )

#     forest = rf.RandomForest(
#         num_trees=60,
#         num_splitting_features=3,
#         bootstrap_sample_size=120,
#         max_depth=10,
#         min_samples=7,
#         min_information=0,
#         num_classifications=3,
#         with_replacement=True
#     )

#     # rf_results = cv.syntheticKFoldCrossValidation(
#     #         df,
#     #         feature_columns_n,
#     #         synth_columns_n,
#     #         forest,
#     #         k_folds,
#     #         0,
#     #         random_state=310,
#     #         preprocess = True
#     #     )

#     models = [forest, regressor]

#     voter = Voter(models, 0)

#     voted_results =  cv.syntheticKFoldCrossValidation(
#             df,
#             feature_columns_n,
#             synth_columns_n,
#             voter,
#             k_folds,
#             900,
#             random_state=310,
#             preprocess = True,
#             asymmetric = True
#         )

#     # print("Custom ordinal regressor")
#     # print("Confusion matrix (actual rows, predicted columns):")
#     # print(ol_results[0])
#     # print("Train accuracy:", ol_results[1])
#     # print("Train macro-F1:", ol_results[2])
#     # print("Test accuracy:", ol_results[3])
#     # print("Test macro-F1:", ol_results[4])

#     # print("Custom random forest")
#     # print("Confusion matrix (actual rows, predicted columns):")
#     # print(rf_results[0])
#     # print("Train accuracy:", rf_results[1])
#     # print("Train macro-F1:", rf_results[2])
#     # print("Test accuracy:", rf_results[3])
#     # print("Test macro-F1:", rf_results[4])

#     print("Voted")
#     print("Confusion matrix (actual rows, predicted columns):")
#     print(voted_results[0])
#     print("Train accuracy:", voted_results[1])
#     print("Train macro-F1:", voted_results[2])
#     print("Test accuracy:", voted_results[3])
#     print("Test macro-F1:", voted_results[4])

def main():
    k_folds = 5
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(
        parent_dir, "Data", "CSVs", "Sorted", "TH_DATA_BY_PROJECT_FINAL.csv"
    )
    df = pd.read_csv(path)

    feature_columns_n = [
        "Initial Assessment", "Income", "Distance To Downtown",
        "Distance To Nearest Transit Stop", "Cost Per Unit",
        "Distance to Nearest Park", "Distance to Nearest Public School",
    ]

    synth_columns_n = [
        "EstProjectCost", "Initial Assessment", "Final Assessment",
        "Income", "Distance To Downtown", "Distance To Nearest Transit Stop",
        "Cost Per Unit", "Distance to Nearest Park",
        "Distance to Nearest Public School", "Classification"
    ]

    # 1. Updated Grid for Balanced Tertiles
    param_grid = {
        'rf_trees': [60, 100, 150],          # RFs plateau early on small balanced data
        'rf_depth': [5, 7, 10],              # Shallower trees prevent overfitting the 180 training points
        'rf_min_samples': [3, 5, 7],
        'rf_bootstrap': [120, 150, 180],     
        
        'olr_lr': [0.01, 0.005, 0.001],      # Kept standard to avoid exploding gradients
        'olr_batch': [32, 64, 72],
        
        'synth_samples': [600, 900, 1200]
    }

    num_random_tests = 20
    best_accuracy = 0  # <--- Changed tracking variable
    best_params = None
    results_log = []

    print(f"Starting Hyperparameter Search ({num_random_tests} iterations)...")
    print("Optimizing for: TEST ACCURACY (Tertile Balanced)")
    print("-" * 50)

    for i in range(num_random_tests):
        p = {
            'rf_trees': random.choice(param_grid['rf_trees']),
            'rf_depth': random.choice(param_grid['rf_depth']),
            'rf_min_samples': random.choice(param_grid['rf_min_samples']),
            'rf_bootstrap': random.choice(param_grid['rf_bootstrap']),
            'olr_lr': random.choice(param_grid['olr_lr']),
            'olr_batch': random.choice(param_grid['olr_batch']),
            'synth_samples': random.choice(param_grid['synth_samples'])
        }

        print(f"Iteration {i+1}/{num_random_tests} | Testing configuration:")
        print(f"RF: {p['rf_trees']} trees, Depth {p['rf_depth']}, MinSamp {p['rf_min_samples']}")
        print(f"OLR: LR {p['olr_lr']}, Batch {p['olr_batch']} | Synth: {p['synth_samples']}")

        regressor = ol.Regressor(
            max_iter=5000, 
            learning_rate=p['olr_lr'], 
            num_classes=3, 
            batch_size=p['olr_batch']
        )
        
        forest = rf.RandomForest(
            num_trees=p['rf_trees'],
            num_splitting_features=3, 
            bootstrap_sample_size=p['rf_bootstrap'],
            max_depth=p['rf_depth'],
            min_samples=p['rf_min_samples'],
            min_information=0,
            num_classifications=3,
            with_replacement=True
        )

        voter = Voter([forest, regressor], 0)

        try:
            voted_results = cv.syntheticKFoldCrossValidation(
                df,
                feature_columns_n,
                synth_columns_n,
                voter,
                k_folds,
                p['synth_samples'],
                random_state=310,
                preprocess=True,
                asymmetric=True
            )
            
            test_acc = voted_results[3] # Extracting Accuracy
            test_f1 = voted_results[4]
            
            print(f"Result: Acc = {test_acc:.3f}, F1 = {test_f1:.3f}\n")
            
            results_log.append({
                'params': p,
                'accuracy': test_acc,
                'f1': test_f1
            })

            # 2. Changed condition to save the best configuration based on Accuracy
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_params = p
                
        except Exception as e:
            print(f"Iteration Failed: {e}\n")

    print("=" * 50)
    print("SEARCH COMPLETE")
    print("=" * 50)
    print(f"Best Test Accuracy: {best_accuracy:.3f}")
    print("Best Hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
