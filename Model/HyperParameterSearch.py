from Model import RandomForest as rf
from Model import OrdinalLogisticRegression as ol
from Model.Voter import Voter
import os
import pandas as pd
from Model import CrossValidation as cv
import numpy as np
import random




def HyperParamSearch():
    """TESTING FOR BEST HYPERPARAMETERS FOR RF AND OLR"""
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

    
    param_grid = {
        'rf_trees': [150],          
        'rf_depth': [5,7],              
        'rf_min_samples': [5,7],
        'rf_bootstrap': [80, 120],     
        
        'olr_lr': [0.01, 0.005, 0.001],      
        'olr_batch': [32, 64, 72],
        'olr_max_iter': [3000, 5000, 10000],
        
        'synth_samples': [600, 900, 1200]
    }

    num_random_tests = 20
    best_accuracy = 0  
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
            'olr_max_iter': random.choice(param_grid['olr_max_iter']),
            'olr_lr': random.choice(param_grid['olr_lr']),
            'olr_batch': random.choice(param_grid['olr_batch']),
            'synth_samples': random.choice(param_grid['synth_samples'])
        }

        print(f"Iteration {i+1}/{num_random_tests} | Testing configuration:")
        print(f"RF: {p['rf_trees']} trees, Depth {p['rf_depth']}, MinSamp {p['rf_min_samples']}")
        print(f"OLR: Iter {p['olr_max_iter']} LR {p['olr_lr']}, Batch {p['olr_batch']} | Synth: {p['synth_samples']}")

        regressor = ol.Regressor(
            max_iter=p["olr_max_iter"], 
            learning_rate=p['olr_lr'], 
            num_classes=3, 
            batch_size=p['olr_batch'],
            seed = 310
        )
        
        forest = rf.RandomForest(
            num_trees=p['rf_trees'],
            num_splitting_features=3, 
            bootstrap_sample_size=p['rf_bootstrap'],
            max_depth=p['rf_depth'],
            min_samples=p['rf_min_samples'],
            min_information=0,
            num_classifications=3,
            with_replacement=True,
            random_state = 310
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
            
            test_acc = voted_results[3] 
            test_f1 = voted_results[4]
            
            print(f"Result: Acc = {test_acc:.3f}, F1 = {test_f1:.3f}\n")
            
            results_log.append({
                'params': p,
                'accuracy': test_acc,
                'f1': test_f1
            })

            
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

def main():

    hyper_search = True


    if hyper_search:
        HyperParamSearch()
        return
    else:
        return

if __name__ == "__main__":
    main()
