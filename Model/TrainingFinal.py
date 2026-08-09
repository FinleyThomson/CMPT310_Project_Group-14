import joblib
import RandomForest as rf
import OrdinalLogisticRegression as ol
import Voter as vo
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0,parent_dir)
from Data.Code import SyntheticData as sd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = (
    REPOSITORY_ROOT / "Data" / "CSVs" / "Sorted" / "TH_DATA_BY_PROJECT_FINAL.csv"
)

def main():

    data = pd.read_csv(DEFAULT_DATA_PATH)

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
                "Classification"
            ]

    forest = rf.RandomForest(num_trees=60,num_splitting_features=3,bootstrap_sample_size=120,max_depth=10,min_samples=7,min_information=0,num_calssifications=3,with_replacement=True)
    regressor = ol.Regressor(max_iter=5000, learning_rate=0.001, num_classes=3, batch_size=72)
    training_set_real = data[synth_columns]
    training_set_synth = sd.multivariateLognormalDistributionGeneration(data, synth_columns,900,310)

    scaler = StandardScaler()

    y_real = training_set_real["Classification"].to_numpy()
    y_synth = training_set_synth["Classification"].to_numpy()
     
    X_mixed = scaler.fit_transform(np.vstack((training_set_real[feature_columns].to_numpy(), training_set_synth[feature_columns].to_numpy())))
    y_mixed = np.hstack((y_real, y_synth))

    len_real = len(y_real)

    X_real = X_mixed[:len_real]
    X_synth = X_mixed[len_real:]

    ensemble = vo.Voter(models=[forest, regressor], num_tree_synth=0)
    ensemble.fit(X_real, y_real, X_synth, y_synth)
    joblib.dump(ensemble, 'ensemble.pkl')

    forest.fit(X_real, y_real)
    joblib.dump(forest, 'random_forest.pkl')

    regressor.fit(X_mixed, y_mixed)
    joblib.dump(regressor, 'ordinal_logistic.pkl')

    joblib.dump(scaler, 'scaler.pkl')

    print("Models saved successfully!")

if __name__ == "__main__":
    main()