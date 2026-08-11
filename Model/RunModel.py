from Model import RandomForest as rf
from Model import OrdinalLogisticRegression as ol
from Model import Voter as vo
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0,parent_dir)
from Data.Code import SyntheticData as sd
import joblib
model_folder_path = parent_dir + '/Model'
if model_folder_path not in sys.path:
    sys.path.append(model_folder_path)

ensemble_1200_olr_real_rf = joblib.load(parent_dir+'/Model/TrainedModels/ensemble_real_rf_1200_synth_olr.pkl')
ensemble_real_only = joblib.load(parent_dir+'/Model/TrainedModels/ensemble_real_only.pkl')
forest = joblib.load(parent_dir+'/Model/TrainedModels/random_forest_real_only.pkl')
regressor_1200 = joblib.load(parent_dir+'/Model/TrainedModels/ordinal_logistic_regressor_1200_synth.pkl')
regressor_real = joblib.load(parent_dir+'/Model/TrainedModels/ordinal_logistic_regressor_real.pkl')
scaler_1200 = joblib.load(parent_dir+'/Model/TrainedModels/scaler_1200.pkl')
scaler_real = joblib.load(parent_dir+'/Model/TrainedModels/scaler_real.pkl')



def runModel(model, X):
    """
    Parameters:
    -----------
    model: (string) either "Random Forest", "Ordinal Logistic Regression", or "Ensemble"
    X: (nparray) data point to be classified

    Returns:
    --------
    classification: (string) the classification of the point
    """
    if model == "olr_1200" or model == "ensemble_1200_olr_real_rf":
        X_scaled = scaler_1200.transform([X])
    else:
        X_scaled = scaler_real.transform([X])

    if model == "rf":
        cls = forest.predict(X_scaled)
        return classes(cls)
    elif model == "olr_1200":
        cls = regressor_1200.predict(X_scaled)
        return classes(cls)
    elif model == "ensemble_1200_olr_real_rf":
        cls = ensemble_1200_olr_real_rf.predict(X_scaled)
        return classes(cls)
    elif model == "olr_real":
        cls = regressor_real.predict(X_scaled)
        return classes(cls)
    elif model == "ensemble_real":
        cls = ensemble_real_only.predict(X_scaled)
        return classes(cls)

def classes(cls):
    if cls == 0:
        return "Low ROI"
    elif cls == 1:
        return "Medium ROI"
    else:
        return "High ROI"