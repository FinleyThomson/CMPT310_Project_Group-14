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
model_folder_path = parent_dir + '\\Model'
if model_folder_path not in sys.path:
    sys.path.append(model_folder_path)

ensemble = joblib.load(parent_dir+'\\Model\\ensemble.pkl')
forest = joblib.load(parent_dir+'\\Model\\random_forest.pkl')
regressor = joblib.load(parent_dir+'\\Model\\ordinal_logistic.pkl')
scaler = joblib.load(parent_dir+'\\Model\\scaler.pkl')

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

    X_scaled = scaler.transform([X])

    if model == "rf":
        cls = forest.predict(X_scaled)
        return classes(cls)
    elif model == "olr":
        cls = regressor.predict(X_scaled)
        return classes(cls)
    elif model == "ensemble":
        cls = ensemble.predict(X_scaled)
        return classes(cls)


def classes(cls):
    if cls == 0:
        return "Low ROI"
    elif cls == 1:
        return "Medium ROI"
    else:
        return "High ROI"