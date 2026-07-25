import numpy as np
import pandas as pd

# df = pd.read_csv("C:\\Users\\finle\\Documents\\CMPT 310\\Project\\Data\\MASTER\\TH_DATA_BY_PROJECT_FINAL.csv")

# FeatureColumns = ["Initial Assessment","Income",
#                        "Distance To Downtown","Distance To Nearest Transit Stop",
#                         "Cost Per Unit","Distance to Nearest Park",
#                        "Distance to Nearest Public School"]

def preprocess(df, FeatureColumns):
    """ Normalizes all of the features in FeatureColumns by column, only standard normalization, no one-hot encoding.
    
        Parameters:
        df (pandas DataFrame): Contains all of the data
        FeatureColumns (list): List of column names 
    
        Returns:
        X (numpy array):  Matrix of size number_of_features x number_of_samples

    """

    FeatureColumnsNormalized = []

    for col in FeatureColumns:

        df[col] = df[col].astype(str).str.replace(",","").astype(float) #strips numbers with thousands comma to convert to floats
        
        mu = df[col].mean()
        sigma = df[col].std()
        
        if sigma == 0:
            sigma ==1
        
        df[col+" Normalized"] = df[col].apply(lambda x: (x-mu)/sigma) #Normalize the features

        FeatureColumnsNormalized.append(col+" Normalized") #Store normalized column names


    X = df[FeatureColumnsNormalized].values #Copy normalized feature values into array

    return X

# X = preprocess(df, FeatureColumns)

# print(type(X))   # Should return: <class 'numpy.ndarray'>
# print(X.dtype)   # Should return: float64