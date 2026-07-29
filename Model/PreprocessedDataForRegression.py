import numpy as np
import pandas as pd
import itertools



# FeatureColumns = ["Initial Assessment","Income",
#                        "Distance To Downtown","Distance To Nearest Transit Stop",
#                         "Cost Per Unit","Distance to Nearest Park",
#                        "Distance to Nearest Public School"]

def preprocess(df, FeatureColumns):
    """ Normalizes all of the features in FeatureColumns by column, only standard normalization, no one-hot encoding.
    
        Parameters
        ----------
        df (pandas DataFrames): Contains all of the data
        FeatureColumns (2D list): Lists of column names 
    
        Returns
        -------
        X (numpy array):  Matrix of size number_of_features x number_of_samples

    """


    # FeatureColumnsNormalized = []

    dfn = pd.DataFrame()

    for col in FeatureColumns:

        df[col] = df[col].astype(str).str.replace(",","").astype(float) #strips numbers with thousands comma to convert to floats
        
        mu = df[col].mean()
        sigma = df[col].std()
        
        if sigma == 0:
            sigma ==1
        
        dfn[col] = df[col].apply(lambda x: (x-mu)/sigma) #Normalize the features

        #FeatureColumnsNormalized.append(col+" Normalized") #Store normalized column names


    #X = dfn[FeatureColumns].values #Copy normalized feature values into array

    return dfn

# X = preprocess(df, FeatureColumns)

# print(type(X))   # Should return: <class 'numpy.ndarray'>
# print(X.dtype)   # Should return: float64