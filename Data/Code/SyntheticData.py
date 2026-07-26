import numpy as np
import pandas as pd

#------------
# Constants
# -----------

COLS = ["EstProjectCost", "Initial Assessment", "Final Assessment", "Income", "Distance To Downtown",
         "Distance To Nearest Transit Stop", "Cost Per Unit", "Distance to Nearest Park", "Distance to Nearest Public School"]

#------------
# Functions
#------------

def multivariateLognormalDistributionGeneration(data, n): 
    """Creates a multivariate lognormal distribution and then exponeniates in order to ensure positivty of features when generating the synthetic dataset.
    
        Parameters
        ----------
        data: (nparray) dataset to create the synthetic set from
        n: (int) number of datapoints in the synthetic set

        Returns
        -------
        df_synthetic: (pandas data frame): dataframe containing the synthetic dataset
    """

    synth_dfs = []

    per_class_n = n // 3

    for label in [0, 1, 2]:

        split_data = data[data.iloc[:,-1] == label]
        
        if len(split_data) < 2:
            continue

        X = split_data[COLS]

        log_X = np.log1p(X)

        means = log_X.mean().values
        covariance_matrix = log_X.cov().values

        lognormal_matrix = np.random.multivariate_normal(means,covariance_matrix, size = per_class_n)
        normal_matrix = np.expm1(lognormal_matrix)
        normal_matrix = np.clip(normal_matrix, a_min = 0, a_max = None) # just to ensure no small negatives due to the exp(X) - 1

        df_synthetic = pd.DataFrame(normal_matrix, columns=COLS)
        df_synthetic = df_synthetic.astype(float)
        df_synthetic["Classification"] = label
        synth_dfs.append(df_synthetic)

    df = pd.concat(synth_dfs, ignore_index = True)

    return df


def assignClasses(data): #assign low (0), medium (1), and high (2) ROI classifications based on calculated ROI

    data["ROI"] = (data["Final Assessment"] - (data["Initial Assessment"] + data["EstProjectCost"]))/(data["Initial Assessment"] + data["EstProjectCost"])

    data["Class"] = pd.qcut(data["ROI"], q = 3, labels = False)

    return data



