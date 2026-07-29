import numpy as np
import pandas as pd

#------------
# Constants
# -----------

# COLS = ["EstProjectCost", "Initial Assessment", "Final Assessment", "Income", "Distance To Downtown",
#          "Distance To Nearest Transit Stop", "Cost Per Unit", "Distance to Nearest Park", "Distance to Nearest Public School"]

#------------
# Functions
#------------

def multivariateLognormalDistributionGeneration(data, synth_cols, thresholds, n, random_state=None):
    """Creates a multivariate lognormal distribution and then exponeniates in order to ensure positivty of features when generating the synthetic dataset.
    
        Parameters
        ----------
        data: (nparray) dataset to create the synthetic set from
        n: (int) number of datapoints in the synthetic set
        synth_cols: (list) list of columns to be used in the generation of synthetic data fron data
        thresholds: (list) list of floats representing upper cut off points for the different classifications
        random_state: (int or None) optional seed for reproducible generation

        Returns
        -------
        df_synthetic: (pandas data frame): dataframe containing the synthetic dataset
    """

    if n < 0:
        raise ValueError("n must be non-negative")

    rng = np.random.default_rng(random_state)
    synth_dfs = []
    base_class_n = n

    for class_index, label in enumerate([0, 1, 2]):

        split_data = data[data.iloc[:,-1] == label]
        
        if len(split_data) < 2:
            continue

        X = split_data[synth_cols]

        log_X = np.log1p(X)

        means = log_X.mean().values
        covariance_matrix = log_X.cov().values

        class_n = base_class_n + (class_index < n % 3)
        lognormal_matrix = rng.multivariate_normal(means,covariance_matrix, size = class_n)
        normal_matrix = np.expm1(lognormal_matrix)
        normal_matrix = np.clip(normal_matrix, a_min = 0, a_max = None) # just to ensure no small negatives due to the exp(X) - 1

        df_synthetic = pd.DataFrame(normal_matrix, columns=synth_cols)
        df_synthetic = df_synthetic.astype(float)
        df_synthetic["Classification"] = assignClass(df_synthetic, thresholds)
        synth_dfs.append(df_synthetic)

    df = pd.concat(synth_dfs, ignore_index = True)

    return df


def assignClasses(data, thresholds): #assign low (0), medium (1), and high (2) ROI classifications based on calculated ROI

    data["ROI"] = (data["Final Assessment"] - (data["Initial Assessment"] + data["EstProjectCost"]))/(data["Initial Assessment"] + data["EstProjectCost"]) 

    num_t = len(thresholds)

    d = np.zeros((num_t,len(data)))

    i = 0

    for t in thresholds:
        d[i] = (data["ROI"] - t)
        i+=1

    ds = pd.concat(d, axis = 1).values

    idxs = np.argmin(ds, axis = 1)

    classes = []

    for i in idxs:
        classes.append(idxs + np.argmin(ds[:,max(i-1, 0):min(i+2, num_t):2]))
    
    data["Classifications"] = classes

    return data


def assignClass(data, thresholds):

    base_cost = data["Initial Assessment"] + data["EstProjectCost"]
    data["ROI"] = (data["Final Assessment"] - base_cost) / base_cost

    data["Classifications"] = np.digitize(data["ROI"], bins=thresholds)

    return data



