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

def multivariateLognormalDistributionGeneration(data, synth_cols, n, random_state=None):
    """Creates a multivariate lognormal distribution and then exponeniates in order to ensure positivty of features when generating the synthetic dataset.
    
        Parameters
        ----------
        data: (nparray) dataset to create the synthetic set from
        n: (int) number of datapoints in the synthetic set
        synth_cols: (list) list of columns to be used in the generation of synthetic data fron data
        random_state: (int or None) optional seed for reproducible generation

        Returns
        -------
        df_synthetic: (pandas data frame): dataframe containing the synthetic dataset
    """

    if n < 0:
        raise ValueError("n must be non-negative")

    rng = np.random.default_rng(random_state)


    generation_cols = [
        col for col in synth_cols if col not in {"Classification", "ROI"}
    ]
    missing_columns = sorted(set(generation_cols) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing synthetic-data columns: {missing_columns}")


    target_classes = data["Classification"].unique()
    samples_per_class = n // len(target_classes)

    synth_pieces = []

    for cls in target_classes:

        X = data[data["Classification"] == cls][generation_cols]
        log_X = np.log1p(X)

        means = log_X.mean().values
        covariance_matrix = log_X.cov().values
        
        lognormal_matrix = rng.multivariate_normal(means,covariance_matrix, size = samples_per_class)
        normal_matrix = np.expm1(lognormal_matrix)
        normal_matrix = np.abs(normal_matrix) # just to ensure no small negatives due to the exp(X) - 1

        df_synthetic = pd.DataFrame(normal_matrix, columns=generation_cols)
        df_synthetic = df_synthetic.astype(float)
        #df_synthetic = assignClass(df_synthetic, thresholds)

        # df_valid = df_synthetic[df_synthetic["Classification"] == cls]

        # take_n = min(samples_per_class, len(df_valid)) #stuff here to ensure we have correct number
        # synth_pieces.append(df_valid.sample(n=samples_per_class, replace=(take_n < samples_per_class)))

        # series = df_synthetic[["ROI","Classification"]].values
        # df_synthetic = df_synthetic.drop(columns=["Classification","ROI"])
        # df_synthetic[["ROI","Classification"]] = series
        
        df_synthetic["Classification"] = cls
        synth_pieces.append(df_synthetic)

    df_final = pd.concat(synth_pieces).sample(frac=1, random_state=random_state)

    return df_final


def assignClasses(data, thresholds): #assign low (0), medium (1), and high (2) ROI Classification based on calculated ROI

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
    
    data["Classification"] = classes

    return data


def assignClass(data, thresholds):

    base_cost = data["Initial Assessment"] + data["EstProjectCost"]
    data["ROI"] = (data["Final Assessment"] - base_cost) / base_cost

    data["Classification"] = np.digitize(data["ROI"], bins=thresholds)

    return data



