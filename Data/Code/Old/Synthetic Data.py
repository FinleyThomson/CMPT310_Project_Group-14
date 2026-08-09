import numpy as np
import pandas as pd

#------------
# Constants
# -----------

COLS = ["EstProjectCost", "Initial Assessment", "Final Assessment", "Income", "Distance to Downtown",
         "Distance to Nearest Transit Stop", "Cost Per Unit", "Distance to Nearest Park", "Distance to Nearest Public School"]

#------------
# Functions
#------------

def multivariateLognormalDistribution(data):

    X = data[COLS]

    means = np.ln(X.mean().values)

    covariance_matrix = np.log(np.cov(X.values))

    normal_matrix = np.random.multivariate_normal(means,covariance_matrix, size = 3000)

    df_synthetic = pd.DataFrame(normal_matrix, columns=COLS)

    return df_synthetic


def assignClasses(data):

    data["ROI"] = (data["Final Assessment"] - (data["Initial Assessment"] + data["EstProjectCost"]))/(data["Initial Assessment"] + data["EstProjectCost"])

    data["Class"] = pd.qcut(data["ROI"], q = 3, labels = False)

    return data


