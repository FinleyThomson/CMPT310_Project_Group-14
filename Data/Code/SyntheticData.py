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

def multivariateLognormalDistribution(data): # lognormal to normal in order to enure positivty of features

    X = data[COLS]

    log_X = np.log1p(X)

    means = log_X.mean().values
    covariance_matrix = log_X.cov().values

    lognormal_matrix = np.random.multivariate_normal(means,covariance_matrix, size = 3000)
    normal_matrix = np.expm1(lognormal_matrix)
    normal_matrix = np.clip(normal_matrix, a_min = 0, a_max = None) # just to ensure no small negatives due to the exp(X) - 1

    df_synthetic = pd.DataFrame(normal_matrix, columns=COLS)
    df_synthetic = df_synthetic.astype(float)

    return df_synthetic


def assignClasses(data): #assign low (0), medium (1), and high (2) ROI classifications based on calculated ROI

    data["ROI"] = (data["Final Assessment"] - (data["Initial Assessment"] + data["EstProjectCost"]))/(data["Initial Assessment"] + data["EstProjectCost"])

    data["Class"] = pd.qcut(data["ROI"], q = 3, labels = False)

    return data

# -------------------
# Running Program
# -------------------


path = "TH_DATA_BY_PROJECT_FINAL.csv" #change directory if needed, defaulting to current directory
df = pd.read_csv(path) 

s_data = multivariateLognormalDistribution(df)
s_data = assignClasses(s_data)

s_data.to_csv("SYNTHETIC_DATA.csv")



