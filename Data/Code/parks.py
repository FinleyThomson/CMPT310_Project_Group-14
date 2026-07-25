import numpy as np
import pandas as pd

df = pd.read_csv("C:/Users/finle/Documents/CMPT 310/Project/Data/MASTER/Code/Schools_20260722.csv")

df["POINT"] = df["POINT"].str.replace(r"[a-zA-Z/(/)]", "", regex=True)
df["POINT"] = df["POINT"].str.split()
#df["coords"] = df["POINT"].apply(lambda x: [coord.split() for coord in x])
#df["coords"] = df["coords"].apply(lambda x: [[float(item) for item in pair] for pair in x])

df["coords"] = df["POINT"].apply(lambda x: [float(item) for item in x])

#df["centroid"] = df["coords"].apply(lambda x: np.mean(x,axis=0))

#df = df.drop(columns = ["POINT","coords"])

df.to_csv("schools.csv")