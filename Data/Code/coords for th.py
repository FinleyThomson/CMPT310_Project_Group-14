import numpy as np
import pandas as pd

df = pd.read_csv("C:/Users/finle/Documents/CMPT 310/Project/Data/MASTER/TH_DATA.csv")

df["cluster_center_lon"] = df.groupby("Cluster_ID")["Longitude"].transform("mean")
df["cluster_center_lat"] = df.groupby("Cluster_ID")["Latitude"].transform("mean")


df["centroid"] = df.apply(lambda row: [row["cluster_center_lon"], row["cluster_center_lat"]], axis=1)


df.drop(columns=["cluster_center_lon", "cluster_center_lat"], inplace=True)

df.to_csv("TH_DATA_CENTROIDS.csv")