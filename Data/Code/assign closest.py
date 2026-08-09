import numpy as np
import pandas as pd
import re
from haversine import haversine



df=pd.read_csv("schools.csv")

th = pd.read_csv("TH_BY_PROJECT.csv")

th_coords = th["Coordinates"].tolist()
feature_coordss = df["coords"].tolist()

i = 0

for p in feature_coordss:
    p = re.sub(r"[^0-9 .-]","",p)
    p = p.split()
    p[0] = float(p[0])
    p[1] = float(p[1])
    feature_coordss[i] = p
    i+=1

i = 0

for p in th_coords:
    p = re.sub(r"[^0-9 .-]","",p)
    p = p.split()
    p[0] = float(p[0])
    p[1] = float(p[1])
    th_coords[i] = p
    i+=1

th.drop(columns = "Coordinates")

th["Coordinates"] = th_coords

def min_distances(th_coord):

    th_lat_lon = (th_coord[1], th_coord[0])

    dists = [

        haversine(th_lat_lon, ([feature[1],feature[0]])) for feature in feature_coordss

    ]

    return min(dists)

th["Distance to Nearest Transit Stop"] = th["Coordinates"].apply(lambda x: min_distances(x))

th.to_csv("TH_BY_PROJECT1.csv")
