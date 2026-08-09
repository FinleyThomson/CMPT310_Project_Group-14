import os
import pandas as pd
import re
from haversine import haversine

def getNearestFeature(coords, feature):

    """
    Takes in a set of coordinates and a given feature and returns the distance to the nearest feature.
    """

    csv = feature + ".csv"

    parent_dir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(
            parent_dir,
            "CSVs",
            "Sorted",
            csv,
        )
    df = pd.read_csv(path)

    feature_coordss = df["coords"].tolist()

   

    i = 0

    for p in feature_coordss:
        p = re.sub(r"[^0-9 .-]","",p)
        p = p.split()
        p[0] = float(p[0])
        p[1] = float(p[1])
        feature_coordss[i] = p
        i+=1

    dist = min_distances(coords, feature_coordss)

    return dist


def min_distances(th_coord, feature_coords):


    dists = [

        haversine(th_coord, ([feature[1],feature[0]])) for feature in feature_coords

        ]

    return min(dists)


