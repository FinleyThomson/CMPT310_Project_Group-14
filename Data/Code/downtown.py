from haversine import haversine
import pandas as pd
import re

th = pd.read_csv("TH_BY_PROJECT.csv")

point = [51.0447331,-114.0718831] #alr in lat lon format

th_coords = th["Coordinates"].tolist()

i=0
for p in th_coords:
    p = re.sub(r"[^0-9 .-]","",p)
    p = p.split()
    p[0] = float(p[0])
    p[1] = float(p[1])
    th_coords[i] = p
    i+=1

th.drop(columns = "Coordinates")

th["Coordinates"] = th_coords

th["downtown"] = th["Coordinates"].apply(lambda x: haversine((x[1],x[0]),point))

th.to_csv("downtown.csv")
