# This script is for finding the centroid coordinate of the property as the coordinates
# are for each unit/section of the complex rather than the entire project. 

import pandas as pd
import os 
import numpy as np
import Coordinate_Conversions.lat_long_2_XYZ as l2e
import Coordinate_Conversions.XYZ_2_latlong as e2l

vec_geodedic_to_ecef = np.vectorize(l2e.geodedic_to_ecef)
cal_sea_level = 1045

def centroid(groups):
    '''finds centroid of each group, where the group is the units/sections that belong to one project'''
    lats = groups['Latitude'].to_numpy()
    lons = groups['Longitude'].to_numpy()

    ecef = vec_geodedic_to_ecef(lats, lons, cal_sea_level)
    ecef = np.array(ecef)
    ecef = ecef.T

    avg = ecef.sum(axis = 0)/len(groups)
    avg_lat_lon = e2l.ecef_to_geodedic_2(avg)

    return avg_lat_lon
    
def tuple_castor(tuple):
    '''Casts tuple to put data in a better format'''
    return tuple[0].item(), tuple[1].item()

def main():
    TH_DATA = pd.read_csv('TH_DATA.CSV')
    TH_BY_PROJECT = pd.read_csv('TH_BY_PROJECT.csv')
    centroids = TH_DATA.groupby('Cluster_ID').apply(centroid)
    TH_BY_PROJECT['Coordinates'] = centroids.apply(tuple_castor)

    TH_BY_PROJECT['Latitude'] = TH_BY_PROJECT['Coordinates'].apply(lambda x: x[0])
    TH_BY_PROJECT['Longitude'] = TH_BY_PROJECT['Coordinates'].apply(lambda x: x[1])

    TH_BY_PROJECT.to_csv('TH_BY_PROJECT_1.csv')

if __name__ == '__main__':
    main()