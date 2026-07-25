# This script is for finding the distance of an address from the nearest transit station. 

import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from math import cos, asin, sqrt, pi

def distance(project, stations):

    lat1 = project['Latitude']
    lon1 = project['Longitude']
    lat2 = stations['LATITUDE'].values
    lon2 = stations['LONGITUDE'].values

    r = 6371 # km
    p = pi / 180
    a = 0.5 - np.cos((lat2-lat1)*p)/2 + np.cos(lat1*p) * np.cos(lat2*p) * (1-np.cos((lon2-lon1)*p))/2

    #print((2 * r * np.arcsin(np.sqrt(a)))*1000)
    return (2 * r * np.arcsin(np.sqrt(a)))*1000
    

def best_station(project, stations):
    
    min = distance(project, stations).min()

    return min

def main():
    
    TH_BY_PROJECT = pd.read_csv('TH_BY_PROJECT_1.csv')

    TRANSIT = pd.read_csv('TRANSIT.csv')
    # ignore inactive transit stops
    TRANSIT = TRANSIT[TRANSIT['STATUS'] == 'ACTIVE']
    TRANSIT = TRANSIT.reset_index()

    # extract lat and lon from string
    TRANSIT['LONGITUDE'] = TRANSIT['POINT'].str.extract(r'(-?\d+(?:\.\d+)?)')
    TRANSIT['LATITUDE'] = TRANSIT['POINT'].str.extract(r'\d+ (\d+(?:\.\d+))?')

    # convert lat and lon to floats
    TRANSIT['LONGITUDE'] = TRANSIT['LONGITUDE'].apply(float)
    TRANSIT['LATITUDE'] = TRANSIT['LATITUDE'].apply(float)  

    TH_BY_PROJECT['Distance To Nearest Transit Stop'] = TH_BY_PROJECT.apply(best_station, stations=TRANSIT, axis = 1)

    TH_BY_PROJECT.to_csv('TH_BY_PROJECT_2.csv')





if __name__ == '__main__':
    main()