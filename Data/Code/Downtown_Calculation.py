# This script is for finding the distance of an address from downtown.

import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from math import cos, asin, sqrt, pi


def distance(DT_COORDS, stations):
    '''Find distance of Down town from all addresses (stations)'''
    lat1 = DT_COORDS[0]
    lon1 = DT_COORDS[1]
    lat2 = stations['Latitude'].values
    lon2 = stations['Longitude'].values

    r = 6371 # km
    p = pi / 180
    a = 0.5 - np.cos((lat2-lat1)*p)/2 + np.cos(lat1*p) * np.cos(lat2*p) * (1-np.cos((lon2-lon1)*p))/2

    #print((2 * r * np.arcsin(np.sqrt(a)))*1000)
    return (2 * r * np.arcsin(np.sqrt(a)))*1000

def best_station(project, stations):
    '''Not necessary for this calcualtions '''
    min = distance(project, stations).min()

    return min

def main():

    TH_BY_PROJECT = pd.read_csv('TH_BY_PROJECT_2.csv')

    # https://geohack.toolforge.org/geohack.php?pagename=Downtown_Calgary&params=51_02_53_N_114_04_17_W_region:CA-AB_type:city(46763)
    DT_COORDS = [51.048056,-114.071389]

    TH_BY_PROJECT['Distance To Downtown'] = distance(DT_COORDS, TH_BY_PROJECT)

    TH_BY_PROJECT.to_csv('TH_BY_PROJECT_3.csv')





if __name__ == '__main__':
    main()