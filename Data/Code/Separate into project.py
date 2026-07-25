import numpy as np
import pandas as pd
import sklearn

from sklearn.cluster import DBSCAN

th_data = pd.read_csv("TH_DATA_NEW.csv",header=0, usecols=['PermitNum','StatusCurrent','AppliedDate','AppliedYear','IssuedDate','CompletedDate','CompletedYear','PermitType','PermitTypeMapped','PermitClass','PermitClassGroup','PermitClassMapped','WorkClass','WorkClassGroup','WorkClassMapped','Description','ApplicantName','ContractorIfNoApplicant','ContractorName','HousingUnits','EstProjectCost','TotalSqFt','OriginalAddress','CommunityCode','CommunityName','Latitude','Longitude','LocationCount','LocationTypes','LocationAddresses','LocationsWKT','LocationsGeoJSON','Point','LocationsGeoJSON2','Point3','Assessed Value','Address Without Unit Number'])

# th_data["ChildAddresses"] = th_data["LocationAddresses"].apply(lambda x: str(x).split(';'))

# th_data["LocationTypesChild"] = th_data["LocationTypes"].apply(lambda x: str(x).split(';'))

# split_address_th = th_data[['PermitNum','StatusCurrent','AppliedDate','AppliedYear','IssuedDate','CompletedDate','YearCompleted','PermitType','PermitTypeMapped','PermitClass','PermitClassGroup','PermitClassMapped','WorkClass','WorkClassGroup','WorkClassMapped','Description','ApplicantName','Contractor if no applicant','ContractorName','HousingUnits','EstProjectCost','ProjectNumber','ProjectNumber Hardcoded','OriginalAddress','CommunityCode','CommunityName','Latitude','Longitude','LocationCount','LocationTypes','LocationAddresses','LocationsWKT','LocationsGeoJSON','Point','ADDRESS','Assessed Value Before','Assessed Value After','LocationTypesChild',"ChildAddresses"]].copy()

# split_address_th = split_address_th.explode(["LocationTypesChild","ChildAddresses"])

# split_address_th.to_csv('split_address_th.csv',index=False)

coords = np.radians(th_data[['Latitude','Longitude']])

coords.to_csv('C:\\Users\\finle\\Documents\\CMPT 310\\Project\\Data\\coords.csv',index=False)

eps = 70 / 6371000 

scan = DBSCAN(eps, min_samples = 1, metric = 'haversine', algorithm = 'ball_tree')

th_data['Cluster_ID'] = scan.fit_predict(coords)

th_data.to_csv('TH_DATA_GROUPED.csv',index=False)