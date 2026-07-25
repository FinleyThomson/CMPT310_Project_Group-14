import numpy as np
import pandas as pd

th_data = pd.read_csv("Townhome_only.csv",header=0, usecols=['PermitNum','StatusCurrent','AppliedDate','AppliedYear','IssuedDate','CompletedDate','YearCompleted','PermitType','PermitTypeMapped','PermitClass','PermitClassGroup','PermitClassMapped','WorkClass','WorkClassGroup','WorkClassMapped','Description','ApplicantName','Contractor if no applicant','ContractorName','HousingUnits','EstProjectCost','ProjectNumber','ProjectNumber Hardcoded','OriginalAddress','CommunityCode','CommunityName','Latitude','Longitude','LocationCount','LocationTypes','LocationAddresses','LocationsWKT','LocationsGeoJSON','Point','ADDRESS','Assessed Value Before','Assessed Value After'])

th_data["ChildAddresses"] = th_data["LocationAddresses"].apply(lambda x: str(x).split(';'))

th_data["LocationTypesChild"] = th_data["LocationTypes"].apply(lambda x: str(x).split(';'))

split_address_th = th_data[['PermitNum','StatusCurrent','AppliedDate','AppliedYear','IssuedDate','CompletedDate','YearCompleted','PermitType','PermitTypeMapped','PermitClass','PermitClassGroup','PermitClassMapped','WorkClass','WorkClassGroup','WorkClassMapped','Description','ApplicantName','Contractor if no applicant','ContractorName','HousingUnits','EstProjectCost','ProjectNumber','ProjectNumber Hardcoded','OriginalAddress','CommunityCode','CommunityName','Latitude','Longitude','LocationCount','LocationTypes','LocationAddresses','LocationsWKT','LocationsGeoJSON','Point','ADDRESS','Assessed Value Before','Assessed Value After','LocationTypesChild',"ChildAddresses"]].copy()

split_address_th = split_address_th.explode(["LocationTypesChild","ChildAddresses"])

split_address_th.to_csv('Csplit_address_th.csv',index=False)