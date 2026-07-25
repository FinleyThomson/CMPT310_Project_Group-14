import numpy as np
import pandas as pd

txt = 'PermitNum,StatusCurrent,AppliedDate,AppliedYear,IssuedDate,CompletedDate,CompletedYear,PermitType,PermitTypeMapped,PermitClass,PermitClassGroup,PermitClassMapped,WorkClass,WorkClassGroup,WorkClassMapped,Description,ApplicantName,ContractorIfNoApplicant,ContractorName,HousingUnits,EstProjectCost,TotalSqFt,OriginalAddress,CommunityCode,CommunityName,Latitude,Longitude,LocationCount,LocationTypes,LocationAddresses,LocationsWKT,LocationsGeoJSON,Point,LocationsGeoJSON2,Point3,Assessed Value,Address Without Unit Number,Cluster_ID'

header = txt.split(',')

assessment = pd.read_csv("Historical_Property_Assessments_(Parcel)_20260629_filtered.csv",header=0, usecols=["ROLL_YEAR","ADDRESS","RE_ASSESSED_VALUE"])

th_data = pd.read_csv("TH_DATA_NEW.csv",header=0, usecols=header)

th_data["ChildAddresses"] = th_data["LocationAddresses"].apply(lambda x: str(x).split(';'))

th_data["LocationTypesChild"] = th_data["LocationTypes"].apply(lambda x: str(x).split(';'))

header.append("ChildAddresses")

header.append("LocationTypesChild")

th_data['ChildAddresses'] = th_data.apply(lambda row: row["ChildAddresses"] + [row["Address Without Unit Number"]], axis = 1)

th_data['LocationTypesChild'] = th_data.apply(lambda row: row["LocationTypesChild"] + ["Original Address Numberless"], axis = 1)

expanded_th = th_data[header].copy()

expanded_th = expanded_th.explode(['ChildAddresses','LocationTypesChild'])

expanded_th["Clean_Addresses"]=expanded_th["ChildAddresses"].str.replace('#','',regex=False).str.strip()

expanded_th['Clean_Addresses'] = expanded_th['Clean_Addresses'].str.upper()

expanded_th = expanded_th.drop_duplicates(subset=['Clean_Addresses'])

assessment['ADDRESS'] = assessment['ADDRESS'].str.upper()

assessment['ROLL_YEAR'] = pd.to_numeric(assessment['ROLL_YEAR'], errors='coerce').astype('Int64')
expanded_th['CompletedYear'] = pd.to_numeric(expanded_th['CompletedYear'], errors='coerce').astype('Int64')


expanded_th['Target_Year_1'] = expanded_th['CompletedYear'] + 1
expanded_th['Target_Year_2'] = expanded_th['CompletedYear'] + 2
expanded_th['Target_Year_3'] = expanded_th['CompletedYear'] + 3

final_child_values_1 = pd.merge(
    expanded_th, 
    assessment, 
    left_on=['Clean_Addresses', 'Target_Year_1'], 
    right_on=['ADDRESS', 'ROLL_YEAR'], 
    how='left'
)

final_child_values_1["Val1"] = final_child_values_1["RE_ASSESSED_VALUE"]


missing1 = final_child_values_1['RE_ASSESSED_VALUE'].isna()


failed1 = final_child_values_1[missing1].drop(columns=["ROLL_YEAR","ADDRESS","RE_ASSESSED_VALUE"])


success1 = final_child_values_1[~missing1]

final_child_values_2 = pd.merge(
     failed1, 
     assessment, 
     left_on=['Clean_Addresses', 'Target_Year_2'], 
     right_on=['ADDRESS', 'ROLL_YEAR'], 
     how='left'
 )

final_child_values_2["Val2"] = final_child_values_2['RE_ASSESSED_VALUE']


missing2 = final_child_values_2['RE_ASSESSED_VALUE'].isna()
failed2 = final_child_values_2[missing2].drop(columns=["ROLL_YEAR","ADDRESS","RE_ASSESSED_VALUE"])

success2 = final_child_values_2[~missing2]

final_child_values_3 = pd.merge(
     failed2, 
     assessment, 
     left_on=['Clean_Addresses', 'Target_Year_3'], 
     right_on=['ADDRESS', 'ROLL_YEAR'], 
     how='left'
 )

final_child_values_3["Val3"] = final_child_values_3['RE_ASSESSED_VALUE']

final = pd.concat([success1,success2,final_child_values_3], ignore_index = True)


final.to_csv('FINAL_ASSESSMENTS_NEW.csv', index=False)





