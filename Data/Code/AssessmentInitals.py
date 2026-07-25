import numpy as np
import pandas as pd

assessment = pd.read_csv("Historical_Property_Assessments_(Parcel)_20260629_filtered.csv",header=0, usecols=["ROLL_YEAR","ADDRESS","RE_ASSESSED_VALUE"], thousands = ',')

th_data = pd.read_csv("THdata.csv",header=0, usecols=["AppliedYear", "YearCompleted","OriginalAddress","LocationTypes" ,"LocationAddresses", "ProjectNumber","Cluster_ID"])

assessment['ROLL_YEAR'] = pd.to_numeric(assessment['ROLL_YEAR'], errors='coerce').astype('Int64')
assessment['ADDRESS'] = assessment['ADDRESS'].str.upper().str.strip()
assessment['RE_ASSESSED_VALUE'] = pd.to_numeric(assessment['RE_ASSESSED_VALUE'], errors='coerce')

th_data["AppliedYear"]=pd.to_numeric(th_data['AppliedYear'], errors='coerce').astype('Int64')
th_data["YearCompleted"]=pd.to_numeric(th_data['YearCompleted'], errors='coerce').astype('Int64')

th_data['OriginalAddress'] = th_data['OriginalAddress'].str.upper().str.strip()

th_data["Clean_Addresses"]=th_data["OriginalAddress"].str.replace('#','',regex=False).str.strip()
th_data['Clean_Addresses'] = th_data['Clean_Addresses'].str.upper()

assessment['ADDRESS'] = assessment['ADDRESS'].str.upper()

th_data['Start_year'] = th_data["AppliedYear"] - 1


final_child_values_1 = pd.merge(
    th_data, 
    assessment, 
    left_on=['Clean_Addresses', 'Start_year'], 
    right_on=['ADDRESS', 'ROLL_YEAR'], 
    how='left'
)


final_child_values_1.to_csv('initial_totals.csv', index=False)





