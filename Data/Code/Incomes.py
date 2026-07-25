import pandas as pd

txt = 'community,Average Household Income Before Taxes,Median Household Income Before Taxes,Average Household Income After Taxes,Median Household Income After Taxes'

header = txt.split(",")

df = pd.read_csv("Incomes.csv",header = 0, usecols = header)

df["CommunityList"] = df["community"].astype(str).str.upper().str.split("/")

df = df.drop(columns = 'community')

df = df.explode("CommunityList")

df.to_csv('INCOME_EXPLODED.csv', index=False)
