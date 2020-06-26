import pandas as pd
import numpy as np

#read_orig_file():
data = pd.read_csv('ballByballData.csv', low_memory=False)
#Get all pair of batting team and bowling team names

'''
column_values = data[["batting_team","bowling_team"]].values.ravel()
unique_values =  pd.unique(column_values)

df = pd.DataFrame({'Team': unique_values})
df.to_csv("ListAllPairs.csv",index=False)
'''

allTeam = data[["batting_team","bowling_team"]]

data=allTeam.drop_duplicates()
data.to_csv("ListAllPairs.csv",index=False)
