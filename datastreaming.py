import numpy as np 
import pandas as pd 
import random 

data=pd.read_csv("IdListAllPairs.csv")
def Id_list(team1,team2):
#def Id_list():
   # team1='Mumbai Indians'
   # team2='Chennai Super Kings'
    IdList=data[(data["batting_team"]== team1)&(data['bowling_team']==team2)&(data['inning']==1)][['inning','match_id']].values.tolist()
    IdList.sort()
    l=len(IdList)
    r=random.randrange(0, l)
    IdList[0]=IdList[r]
    return IdList[0]
    
