import numpy as np 
import pandas as pd 

df=pd.read_csv("ListTeam.csv")
def team_list():
	team = df['Team'].unique().tolist() 
	return team

if __name__ == '__main__':
    print(team_list())
