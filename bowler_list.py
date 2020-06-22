import numpy as np 
import pandas as pd 

df=pd.read_csv("ListBowler.csv")
def bowler_list():
	bowler = df['Bowler'].unique().tolist() 
	return bowler

if __name__ == '__main__':
    print(bowler_list())
