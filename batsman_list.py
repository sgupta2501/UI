import numpy as np 
import pandas as pd 

df=pd.read_csv("ListBatsman.csv")
def batsman_list():
	batsman = df['Batsman'].unique().tolist() 
	return batsman

if __name__ == '__main__':
    print(batsman_list())
