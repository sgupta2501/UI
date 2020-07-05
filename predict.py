import numpy as np
import pandas as pd
import csv  
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    
from sklearn.metrics import accuracy_score
import pickle
from sklearn.utils import resample


def predict_over_1(x_1, over,y):

    filename = 'finalized_model_mlp_1.sav'        
    # load the model from disk
    #print("Using mutilayer perceptron")
    loaded_model = pickle.load(open(filename, 'rb'))
    #print("x_1", x_1)
    result_prob_1_mlp = loaded_model.predict_proba(x_1)
    #print("For +4 over", result_prob_1)
   # y=[1]
 #   loaded_model = loaded_model.fit_one(x_1, y) 
    loaded_model = loaded_model.partial_fit(x_1, y)
    pickle.dump(loaded_model, open(filename, 'wb'))
    
    return round(result_prob_1_mlp[0][1],2)


def predict_over_2(x_2, over,y):

    filename = 'finalized_model_mlp_2.sav'        
    # load the model from disk
    #print("Using mutilayer perceptron")
    loaded_model = pickle.load(open(filename, 'rb'))
    #print("x_2", x_2)
    result_prob_2_mlp = loaded_model.predict_proba(x_2)
    #print("For +2 over", result_prob_2)    
    #y=[1]
    loaded_model = loaded_model.partial_fit(x_2, y)
    pickle.dump(loaded_model, open(filename, 'wb'))

    return round(result_prob_2_mlp[0][1],2)

def predict_over_3(x_3, over,y):

    filename = 'finalized_model_mlp_3.sav'            
    # load the model from disk
    #print("Using mutilayer perceptron")
    loaded_model = pickle.load(open(filename, 'rb'))
    #print("x_3", x_3)
    result_prob_3_mlp = loaded_model.predict_proba(x_3)
    #print("For +3 over", result_prob_3)    
   # y=[1]
    loaded_model = loaded_model.partial_fit(x_3, y)
    pickle.dump(loaded_model, open(filename, 'wb'))

    return round(result_prob_3_mlp[0][1],2)

def predict_over_4(x_4, over,y):

    filename = 'finalized_model_mlp_4.sav'        
    # load the model from disk
    #print("Using mutilayer perceptron")
    loaded_model = pickle.load(open(filename, 'rb'))
    #print("x_4", x_4)
    result_prob_4_mlp = loaded_model.predict_proba(x_4)
    #print("For +4 over", result_prob_4)
   # y=[1]
    loaded_model = loaded_model.partial_fit(x_4
                                            , y)
    pickle.dump(loaded_model, open(filename, 'wb'))

    return round(result_prob_4_mlp[0][1],2)

    
def bowler_prob_in_a_over(bowler, over):
    
    if over > 20:
        num=0
        den=1
    else:
        data_bowler = pd.read_csv('Bowler_WicketsPerOver.csv')
        str_wicket="Wicket_Over_"+ str(over)

        den=data_bowler[(data_bowler["bowler"]== bowler)]["Tot_Match_Played"].values
        num=data_bowler[(data_bowler["bowler"]== bowler)][str_wicket].values
    #print("bowler ", bowler, "over ", over, "den", den, "num", num)
    if (den==0):
        p1=0
    else:
        p1=num/den
    return p1

def batsman_prob_in_a_over(batsman, over):
    if over > 20:
        num=0
        den=1
    else:
        data_batsman = pd.read_csv('Batsman_OutPerOver.csv')
        str_out="out_over_"+ str(over)

        num=data_batsman[(data_batsman["batsman"]== batsman)][str_out].values
        den=data_batsman[(data_batsman["batsman"]== batsman)]["Tot_Match_Played"].values

    if (den==0):
        p=0
    else:
        p=num/den
    #print(p)
    return p

def Myrun(bowler, batsman, non_striker, over, tot_wicket_till_now, over_last_wicket, plus,y):
    
    diff = over-over_last_wicket
    #cal p1
    p1=bowler_prob_in_a_over(bowler, over)
    
    p51 = tot_wicket_till_now/(diff +1)
    p52 = tot_wicket_till_now/(diff +2)
    p53 = tot_wicket_till_now/(diff +3)
    p54 = tot_wicket_till_now/(diff +4)
    
    #cal p2
    p21=bowler_prob_in_a_over(bowler, over+1)
    p22=bowler_prob_in_a_over(bowler, over+2)
    p23=bowler_prob_in_a_over(bowler, over+3)
    p24=bowler_prob_in_a_over(bowler, over+4)
    
    p31=batsman_prob_in_a_over(batsman, over+1)
    p32=batsman_prob_in_a_over(batsman, over+2)
    p33=batsman_prob_in_a_over(batsman, over+3)
    p34=batsman_prob_in_a_over(batsman, over+4)

    p41=batsman_prob_in_a_over(non_striker, over+1)
    p42=batsman_prob_in_a_over(non_striker, over+2)
    p43=batsman_prob_in_a_over(non_striker, over+3)
    p44=batsman_prob_in_a_over(non_striker, over+4)

    if (plus==1):
        tmp=p1*p51
        #print("p1",p1,"p51",p51,"p21",p21,"p31",p31,"p41",p41,"p1*p51",tmp)
        x_1= [tmp,p21,p31, p41]
        ans=predict_over_1(np.asarray(x_1).reshape(1,4),over,y)
    elif (plus==2):
        tmp=p1*p52
        x_2= [tmp,p22,p32, p42]
        ans=predict_over_2(np.asarray(x_2).reshape(1,4),over,y)
    elif (plus==3):
        tmp=p1*p53
        x_3= [tmp,p23,p33, p43]
        ans=predict_over_3(np.asarray(x_3).reshape(1,4), over,y)
    elif (plus==4):
        tmp=p1*p54
        x_4= [tmp,p24,p34, p44]
        ans=predict_over_4(np.asarray(x_4).reshape(1,4),over,y)

    #a={'ans':ans}
    #return a
    return ans

'''
bowler='R Bhatia'
batsman='TS Mills'
non_striker='Yuvraj Singh'
over=5
tot_wicket_till_now=1
over_last_wicket=1

print(Myrun(bowler, batsman, non_striker, over, tot_wicket_till_now, over_last_wicket,1))
print(Myrun(bowler, batsman, non_striker, over, tot_wicket_till_now, over_last_wicket,2))
print(Myrun(bowler, batsman, non_striker, over, tot_wicket_till_now, over_last_wicket,3))
print(Myrun(bowler, batsman, non_striker, over, tot_wicket_till_now, over_last_wicket,4))
'''

