import numpy as np
import pandas as pd
import csv  
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    
from sklearn.metrics import accuracy_score
import pickle
from sklearn.utils import resample

    
def train_data_mlp_1(x,y,filename):
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/4,random_state=0)

    #feature scalling
    sc=StandardScaler()
    
    #fitting simple linear regression to the training set    
    x_train=sc.fit_transform(x_train)
    x_test=sc.fit_transform(x_test)
    
    #Multi layer pereptron
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=1000)
    clf.fit(x_train, y_train)
    
    ypred = clf.predict(x_test)
    # Calculate the absolute errors
    errors = abs(ypred - y_test)# Print out the mean absolute error (mae)
    print('Multi layer pereptron  Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    #accuracy
    acc = accuracy_score(y_true=y_test, y_pred=ypred)
    print('Multi layer pereptron   Acc: {:.4f}'.format(acc))
    #confusion matrix
    print('Multi layer pereptron   Confusion matrix:')
    print(confusion_matrix(y_test, ypred))

    pickle.dump(clf, open(filename, 'wb'))
    # some time later...
    # load the model from disk
    print("loading model...")
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(x_test, y_test)
    print(result)    

def train_data_mlp(x,y):
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/4,random_state=0)

    #feature scalling
    sc=StandardScaler()
    
    #fitting simple linear regression to the training set    
    x_train=sc.fit_transform(x_train)
    x_test=sc.fit_transform(x_test)
    
    #Multi layer pereptron
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=1000)
    clf.fit(x_train, y_train)
    
    ypred = clf.predict(x_test)
    # Calculate the absolute errors
    errors = abs(ypred - y_test)# Print out the mean absolute error (mae)
    print('Multi layer pereptron  Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    #accuracy
    acc = accuracy_score(y_true=y_test, y_pred=ypred)
    print('Multi layer pereptron   Acc: {:.4f}'.format(acc))
    #confusion matrix
    print('Multi layer pereptron   Confusion matrix:')
    print(confusion_matrix(y_test, ypred))

    
def call_train(filename,col1,col2,col3,col4,label):
    data = pd.read_csv(filename)  
    
    #remove_class_imbalance
    #16% data is 1, rest is 0
    cnt_label=data[label].value_counts()
    print("count of data labels before up sampling of minority class", cnt_label)
    # Separate majority and minority classes
    data_majority = data[data[label]==0]
    data_minority = data[data[label]==1]
 
    # Upsample minority class
    data_minority_upsampled = resample(data_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=cnt_label[0],    # to match majority class
                                 random_state=123) # reproducible results
    
    # Combine majority class with upsampled minority class
    data_upsampled = pd.concat([data_majority, data_minority_upsampled])
    
    cnt_label=data_upsampled[label].value_counts()
    print("count of data labels after up sampling of minority class", cnt_label)
    
    p1_idx=data_upsampled.columns.get_loc(col1)
    p21_idx=data_upsampled.columns.get_loc(col2)
    p31_idx=data_upsampled.columns.get_loc(col3)
    p41_idx=data_upsampled.columns.get_loc(col4)
    y_idx=data_upsampled.columns.get_loc(label)
    
    x_1= data_upsampled.iloc[:,[p1_idx,p21_idx,p31_idx,p41_idx]].values
    y= data_upsampled.iloc[:,y_idx].values
    
    train_data_mlp(x_1,y,modelfilename2)      
    

def main():    
    #train for over +1
    call_train('yx_1.csv',"p1","p21","p31","p41","y",'finalized_model_mlp_1.sav')
    #train for over +2
    call_train('yx_2.csv',"p1","p22","p32","p42","y",'finalized_model_mlp_2.sav')
    #train for over +3
    call_train('yx_3.csv',"p1","p23","p33","p43","y",'finalized_model_mlp_3.sav')
    #train for over +4
    call_train('yx_4.csv',"p1","p24","p34","p44","y",'finalized_model_mlp_4.sav')



def predict_over_1(x_1, over):
    #mlp
    #filename = 'finalized_model_mlp_1.sav'        
    # load the model from disk
    #print("Using mutilayer perceptron")
    #print(x_1)
    #loaded_model = pickle.load(open(filename, 'rb'))
    #nsamples, nx, ny = np.asarray(x_1).shape
    #print(x_1)
    #print(np.asarray(x_1))

        
    filename='yx_1.csv'
    col1="p1"
    col2="p21"
    col3="p31"
    col4="p41"
    label="y"
    data = pd.read_csv(filename)  
    
    #remove_class_imbalance
    #16% data is 1, rest is 0
    cnt_label=data[label].value_counts()
    #print("count of data labels before up sampling of minority class", cnt_label)
    # Separate majority and minority classes
    data_majority = data[data[label]==0]
    data_minority = data[data[label]==1]
 
    # Upsample minority class
    data_minority_upsampled = resample(data_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=cnt_label[0],    # to match majority class
                                 random_state=123) # reproducible results
    
    # Combine majority class with upsampled minority class
    data_upsampled = pd.concat([data_majority, data_minority_upsampled])
    
    cnt_label=data_upsampled[label].value_counts()
    #print("count of data labels after up sampling of minority class", cnt_label)
    
    p1_idx=data_upsampled.columns.get_loc(col1)
    p21_idx=data_upsampled.columns.get_loc(col2)
    p31_idx=data_upsampled.columns.get_loc(col3)
    p41_idx=data_upsampled.columns.get_loc(col4)
    y_idx=data_upsampled.columns.get_loc(label)
    
    x= data_upsampled.iloc[:,[p1_idx,p21_idx,p31_idx,p41_idx]].values
    y= data_upsampled.iloc[:,y_idx].values
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/4,random_state=0)

    #feature scalling
    sc=StandardScaler()
    
    #fitting simple linear regression to the training set    
    x_train=sc.fit_transform(x_train)
    x_test=sc.fit_transform(x_test)
    
    #Multi layer pereptron
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=1000)
    clf.fit(x_train, y_train)
    
    ypred = clf.predict(x_test)
    #print(x_test)
    #print(x_1)
    result_prob_1_mlp = clf.predict_proba(np.asarray(x_1))
    #print("For +1 over", result_prob_1)    

    return round(result_prob_1_mlp[0][1],2)

def predict_over_2(x_2, over):

    filename = 'finalized_model_mlp_2.sav'        
    # load the model from disk
    #print("Using mutilayer perceptron")
    loaded_model = pickle.load(open(filename, 'rb'))
    result_prob_2_mlp = loaded_model.predict_proba(x_2)
    #print("For +2 over", result_prob_2)    

    return round(result_prob_2_mlp[0][1],2)

def predict_over_3(x_3, over):

    filename = 'finalized_model_mlp_3.sav'            
    # load the model from disk
    #print("Using mutilayer perceptron")
    loaded_model = pickle.load(open(filename, 'rb'))
    result_prob_3_mlp = loaded_model.predict_proba(x_3)
    #print("For +3 over", result_prob_3)    

    return round(result_prob_3_mlp[0][1],2)

def predict_over_4(x_4, over):

    filename = 'finalized_model_mlp_4.sav'        
    # load the model from disk
    #print("Using mutilayer perceptron")
    loaded_model = pickle.load(open(filename, 'rb'))
    result_prob_4_mlp = loaded_model.predict_proba(x_4)
    #print("For +4 over", result_prob_4)

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

def Myrun(bowler, batsman, non_striker, over, tot_wicket_till_now, over_last_wicket, plus):
    
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
        x_1= [tmp,p21,p31, p41]
        ans=predict_over_1(np.asarray(x_1).reshape(1,4),over)
    elif (plus==2):
        tmp=p1*p52
        x_2= [tmp,p22,p32, p42]
        ans=predict_over_2(np.asarray(x_2).reshape(1,4),over)
    elif (plus==3):
        tmp=p1*p53
        x_3= [tmp,p23,p33, p43]
        ans=predict_over_3(np.asarray(x_3).reshape(1,4), over)
    elif (plus==4):
        tmp=p1*p54
        x_4= [tmp,p24,p34, p44]
        ans=predict_over_4(np.asarray(x_4).reshape(1,4),over)

    return ans


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


