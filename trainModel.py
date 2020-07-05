import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import csv  
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    
from sklearn.metrics import accuracy_score
import pickle
from sklearn.utils import resample

def train_data_logreg(x,y,filename):

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/4,random_state=0)

    #feature scalling
    sc=StandardScaler()
    
    #fitting simple linear regression to the training set    
    x_train=sc.fit_transform(x_train)
    x_test=sc.fit_transform(x_test)    
    
    #logistic regression
    from sklearn.linear_model import LogisticRegression
    logreg=LogisticRegression(solver='lbfgs')
    logreg.fit(x_train,y_train)
    #predicting the test set resuts
    ypred=logreg.predict(x_test)
    # Calculate the absolute errors
    errors = abs(ypred - y_test)# Print out the mean absolute error (mae)
    #print('logistic regression  Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    #accuracy
    acc = accuracy_score(y_true=y_test, y_pred=ypred)
    #print('logistic regression   Acc: {:.4f}'.format(acc))
    #confusion matrix
    #print('logistic regression   Confusion matrix:')
    #print(confusion_matrix(y_test, ypred))

    pickle.dump(logreg, open(filename, 'wb'))
    # some time later...
    # load the model from disk
    #print("loading model...")
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(x_test, y_test)
    #print(result)    
    
def train_data_mlp(x,y,filename):
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/4,random_state=0)

    #feature scalling
    sc=StandardScaler()
    
    #fitting simple linear regression to the training set    
    x_train=sc.fit_transform(x_train)
    x_test=sc.fit_transform(x_test)
    
    #Multi layer pereptron
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=1000,warm_start=False)
    clf.fit(x_train, y_train)
    ypred = clf.predict(x_test)
    # Calculate the absolute errors
    errors = abs(ypred - y_test)# Print out the mean absolute error (mae)
    #print('Multi layer pereptron  Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    #accuracy
    acc = accuracy_score(y_true=y_test, y_pred=ypred)
    #print('Multi layer pereptron   Acc: {:.4f}'.format(acc))
    #confusion matrix
    #print('Multi layer pereptron   Confusion matrix:')
    #print(confusion_matrix(y_test, ypred))

    pickle.dump(clf, open(filename, 'wb'))
    # some time later...
    # load the model from disk
    #print("loading model...")
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(x_test, y_test)
    #print(result)    


    
def call_train(filename,col1,col2,col3,col4,label,modelfilename1,modelfilename2):
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
    
    x_1= data_upsampled.iloc[:,[p1_idx,p21_idx,p31_idx,p41_idx]].values
    y= data_upsampled.iloc[:,y_idx].values
    
    #train_data_logreg(x_1,y,modelfilename1)
    train_data_mlp(x_1,y,modelfilename2)      
    

def main():    
    #train for over +1
    call_train('yx_1.csv',"p1","p21","p31","p41","y",'finalized_model_logreg_1.sav','finalized_model_mlp_1.sav')
    #train for over +2
    call_train('yx_2.csv',"p1","p22","p32","p42","y",'finalized_model_logreg_2.sav','finalized_model_mlp_2.sav')
    #train for over +3
    call_train('yx_3.csv',"p1","p23","p33","p43","y",'finalized_model_logreg_3.sav','finalized_model_mlp_3.sav')
    #train for over +4
    call_train('yx_4.csv',"p1","p24","p34","p44","y",'finalized_model_logreg_4.sav','finalized_model_mlp_4.sav')

    
#main()
