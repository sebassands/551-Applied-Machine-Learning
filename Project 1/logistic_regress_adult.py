# Functions to prepare data
# COMP 551 mini project 1
# Lia Formenti
# Feb 4 / 2020

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from processing_functions import one_hot_training, remove_bad_values, evaluate_acc, splitData
from models_new import LogisticRegression1

filename = 'adult.data'

data = pd.read_csv(filename, header=None)
data.head(n=5)

cleaned_df = remove_bad_values(data)
df = one_hot_training(cleaned_df)


y1 = df['Outcome']

Y = pd.DataFrame(y1)

X = df.drop(['Outcome'], axis=1) 
X = X.drop(['cont-2'], axis=1) #REMOVING fnl_wgt feature

X_train, X_test = splitData(X, 0.2)
y_train, y_test = splitData(Y, 0.2)

X_train = X_train.to_numpy(dtype=float)
X_test = X_test.to_numpy(dtype=float)
y_train = y_train.to_numpy(dtype=float)
y_test = y_test.to_numpy(dtype=float)

   
regressor = LogisticRegression1(learning_rate=0.0001, n_iters=50000,max_iter=100, min_err=0.000001, verbose=True)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
    #predictions = predictions.ravel()

print ('Logistic Regression Accuracy:', evaluate_acc(y_test, predictions))
    
    #acc = np.array([evaluate_acc(y_test, predictions)])
   
   # def Average(lst): 
    #    return sum(lst) / len(lst)
    
    #acc = evaluate_acc(y_test, predictions)
    #accs.append(acc)
    #averageAccuracy = Average(accs)
    #print('average accuracy until fold',_+1,':', Average(accs))
    #plt.plot(accs)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score 
print(classification_report (y_test, predictions))
print (accuracy_score(y_test, predictions))

#from processResults import crossVal

#crossVal(X_train, y_train, LogisticRegression1(learning_rate=0.0001, n_iters=10,max_iter=100, min_err=0.000001), 5)

def crossVal(X, y, mlInst, K):
    #Takes in training data, and machine learning algorithm instance (eg. logistic regression class instance initiated with desired hyperparameters), and number of folds. 
    #The ML algorithm should have a class method fit to train and a class method predict to test. Performs K fold cross validation on data and outputs average train accuracy and average validation accuracy 
    valAccs = []
    trainAccs = []
    avtrainAccs = []
    avValAccs = []
    
    # Get the indices that denote the end of a partition
    N = X.shape[0]
    partitionSize = np.floor(N/K).astype('int')
    partitionEnds = [i*partitionSize for i in range(K+1)]
  
    for i in range(K):
        print('fold: ', i+1)
        # Assign folds to training and validation data by slicing arrays
        XVal = X[partitionEnds[i]:partitionEnds[i+1], :]
        yVal = y[partitionEnds[i]:partitionEnds[i+1]] 
        XTrainP1 = X[:partitionEnds[i], :]
        XTrainP2 = X[partitionEnds[i+1]:, :]
        XTrain = np.concatenate((XTrainP1, XTrainP2), axis=0)
        yTrainP1 = y[:partitionEnds[i]]
        yTrainP2 = y[partitionEnds[i+1]:]
        yTrain = np.concatenate((yTrainP1, yTrainP2), axis=0)
        # Use this to check prior probabilities of each fold (should be ~ same for each fold)
        # must have Lia's models.py script in current directory and imported
        # mlInst._Gaussian_Naive_Bayes__getCatLogPriors(2, yVal)
        # print(np.exp(mlInst._Gaussian_Naive_Bayes__logPriors))
        mlInst.fit(XTrain, yTrain)
        trainPreds = mlInst.predict(XTrain)
        trainAccs.append(evaluate_acc(yTrain, trainPreds))
        valPreds = mlInst.predict(XVal)
        valAccs.append(evaluate_acc(yVal, valPreds))
        
    avtrainAccs.append(sum(trainAccs)/len(trainAccs))
    avValAccs.append(sum(valAccs)/len(valAccs))
    print ('train',avtrainAccs, 'valid',avValAccs)
    
    return  avtrainAccs, avValAccs

vallist=[]
trainlist=[]
Niters = [50000]
for n_iters in Niters:
    avtrainAccs,avValAccs=crossVal(X_train, y_train, LogisticRegression1(learning_rate=0.001, n_iters=n_iters ,max_iter=n_iters, min_err=0.000001), 5)
    vallist.append(avValAccs)
    trainlist.append(avtrainAccs)
#plt.xscale('log')
    

plt.xlabel('Iterations GD', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.plot(Niters,trainlist, label="Training")
plt.plot(Niters,vallist,label="Validation",linestyle='--')
plt.legend(loc="upper left")
plt.show()


