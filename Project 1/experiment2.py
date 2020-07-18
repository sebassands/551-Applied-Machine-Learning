import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from LogisticRegression import evaluate_acc
from LogisticRegression import LogisticRegression1
from LogisticRegression import X_train
from LogisticRegression import y_train
def crossVal(X, y, mlInst, K):
    ''' Takes in training data, and machine learning algorithm instance (eg. logistic regression class instance initiated with desired hyperparameters), and number of folds. The ML algorithm should have a class method fit to train and a class method predict to test. Performs K fold cross validation on data and outputs average train accuracy and average validation accuracy '''
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
Niters = [10,15,25,50,100,200,300,400,500,700,800,1000,2000,4000,5000,10000,15000,20000]
for n_iters in Niters:
    avtrainAccs,avValAccs=crossVal(X_train, y_train, LogisticRegression1(learning_rate=0.0001, n_iters=n_iters ,max_iter=100000, min_err=0.000001), 5)
    vallist.append(avValAccs)
    trainlist.append(avtrainAccs)
#plt.xscale('log')
    

plt.xlabel('Iterations GD', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.plot(Niters,trainlist, label="Training")
plt.plot(Niters,vallist,label="Validation",linestyle='--')
plt.legend(loc="upper left")




    
    