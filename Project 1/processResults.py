# Functions to process results of machine learning models
# COMP 551 mini project 1
# Feb. 8, 2020

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_acc(y_true, y_prediction):
    accuracy = np.sum(y_true.T.ravel()==y_prediction)/len(y_true.T.ravel())
    #accuracy=np.mean(y_true==y_prediction)
    return accuracy

def crossVal(X, y, mlInst, K):
    ''' Takes in training data, and machine learning algorithm instance (eg. logistic regression class instance initiated with desired hyperparameters), and number of folds. 
    The ML algorithm should have a class method fit to train and a class method predict to test. Performs K fold cross validation on data and outputs average train accuracy and average validation accuracy '''
    
    valAccs = []
    trainAccs = []
    # Get the indices that denote the end of a partition
    N = X.shape[0]
    partitionSize = np.floor(N/K).astype('int')
    partitionEnds = [i*partitionSize for i in range(K+1)]
    partitionEnds[-1] = N # last partition end should be no. rows 
    X_train = X.to_numpy(dtype=float)
    y_train = y.to_numpy(dtype=float)


    for i in range(K):
        print('fold: ', i+1)
        # Assign folds to training and validation data by slicing arrays
        XVal = X_train[partitionEnds[i]:partitionEnds[i+1], :]
        yVal = y_train[partitionEnds[i]:partitionEnds[i+1]] 
        XTrainP1 = X_train[:partitionEnds[i], :]
        XTrainP2 = X_train[partitionEnds[i+1]:, :]
        XTrain = np.concatenate((XTrainP1, XTrainP2), axis=0)
        yTrainP1 = y_train[:partitionEnds[i]]
        yTrainP2 = y_train[partitionEnds[i+1]:]
        yTrain = np.concatenate((yTrainP1, yTrainP2), axis=0)
       
        yTrain = pd.DataFrame(yTrain)
        XTrain = pd.DataFrame(XTrain)
        XVal = pd.DataFrame(XVal)
        # Use this to check prior probabilities of each fold (should be ~ same for each fold)
        # must have Lia's models.py script in current directory and imported
        # mlInst._Gaussian_Naive_Bayes__getCatLogPriors(2, yVal)
        # print(np.exp(mlInst._Gaussian_Naive_Bayes__logPriors))
        costDiffs = mlInst.fit(XTrain, yTrain)
        # plt.plot(np.arange(1, len(costDiffs) + 1), costDiffs, '.', label=str(i+1))
        trainPreds = mlInst.predict(XTrain)
        trainAccs.append(evaluate_acc(np.asarray(yTrain), trainPreds))
        valPreds = mlInst.predict(XVal)
        valAccs.append(evaluate_acc(yVal, valPreds))
    # plt.yscale('log')
    # plt.show()
    # print(accs)
    # Return mean train accuracy and mean validation accuracy
    return sum(trainAccs)/len(trainAccs), sum(valAccs)/len(valAccs)
