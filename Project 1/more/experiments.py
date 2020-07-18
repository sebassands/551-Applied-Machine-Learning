# Writing up Task 3 substep 3 experiment for COMP 551 mini project 1
# February 8, 2020

import numpy as np
import matplotlib.pyplot as plt
import prepData as prep
import models as ml
import processResults as pro


def checkLogRegAndLearningRates(XTrain, yTrain, XTest, yTest, learningRates, maxIters, minError):
    '''Cross validate all learning rates in list learning rates on train data, given max number of iterations in grad descent and min error. Prints best learning rate and accuracies on train, validation and test.'''
    valAccsPerRate = []
    trainAccsPerRate = []
    # Try all learning rates and find the best
    for lr in learningRates:
        print('\nTesting learning rate: ', lr)
        LR = ml.LogisticRegression1(lr, maxIters, maxIters, minError, verbose=True)
        trainAcc, valAcc = pro.crossVal(XTrain, yTrain, LR, 5)
        trainAccsPerRate.append(trainAcc)
        valAccsPerRate.append(valAcc)
    indexOfOptimum = valAccsPerRate.index(max(valAccsPerRate))
    bestLR = ml.LogisticRegression1(learningRates[indexOfOptimum], maxIters, maxIters, minError)
    bestLR.fit(XTrain, yTrain)
    preds = bestLR.predict(XTest)
    testAcc = pro.evaluate_acc(yTest, preds)
    print('Accuracies for learning rate: ', learningRates[indexOfOptimum])
    print('Train:', trainAccsPerRate[indexOfOptimum])
    print('Validation:', valAccsPerRate[indexOfOptimum])
    print('Test:', testAcc)
    plt.plot(learningRates, trainAccsPerRate, 'o-b', label='Train')
    plt.plot(learningRates, valAccsPerRate, 'o-r', label='Validation')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.legend()
    plt.show()



def checkNB(XTrain, yTrain, XTest, yTest, nb):
    ''' Takes in training and test data, and a NB class instance. Gets training accuracy and validation accuracy from cross validation. Predicts on XTest and ouputs test accuracy. Returns training, validation and test accuracy '''
    trainAcc, valAcc = pro.crossVal(XTrain, yTrain, nb, 5)
    print('Cross train accuracy: ', trainAcc)
    print('Cross validation accuracy: ', valAcc)
    preds = nb.predict(XTest)
    testAcc = pro.evaluate_acc(yTest, preds)
    print('Test accuracy: ', testAcc)
    return trainAcc, valAcc, testAcc

def getExperiment3Data(prepFcn, fracs, nb, LR):
    # Will hold accuracies per fraction for naive bayes and LR
    nbTestAccs = []
    LRTestAccs = []
    Ns = [] # hold number of instances for each iteration
    XTrain, yTrain, XTest, yTest = prepFcn()
    for frac in fracs:
        print('Frac', frac)
        # Keep fraction, frac, of the training data
        # splitRow = np.ceil(len(yTrain)*frac).astype('int')
        # XTrain = XTrain[:splitRow]
        # yTrain = yTrain[:splitRow]
        print('Frac', frac)
        X = XTrain[:frac]
        y = yTrain[:frac]
        print(X.shape, y.shape)
    
        # Ns.append(len(yTrain))
        # Get naive bayes accuracy for this fraction
        nb.fit(X, y)
        nbPreds = nb.predict(XTest)
        nbTestAccs.append(pro.evaluate_acc(yTest, nbPreds))
    
        # Get LR accuracy for this fraction
        LR.fit(X, y)
        LRPreds = LR.predict(XTest)
        LRTestAccs.append(pro.evaluate_acc(yTest, LRPreds))

    return fracs, nbTestAccs, LRTestAccs
    
        
def experiment3():
    
    iInsts, inbAccs, iLRAccs = getExperiment3Data(prep.prepIonosphere, [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100], ml.Gaussian_Naive_Bayes(), ml.LogisticRegression1(0.0001, 200000,200000,1e-6,verbose=False))
    mInsts, mnbAccs, mLRAccs = getExperiment3Data(prep.prepMagicGamma, [10, 100, 1000, 10000, 15000], ml.Gaussian_Naive_Bayes(), ml.LogisticRegression1(0.000275, 200000, 200000, 1e-6, verbose=False))
    rInsts, rnbAccs, rLRAccs = getExperiment3Data(prep.prepRocks,  [10,20,30,40,50,60,70,80, 100, 150, 200], ml.Gaussian_Naive_Bayes(), ml.LogisticRegression1(0.01, 200000,200000,1e-6,verbose=False))
#     plt.plot(rInsts, rnbAccs, '.b', label='Naive Bayes')
#     plt.plot(rInsts, rLRAccs, '+r', label='LogisticRegression')
#     plt.xlabel('Number of Instances')
#     plt.ylabel('Test Accuracy')
#     plt.title('Connectionist Bench Dataset')
#
    fig, axes = plt.subplots(2,2)
    axes[0,0].plot(iInsts, inbAccs, '.b', label='Naive Bayes')
    axes[0,0].plot(iInsts, iLRAccs, '+r', label='Logistic Regression')

    axes[0,1].plot(mInsts, mnbAccs, '.b', label='Naive Bayes')
    axes[0,1].plot(mInsts, mLRAccs, '+r', label='Logistic Regression')

    axes[1,0].plot(rInsts, rnbAccs, '.b', label='Naive Bayes')    
    axes[1,0].plot(rInsts, rLRAccs, '+r', label='Logistic Regression')
    # Set axes labels
    for i in range(2):
        for j in range(2):
             axes[i,j].set_xlabel('Number of Instances')
             axes[i,j].set_ylabel('Accuracy')
             axes[i,j].set_xscale('log')
             axes[i,j].legend()
# 

    plt.show()
    
