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

