# main program for analyzing magic gamma telescope data from UCI
# For COMP551 mini-project 1
# Feb. 1 / 2020
# Lia Formenti
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate 
#import magicGammaFunctions as mgf
import prepData as prep
import models as ml
import processResults as pro
import experiments as exp

######  TREVOR AND SEB: Copy the function calls for the Ionosphere dataset for your dataset. Record the train val and test accs for NB, and those same values for log reg once you hone in on a learning rate. 
# 100 000 - 200 000 iterations
# 1e-6 min error
# Three cases for learning rates: 
    # Too large - takes max_iters and acc should be crap
    # Right ballpark - doesn't take max_iters
    # Too small - Only does like 1 iteration
# Add an encosed function called prepDatasetName to prepData.py
# This function should return XTrain, YTrain, XTest, YTest as numpy arrays

### Magic Gamma dataset ###

# magicXTrain, magicYTrain, magicXTest, magicYTest = prep.prepMagicGamma()

# print('Magic Gamma dataset Naive Bayes')
#  
# exp.checkNB(magicXTrain, magicYTrain, magicXTest, magicYTest, ml.Gaussian_Naive_Bayes())

# print('Magic Gamma dataset logistic regression')
# learningRates = [0.0001, 0.000125, 0.00015, 0.000175, 0.0002, 0.000225, 0.00025, 0.000275, 0.0003, 0.00035] # 0.001 is too large
# learningRates = [0.0001, 0.000125, 0.00015, 0.000175, 0.0002, 0.000225, 0.00025, 0.000275] # 0.001 is too large
# exp.checkLogRegAndLearningRates(magicXTrain, magicYTrain, magicXTest, magicYTest, learningRates, 100000, 10e-5)

##### Ionosphere dataset #####

# ionXTrain, ionYTrain, ionXTest, ionYTest = prep.prepIonosphere()
# 
# print('Ionosphere dataset, Gaussian Naive Bayes')
# 
# exp.checkNB(ionXTrain, ionYTrain, ionXTest, ionYTest, ml.Gaussian_Naive_Bayes()) 

# learningRates = [0.1, 0.05, 0.01, 0.005, 0.0025, 0.001, 0.00075, 0.0005, 0.0001,0.00005]
# learningRates=[2.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
# learningRates = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
# The uncommented learning rates are a good start point
# exp.checkLogRegAndLearningRates(ionXTrain, ionYTrain, ionXTest, ionYTest, learningRates, 200000, 1e-6)


##### Adult (census income) dataset #####

adultXTrain, adultYTrain, adultXTest, adultYTest = prep.prepAdultNB()

# 
print('Adult dataset, Gaussian Naive Bayes')
# 
exp.checkNB(adultXTrain, adultYTrain, adultXTest, adultYTest, ml.Combined_Naive_Bayes()) 

# learningRates = [0.1, 0.05, 0.01, 0.005, 0.0025, 0.001, 0.00075, 0.0005, 0.0001,0.00005]
# learningRates=[2.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
#learningRates = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
#The uncommented learning rates are a good start point
#exp.checkLogRegAndLearningRates(adultXTrain, adultYTrain, adultXTest, adultYTest, learningRates, 200000, 1e-6)


