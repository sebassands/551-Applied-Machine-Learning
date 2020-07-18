# Functions to prepare data
# COMP 551 mini project 1
# Lia Formenti
# Feb 4 / 2020

#Trevor: added the functions one_hot_encoding, prepAdultLR, and prepAdultNB

import numpy as np
import pandas as pd
#import magicGammaFunctions as mgf

##### Helper Functions #####

# Takes in a dataframe. Randomizes the rows.
# Splits the data into train and test based on testFrac,
# the fraction of data you want in the test set
def splitData(data, testFrac):
    # randomize data and reset index (getting rid of old index)
    data = data.sample(frac=1).reset_index(drop=True)
    splitRow = np.ceil(len(data)*(1-testFrac)).astype('int')
    trainData = data[:splitRow]
    testData = data[splitRow:]
    # You should instigate a warning if training or test contains way more True than false!
    return trainData, testData

# takes in pandas frames for the design matrix and classes.
# Outputs inputs as numpy arrays
# Types of y to be decided - I think int is most general and less error prone than float.
def createDesignArray(designFrame, classFrame):
    return designFrame.to_numpy(), classFrame.to_numpy()

# Lia's version of one hot encoding. Not used.
def oneHot(y):
    N, C = y.shape[0], np.max(y)
    yHot = np.zeros((N,C))
    yHot[np.arange(N), y.to_numpy(dtype=int)-1] = 1
    return yHot

##### Magic Gamma Dataset ##### 

def prepMagicGamma():
    # Load feature names, and frame with design matrix and classes
    magicFeatures = mgf.returnFeatureNamesList()
    magicFrame = mgf.loadMagic()
    
    # One hot encode the classes
    # oneHotYMatrix = oneHot(magicFrame.loc[:, magicFeatures[-1]])
    # Replace 'class' column of frame with one hot encoded matrix
    # magicFrame = magicFrame.drop(labels='class', axis=1)
    # magicFrame = pd.concat([magicFrame, pd.DataFrame(oneHotYMatrix)], axis=1)
    
    trainFrame, testFrame = splitData(magicFrame, 0.2)
    # yCols = oneHotYMatrix.shape[1]
    yCols = 1 # If not one hot encoding, class assignments are only 1 column.
    XTrain, yTrain = createDesignArray(trainFrame.iloc[:,:-yCols], trainFrame.iloc[:,-yCols:])
    
    XTest, yTest= createDesignArray(testFrame.iloc[:,:-yCols], testFrame.iloc[:,-yCols:])
    return XTrain, yTrain, XTest, yTest

##### Ionosphere dataset #####

def prepIonosphere():

    df= pd.read_csv(r'teammateCode/ionosphere.data', header=None)
    df.head(n=5)

    names = [("Feature "+str(i+1)) for i in range(34)]
    names.append("Outcome")


    temp = {}

    for i in range(35):
        temp[i]=names[i]

    df = df.rename(columns=temp)
    del(names,i,temp)
    y = df['Outcome']
    y1 = [1 if i == 'g' else 0 for  i in y]
    Y = pd.DataFrame(y1)
    #Y = y2.to_numpy(dtype=float)
    del(y,y1)

    X = df.drop(['Outcome','Feature 2'], axis=1)
    #X = dfx.to_numpy(dtype=float)
    #del(dfx)

    X_train, X_test = splitData(X, 0.2)
    y_train, y_test = splitData(Y, 0.2)

    X_train = X_train.to_numpy(dtype=float)
    X_test = X_test.to_numpy(dtype=float)
    y_train = y_train.to_numpy(dtype=float)
    y_test = y_test.to_numpy(dtype=float)
    return X_train, y_train, X_test, y_test

##### Lung Cancer Dataset #####
def prepLung():
    missing_values = ["n/a", "na", "--", "?"]
    df= pd.read_csv(r'lung-cancer.data', header=None,na_values = missing_values)
    df.head(n=5)
    
    names = [("Feature "+str(i+1)) for i in range(56)]
    names.append("Outcome")
    
    
    temp = {}
    
    for i in range(57):
        temp[i]=names[i]
    
    df = df.rename(columns=temp)
    del(names,i,temp)
    
    for i in range (56):
        median = df[(("Feature "+str(i+1)))].median()
        df[(("Feature "+str(i+1)) )].fillna(median, inplace=True)

def prepAdultLR():

    filename = 'adult.data'

    data = pd.read_csv(filename, header=None)
    data.head(n=5)


    df = one_hot_training(data)

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

    return X_train, y_train, X_test, y_test

def prepAdultNB():

    filename = 'adult.data'

    data = pd.read_csv(filename, header=None)


    df = one_hot_training(data)

    y1 = df['Outcome']

    Y = pd.DataFrame(y1)

    X = df.drop(['Outcome'], axis=1) 
    X = X.drop(['cont-2'], axis=1) #REMOVING fnl_wgt feature

    X_train, X_test = splitData(X, 0.2)
    y_train, y_test = splitData(Y, 0.2)
    
    return X_train, y_train, X_test, y_test



def one_hot_training(df):
    # This function 1. removes all rows which have a '?' as an entry in any column
    #               2. one-hot encodes all categorical columns and removes the original categorical columns for training
    # It also renames the continuous columns 'cont-1', 'cont-2', etc
    #                 the one-hot encoded columns 'cat-1', 'cat-2', etc
    #                 the output column 'output'
    for col in df.columns:
        if df.dtypes[col] == np.object:
            df = df[~df[col].str.contains('\?')]

    categorical_columns = df.select_dtypes(include=[np.object])
    continuous_columns = df._get_numeric_data()
    one_hot = pd.get_dummies(categorical_columns[categorical_columns.columns])

    name_cont = [("cont-"+str(i+1)) for i in range(len(continuous_columns.columns))] #name all of the continuous columns
    name_cat = [("cat-"+str(i+1)) for i in range(len(one_hot.columns) - 2)] #name all of the one-hot categorical columns
    names = name_cont + name_cat
    names.append('Outcome')
    df_one_hot = pd.concat([continuous_columns, one_hot], axis=1)
    df_one_hot = df_one_hot.iloc[:, :-1] #removes the last one hot encoding of outcome (all we need is one column for it, not two) --> <=50K = 1, >50K = 0
    df_one_hot.columns = names

    return df_one_hot