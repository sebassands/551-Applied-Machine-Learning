import pandas as pd
import numpy as np
'''
def one_hot_training(df):
	# This function one-hot encodes all categorical columns and removes the original categorical columns for training
	# It also renames the continuous columns 'cont-1', 'cont-2', etc
	# 				  the one-hot encoded columns 'cat-1', 'cat-2', etc
	#				  the output column 'output'

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
'''

def remove_bad_values(df): # This function removes all rows which have a '?' as an entry in any column
	# This function removes all rows which have a '?' as an entry in any column
	for col in df.columns:
		if df.dtypes[col] == np.object:
			df = df[~df[col].str.contains('\?')]

	return df 

def evaluate_acc(y_true, y_prediction):
    accuracy = np.sum(y_true.T.ravel()==y_prediction)/len(y_true.T.ravel())
    #accuracy=np.mean(y_true==y_prediction)
    return accuracy

def splitData(data, testFrac):
    # randomize data and reset index (getting rid of old index)
    data = data.sample(frac=1).reset_index(drop=True)
    splitRow = np.ceil(len(data)*(1-testFrac)).astype('int')
    trainData = data[:splitRow]
    testData = data[splitRow:]
    # You should instigate a warning if training or test contains way more True than false!
    return trainData, testData

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