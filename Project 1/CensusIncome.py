import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from pandas.plotting import scatter_matrix
#from LogisticRegression import LogisticRegression1

# A bunch of modules:

def value_count_df(df):
	for col in df:
		print(col)
		print(df[col].value_counts())
		print(' ')

def data_summary(df):
	# This function first splits the columns up into categorical columns and numerical columns,
	# For categorical columns it prints how many entries are in each category.
	# For numerical columns it prints out various statistics: total count, mean, std, min, max, etc.

	categorical_columns = df.select_dtypes(include=[np.object])
	numerical_columns = df._get_numeric_data()

	for col in categorical_columns:
		print(col)
		print(df[col].value_counts())
		print(' ')

	for col in numerical_columns:
		print(col)
		print(df[col].describe())
		print(' ')


def categorical_histograms(df):
	# This function plots the categorical columns in histograms (technically bar plots)

	categorical_columns = df.select_dtypes(include=[np.object])
	cat_col_list = list(categorical_columns)

	fig, ax = plt.subplots(1, len(cat_col_list))

	for i, categorical_columns in enumerate(df[cat_col_list]):
		df[categorical_columns].value_counts().plot(kind='bar', ax=ax[i]).set_title(categorical_columns)
	
	plt.show()

def one_hot_plot(df, x): # x should be income_ >50K in this case
	# This function will plot cool stuff but i need
	# to figure out how to loop through what I actually want...
	categorical_columns = df.select_dtypes(include=[np.object])
	cat_col_list = list(categorical_columns)


	sb.set_style('whitegrid')
	sb.countplot(x='income_ >50K', hue='sex', data=final_df, palette='RdBu_r')
	plt.legend(loc = 'upper right')
	plt.show()

	
def remove_bad_values(df): # This function removes all rows which have a '?' as an entry in any column
	# This function removes all rows which have a '?' as an entry in any column
	for col in df.columns:
		if df.dtypes[col] == np.object:
			df = df[~df[col].str.contains('\?')]

	return df 

def splitData(data, testFrac):
    # randomize data and reset index (getting rid of old index)
    data = data.sample(frac=1).reset_index(drop=True)
    splitRow = np.ceil(len(data)*(1-testFrac)).astype('int')
    trainData = data[:splitRow]
    testData = data[splitRow:]
    # You should instigate a warning if training or test contains way more True than false!
    return trainData, testData

def evaluate_acc(y_true, y_prediction):
    accuracy = np.sum(y_true.T.ravel()==y_prediction)/len(y_true.T.ravel())
    #accuracy=np.mean(y_true==y_prediction)
    return accuracy

def income_correlation(df):
	# This function overlays categorical values based on income
	# Doesn't really work (can't figure out the legend) 
	# and probably not that useful anyways :)

	for col in df.columns:
		grouped = df.groupby(col)
		for key, group in grouped:
			group.code.hist(alpha=0.4, label=key)
			title = plt.title()
			plt.legend()
			plt.show()

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


def one_hot_plotting(df):
	# This function one-hot encodes all categorical columns and keeps the original columns for plotting 

	categorical_columns = df.select_dtypes(include=[np.object])
	one_hot = pd.get_dummies(categorical_columns[categorical_columns.columns])
	df_one_hot = pd.concat([df, one_hot], axis=1)

	return df_one_hot



#begin: 

filename = 'adult.data'

names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital status', 
		 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
		 'hours-per-week', 'native-country', 'income']

data = pd.read_csv(filename, names=names, index_col=False)

cleaned_df = remove_bad_values(data)
final_df = one_hot_plotting(cleaned_df)
'''

sb.set_style('whitegrid')
sb.countplot(x='income_ >50K', hue='sex', data=final_df, palette='RdBu_r')
plt.legend(loc = 'upper right')
plt.show()
sb.countplot(x='income_ >50K', hue='age', data=final_df, palette='RdBu_r')
plt.legend(loc = 'upper right')
plt.show()
sb.countplot(x='income_ >50K', hue='workclass', data=final_df, palette='RdBu_r')
plt.legend(loc = 'upper right')
plt.show()
sb.countplot(x='income_ >50K', hue='education', data=final_df, palette='RdBu_r')
plt.legend(loc = 'upper right')
plt.show()
sb.countplot(x='income_ >50K', hue='marital status', data=final_df, palette='RdBu_r')
plt.legend(loc = 'upper right')
plt.show()
sb.countplot(x='income_ >50K', hue='relationship', data=final_df, palette='RdBu_r')
plt.legend(loc = 'upper right')
plt.show()
sb.countplot(x='income_ >50K', hue='race', data=final_df, palette='RdBu_r')
plt.legend(loc = 'upper right')
plt.show()
sb.countplot(x='income_ >50K', hue='native-country', data=final_df, palette='RdBu_r')
plt.legend(loc = 'upper right')
plt.show()
sb.countplot(x='income_ >50K', hue='income', data=final_df, palette='RdBu_r')
plt.legend(loc = 'upper right')
plt.show()
'''
'''
data.hist()
plt.show()
remove_some_columns_df = data.drop(['fnlwgt','capital-gain', 'capital-loss'], axis=1)
remove_some_columns_df.boxplot()
plt.show()


data.groupby('sex').age.hist(alpha=0.4)
scatter_matrix(data, alpha=0.2, figsize=(14, 14), diagonal='kde')

plt.show()
'''



categorical_histograms(final_df)
data_summary(final_df)

value_count_df(cleaned_df)