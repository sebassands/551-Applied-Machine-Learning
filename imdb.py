from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_files
import numpy as np
import time


############## TEST OVER MULTIPLE VALUES FOR RANDOM_STATE TO ENSURE STABLE RESULTS ##############


imdb_train = load_files("aclimdbdata/train/")
imdb_test = load_files("aclimdbdata/test/")

X_train, X_val, y_train, y_val  = train_test_split(imdb_train.data, imdb_train.target, test_size=0.2, random_state=1)

X_train_full = imdb_train.data
y_train_full = imdb_train.target
X_test = imdb_test.data
y_test = imdb_test.target


############################# DEFAULT HYPER-PARAMETERS #############################

clf_nb = Pipeline([
					('vect', CountVectorizer()),
					('tfidf', TfidfTransformer()),
					('clf', MultinomialNB()),
					])

clf_adaboost = Pipeline([
					('vect', CountVectorizer()),
					('tfidf', TfidfTransformer()),
					('clf', AdaBoostClassifier()),
					]) 

clf_logistic = Pipeline([
					('vect', CountVectorizer()),
					('tfidf', TfidfTransformer()),
					('clf', LogisticRegression()),
					]) 

clf_trees = Pipeline([
					('vect', CountVectorizer()),
					('tfidf', TfidfTransformer()),
					('clf', DecisionTreeClassifier()),
					]) 

clf_forest = Pipeline([
					('vect', CountVectorizer()),
					('tfidf', TfidfTransformer()),
					('clf', RandomForestClassifier()),
					]) 

clf_svm = Pipeline([
					('vect', CountVectorizer()),
					('tfidf', TfidfTransformer()),
					('clf', LinearSVC()),
					]) 


############################## TUNED HYPER-PARAMETERS ##############################

clf_nb_tuned = Pipeline([('vect', CountVectorizer()),
						 ('tfidf', TfidfTransformer()),
					   	 ('clf', MultinomialNB()),
					])

clf_adaboost_tuned = Pipeline([
				('vect', CountVectorizer(max_df=0.5, stop_words= 'english')),
				('tfidf', TfidfTransformer()),
				('clf', AdaBoostClassifier(base_estimator= DecisionTreeClassifier(max_depth=1),
										   n_estimators= 1250,
										   learning_rate= .5,
										   algorithm= 'SAMME.R'
												   )),
					]) 

clf_logistic_tuned = Pipeline([
				('vect', CountVectorizer()),
				('tfidf', TfidfTransformer()),
				('clf', LogisticRegression(penalty='l2',
										   C=8.0,
										   solver= 'liblinear',
										   dual=True
										   )),
					]) 

clf_trees_tuned = Pipeline([
			('vect', CountVectorizer()),
			('tfidf', TfidfTransformer()),
			('clf', DecisionTreeClassifier(ccp_alpha= 0.0001,
										   criterion= 'entropy',
										   max_depth = 100, 
										   min_samples_leaf= 10, 
										   splitter= 'random',
										   min_samples_split=300,
										   max_leaf_nodes= 200
										   )),
					]) 

clf_forest_tuned = Pipeline([
			('vect', CountVectorizer()),
			('tfidf', TfidfTransformer()),
			('clf', RandomForestClassifier(n_estimators= 400,
										   ccp_alpha= 0.0001,
										   criterion= 'entropy',
										   max_depth= 100,
										   min_samples_leaf= 10,
										   min_samples_split=400,
										   max_leaf_nodes= 300
										   )),
					]) 

clf_svm_tuned = Pipeline([
						('vect', CountVectorizer(max_df=0.2, ngram_range=(1,2))),
						('tfidf', TfidfTransformer(sublinear_tf=True)),
						('clf', LinearSVC(penalty='l2',
										  loss= 'squared_hinge',
										  dual=True,
										  C=1.0
										  )),
					]) 



############################## Train and Predict ##############################


# Run with default hyper-parameters:

clf_dict = {#'naive bayes': clf_nb, 
			#'adaboost': clf_adaboost, 
			#'logistic regression':clf_logistic, 
			#'decision trees': clf_trees, 
			#'random forest': clf_forest, 
			'svm': clf_svm
			}

for name, clf in clf_dict.items():

	start = time.time()
	clf.fit(X_train, y_train)
	print(f'{name} training time: {(time.time() - start):.2f} sec')

	start = time.time()
	predicted_train = clf.predict(X_train)
	train_acc = accuracy_score(y_train, predicted_train)
	print(f'{name} training prediction time: {(time.time()-start):.2f} sec')
	print(f'{name} bayes training accuracy: ', train_acc)

	start = time.time()
	predicted_val = clf.predict(X_val)
	val_acc = accuracy_score(y_val, predicted_val)
	print(f'{name} validation prediction time: {(time.time() - start):.2f} sec')
	print(f'{name} validation accuracy: ', val_acc)
	print('')



# Run with tuned hyper-parameters:
'''
clf_tuned = {'naive bayes': clf_nb_tuned, 
			 'adaboost': clf_adaboost_tuned, 
			 'logistic regression':clf_logistic_tuned, 
			 'decision trees': clf_trees_tuned, 
			 'random forest': clf_forest_tuned, 
			 'svm': clf_svm_tuned
			 }

for name, clf in clf_tuned.items():
	
	start = time.time()
	clf.fit(X_train, y_train)
	print(f'{name} tuned training time: {(time.time() - start):.2f} sec')

	start = time.time()
	predicted_tuned_train = clf.predict(X_train)
	tuned_train_acc = accuracy_score(y_train, predicted_tuned_train)
	print(f'{name} tuned training prediction time: {(time.time()-start):.2f} sec')
	print(f'{name} tuned training accuracy: ', tuned_train_acc)

	start = time.time()
	predicted_tuned_val = clf.predict(X_val)
	tuned_val_acc = accuracy_score(y_val, predicted_tuned_val)
	print(f'{name} tuned alidation prediction time: {(time.time() - start):.2f} sec')
	print(f'{name} tuned validation accuracy: ', tuned_val_acc)
	print('')

	start = time.time()
	clf.fit(X_train_full, y_train_full)
	print(f'{name} tuned full training time: {(time.time() - start):.2f} sec')

	start = time.time()
	predicted_tuned_test = clf.predict(X_test)
	tuned_val_acc = accuracy_score(y_test, predicted_tuned_test)
	print(f'{name} tuned testing prediction time: {(time.time() - start):.2f} sec')
	print(f'{name} tuned testing accuracy: ', tuned_val_acc)
	print('')'''