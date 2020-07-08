# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:23:42 2020

@author: sarena2
"""

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.tree	import	DecisionTreeClassifier
from sklearn.tree	import	export_graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn import tree
import time
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

imdbTrain = load_files("train/")
imdbTest = load_files("test/")

# Decision tree on Imdb dataset ############# ORIGINAL HYPER-PARAMETERS FROM SKLEARN###################################
X_train, X_val, y_train, y_val  = train_test_split(imdbTrain.data, imdbTrain.target, test_size=0.2, random_state=1)

start = time.time()
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    #('sc',  StandardScaler()),
    #('pca', decomposition.PCA()),
    ('clf', RandomForestClassifier()),
    ])
text_clf.fit(X_train, y_train)
print(f'training time: {time.time()-start}')
#Decision Trees (accuracy)

start = time.time()
predicted = text_clf.predict(X_train)
dttacc=accuracy_score(y_train, predicted)
print(f'training prediction time: {time.time()-start}')
print('Decision Tree TRAINING accuracy' ,dttacc)

start = time.time()
predicted = text_clf.predict(X_val)
dtacc=accuracy_score(y_val, predicted)
print(f'testing time: {time.time()-start}')
print('Decision Tree VALIDATION accuracy' ,dtacc)

#CV_Result = cross_val_score(text_clf, imdbTest.data, imdbTest.target, cv=5, n_jobs=-1)
#print(); print(CV_Result)
#print(); print(CV_Result.mean())
#print(); print(CV_Result.std())

############################visualizing the hyper-parameters
CounVec = CountVectorizer()
TFIDF1 = TfidfTransformer()

TFIDF = TfidfVectorizer()

text_clf = Pipeline(steps=[
    # ('vect', CounVec),
    # ('tfidf1', TFIDF1),
    ('tfidf', TFIDF),
    #('sc',  StandardScaler()),
    #('pca', decomposition.PCA()),
    ('clf', RandomForestClassifier())
    ])

parameters = {
     'clf__criterion': ['gini','entropy'],
     'clf__min_samples_split': [10,20,30,40,50],
     'clf__max_depth': [100,50,200],
     'clf__min_samples_leaf': [1,5,10,30,50],
     'clf__ccp_alpha': [0.0001,0.00001,0.00005,0.0005,0.000001],
     'clf__max_leaf_nodes': [30,50,80,100,200,300]
    
    
      }

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
gs_clf.fit(imdbTrain.data, imdbTrain.target)
grid_scores = gs_clf.cv_results_

best_hyperparams = gs_clf.best_params_
print('Best hyerparameters:\n', best_hyperparams)

print(gs_clf.best_score_)

mean_test_score = gs_clf.cv_results_['mean_test_score']


# Decision tree on Imdb dataset #######With Tuned Hyperparameters#############

start = time.time()
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    #('sc',  StandardScaler()),
    #('pca', decomposition.PCA()),
    ('clf', RandomForestClassifier( n_estimators= 400,ccp_alpha= 0.0001,criterion= 'entropy',max_depth= 100,min_samples_leaf= 10,min_samples_split=400 ,max_leaf_nodes= 300)),
    ])
text_clf.fit(imdbTrain.data, imdbTrain.target)
print(f'training time: {time.time()-start}')
#Decision Trees (accuracy)

start = time.time()
predicted = text_clf.predict(imdbTrain.data)
dttacc=accuracy_score(imdbTrain.target, predicted)
print(f'training prediction time: {time.time()-start}')
print('Decision Tree training accuracy' ,dttacc)

start = time.time()
predicted = text_clf.predict(imdbTest.data)
dtacc=accuracy_score(imdbTest.target, predicted)
print(f'testing time: {time.time()-start}')
print('Decision Tree testing accuracy' ,dtacc)