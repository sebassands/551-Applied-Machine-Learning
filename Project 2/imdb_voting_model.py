# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:58:53 2020

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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


imdbTrain = load_files("train/")
imdbTest = load_files("test/")




clf1 = Pipeline([
                         ('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', LogisticRegression(penalty='l2',
                                           solver='liblinear',
                                           dual=True,
                                           C=20))
                         ])


clf4 = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    #('sc',  StandardScaler()),
    #('pca', decomposition.PCA()),
    ('clf', DecisionTreeClassifier( ccp_alpha= 0.0001,criterion= 'entropy', max_depth = 100, min_samples_leaf= 10, splitter= 'random',min_samples_split=300,max_leaf_nodes= 200)),
    ])


clf3 =  Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    #('sc',  StandardScaler()),
    #('pca', decomposition.PCA()),
    ('clf', MultinomialNB() ),
    ])

clf2 = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    #('sc',  StandardScaler()),
    #('pca', decomposition.PCA()),
    ('clf', RandomForestClassifier( n_estimators= 400,ccp_alpha= 0.0001,criterion= 'entropy',max_depth= 100,min_samples_leaf= 10,min_samples_split=400 ,max_leaf_nodes= 300)),
    ])

clf5 = Pipeline([
                         ('vect', CountVectorizer(max_df=0.2, ngram_range=(1,2))),
                         ('tfidf', TfidfTransformer(sublinear_tf=True)),
                         ('clf', LinearSVC(penalty='l2', dual=True, loss='squared_hinge', C=1.0)) 
])

clf6 = Pipeline([
				('vect', CountVectorizer(max_df=0.5, stop_words= 'english')),
				#('vect', CountVectorizer()),
				('tfidf', TfidfTransformer()),
				('clf', AdaBoostClassifier(base_estimator= DecisionTreeClassifier(max_depth=1),
										   n_estimators= 1250,
										   learning_rate= .5,
										   algorithm= 'SAMME.R'
												   )),
					])

#eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('mnb', clf3),('SVM', clf5)], voting='hard')

start = time.time()

vot = VotingClassifier(estimators=[('lr', clf1),('Ad', clf6),('SVM', clf5)], voting='hard')
clss =vot.fit(imdbTrain.data, imdbTrain.target)
print(f'training time: {time.time()-start}')
#Decision Trees (accuracy)

start = time.time()
predicted = vot.predict(imdbTrain.data)
dttacc=accuracy_score(imdbTrain.target, predicted)
print(f'training prediction time: {time.time()-start}')
print('Voting TRAINING accuracy' ,dttacc)

start = time.time()
predicted = vot.predict(imdbTest.data)
dttacc=accuracy_score(imdbTest.target, predicted)
print(f'training prediction time: {time.time()-start}')
print('Voting TESTING accuracy' ,dttacc)

