# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:35:44 2020

@author: sarena2
"""

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import seaborn as sb
from sklearn.metrics import plot_confusion_matrix

twenty_train = fetch_20newsgroups(subset='train', remove=(['headers', 'footers', 'quotes']))
twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42,remove=(['headers', 'footers', 'quotes']))


clf1 = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(solver='lbfgs', penalty='l2', max_iter=1000, verbose=True, C=17))
])

clf2 = Pipeline([
    ('vect', CountVectorizer(max_df=0.5,ngram_range=(1,2),stop_words='english')),
    ('tfidf', TfidfTransformer(sublinear_tf=True)),
    #('sc',  StandardScaler()),
    #('pca', decomposition.PCA()),
    ('clf', RandomForestClassifier(n_estimators=700,max_leaf_nodes=400,ccp_alpha=0.00001,min_samples_split=60,criterion= 'gini', max_depth = 500, min_samples_leaf= 1)),
    ])

clf3 = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    #('sc',  StandardScaler()),
    #('pca', decomposition.PCA()),
    ('clf', MultinomialNB() ),
    ])

clf4 = Pipeline([
    ('vect', CountVectorizer(max_df=0.5,ngram_range=(1,2),stop_words='english')),
    ('tfidf', TfidfTransformer(sublinear_tf=True)),
    #('sc',  StandardScaler()),
    #('pca', decomposition.PCA()),
    ('clf', DecisionTreeClassifier(max_leaf_nodes=300,ccp_alpha=0.00001,min_samples_split=200,criterion= 'gini', max_depth = 3000, min_samples_leaf= 1)),
    ])

clf5 = Pipeline([
                        ('vect', CountVectorizer(ngram_range=(1,2), max_df=0.2)),
                        ('tfidf', TfidfTransformer(sublinear_tf=True)),
                        ('clf', LinearSVC(penalty='l2', C=20.0, dual=True, loss='squared_hinge'))
])

clf6 = Pipeline([
				('vect', CountVectorizer(max_df=0.5, stop_words= 'english')),
				#('vect', CountVectorizer()),
				('tfidf', TfidfTransformer()),
				('clf', AdaBoostClassifier(base_estimator= DecisionTreeClassifier(max_depth=2),
										   n_estimators= 10000,
										   learning_rate= 1,
										   algorithm= 'SAMME'
												   )),
					])

#eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('mnb', clf3),('DT', clf4),('SVM', clf5)], voting='hard')

start = time.time()

vot = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2),('SVM', clf5)], voting='hard')
clss=vot.fit(twenty_train.data, twenty_train.target)
print(f'training time: {time.time()-start}')
#Decision Trees (accuracy)

start = time.time()
predicted = vot.predict(twenty_train.data)
dttacc=accuracy_score(twenty_train.target, predicted)
print(f'training prediction time: {time.time()-start}')
print('Voting TRAINING accuracy' ,dttacc)

start = time.time()
predicted = vot.predict(twenty_test.data)
dttacc=accuracy_score(twenty_test.target, predicted)
print(f'training prediction time: {time.time()-start}')
print('Voting TESTING accuracy' ,dttacc)



# Confusion Matrix
cfmatrix = confusion_matrix(twenty_test.target, predicted, normalize='all')
cfdataframe = pd.DataFrame(cfmatrix,index = [i for i in "ABCDEFGHIJKLMNOPQRST"],
                  columns = [i for i in "ABCDEFGHIJKLMNOPQRST"])
plt.figure(figsize = (20,20))
sb.heatmap(cfdataframe, annot=True)
plt.savefig('confusm.pdf')

