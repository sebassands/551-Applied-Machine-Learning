# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:42:01 2020

@author: sarena2
"""

import numpy as np
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
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

twenty_train = fetch_20newsgroups(subset='train', remove=(['headers', 'footers', 'quotes']))
twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42,remove=(['headers', 'footers', 'quotes']))


#Decision trees
###validation set partition

X_train, X_val, y_train, y_val  = train_test_split(twenty_train.data, twenty_train.target, test_size=0.2, random_state=1)

# Random Forest on twenty dataset ############# ORIGINAL HYPER-PARAMETERS FROM SKLEARN###################################

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
print('Random Forest TRAINING accuracy' ,dttacc)

start = time.time()
predicted = text_clf.predict(X_val)
dtacc=accuracy_score(y_val, predicted)
print(f'validation time: {time.time()-start}')
print('Random Forest VALIDATION accuracy' ,dtacc)

#####################################Grid Search
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
    'clf__max_depth': [1000,500,2000,1500],
    'clf__min_samples_leaf': [1,5,10,30,50],
    'clf__ccp_alpha': [0.0001,0.00001,0.00005,0.0005,0.000001],
    'clf__max_leaf_nodes': [30,50,80,100,200,300,400,500],
    'clf__n_estimators': [300,500, 700,850,1000],
    'tfidf__max_features': (None, 50000),
    'tfidf__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2')
    
      }

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
gs_clf.fit(twenty_train.data, twenty_train.target)

#predicted = text_clf.predict(imdbTest.data)
#dtacc=accuracy_score(imdbTest.target, predicted)
#print('Decision Tree testing accuracy' ,dtacc)

best_hyperparams = gs_clf.best_params_
print('Best hyerparameters:\n', best_hyperparams)

print(gs_clf.cv_results_)
print(gs_clf.best_score_)


# Random Forest on twenty dataset ############# Tuned HYPER-PARAMETERS FROM SKLEARN###################################

start = time.time()
text_clf = Pipeline([
    ('vect', CountVectorizer(max_df=0.5,ngram_range=(1,2),stop_words='english')),
    ('tfidf', TfidfTransformer(sublinear_tf=True)),
    #('sc',  StandardScaler()),
    #('pca', decomposition.PCA()),
    ('clf', RandomForestClassifier(n_estimators=700,max_leaf_nodes=400,ccp_alpha=0.00001,min_samples_split=60,criterion= 'gini', max_depth = 500, min_samples_leaf= 1)),
    ])
text_clf.fit(X_train, y_train)
print(f'training time: {time.time()-start}')
#Decision Trees (accuracy)

start = time.time()
predicted = text_clf.predict(twenty_train.data)
dttacc=accuracy_score(twenty_train.target, predicted)
print(f'training prediction time: {time.time()-start}')
print('Random Forest TRAINING accuracy' ,dttacc)

start = time.time()
predicted = text_clf.predict(twenty_test.data)
dtacc=accuracy_score(twenty_test.target, predicted)
print(f'testing time: {time.time()-start}')
print('Random Forest TESTING accuracy' ,dtacc)



