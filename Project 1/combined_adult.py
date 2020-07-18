import numpy as np
from Combined_NB import Combined_Naive_Bayes
import pandas as pd
from processing_functions import one_hot_training, remove_bad_values, evaluate_acc, splitData
import collections

filename = 'adult.data'

data = pd.read_csv(filename, header=None)
df = one_hot_training(data)

y1 = df['Outcome']
Y = pd.DataFrame(y1)
X = df.drop(['Outcome'], axis=1)
X = X.drop(['cont-2'], axis=1) #REMOVING fnl_wgt feature

X_train, X_test = splitData(X, 0.2)
y_train, y_test = splitData(Y, 0.2)
y_test = y_test.to_numpy(dtype=float)


nb = Combined_Naive_Bayes()

nb.fit(X_train, y_train)

predictions = nb.predict(X_test)

print ('Naive Bayes Accuracy:', evaluate_acc(y_test, predictions))
 