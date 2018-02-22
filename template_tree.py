from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import preprocessing
import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from numpy import argmax
import re

# Model Template
x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.process("images.npy","labels.npy")

model = tree.DecisionTreeClassifier()

model.fit(x_train, y_train)


print ("model")
print (model)
print("====================")
expected = y_test
print ("expected")
print (expected)
print("====================")
predicted = model.predict(x_test)
print ("predicted")
print (predicted)
print("====================")
print("metrics 1")
print(metrics.classification_report(expected, predicted))
print("====================")
print("metrics 2")

y_act = list(map(np.argmax, expected))
y_pred = list(map(np.argmax, predicted))
#print y_act
#print y_pred
print(confusion_matrix(y_act,y_pred, [0,1,2,3,4,5,6,7,8,9]))
# inv_exp = format(np.argmax(expected, axis=0))
# re.sub("\s+", ",", inv_exp.strip())

# inv_pre = format(np.argmax(predicted, axis=0))
# re.sub("\s+", ",", inv_pre.strip())
# print(metrics.confusion_matrix(inv_exp, inv_pre))
