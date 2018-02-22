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

# Model Template
x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.process("images.npy","labels.npy")

model = tree.DecisionTreeClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)
accuracy_score(y_test, y_predict)

expected = y_test
predicted = model.predict(x_test)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
#print(metrics.confusion_matrix(expected, predicted))


