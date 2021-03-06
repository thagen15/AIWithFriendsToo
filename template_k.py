from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors

# Model Template
x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.process("images.npy","labels.npy")

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(x_train, y_train)
#print(knn.fit(x_train, y_train))

# print("====================50
expected = y_test
# print ("expected")
# print (expected)
# print("====================")
predicted = knn.predict(x_test)
# print ("predicted")
# print (predicted)


a = knn.kneighbors(n_neighbors=3, return_distance=False)
preprocessing.showImage(x_train[a[0][0]])
preprocessing.showImage(x_train[a[0][1]])
preprocessing.showImage(x_train[a[0][2]])



#print(knn.kneighbors(X=[[784],[1]], n_neighbors=3, return_distance=True))
print("n=3")
print("====================")
print("metrics 1")
print(metrics.classification_report(expected, predicted))
print("====================")
print ("metric 2")
y_act = list(map(np.argmax, expected))
y_pred = list(map(np.argmax, predicted))
print(confusion_matrix(y_act,y_pred, [0,1,2,3,4,5,6,7,8,9]))