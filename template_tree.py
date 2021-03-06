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
# val = int(raw_input("Max Depth: "))
model = tree.DecisionTreeClassifier(max_depth = 10)
#4 different types of feature extraction

#1 avg pixel values for each number
def getAvgPixelIntensity(x_set):
    pixelIntensity =0
    pics = []
    #print(x_set.shape)
    for picture in x_set:
        sum = 0
        for pixel in picture:
            sum += pixel
        pixelIntensity +=sum
        pics.append(pixelIntensity/len(picture))


    pics = np.array(pics)

    #print(pics.shape)
    return pics

def getIntensitySum(x_set):
    pixelIntensity =0
    pics = []
    #print(x_set.shape)
    for picture in x_set:
        sum = 0
        for pixel in picture:
            sum += pixel
        pixelIntensity +=sum
        pics.append(pixelIntensity)


    pics = np.array(pics)

    #print(pics.shape)
    return pics

def getIndiceSum(x_set):
    pics = []

    for picture in x_set:
        indices = 0

        for i in range(len(picture)):
            if picture[i] >175:
                indices = indices + i
        pics.append(indices)

    pics = np.array(pics)
    print(pics.shape)
    return pics

#3 Black pixels - white pixels
def blackMinusWhite(x_set):
    pics = []
    for picture in x_set:
        whites = 0
        blacks = 0
        for pixel in picture:
            if pixel >= 200:
                blacks += 1
            else:
                whites+=1
        pics.append(blacks-whites)
    pics = np.array(pics, dtype=object)
    return pics
#3 number of black pixels
def getWhiteBlackRatio(x_set):
    pics = []
    for picture in x_set:
        whites = 0
        blacks = 0
        for pixel in picture:
            if pixel<100:
                whites +=1
            if pixel > 150:
                blacks += 1
        pics.append(whites/blacks)
    pics = np.array(pics, dtype=object)
    return pics

#4 Sort picture by pixel intensity
def sortByIntesnity(x_set):
    for picture in x_set:
        sorted(picture)
    return x_set

#### AVG PIXEL INTENSITY###
x_train_avg = getAvgPixelIntensity(x_train).reshape(-1,1)
x_test_avg = getAvgPixelIntensity(x_test).reshape(-1,1)
x_val_avg = getAvgPixelIntensity(x_val).reshape(-1,1)
#print(x_train_avg.shape)

#print(x_train.shape)
############################

###Intensities###
##Change the second number in reshape() if the number doesn't match the labels length
x_train_numInts = blackMinusWhite(x_train).reshape(-1,1)
x_test_numInts = blackMinusWhite(x_test).reshape(-1,1)
x_val_numInts = blackMinusWhite(x_val).reshape(-1,1)
###########################

#### Black White Ratio###
x_train_i_sum = getIndiceSum(x_train).reshape(-1,1)
x_test_i_sum = getIndiceSum(x_test).reshape(-1,1)
x_val_i_sum = getIndiceSum(x_val).reshape(-1,1)
############################

#### Black White Ratio###
x_train_sum = getIntensitySum(x_train).reshape(-1,1)
x_test_sum = getIntensitySum(x_test).reshape(-1,1)
x_val_sum = getIntensitySum(x_val).reshape(-1,1)
############################

#### SORTED BY PIXEL DENSITY###
x_train_sorted = sortByIntesnity(x_train)
x_test_sorted = sortByIntesnity(x_test)
x_val_sorted = sortByIntesnity(x_val)
############################

# print(x_train[0])
# print(x_train.shape)
# print(y_train.shape)

x_train = np.column_stack((x_train, x_train_avg, x_train_numInts, x_train_sorted, x_train_sum, x_train_i_sum))
x_test = np.column_stack((x_test, x_test_avg, x_test_numInts, x_test_sorted, x_test_sum, x_test_i_sum))
x_val = np.column_stack((x_val, x_val_avg,x_val_numInts, x_val_sorted, x_val_sum, x_val_i_sum))

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
