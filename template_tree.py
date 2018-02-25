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
model = tree.DecisionTreeClassifier()
#4 different types of feature extraction

#1 avg pixel values for each number
def getAvgPixelIntensity(x_set):
    pixelIntensity =0
    pics = []
    for picture in x_set:
        sum = 0
        for pixel in picture:
            sum += pixel
        pixelIntensity +=sum
        pics.append(pixelIntensity/len(picture))
    pics = np.array(pics)
    return pics

#2 Break picture down into 9 sections and assign each section mostly black or white
def ticTacToe(x_set):
    pics = []
    ###Doesn't really work, Leaving this for Parm
    for picture in x_set:
        picture.reshape(28,28)
        picture = np.hsplit(picture,2)


        for submatrice in picture:
            whites = 0
            blacks = 0
            for pixel in submatrice:
                if pixel<100:
                    whites +=1
                if pixel > 150:
                    blacks += 1
            if blacks/whites > .05:
                pics.append(1)
            else:
                pics.append(0)
    pics = np.array(pics)
    #print(pics)
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
    pics = np.array(pics)
    return pics

#4 Sort picture by pixel intensity
def sortByIntesnity(x_set):
    for picture in x_set:
        sorted(picture)
    return x_set



#### AVG PIXEL INTENSITY###
# x_train = getAvgPixelIntensity(x_train).reshape(-1,1)
# x_test = getAvgPixelIntensity(x_test).reshape(-1,1)
# x_val = getAvgPixelIntensity(x_val).reshape(-1,1)
############################

### TIC TAC TOE###
##Change the second number in reshape() if the number doesn't match the labels length
#x_train = ticTacToe(x_train).reshape(-1,2)
#x_test = ticTacToe(x_test).reshape(-1,2)
#x_val = ticTacToe(x_val).reshape(-1,2)
############################

#### Black White Ratio###
x_train = getWhiteBlackRatio(x_train).reshape(-1,1)
x_test = getWhiteBlackRatio(x_test).reshape(-1,1)
x_val = getWhiteBlackRatio(x_val).reshape(-1,1)
############################

#### SORTED BY PIXEL DENSITY###
# x_train = sortByIntesnity(x_train)
# x_test = sortByIntesnity(x_test)
# x_val = sortByIntesnity(x_val)
############################


print(x_train.shape)
print(y_train.shape)

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
