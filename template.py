from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

# Model Template
x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.process("images.npy","labels.npy")

model = Sequential() # declare model

model.add(Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
model.add(Activation('relu'))



#
model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('tanh'))

model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('relu'))
#
model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('tanh'))


model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('softmax'))


# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
history = model.fit(x_train, y_train,
                    validation_data = (x_val, y_val),
                    epochs=400,
                    batch_size=512)


# Report Results

print(history.history)
predict_arr = model.predict(x_test)
#print (predict_arr.shape)
y_act = list(map(np.argmax, y_test))
y_pred = list(map(np.argmax, predict_arr))
print("Confusion Matrix")
print(confusion_matrix(y_act,y_pred, [0,1,2,3,4,5,6,7,8,9]))

#print("Evaluating performance...")
#precision = precision_score(y_test, predict_arr, average=None)	# Calculate the precision
#recall    = recall_score(y_test, predict_arr, average=None)	# Calculate the recall
#f1        = f1_score(y_test, predict_arr, average=None)		# Calculate f1

# model.predict(x_test,y_test)
