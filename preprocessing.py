import numpy as np
import keras

# def returnVectors(imagePath, labelPath):

images = np.load("images.npy")
image_vectors = []
#Flatten each 28x28 matrix
for image in images:
    image_vector = np.reshape(image,-1)
    image_vectors.append(image_vector)

image_vectors = np.array(image_vectors)
print image_vectors.shape
labels = np.load("labels.npy")

label_vectors = []
#make one hot encoding with labels
for label in labels:
    label_vector = keras.utils.to_categorical(label)
    label_vectors.append(label_vector)
label_vectors = np.array(label_vectors)
print label_vectors.shape

data = np.column_stack((label_vectors, image_vectors))
print data[0]
#     return data
#
# if __name__== "__main__":
#     returnVectors()
