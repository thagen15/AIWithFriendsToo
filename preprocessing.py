import numpy as np
import numpy.random
import keras
import matplotlib.pyplot as plt
import tensorflow as tf

def showImage(img):
    image = img.reshape([28,28])
    plt.gray()
    plt.imshow(image)
    plt.show()
# def returnVectors(imagePath, labelPath):
def printImage(image):
    count=0
    thingToPrint = ""
    for bits in image:
        count+=1
        thingToPrint += str(bits)+" "
        if count%28==0:
            print thingToPrint
            thingToPrint =""


images = np.load("images.npy")
#print images[4]
image_vectors = []
# #Flatten each 28x28 matrix
# for image in images:
#     image_vector = np.reshape(image,-1)
#     image_vectors.append(image_vector)

image_vectors = images.reshape(6500, 784)
image_vectors = np.array(image_vectors)
print image_vectors[0].shape
#print image_vectors.shape
#labels = np.load("labels.npy")

labels = np.load("labels.npy")
labels_flat = labels.reshape(6500, 1)
label_vectors = []
one_hot_labels = keras.utils.to_categorical(labels_flat, num_classes=10)
# #make one hot encoding with labels
# for label in labels:
#     label_vector = keras.utils.to_categorical(label, num_classes = 10)
#     label_vectors.append(label_vector)
label_vectors = np.array(one_hot_labels)
print label_vectors[1].shape
#print label_vectors.shape
#print label_vectors[0]

data = np.column_stack((label_vectors, image_vectors))
np.random.shuffle(data)

one_hot_labels = data[:,0:10]
flattend_images = data[:,10:]
#one_hot_labels, flattend_images = np.split(data,2,10)
print one_hot_labels[1]
print ""
showImage(flattend_images[1])
#print data[1]

#     return data
#
# if __name__== "__main__":
#     returnVectors()
