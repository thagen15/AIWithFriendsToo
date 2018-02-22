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

def process(imagePath, labelsPath):
    images = np.load(imagePath)
    image_vectors = []

    image_vectors = images.reshape(6500, 784)
    image_vectors = np.array(image_vectors)
    print image_vectors[0].shape

    labels = np.load(labelsPath)
    labels_flat = labels.reshape(6500, 1)
    label_vectors = []
    one_hot_labels = keras.utils.to_categorical(labels_flat, num_classes=10)
    label_vectors = np.array(one_hot_labels)
    print label_vectors[1].shape

    data = np.column_stack((label_vectors, image_vectors))
    np.random.shuffle(data)

    one_hot_labels = data[:,0:10]
    flattend_images = data[:,10:]
    print one_hot_labels[0]
    print ""
    showImage(flattend_images[0])

    data_size = 6500
    training_size = int(data_size * 0.60)
    print ("Training Set size = ", training_size)
    val_size = int(data_size * (0.65 + 0.15))
    print ("Validation set index = ", val_size)

    num_classes = np.unique(labels).shape[0] #which is 10
    x_train, x_val, x_test = flattend_images[:training_size], flattend_images[training_size:val_size], flattend_images[val_size:]
    y_train, y_val, y_test = one_hot_labels[:training_size], one_hot_labels[training_size: val_size], one_hot_labels[val_size:]
    return x_train, y_train, x_val, y_val, x_test, y_test
