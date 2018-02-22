import preprocessing

images, labels = preprocessing.returnVectors("images.npy","labels.npy")

print(len(images), len(labels))
