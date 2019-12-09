import h5py
import tensorflow as tf
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dropout
from keras.layers import Dense, Flatten, Input
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import random
import numpy
from skimage.io import imread
random.seed(123)

def buildModel():
    model = Sequential()
    model.add(Dense(256, input_shape=(16,), activation=tf.nn.relu))
    model.add(Dense(64, activation=tf.nn.relu))
    model.add(Dense(2, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def GetData():
    survival = open("./../data/survival/survivalCRC.csv")
    next(survival)
    d = {}
    for line in survival:
        values = line.split(",")
        d[int(values[0])] = int(values[3])

    f = open("./../data/survival/hex.txt")
    next(f)
    X = []
    y = []
    for line in f:
        values = line.split(",")
        id = int(values[0])
        rank = int(values[1])
        if (rank == 0 and id in d):
            X.append([float(v) for v in values[2:]])
            y.append(d[id])
    return (numpy.array(X), numpy.array(y))

X, y = GetData()
print(X.shape)
print(y.shape)


accuracy = 0
for leave in range(X.shape[0]):
    print("Left " + str(leave))
    X_out = numpy.delete(X, leave, 0)
    y_out = numpy.delete(y, leave)
    model = buildModel()
    earlyStopper = EarlyStopping(monitor='val_loss', patience=50)
    model.fit(X_out, y_out, shuffle=True, validation_split = 0.2, epochs = 500, callbacks=[earlyStopper], verbose = 0)
    p = model.predict(numpy.expand_dims(X[leave], axis=0))
    print(p.argmax(), y[leave])
    if (p.argmax() == y[leave]):
        accuracy += 1

print("Accuracy: " + str(accuracy / float(X.shape[0])))
print("0 vs 1 ratio: " + str((y == 0).sum() / float(y.shape[0])))

