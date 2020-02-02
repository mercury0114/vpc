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

def addConvolutionalLayer(layer, newDepth = None):
    print(layer.get_shape().as_list())
    if newDepth is None:
	    newDepth = layer.get_shape().as_list()[3] * 2
    newLayer = Conv2D(newDepth, kernel_size = (3, 3),
					activation = 'relu', padding='same') (layer)
    newLayer = Dropout(0.2) (newLayer)
    return Conv2D(newDepth, kernel_size = (3, 3),
    		activation = 'relu', padding='same') (newLayer)

def buildModel():
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3,3), input_shape=(1500,1500,1)))
    model.add(MaxPooling2D(pool_size=(4,4)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, kernel_size=(3,3)))
    model.add(MaxPooling2D(pool_size=(4,4)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, kernel_size=(3,3)))
    model.add(MaxPooling2D(pool_size=(4,4)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def GetTrainingTestData():
    f = open("./../data/survivalBalanced.txt")
    next(f)
    zeroX = []
    zeroY = []
    oneX = []
    oneY = []
    for line in f:
        values = line.split(",")
        id = values[1]
        v = int(values[3])
        image = imread("./../data/masksBalancedMariaus/" + id)
        array = numpy.array(image).astype(float) / 255
        if (v == 0):
            zeroY.append(v)
            zeroX.append(array)
        else:
            oneY.append(v)
            oneX.append(array)

    zeroX = numpy.array(zeroX)
    oneX = numpy.array(oneX)
    zeroY = numpy.array(zeroY)
    oneY = numpy.array(oneY)

    zeroXTrain, zeroXTest, zeroYTrain, zeroYTest = train_test_split(zeroX, zeroY, test_size=0.3, random_state=0)
    
    oneXTrain, oneXTest, oneYTrain, oneYTest = train_test_split(oneX, oneY, test_size=0.3, random_state=0)
    
    X_train = numpy.concatenate((zeroXTrain, oneXTrain))
    X_train = numpy.expand_dims(X_train, axis = 3)
    y_train = numpy.concatenate((zeroYTrain, oneYTrain))
    X_test = numpy.concatenate((zeroXTest, oneXTest))
    X_test = numpy.expand_dims(X_test, axis = 3)
    y_test = numpy.concatenate((zeroYTest, oneYTest))
    return (X_train, X_test, y_train, y_test)

print("Getting training and testing data")
X_train, X_test, y_train, y_test = GetTrainingTestData()

print("Building model")
model = buildModel()
print(model.summary())

print("Training model")
earlyStopper = EarlyStopping(monitor='val_loss', patience=25, verbose=1)
checkPointer = ModelCheckpoint('./../data/tempPrediction.h5', verbose=1, save_best_only=True)
model.fit(X_train, y_train, batch_size = 8, shuffle=True, validation_split = 0.3,
	  epochs = 1000, callbacks=[earlyStopper, checkPointer])
model.save('./../data/prediction.h5')

print("Evaluation:")
print("Train 0 vs 1 ratio: " + str(float((y_train == 0).sum()) / len(y_train)))
print("Test 0 vs 1 ratio: " + str(float((y_test == 0).sum()) / len(y_test)))
print(model.evaluate(X_test, y_test, batch_size = 8))
