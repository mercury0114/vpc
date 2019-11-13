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
    model.add(Conv2D(64, kernel_size=(3,3), kernel_initializer='random_normal',
              input_shape=(1500,1500,3)))
    model.add(Conv2D(64, kernel_size=(3,3), kernel_initializer='random_normal'))
    model.add(MaxPooling2D(pool_size=(4,4)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, kernel_size=(3,3), kernel_initializer='random_normal'))
    model.add(Conv2D(128, kernel_size=(3,3), kernel_initializer='random_normal'))
    model.add(MaxPooling2D(pool_size=(4,4)))
    model.add(Dropout(0.2))
    model.add(Conv2D(256, kernel_size=(3,3),kernel_initializer='random_normal'))
    model.add(Conv2D(256, kernel_size=(3,3),kernel_initializer='random_normal'))
    model.add(MaxPooling2D(pool_size=(4,4)))
    model.add(Conv2D(512, kernel_size=(3,3),kernel_initializer='random_normal'))
    model.add(Conv2D(512, kernel_size=(3,3),kernel_initializer='random_normal'))
    model.add(Flatten())
    model.add(Dense(256, activation=tf.nn.relu, kernel_initializer='random_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation=tf.nn.relu,kernel_initializer='random_normal'))
    model.add(Dense(2, activation=tf.nn.softmax, kernel_initializer='random_normal'))
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
        image = imread("./../data/1500Balanced/" + id)
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

    zeroXTrain, zeroXValidation, zeroYTrain, zeroYValidation = train_test_split(zeroXTrain, zeroYTrain, test_size=0.3)
    
    oneXTrain, oneXTest, oneYTrain, oneYTest = train_test_split(oneX, oneY, test_size=0.3, random_state=0)

    oneXTrain, oneXValidation, oneYTrain, oneYValidation = train_test_split(oneXTrain, oneYTrain, test_size=0.3)
    
    X_train = numpy.concatenate((zeroXTrain, oneXTrain))
    y_train = numpy.concatenate((zeroYTrain, oneYTrain))
    X_validation = numpy.concatenate((zeroXValidation, oneXValidation))
    y_validation = numpy.concatenate((zeroYValidation, oneYValidation))
    X_test = numpy.concatenate((zeroXTest, oneXTest))
    y_test = numpy.concatenate((zeroYTest, oneYTest))
    return (X_train, X_validation, X_test, y_train, y_validation, y_test)

print("Getting training and testing data")
X_train, X_validation, X_test, y_train, y_validation, y_test = GetTrainingTestData()

print("Building model")
model = buildModel()
print(model.summary())

print("Training model")
earlyStopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
checkPointer = ModelCheckpoint('./../data/tempPrediction.h5', verbose=1, save_best_only=True)
model.fit(X_train, y_train, batch_size = 1, shuffle=True, validation_data=(X_validation, y_validation),
	  epochs = 100, callbacks=[earlyStopper, checkPointer])
model.save('./../data/prediction.h5')

print("Evaluation:")
print("Train 0 vs 1 ratio: " + str(float((y_train == 0).sum()) / len(y_train)))
print("Train size: " + str(len(y_train))) 
print("Validation 0 vs 1 ratio: " + str(float((y_validation == 0).sum()) / len(y_validation)))
print("Validation size: " + str(len(y_validation)))
print("Test 0 vs 1 ratio: " + str(float((y_test == 0).sum()) / len(y_test)))
print("Test size: " + str(len(y_test)))
print(model.evaluate(X_test, y_test, batch_size = 4))
