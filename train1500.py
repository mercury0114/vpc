import h5py
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dropout
from keras.layers import Dense, Flatten, Input
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
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
    s = Input((1500, 1500, 1))
    d1 = addConvolutionalLayer(s, 32)
    d2 = addConvolutionalLayer(MaxPooling2D(4)(d1))
    d3 = addConvolutionalLayer(MaxPooling2D(4)(d2))
    d4 = addConvolutionalLayer(MaxPooling2D(4)(d3))
    f = Flatten()(d4)
    dense = Dense(12, activation='relu') (f)
    o = Dense(1, activation='softmax') (dense)
    model = Model(inputs = [s], outputs = [o])
    model.compile(optimizer='adam', loss='binary_crossentropy',
				  metrics=['accuracy'])
    return model

print("Getting training and testing data")
f = open("./../data/survivalBalanced.txt")
next(f)
X = []
y = []
for line in f:
    values = line.split(",")
    id = values[1]
    y.append(int(values[3]))
    image = imread("./../data/masksBalanced/" + id)
    X.append(numpy.array(image).astype(float) / 255)

X = numpy.array(X)
X = numpy.expand_dims(X, axis = 3)
y = numpy.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("Building model")
model = buildModel()
model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
print(model.summary())


print("Training model")
earlyStopper = EarlyStopping(monitor='val_loss', patience=45, verbose=1)
checkPointer = ModelCheckpoint('./../data/tempPrediction.h5', verbose=1, save_best_only=True)
model.fit(X_train, y_train, validation_split = 0.2, batch_size = 2,
	  epochs = 300, callbacks=[earlyStopper, checkPointer])
model.save('./prediction.h5')
