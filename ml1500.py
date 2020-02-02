import csv
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense


from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

f = open("./../data/survivalBalanced.txt")
next(f)
survival = [line.split(",") for line in f]
output = {}
for entry in survival:
    output[entry[1]] = int(entry[3])


f = open("./../data/statisticsBalancedMariaus.txt", "r")
next(f)
data = [line.split(",") for line in f]
dictionary = {}
for row in data:
    if (row[0] in output):
        dictionary[row[0]] = row[1:]

X = []
y = []
for key in output.keys():
    X.append(dictionary[key])
    y.append(output[key])

X = numpy.array(X)
X = normalize(X, axis = 0)
y = numpy.array(y)

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.25)

import keras
import tensorflow as tf
from keras.layers import *
from keras.models import Model

model = Sequential()
model.add(Dense(256, activation ='relu',
                kernel_initializer='random_uniform',
                bias_initializer='random_uniform',
                input_dim=16))
model.add(Dense(64, activation ='relu',
                kernel_initializer='random_uniform',
                bias_initializer='random_uniform'))
model.add(Dense(2, activation='softmax',
                kernel_initializer='random_uniform',
                bias_initializer='random_uniform'))

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.predict_proba(testX))

print("Training model")
earlyStopper = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
model.fit(trainX, trainY, shuffle=True,
          epochs = 10000, callbacks=[earlyStopper], verbose=0)

print("Evaluation:")

print("Train size: " + str(len(trainY)))
print("Train 0 vs 1 ratio: " + str(float((trainY == 0).sum()) / len(trainY)))
print("Training accuracy:")
print(model.evaluate(trainX, trainY))

print("Test size: " + str(len(testY)))
print("Test 0 vs 1 ratio: " + str(float((testY == 0).sum()) / len(testY)))
print("Test accuracy:")
print(model.evaluate(testX, testY))

print(model.predict_proba(testX))
