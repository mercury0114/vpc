import csv
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense

f = open("./../data/survivalBalanced.txt")
next(f)
survival = [line.split(",") for line in f]
output = {}
for entry in survival:
    output[entry[1]] = int(entry[3])


f = open("./../data/statisticsBalanced.txt", "r")
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

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.3)

print(trainX.shape)
print(testX.shape)

import keras
import tensorflow as tf
from keras.layers import *
from keras.models import Model

model = Sequential()
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training model")
earlyStopper = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
model.fit(trainX, trainY, validation_split = 0.3,
      epochs = 1000, callbacks=[earlyStopper])

print("Evaluation:")
print("Train 0 vs 1 ratio: " + str(float((trainY == 0).sum()) / len(trainY)))
print("Test 0 vs 1 ratio: " + str(float((testY == 0).sum()) / len(testY)))
print(model.evaluate(testX, testY))
