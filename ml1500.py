import csv
import numpy
from random import choices, sample
from sklearn.preprocessing import normalize


f = open("./../data/survival1500.txt")
next(f)
survival = [line.split(",") for line in f]
output = {}
for entry in survival:
    output[int(entry[1])] = int(entry[3])

# TODO(mariusl): fix
del output[29438] # For some reason nothing was computed for this

f = open("./../data/statistics1500.txt", "r")
next(f)
data = [[float(x) for x in line.split(",")] for line in f]
dictionary = {}
for row in data:
    if (int(row[0]) in output):
        dictionary[int(row[0])] = row[1:]

X = []
y = []
for key in output.keys():
    X.append(dictionary[key])
    y.append(output[key])

X = numpy.array(X)
X = normalize(X, axis = 0)
y = numpy.array(y)

sample = choices([True,False], weights=[0.75, 0.25], k = len(X))
sample = numpy.array(sample)

trainX = X[sample]
trainy = y[sample]

testX = X[sample == False]
testy = y[sample == False]

"""
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors = 1)
kn.fit(trainX, trainy)
"""

import keras
import tensorflow as tf
from keras.layers import *
from keras.models import Model

s = Input((16,))
d1 = Dense(16, activation = 'relu') (s)
d2 = Dense(16, activation = 'relu') (d1)
o = Dense(1, activation = 'sigmoid') (d2)
model = Model(inputs = [s], outputs = [o])

model.compile(optimizer='adam',
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

model.fit(trainX, trainy, epochs=50, batch_size=1)

predictions = model.predict(testX)

