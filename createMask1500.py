from extractor import *
from keras import backend as K
from keras.models import load_model
from skeletonizer import *
from skimage.io import imread
import tensorflow as tf
import os
import numpy
import sys

def computeRawCollagenMask(model, image, rawCollagenFile):
    if (os.path.isfile(rawCollagenFile)):
        print(rawCollagenFile + " already exists")
        return
    s = 256
    rawCollagen = memmap(rawCollagenFile, shape=(image.shape[0], image.shape[1]), dtype='uint8')
    for x in range(0, image.shape[0] - (s-1), s/2):
        for y in range(0, image.shape[1] - (s-1), s/2):
            patch = image[x:x+s, y:y+s]
            current = numpy.array(rawCollagen[x:x+s, y:y+s])
            rawCollagen[x:x+s, y:y+s] = current | extractCollagen(patch, model)
    del rawCollagen

def mean_iou(y_true, y_pred):
    prec = []
    for t in numpy.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

print("Loading model")
model = load_model(sys.argv[1], custom_objects={'mean_iou' : mean_iou})
print("Model loaded")

assert(os.path.isdir(sys.argv[2]))
files = os.listdir("./../data/1500/")
count = 0
for file in files:
    count += 1
    print("Computation nr " + str(count) + " for " + file)
    outdir = "".join([sys.argv[2], str(file.split(".")[0]), '/'])
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    print("Extracting collagen")
    image = imread("./../data/1500/" + file)
    rawCollagenFile = outdir + "raw.tiff"
    computeRawCollagenMask(model, image, rawCollagenFile)

    print("Removing small blops")
    collagenWithoutBlopsFile = outdir + "without_blops.tiff"
    removeSmallCollagenBlops(rawCollagenFile, collagenWithoutBlopsFile, 100)

    print("Computing skeleton")
    skeletonFile = outdir + "skeleton.tiff"
    computeSkeleton(collagenWithoutBlopsFile, skeletonFile)

    print("Partitioning skeleton")
    partitionFile = outdir + "partition.tiff"
    partitionSkeleton(skeletonFile, partitionFile)

    print("Removing short skeleton segments")
    shortSkeletonRemovedFile = outdir + "short_removed.tiff"
    removeSmallCollagenBlops(partitionFile, shortSkeletonRemovedFile, 5)

    print("Labelling partitioned skeleton")
    labelledFile = outdir + "labels.tiff"
    numberOfParts = labelSkeletonParts(shortSkeletonRemovedFile, labelledFile)
    print("Labelled into " + str(numberOfParts) + " parts.")

    print("Splitting collagen into fibers")
    fibersFile = outdir + "fibers.tiff"
    makeFibersFromSkeleton(collagenWithoutBlopsFile, labelledFile, fibersFile)
