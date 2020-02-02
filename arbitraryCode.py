from extractor import *
from keras.models import load_model
from skeletonizer import *
from skimage.io import imread
import os

from fiber import length
import numpy

a = numpy.array([[1,1,1],[0,0,0]])
print(length(a))
b = numpy.array([[1,0,1],[0,1,0],[1,0,1]])
print(length(b))

"""
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


print("Loading model")
model = load_model("./../data/model.h5")
print("Model loaded")


files = os.listdir("./../data/1500/")

for file in files:
    gID = os.path.splitext(file)[0]
    outdir = "".join(["./../data/masks1500/", str(gID), '/'])
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    print("Extracting collagen")
    image = imread("./../data/1500/" + file)
    rawCollagenFile = outdir + gID
    computeRawCollagenMask(model, image, rawCollagenFile)

    print("Removing small blops")
    collagenWithoutBlopsFile = outdir + "without_blops.tiff"
    removeSmallCollagenBlops(rawCollagenFile, collagenWithoutBlopsFile, 100)

    print("Filling holes in collagen")
    collagenHolesFilledFile = outdir + "holes_filled.tiff"
    fillHoles(collagenWithoutBlopsFile, collagenHolesFilledFile)

    print("Computing skeleton")
    skeletonFile = outdir + "skeleton.tiff"
    computeSkeleton(collagenHolesFilledFile, skeletonFile)

    print("Partitioning skeleton")
    partitionFile = outdir + "partition.tiff"
    partitionSkeleton(skeletonFile, partitionFile)

    print("Labelling partitioned skeleton")
    labelledFile = outdir + "labels.tiff"
    numberOfParts = labelSkeletonParts(partitionFile, labelledFile)
    print("Labelled into " + str(numberOfParts) + " parts.")

    print("Splitting collagen into fibers")
    fibersFile = outdir + "fibers.tiff"
    makeFibersFromSkeleton(collagenHolesFilledFile, labelledFile, fibersFile)
"""
