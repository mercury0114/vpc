import openslide
import os
import numpy
from tifffile import memmap
from keras.models import load_model
from scipy.misc import imsave
from extractor import computeRawCollagenMask
from extractor import removeSmallCollagenBlops
from extractor import fillHoles
import os
import sys

sys.path.append("./../data")
from data import *

grids = readGridIDs("./../data/gIDs.txt")

print("Loading model")
model = load_model("./../data/model.h5")
print("Model loaded")

for gID in grids:
    print("Computing for gID " + str(gID))
    print("Getting openslide")
    slide = getOpenSlide(gID)
    masksDir = "".join(['../data/masks/', str(gID), '/'])
    if not os.path.exists(masksDir):
        os.makedirs(masksDir)

    print("Computing raw collagen mask")
    rawCollagenFile = masksDir + "raw.tiff"
    computeRawCollagenMask(slide, rawCollagenFile, model)

    print("Removing small blops")
    collagenWithoutBlopsFile = masksDir + "without_blops.tiff"
    removeSmallCollagenBlops(rawCollagenFile, collagenWithoutBlopsFile)

    print("Filling holes in collagen")
    collagenHolesFilledFile = masksDir + "holes_filled.tiff"
    fillHoles(collagenWithoutBlopsFile, collagenHolesFilledFile)

    print("Computing skeleton")
    skeletonFile = masksDir + "skeleton.tiff"
    computeSkeleton(collagenHolesFilledFile, skeletonFile)

    print("Labelling partitioned skeleton")
    labelledFile = masksDir + "labels.tiff"
    numberOfParts = labelSkeletonParts(partitionFile, labelledFile)
    print("Labelled into " + str(numberOfParts) + " parts.")

    print("Splitting collagen into fibers")
    fibersFile = masksDir + "fibers.tiff"
    makeFibersFromSkeleton(collagenHolesFilledFile, labelledFile, fibersFile)

    print("Done with " + str(gID))
print("DONE with ALL")

