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
    print("Getting openslide")
    slide = getOpenSlide(gID)
    afnam = slide.properties['aperio.Filename']
    print(afnam)
    outdir = "".join(['../data/masks/', afnam, '/'])
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Computing raw collagen mask")
    rawCollagenFile = outdir + "raw.tiff"
    computeRawCollagenMask(slide, rawCollagenFile, model)

    print("Removing small blops")
    collagenWithoutBlopsFile = outdir + "without_blops.tiff"
    removeSmallCollagenBlops(rawCollagenFile, collagenWithoutBlopsFile)

    print("Filling holes in collagen")
    collagenHolesFilledFile = outdir + "holes_filled.tiff"
    fillHoles(collagenWithoutBlopsFile, collagenHolesFilledFile)

    print("Done with " + afnam)
print("DONE with ALL")

