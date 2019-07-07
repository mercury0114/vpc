import openslide
import os
import numpy
from tifffile import memmap
from keras.models import load_model
from scipy.misc import imsave
from extractor import extractCollagen
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
	slide = memmap("./../data/smaller.tiff", dtype='uint8')
	outfname = "./../data/masks/smaller.tiff"
	if os.path.isfile(outfname):
		print("Mask already exists")
		continue
	collagen = memmap(outfname,
	           shape=(slide.shape[0], slide.shape[1]),
	           dtype='uint8',
	           bigtiff = True)

	s = 256

	print("Computing collagen mask")
	for x in range(0, slide.shape[0] - (s-1), s/2):
		print(x)
		for y in range(0, slide.shape[1] - (s-1), s/2):
			patch = slide[x:x+s, y:y+s]
			patch = numpy.array(patch)[:,:,:3]
			current = numpy.array(collagen[x:x+s, y:y+s])
			collagen[x:x+s, y:y+s] = current | extractCollagen(patch, model)
	
	collagen[collagen==1] = 255
	del collagen
	print("Done with mask")

print("DONE with ALL")

