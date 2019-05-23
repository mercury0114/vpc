import openslide
import os
import numpy
from tifffile import memmap
from keras.models import load_model
from scipy.misc import imsave
from extractor import *
from data import *
import sys

print("Reading grid ID from cmd")
grids = readGridIDs(sys.argv[1:])
print(grids)

print("Loading model")
model = load_model("./../data/model.h5")
print(model.summary())
sys.stdout.flush()

for gID in grids:
	print("Getting openslide")
	gID = int(gID)
	slide = getOpenSlide(gID)
	afnam = slide.properties['aperio.Filename']
	print(afnam)
	sys.stdout.flush()
	outfname = "".join(['../data/masks/',afnam,'_msk.svs'])
	if os.path.isfile(outfname):
		print("Mask already exists")
		continue
	collagen = memmap(outfname,
	           shape=slide.dimensions,
	           dtype='uint8',
	           bigtiff = True)

	s = 256

	print("Computing collagen mask")
	for x in range(0, slide.dimensions[0] - (s-1), s/2):
		for y in range(0, slide.dimensions[1] - (s-1), s/2):
			patch = slide.read_region(location = (x, y), size = (s, s), level = 0)
			patch = numpy.array(patch)[:,:,:3]
			current = numpy.array(collagen[x:x+s, y:y+s])
			collagen[x:x+s, y:y+s] = current | extractCollagen(patch, model)
	del collagen
	print("Done with mask")

print("DONE with ALL")

