import openslide
import os
import numpy
from tifffile import memmap
from keras.models import load_model
from scipy.misc import imsave
from extractor import *
from data import *
import sys

print("Getting openslide")
slide = getOpenSlide(8207)
print("Slide got")

if os.path.isfile("./../data/mask.svs"):
	os.remove("./../data/mask.svs")
collagen = memmap("./../data/mask.svs",
            shape=slide.dimensions,
            dtype='uint8',
			bigtiff = True)

s = 256
model = load_model("./../data/model.h5")

print("Model loaded")
print(model.summary())

sys.stdout.flush()

print("Computing collagen mask")
for x in range(0, slide.dimensions[0] - (s-1), s/2):
	for y in range(0, slide.dimensions[1] - (s-1), s/2):
		patch = slide.read_region(location = (x, y), size = (s, s), level = 0)
		patch = numpy.array(patch)[:,:,:3]
		current = numpy.array(collagen[x:x+s, y:y+s])
		collagen[x:x+s, y:y+s] = current | extractCollagen(patch, model)
del collagen
print("Done with mask")
		
