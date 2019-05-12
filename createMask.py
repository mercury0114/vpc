import openslide
import os
import numpy
from tifffile import memmap
from keras.models import load_model
from scipy.misc import imsave
from extractor import *

slide = openslide.open_slide("./../data/image.svs")

if os.path.isfile("./../data/mask.svs"):
	os.remove("./../data/mask.svs")
collagen = memmap("./../data/mask.svs",
            shape=slide.dimensions,
            dtype='uint8',
			bigtiff = True)

s = 256
model = load_model("./../data/model.h5")

print("Computing collagen mask")
for x in range(0, slide.dimensions[0] - (s-1), s/2):
	for y in range(0, slide.dimensions[1] - (s-1), s/2):
		patch = slide.read_region(location = (x, y), size = (s, s), level = 0)
		patch = numpy.array(patch)[:,:,:3]
		w = model.predict(patch.reshape(1, s, s, 3))
		w[w <= 0.5] = 0
		w[w > 0.5] = 255
		w = w.astype(np.uint8)
		current = numpy.array(collagen[x:x+s, y:y+s])
		collagen[x:x + s, y:y + s] = current | w.reshape(s, s)
del collagen
		
