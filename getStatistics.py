import openslide
import os
import numpy
from numpy import memmap
from scipy.misc import imsave
from extractor import *
from PIL import Image, ImageDraw


def drawhex(w=260,fill=1,inv=0):
    h = int(np.sqrt(3)*.5*260 / 2) * 2
    polygon = [(0,h/2),(w/4,0),(3*w/4,0),(w,h/2),(3*w/4,h),(w/4,h)]
    img_ = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img_).polygon(polygon, outline=1, fill=fill)
    img = np.array(img_)
    if inv != 0:
        img = ImageOps.invert(img_)
        img = np.array(img) / 255.
    return(img.astype(np.uint8))
emptyHex = drawhex()


# TODO(mariusl): get actual collagen and centers slides
slide = openslide.open_slide("image.svs")
centers = [[1000, 1000], [2000, 2000]]

with open("features.txt", "w+") as f:
	for c in centers:
		print(c)
		patch = slide.read_region(location = (c[0] - 130, c[1] - 112), size = (260, 244), level = 0)
		patch = numpy.array(patch)
		patch[patch == 255] = 1
		collagen = emptyHex & patch
		skeleton = skeletonize(collagen)
		statistics = [averageWidth(skeleton, collagen), averageCurvature(skeleton)]
		f.write(','.join(str(e) for e in statistics))
		f.write('\n')
		f.flush()
f.close()
