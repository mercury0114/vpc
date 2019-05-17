import openslide
import os
import numpy
import mahotas
from numpy import memmap
from scipy.misc import imsave
from skeletonizer import *
from extractor import *
from PIL import Image, ImageDraw
from tifffile import memmap
from skimage.morphology import skeletonize
from data import *

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
mask = memmap("./../data/mask.svs",
			dtype='uint8')


centers = getGrid(8207)


print("Calculating statitics")
with open("./../data/features_8207.txt", "w+") as f:
	f.write("width,length,curvature,area,connectivity,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13\n")
	for center in centers:
		c = [int(e) for e in center]
		if (c[0] >= 112 and c[0] + 112 <= mask.shape[0] and c[1] >= 130 and c[1] + 130 <= mask.shape[1]):
			patch = mask[c[0] - 112 : c[0] + 112, c[1] - 130 : c[1] + 130]
			patch = numpy.array(patch)
			collagen = emptyHex & patch
			skeleton = skeletonize(collagen)

			statistics = []
			statistics.append(averageWidth(skeleton, collagen))
			statistics.append(averageEdgeLength(skeleton))
			statistics.append(averageCurvature(skeleton))
			statistics.append(collagenAreaRatio(collagen))
			statistics.append(collagenConnectivityRatio(collagen))
			statistics.extend(mahotas.features.haralick(collagen).mean(0))

			f.write(','.join(str(e) for e in statistics))
			f.write('\n')
			f.flush()
f.close()

print("Statistics calculated")
