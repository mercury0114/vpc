import sys
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

sys.path.append("./../data")
from data import *
import time

def drawhex():
    w = 260
    h = int(numpy.sqrt(3)*.5*260 / 2) * 2
    polygon = [(0,h/2),(w/4,0),(3*w/4,0),(w,h/2),(3*w/4,h),(w/4,h)]
    img_ = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img_).polygon(polygon, fill=1)
    img = numpy.array(img_).astype('uint32')
    img[img == 1] = 0xFFFFFFFF
    return img
emptyHex = drawhex()

grids = readGridIDs("./../data/gIDs.txt")
	
for gID in grids:
    print("Dealing with " + str(gID))
    fibers = memmap("./../data/masks/" + str(gID) + "/fibers.tiff")
    centers = getGrid(gID)
    print("Calculating statitics")
    f = open("./../data/statistics/" + str(gID), "w")
    f.write("width,length,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13\n")
    for center in centers:
        c = [int(e) for e in center]
        if (c[0] >= 112 and c[0] + 112 <= fibers.shape[0] and c[1] >= 130 and c[1] + 130 <= fibers.shape[1]):
            patch = fibers[c[0] - 112 : c[0] + 112, c[1] - 130 : c[1] + 130]
            patch = numpy.array(patch)
            hexagon = emptyHex & patch
            print(hexagon.max())

			#statistics = []
			#statistics.append(averageWidth(skeleton, collagen))
			#statistics.append(averageEdgeLength(skeleton))
			#statistics.append(averageCurvature(skeleton))
			#statistics.append(collagenAreaRatio(collagen))
			#statistics.append(collagenConnectivityRatio(collagen))
			#statistics.extend([e for e in mahotas.features.haralick(collagen).mean(0)])

	        #f.write(','.join(str(e) for e in [gID]+statistics))
	        #f.write('\n')
	        #f.flush()
    f.close()
    print('DONE with ', gID)

print("Statistics calculated")
