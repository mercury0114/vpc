import sys
import openslide
import os
import numpy
import mahotas
from numpy import memmap
from scipy.misc import imsave
from skeletonizer import *
from extractor import *
from fiber import extractOneFiber
from fiber import length
from fiber import width
from PIL import Image, ImageDraw
from tifffile import memmap
from skimage.morphology import skeletonize

sys.path.append("./../data")
from data import *
import time

w = 260
h = int(numpy.sqrt(3) * .5 * 260 / 2) * 2
def drawhex():
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
    f = open("./../data/statistics/" + str(gID), "w")
    f.write("hexX,hexY,distance,fiberId,width,length,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13\n")
    f.flush()
    i = 0
    for center in centers:
        i += 1
        print(str(i) + " out of " + str(len(centers)))
        c = [int(e) for e in center]
        if (c[0] >= h/2 and c[0] + h/2 <= fibers.shape[0] and c[1] >= w/2 and c[1] + w/2 <= fibers.shape[1]):
            patch = fibers[c[0] - h/2 : c[0] + h/2, c[1] - w/2 : c[1] + w/2]
            patch = numpy.array(patch)
            hexagon = emptyHex & patch
            processedFiber = {}
            for x in range(hexagon.shape[0]):
                for y in range(hexagon.shape[1]):
                    fiberId = hexagon[x,y]
                    if (fiberId > 0 and not fiberId in processedFiber):
                        print("Extracting")
                        fiber = extractOneFiber(fibers, c[0] - h/2 + x, c[1] - w/2 + y)
                        print(fiber.shape)
                        statistics = c[:]
                        statistics.append(fiberId)
                        statistics.append(length(fiber))
                        statistics.append(width(fiber))
                        statistics.extend([e for e in mahotas.features.haralick(fiber).mean(0)])
                        processedFiber[fiberId] = True
                        f.write(','.join(str(e) for e in statistics))
                        f.write('\n')
                        f.flush()
    f.close()
    print('DONE with ', gID)

print("Statistics calculated")
