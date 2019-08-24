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
from fiber import areaRatio
from PIL import Image, ImageDraw
from tifffile import memmap
from skimage.morphology import skeletonize

sys.path.append("./../data")
from data import *
import time

w = 260
h = int(numpy.sqrt(3) * .5 * 260 / 2) * 2
def drawhex():
    polygon = [(0,h//2),(w//4,0),(3*w//4,0),(w,h//2),(3*w//4,h),(w//4,h)]
    img_ = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img_).polygon(polygon, fill=1)
    img = numpy.array(img_).astype('uint32')
    img[img == 1] = 0xFFFFFFFF
    return img
emptyHex = drawhex()

statisticsHeader ="hexX,hexY,class,distance,length,width,areaRatio,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13\n"

def flushStatisticsToFile(collagen, center, f):
    statistics = center[:]
    statistics.append(length(collagen))
    statistics.append(width(collagen))
    statistics.append(areaRatio(collagen))
    copy = numpy.copy(collagen)
    copy[copy > 0] = 1
    statistics.extend([e for e in mahotas.features.haralick(copy).mean(0)])
    f.write(','.join(str(e) for e in statistics))
    f.write('\n')
    f.flush()

grids = ['8286']

start = time.time()
for gID in grids:
    print("Dealing with " + str(gID))
    fibers = memmap("./../data/masks/" + str(gID) + "/fibers.tiff")
    grid = getGrid(gID)
    directory = "./../data/statistics/" + str(gID) + "/"
    if not os.path.isdir(directory):
        os.mkdir(directory)

    hexFile = open(directory + "hex.txt", "w")
    fiberFile = open(directory + "fiber.txt", "w")
    hexFile.write(statisticsHeader)
    fiberFile.write(statisticsHeader)
    
    for center in grid:
        c = [int(e) for e in center]
        if (c[2] != 3 and c[2] != 5 and 
            c[0] >= h//2 and c[0] + h//2 <= fibers.shape[0] and
            c[1] >= w//2 and c[1] + w//2 <= fibers.shape[1]):
            patch = fibers[c[0] - h//2 : c[0] + h//2, c[1] - w//2 : c[1] + w//2]
            patch = numpy.array(patch)
            hexagon = emptyHex & patch
            if (hexagon.max() > 0):
                flushStatisticsToFile(hexagon, c, hexFile)

                processedFiber = {}
                nonzero = hexagon.nonzero()
                for n in range(len(nonzero[0])):
                    fiberId = hexagon[nonzero[0][n], nonzero[1][n]]
                    if fiberId not in processedFiber:
                        fiber = extractOneFiber(fibers, c[0] - h//2 + nonzero[0][n], c[1] - w//2 + nonzero[1][n])
                        flushStatisticsToFile(fiber, c, fiberFile)
                        processedFiber[fiberId] = True
    
    hexFile.close()
    fiberFile.close()
    print('DONE with ', gID)

print(time.time() - start)
print("Statistics calculated")
