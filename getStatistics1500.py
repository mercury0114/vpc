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
from tifffile import memmap
import time

def getStatistics(collagen):
    statistics = []
    statistics.append(length(collagen))
    statistics.append(width(collagen))
    statistics.append(areaRatio(collagen))
    copy = numpy.copy(collagen)
    copy[copy > 0] = 1
    statistics.extend([e for e in mahotas.features.haralick(copy).mean(0)])
    return numpy.array(statistics)

def flushToFile(statistics, f):
    f.write(','.join(str(e) for e in statistics))
    f.write('\n')
    f.flush()

header = "id,length,width,areaRatio,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13\n"
statisticsFile = open("./../data/statistics1500.txt", "w")
statisticsFile.write(header)

grids = os.listdir("./../data/masks1500/")
for gID in grids:
    start = time.time()
    fibers = memmap("./../data/masks1500/" + str(gID) + "/fibers.tiff")
    nonzero = numpy.nonzero(fibers)

    computedFiberStatistics = {}
    for n in range(len(nonzero[0])):
        fiberId = fibers[nonzero[0][n], nonzero[1][n]]
        if fiberId not in computedFiberStatistics:
            fiber = extractOneFiber(fibers, nonzero[0][n], nonzero[1][n])
            # TODO(mariusl): fix this hack by removing small skeleton segments
            if (fiber.shape[0] * fiber.shape[1] > 16):
                computedFiberStatistics[fiberId] = getStatistics(fiber)

    # Must match the statistics header length minus id
    average = numpy.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    for value in computedFiberStatistics.values():
        average += value
    average /= len(computedFiberStatistics)
    
    flushToFile([int(gID)] + list(average), statisticsFile)
    print(time.time() - start)
    print('DONE with ', gID)

statisticsFile.close()
print("Statistics calculated")
