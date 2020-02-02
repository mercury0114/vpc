from tifffile import memmap
from skimage.morphology import skeletonize
import math
import numpy
import itertools

def findNeighbours(image, x, y):
    neighbours = []
    for x0, y0 in itertools.product((-1, 0, 1), (-1, 0, 1)):
        if (x + x0 >= 0 and x + x0 < image.shape[0] and
            y + y0 >= 0 and y + y0 < image.shape[1] and image[x + x0, y + y0]):
            if (x0 != 0 or y0 != 0):
                neighbours.append((x + x0, y + y0))
    return neighbours

def extractOneFiber(collagen, x, y):
    fiberId = collagen[x, y]
    xMin = xMax = x
    yMin = yMax = y
    stack = [(x,y)]
    visited = {}
    while len(stack) > 0:
        x,y = stack.pop()
        visited[x, y] = True
        xMin = min(xMin, x)
        yMin = min(yMin, y)
        xMax = max(xMax, x)
        yMax = max(yMax, y)
        neighbours = findNeighbours(collagen, x, y)
        for n in neighbours:
            if collagen[n] == fiberId and not n in visited:
                stack.append(n)
    fiber = numpy.zeros(shape=(xMax+1-xMin, yMax+1-yMin), dtype='uint32')
    fiber[:,:] = collagen[xMin : xMax + 1, yMin : yMax + 1]
    fiber[fiber != fiberId] = 0
    fiber[fiber == fiberId] = 255
    return fiber

def length(fiber):
    copy = numpy.copy(fiber)
    copy[copy != 0] = 1
    skeleton = skeletonize(copy).astype(int)
    pixels = numpy.nonzero(skeleton)
    totalLength = 0.0
    for i in range(len(pixels[0])):
        x, y = pixels[0][i], pixels[1][i]
        neighbours = findNeighbours(skeleton, x, y)
        length = 0.0
        for n in neighbours:
            length += math.sqrt((x - n[0])**2 + (y - n[1])**2)
        if (len(neighbours) > 0):
            length /= (float(len(neighbours)))
        totalLength += length
    return totalLength

def width(fiber):
    return (fiber > 0).sum() // length(fiber)

def areaRatio(fiber):
    return float((fiber > 0).sum()) / (fiber.shape[0] * fiber.shape[1])




