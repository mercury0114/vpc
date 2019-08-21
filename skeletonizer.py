from __future__ import division
import itertools
import numpy
import scipy
import cv2

from collections import deque
from skimage import io
from skimage.morphology import skeletonize
from math import sqrt
from PIL import Image, ImageDraw
from tifffile import memmap
import os

def computeSkeleton(collagenFile, skeletonFile):
    if (os.path.isfile(skeletonFile)): 
        print(skeletonFile + " already exists.")
        return
    
    collagen = memmap(collagenFile, dtype='uint8')
    skeleton = memmap(skeletonFile, dtype='uint8', shape=collagen.shape)
    skeleton[:,:] = collagen
    del collagen
    skeleton[skeleton == 255] = 1
    skeleton[:,:] = skeletonize(skeleton)
    skeleton[skeleton == 1] = 255
    del skeleton

def partitionSkeleton(skeletonFile, partitionFile):
    if (os.path.isfile(partitionFile)): 
        print(partitionFile + " already exists.")
        return

    skeleton = memmap(skeletonFile)
    partition = memmap(partitionFile, shape=skeleton.shape, dtype='uint8')
    partition[:,:] = skeleton
    del skeleton
    pixels = numpy.nonzero(partition)
    for turn in [1,2,3]:
        for pixel in range(len(pixels[0])):
            n = countNeighbours(partition, pixels[0][pixel], pixels[1][pixel])
            if (n > 2):
                partition[pixels[0][pixel], pixels[1][pixel]] = 0
    del partition

def countNeighbours(image, x, y):
    count = 0
    for x0, y0 in itertools.product((-1, 0, 1), (-1, 0, 1)):
        if (x + x0 >= 0 and x + x0 < image.shape[0] and y + y0 >= 0 and y + y0 < image.shape[1]):
            if (x0 != 0 or y0 != 0):
                if (image[x+x0, y+y0]):
                    count += 1
    return count;

def labelSkeletonParts(partitionFile, labelledFile):
    if (os.path.isfile(labelledFile)): 
        print(labelledFile + " already exists.")
        return
    
    partition = memmap(partitionFile)
    labelled = memmap(labelledFile, shape = partition.shape, dtype='uint32')
    pixels = numpy.nonzero(partition)
    labelNumber = 0
    for p in range(len(pixels[0])):
        if (labelled[pixels[0][p], pixels[1][p]] == 0):
            labelNumber += 1
            stack = [(pixels[0][p],pixels[1][p])]
            while(len(stack) > 0):
                x,y = stack.pop()
                labelled[x,y] = labelNumber
                neighbours = findNeighbours(partition, x, y)
                for n in neighbours:
                    if (labelled[n[0], n[1]] == 0):
                        stack.append(n)
    del partition
    del labelled
    return labelNumber

def makeFibersFromSkeleton(collagenFile, labelledSkeletonFile, fibersFile):
    if (os.path.isfile(fibersFile)):
        print(fibersFile + " already exists.")
        return

    collagen = memmap(collagenFile)
    skeleton = memmap(labelledSkeletonFile)
    assert(collagen.shape == skeleton.shape), "shapes do not match"
    fibers = memmap(fibersFile, shape=collagen.shape, dtype='uint32')
    fibers[:,:] = skeleton
    del skeleton

    # Perfoming a breath first search to assign a value to each collagen
    # pixels based on the value of the closest skeleton label.
    pixels = numpy.nonzero(fibers)
    queue = deque()
    for p in range(len(pixels[0])):
        queue.appendleft((pixels[0][p], pixels[1][p]))

    while (len(queue) > 0):
        x,y = queue.pop()
        neighbours = findNeighbours(collagen, x, y)
        for n in neighbours:
            if (fibers[n[0], n[1]] == 0):
                fibers[n[0], n[1]] = fibers[x,y]
                queue.appendleft(n)

    del collagen
    del fibers



def junction(skeleton, i, j):
	return countNeighbours(skeleton, i, j) > 2

def corner(skeleton, i, j):
	return countNeighbours(skeleton, i, j) == 1

def findNeighbours(skeleton, x, y):
    neighbours = []
    for x0, y0 in itertools.product((-1, 0, 1), (-1, 0, 1)):
        if (x + x0 >= 0 and x + x0 < skeleton.shape[0] and
            y + y0 >= 0 and y + y0 < skeleton.shape[1] and skeleton[x + x0, y + y0]):
            if (x != 0 or y != 0):
                neighbours.append((x + x0, y + y0))
    return neighbours

def findNeighbour(skeleton, i, j):
    neighbours = findNeighbours(skeleton, i, j)
    if (len(neighbours) == 0):
        return False
    return neighbours[0]

def countAndRemovePixels(skeleton, pixel):
	count = 0
	while(pixel):
		count += 1
		skeleton[pixel[0], pixel[1]] = False
		pixel = findNeighbour(skeleton, pixel[0], pixel[1])
	return count

def findOtherCorner(skeleton, pixel):
	n = findNeighbour(skeleton, pixel[0], pixel[1])
	while(n):
		skeleton[pixel[0], pixel[1]] = False
		pixel = n
		n = findNeighbour(skeleton, pixel[0], pixel[1])
	skeleton[pixel[0], pixel[1]] = False
	return pixel

def triangle(skeleton, i, j):
	if (i == 0 or i == skeleton.shape[0] - 1 or j == 0 or j == skeleton.shape[1] - 1):
		return False
	if (skeleton[i - 1, j] and skeleton[i, j - 1]):
		return True
	if (skeleton[i, j - 1] & skeleton[i + 1, j]):
		return True
	if (skeleton[i + 1, j] & skeleton[i, j + 1]):
		return True
	if (skeleton[i, j + 1] & skeleton[i - 1, j]):
		return True
	return False

def averageEdgeLength(s):
	skeleton = numpy.copy(s)
	# Splitting skeleton into edges
	for i,j in itertools.product(range(skeleton.shape[0]), range(skeleton.shape[1])):
		if skeleton[i, j] and triangle(skeleton, i, j):
			skeleton[i, j] = 0
	for i,j in itertools.product(range(skeleton.shape[0]), range(skeleton.shape[1])):
		if skeleton[i, j] and junction(skeleton, i, j):
			skeleton[i, j] = 0

	nEdges = 0
	pixelCount = 0
	for i,j in itertools.product(range(skeleton.shape[0]), range(skeleton.shape[1])):
		if skeleton[i,j] and corner(skeleton, i, j):
			nPixels = countAndRemovePixels(skeleton, (i, j))
			if (nPixels > 10):
				nEdges += 1
				pixelCount += nPixels
	if (nEdges == 0):
		return 0.0
	return pixelCount / nEdges

def getCurvature(s, pixel):
	length = 0
	first = numpy.copy(pixel)
	n = findNeighbour(s, pixel[0], pixel[1])
	while(n):
		s[pixel[0], pixel[1]] = False
		length += sqrt((pixel[0] - n[0])**2 + (pixel[1] - n[1])**2)
		pixel = n
		n = findNeighbour(s, pixel[0], pixel[1])
	s[pixel[0], pixel[1]] = False
	if (length == 0):
		return 1
	return sqrt((first[0] - pixel[0])**2 + (first[1] - pixel[1])**2) / length

def averageCurvature(s):
	skeleton = numpy.copy(s)
	# Splitting skeleton into edges
	for i,j in itertools.product(range(skeleton.shape[0]), range(skeleton.shape[1])):
		if skeleton[i, j] and triangle(skeleton, i, j):
			skeleton[i, j] = 0
	for i,j in itertools.product(range(skeleton.shape[0]), range(skeleton.shape[1])):
		if skeleton[i, j] and junction(skeleton, i, j):
			skeleton[i, j] = 0

	curvature = 0
	nEdges = 0
	for i,j in itertools.product(range(skeleton.shape[0]), range(skeleton.shape[1])):
		if skeleton[i,j] and corner(skeleton, i, j):
			curvature += getCurvature(skeleton, (i, j))
			nEdges += 1
	if (nEdges == 0):
		return 1
	return curvature / nEdges

def shortestDistance(i, j, collagen):
	bottom = (max(0, i - 50), max(0, j - 50))
	top = (min(i + 50, collagen.shape[0]), min(j + 50, collagen.shape[1]))
	area = numpy.array(collagen[bottom[0] : top[0], bottom[1] : top[1]])
	areaCopy = numpy.copy(area)
	areaCopy[area == 0] = 1
	areaCopy[area != 0] = 0
	pixels = numpy.stack(numpy.nonzero(areaCopy), axis = -1)
	# pixels = cv2.findNonZero(areaCopy)
	if (len(pixels) == 0):
		return 60
	centerX = area.shape[0] / 2
	centerY = area.shape[1] / 2
	distances = numpy.sqrt((pixels[:,0] - centerX)**2 + (pixels[:,1] - centerY)**2)
	return min(distances)

def averageWidth(skeleton, collagen):
	width = 0
	count = 0
	for i,j in itertools.product(range(skeleton.shape[0]), range(skeleton.shape[1])):
		if (skeleton[i,j]):
			count += 1
			width += shortestDistance(i, j, collagen)
	if (count == 0):
		return 0.0
	return width / count

def drawhex(w=260,fill=1,inv=0):
    h = int(numpy.sqrt(3)*.5*260 / 2) * 2
    polygon = [(0,h/2),(w/4,0),(3*w/4,0),(w,h/2),(3*w/4,h),(w/4,h)]
    img_ = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img_).polygon(polygon, outline=1, fill=fill)
    img = numpy.array(img_)
    if inv != 0:
        img = ImageOps.invert(img_)
        img = numpy.array(img) / 255.
    return(img.astype(numpy.uint8))
emptyHex = drawhex()

def depth(collagen, centerX, centerY):
	print(emptyHex.shape)
	return 0.0



# image = io.imread("mask.png")
# image[image == 255] = 1
# skeleton = skeletonize(image)
