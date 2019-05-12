from __future__ import division
import itertools
import numpy as np
import scipy
import cv2

from skimage import io
from skimage.morphology import skeletonize
from math import sqrt
def countPixels(skeleton, i, j):
	count = 0
	for x, y in itertools.product((-1, 0, 1), (-1, 0, 1)):
		if (i + x >= 0 and i + x < skeleton.shape[0] and j + y >= 0 and j + y < skeleton.shape[1]):
			count += skeleton[i + x, j + y]
	return count

def junction(skeleton, i, j):
	return countPixels(skeleton, i, j) > 3

def corner(skeleton, i, j):
	return countPixels(skeleton, i, j) == 2

def findNeighbour(skeleton, i, j):
    for x, y in itertools.product((-1, 0, 1), (-1, 0, 1)):
        if (i + x >= 0 and i + x < skeleton.shape[0] and
            j + y >= 0 and j + y < skeleton.shape[1] and skeleton[i + x, j + y]):
            if (x != 0 or y != 0):
                return (i + x, j + y)
    return False

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
	skeleton = np.copy(s)
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
	first = np.copy(pixel)
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
	skeleton = np.copy(s)
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
	area = np.array(collagen[bottom[0] : top[0], bottom[1] : top[1]])
	areaCopy = np.copy(area)
	areaCopy[area == 0] = 1
	areaCopy[area != 0] = 0
	pixels = np.stack(np.nonzero(areaCopy), axis = -1)
	# pixels = cv2.findNonZero(areaCopy)
	if (len(pixels) == 0):
		return 60
	centerX = area.shape[0] / 2
	centerY = area.shape[1] / 2
	distances = np.sqrt((pixels[:,0] - centerX)**2 + (pixels[:,1] - centerY)**2)
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

# image = io.imread("mask.png")
# image[image == 255] = 1
# skeleton = skeletonize(image)
