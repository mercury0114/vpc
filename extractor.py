from __future__ import division
import numpy as np
import collections
import cv2
from skimage.io import imread
from scipy.ndimage import label
from PIL import Image, ImageDraw

def sliding_window(image, stepSize, windowSize):
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def extractCollagen(patch, model):
	w = model.predict(patch.reshape(1, 256, 256, 3))
	threshold = 0.5
	w[w <= threshold] = 0
	w[w > threshold] = 25
	w = w.astype(np.uint8)
	return w.reshape(256, 256)

def extractCollagenWholeImage(imageFile, model):
	imh,imw = 256,256
	sSize = 128
	wSize = 256
	im1 = imread(imageFile)
	d0 = 256 - (im1.shape[0] % 256)
	d1 = 256 - (im1.shape[1] % 256)
	im2 = np.zeros(shape = (im1.shape[0]+d0,im1.shape[1]+d1, 3),dtype = np.uint8)
	im2[:im1.shape[0],:im1.shape[1]] = im1[:, :, :3]
	im3 = np.zeros(shape = (im2.shape[0],im2.shape[1]),dtype = np.int)
	for (x,y,window) in sliding_window(im2,stepSize=sSize,windowSize=(wSize,wSize,3)):
		if window.shape[0] == wSize and window.shape[1] == wSize:
			window = window.astype(np.float) / 255.
			w = model.predict(window.reshape(1,wSize,wSize,3), batch_size=1)
			w[w <= 0.5] = 0
			w[w > 0.5] = 255
			w = w.astype(np.uint8)
			im3[y:y+wSize,x:x+wSize] += w.reshape(wSize,wSize)
	im3 = im3[:-d0,:-d1]
	im3[im3 > 255] = 255
	# kernel = np.ones((5, 5), np.uint8)
	# collagen = cv2.dilate(im3, kernel, iterations = 1)
	# return cv2.erode(collagen, kernel, iterations = 1)
	return im3

def collagenAreaRatio(collagen):
	count = 0
	for x, y in np.ndindex(collagen.shape):
		if (collagen[x,y] == 255):
			count += 1
	return count / (collagen.shape[0] * collagen.shape[1])

def collagenConnectivityRatio(collagen):
	sum = 0
	labelled, num = label(collagen)
	for i in range(1, num, 1):
		count = np.count_nonzero(labelled == i)
		sum += (count * count)
	totalArea = np.count_nonzero(collagen);
	return sum / (totalArea * totalArea)
