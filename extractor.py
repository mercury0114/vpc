import numpy
import os
from scipy.ndimage import label
from PIL import Image, ImageDraw
from tifffile import memmap
from skimage.morphology import skeletonize
import numpy
import cv2

s = 256

def removeSmallCollagenBlops(oldFile, newFile):
    assert(os.path.isfile(oldFile)), oldFile + " does not exist"
    if (os.path.isfile(newFile)):
        print(newFile + " already exists")
        return
    
    success = False
    try:
        oldCollagen = memmap(oldFile, dtype='uint8')
        newCollagen = memmap(newFile, shape = oldCollagen.shape, dtype='uint8')
        for x in range(0, oldCollagen.shape[0] - (s-1), s/2):
            for y in range(0, oldCollagen.shape[1] - (s-1), s/2):
                patch = numpy.array(oldCollagen[x:x+s, y:y+s])
                current = numpy.array(newCollagen[x:x+s, y:y+s])
                newCollagen[x:x+s, y:y+s] = current | removeBlops(patch)
        del oldCollagen
        del newCollagen
        success = True
    finally:
        if (not success):
            os.remove(newFile)

def removeBlops(patch):
    labelled, num = label(patch)
    counts = [0] * (num + 1)
    for x in range(patch.shape[0]):
        for y in range(patch.shape[1]):
            counts[labelled[x,y]] += 1

    for x in range(patch.shape[0]):
        for y in range(patch.shape[1]):
            if (counts[labelled[x,y]] < 100):
                patch[x,y] = 0
    return patch

def computeRawCollagenMask(slide, rawCollagenFile, model):
    if (os.path.isfile(rawCollagenFile)):
        print(rawCollagenFile + " already exists.")
        return

    success = False
    try:
        rawCollagen = memmap(rawCollagenFile, shape=slide.dimensions, dtype='uint8')
        for x in range(0, slide.dimensions[0] - (s-1), s/2):
            for y in range(0, slide.dimensions[1] - (s-1), s/2):
                patch = slide.read_region(location = (x, y), size = (s, s), level = 0)
                patch = numpy.array(patch)[:,:,:3]
                current = numpy.array(rawCollagen[x:x+s, y:y+s])
                # openslide uses (width, height) dimensions, tifffile.memmap (height, width).
                # Thus, I am transposing the extracted collagen.
                rawCollagen[x:x+s, y:y+s] = current | numpy.transpose(extractCollagen(patch, model))
        del rawCollagen
        success = True
    finally:
        if (not success):
            os.remove(rawCollagenFile)

def extractCollagen(patch, model):
    normalised = patch.astype(numpy.float) / 255
    w = model.predict(normalised.reshape(1, 256, 256, 3))
    threshold = 0.5
    w[w <= threshold] = 0
    w[w > threshold] = 1
    w = w.astype(numpy.uint8)
    w = w.reshape(256, 256)
    return w

def fillHoles(oldCollagenFile, newCollagenFile):
    if (os.path.isfile(newCollagenFile)):
        print(newCollagenFile + " already exists.")
        return
    success = False
    try:
        oldCollagen = memmap(oldCollagenFile, dtype='uint8')
        newCollagen = memmap(newCollagenFile, shape=oldCollagen.shape, dtype='uint8')
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        newCollagen[:,:] = cv2.morphologyEx(oldCollagen, cv2.MORPH_CLOSE, kernel)
        newCollagen[newCollagen == 1] = 255
        del oldCollagen
        del newCollagen
        success = True
    finally:
        if (not success):
            os.remove(newCollagenFile)

def computeSkeleton(collagenFile, skeletonFile):
    if (os.path.isfile(skeletonFile)): 
        print(skeletonFile + " already exists.")
        return
    success = False
    try:
        collagen = memmap(collagenFile, dtype='uint8')
        skeleton = memmap(skeletonFile, dtype='uint8', shape=collagen.shape)
        skeleton[:,:] = collagen
        del collagen
        skeleton[skeleton == 255] = 1
        skeleton[:,:] = skeletonize(skeleton)
        skeleton[skeleton == 1] = 255
        del skeleton
        success = True
    finally:
        if (not success):
            os.remove(skeletonFile)


def partitionSkeleton(skeletonFile, partitionFile):
    if (os.path.isfile(partitionFile)): 
        print(partitionFile + " already exists.")
        return
    success = False
    try:
        skeleton = memmap(skeletonFile)
        partition = memmap(partitionFile, shape=skeleton.shape, dtype='uint8')
        partition[:,:] = skeleton
        del skeleton
        pixels = numpy.nonzero(partition)
        for turn in [1,2,3]:
            for pixel in range(len(pixels[0])):
                n = countNeighbours(pixels[0][pixel], pixels[1][pixel], partition)
                if (n > 2 or n == 0):
                    partition[pixels[0][pixel], pixels[1][pixel]] = 0
        del partition
        success = True
    finally:
        if not success:
            os.remove(partitionFile)

def countNeighbours(x, y, image):
    count = 0
    for x0 in [-1,0,1]:
        for y0 in [-1,0,1]:
            if (x + x0 >= 0 and x + x0 < image.shape[0] and y + y0 >= 0 and y + y0 < image.shape[1]):
                if (x0 != 0 or y0 != 0):
                    if (image[x+x0, y+y0]):
                        count += 1
    return count;
        
    
def collagenAreaRatio(collagen):
    count = 0
    for x, y in numpy.ndindex(collagen.shape):
        if (collagen[x,y] == 1):
            count += 1
    return float(count) / (collagen.shape[0] * collagen.shape[1])

def collagenConnectivityRatio(collagen):
    sum = 0
    labelled, num = label(collagen)
    for i in range(1, num, 1):
        count = numpy.count_nonzero(labelled == i)
        sum += (count * count)
    totalArea = numpy.count_nonzero(collagen);
    if (totalArea == 0):
        return 0.0
    return float(sum) / (totalArea * totalArea)
