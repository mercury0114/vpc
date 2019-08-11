from extractor import *
from skeletonizer import *

outdir = "./../data/masks/smaller/"
rawCollagenFile = outdir + "raw.tiff"

print("Removing small blops")
collagenWithoutBlopsFile = outdir + "without_blops.tiff"
removeSmallCollagenBlops(rawCollagenFile, collagenWithoutBlopsFile)

print("Filling holes in collagen")
collagenHolesFilledFile = outdir + "holes_filled.tiff"
fillHoles(collagenWithoutBlopsFile, collagenHolesFilledFile)

print("Computing skeleton")
skeletonFile = outdir + "skeleton.tiff"
computeSkeleton(collagenHolesFilledFile, skeletonFile)

print("Partitioning skeleton")
partitionFile = outdir + "partition.tiff"
partitionSkeleton(skeletonFile, partitionFile)

print("Labelling partitioned skeleton")
labelledFile = outdir + "labels.tiff"
numberOfParts = labelSkeletonParts(partitionFile, labelledFile)
print("Labelled into " + str(numberOfParts) + " parts.")

print("Splitting collagen into fibers")
fibersFile = outdir + "fibers.tiff"
makeFibersFromSkeleton(collagenHolesFilledFile, labelledFile, fibersFile)


from tifffile import memmap
from fiber import extractOneFiber
from fiber import length
from fiber import width
from PIL import Image
p = memmap("./../data/masks/smaller/pvz.tiff")
fiber = extractOneFiber(p, 572, 350)
Image.fromarray(fiber).save("./../data/masks/smaller/temp.png")
