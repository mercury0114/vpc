from extractor import *
from skeletonizer import *

import sys
sys.path.append("./../data")
from data import *
print(readGridIDs())
assert(False)

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
