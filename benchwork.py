import csv
import openslide
import numpy as np
import pandas
import os
from extractor import *
from skeletonizer import *
from keras.models import load_model
from scipy.misc import imsave

centers = pandas.read_csv("centers.txt")
slide = openslide.open_slide("image.svs")
for i in range(centers.shape[0]):
	patch = slide.read_region(location = 
							  (centers.iloc[i,0] - 128, centers.iloc[i,1] - 128),
							  size = (256, 256), level = 0)
	patch = np.array(patch)[:,:,:3]
	imsave("patches/image" + str(i) + ".png", patch)

model = load_model("./model.h5")
with open("features.txt", "wb") as csvFile:
	writer = csv.writer(csvFile)
	for file in sorted(os.listdir("patches")):
		print('Computing features for ', file)
		collagen = extractCollagen("patches/" + file, model)
		imsave("collagen/" + file, collagen)
		collagen[collagen == 255] = 1
		skeleton = skeletonize(collagen)
		writer.writerow([averageWidth(skeleton, collagen), averageCurvature(skeleton)])
csvFile.close()

		
	



