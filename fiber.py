from tifffile import memmap
from scipy.ndimage import label
from scipy.misc import imsave

class Fiber:
	def __init__(self, maskFilePath, xFrom, xTo, yFrom, yTo, x, y):
		self.maskFilePath = maskFilePath
		self.xFrom = xFrom
		self.xTo = xTo
		self.yFrom = yFrom
		self.yTo = yTo
		self.x = x
		self.y = y

	def drawFiber(self, outputFilePath):
		mask = memmap(self.maskFilePath, dtype='uint8')
		box = mask[self.xFrom : self.xTo, self.yFrom : self.yTo]
		labelled, num = label(box)
		v = labelled[self.x - self.xFrom, self.y - self.yFrom]
		box[labelled == v] = 255
		box[labelled != v] = 0
		imsave(outputFilePath, box)
