from tifffile import memmap
import numpy
import os
from PIL import Image

def rotate(degrees, id):
    image = Image.open("./../data/1500/" + id + ".png")
    rot = image.rotate(degrees)
    name = "./../data/1500Balanced/" + id + "_" + str(degrees)
    rot.save(name + ".png")
    os.system("mv " + name + ".png " + name)

fread = open("./../data/survival1500.txt")
fwrite = open("./../data/survivalBalanced.txt", mode="w")
fwrite.write(next(fread))

for line in fread:
    split = line.split(",")
    id = split[1]
    print(id)
    status = int(split[3])
    os.system("cp ./../data/1500/" + id + ".png ./../data/1500Balanced/" + id)
    fwrite.write(line)
    if (status == 1):
        split[1] = id + "_90"
        fwrite.write(",".join(split))
        rotate(90, id)

        split[1] = id + "_180"
        fwrite.write(",".join(split))
        rotate(180, id)
        
        split[1] = id + "_270"
        fwrite.write(",".join(split))
        rotate(270, id)

