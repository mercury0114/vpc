from tifffile import memmap
import numpy
import os

def rotate(degrees, id):
    mask = memmap("./../data/masksBalanced/" + id)
    rot = memmap("./../data/masksBalanced/" + id + "_" + str(degrees), dtype='uint8', shape=mask.shape)
    rot[:,:] = mask
    while (degrees > 0):
        rot[:,:] = numpy.rot90(rot)
        degrees -= 90
    del(mask)
    del(rot)

fread = open("./../data/survival1500.txt")
fwrite = open("./../data/survivalBalanced.txt", mode="w")
fwrite.write(next(fread))

for line in fread:
    split = line.split(",")
    id = split[1]
    print(id)
    status = int(split[3])
    os.system("cp ./../data/masks1500/" + id + "/without_blops.tiff ./../data/masksBalanced/" + id)
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

