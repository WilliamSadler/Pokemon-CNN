from tkinter import filedialog
from tkinter import *
import os

from PIL import Image, ImageOps

IMAGE_SIZE = 50,50

root = Tk()
root.withdraw()
folder_selected = filedialog.askdirectory()

images = []

#Find all images in selected folder
for f in os.listdir(folder_selected):

    suffix = os.path.splitext(f)[1]

    # check if the image ends with png
    if (suffix in [".png", ".jpg"]):
        images.append(f)

#Create temp folder to store downscaled images

greyscale_flat = []

for i in images:
    im = Image.open(folder_selected + "/" + i)
    greyscale = ImageOps.grayscale(im)
    greyscale_resized = greyscale.resize(IMAGE_SIZE, Image.ANTIALIAS)
    #greyscale_resized.save(folder_selected + "/downscaled/" + i, dpi=(50,50))

    pix = greyscale_resized.load()
    pixels = []
    for x in range(0, 50):
        for y in range(0, 50):
            pixels.append(pix[x, y]/255)


    greyscale_flat.append(pixels)


import numpy
numpy_array = numpy.asarray(greyscale_flat)

numpy.savetxt(folder_selected + "/output.csv", numpy_array, delimiter=",")

print("done")