# pragma: no cover

# sngl3
#   load single image as 3 layers rgb
# snglv
#   load single image as a row vector
# mltpl3
#   load multiple images as 3 layers rgb
# mltplv
#   load mulitple images as an array where each row is an image

import glob

import numpy as np
from matplotlib import image


def sngl3(file):
    err1 = False
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        img = image.imread(file)
        if img is None:
            err1 = True
    else:
        err1 = True

    if err1:
        print(file + " is not an image file")
    else:
        return img


def snglv(file):
    err1 = False
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        img = image.imread(file)
        if img is None:
            err1 = True
    else:
        err1 = True

    if err1:
        print(file + " is not an image file")
    else:
        img = np.reshape(img, (1, -1))  # the unspecified value is inferred as ..
        return img


def mltpl3(folder):
    images = []
    for f in glob.glob(folder):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            img = image.imread(f)
            if img is not None:
                images.append(img)
    return images


def mltplv(folder):
    images = []
    for f in glob.glob(folder):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            img = image.imread(f)
            if img is not None:
                img = np.reshape(img, (1, -1))
                images.append(img)
    return images
