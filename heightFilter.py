import os
from math import floor, log10

import cv2
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg as linalg
from scipy import ndimage
import shapely
import skimage as sm
import skimage.io
import skimage.measure
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
import tifffile
from skimage.draw import circle_perimeter

import cellProperties as cell
import findGoodCells as fi


def heightScale(z0, z):

    d = 13

    if z0 > z - 5:
        scale = 1
    elif z > z0 + d:
        scale = 0
    else:
        scale = 1 - abs(z0 - z + 5) / (d - 5)

    return scale


f = open("pythonText.txt", "r")

filenames = f.read()
filenames = filenames.split(", ")

for filename in filenames:

    vidFile = f"dat/{filename}/EcadHeight{filename}.tif"

    height = sm.io.imread(vidFile).astype(int)

    vidFile = f"dat/{filename}/3dFilter{filename}.tif"

    H2 = sm.io.imread(vidFile).astype(int)

    (T, Z, X, Y) = H2.shape

    for t in range(T):
        print(t)
        for x in range(X):
            for y in range(Y):

                z0 = height[t, x, y]

                for z in range(Z):

                    scale = heightScale(z0, z)
                    H2[t, z, x, y] = H2[t, z, x, y] * scale

    H2 = np.asarray(H2, "uint8")
    tifffile.imwrite(f"dat/{filename}/H2HeightScale{filename}.tif", H2)

