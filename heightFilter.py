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

    d = 10

    if z0 > z - 5:
        scale = 1
    elif z > z0 + d:
        scale = 0
    else:
        scale = 1 - abs(z0 - z + 5) / (d - 5)

    return scale


def surface(p):

    n = len(p) - 4

    localMax = []
    for i in range(n):
        q = p[i : i + 5]
        localMax.append(max(q))

    Max = localMax[0]
    for i in range(n):
        if Max < localMax[i]:
            Max = localMax[i]
        elif Max < 250:
            continue
        else:
            return Max

    return Max


f = open("pythonText.txt", "r")

filenames = f.read()
filenames = filenames.split(", ")

background = 4

for filename in filenames:

    # Surface height

    varFile = f"dat/{filename}/varEcad{filename}.tif"
    variance = sm.io.imread(varFile).astype(int)

    (T, Z, X, Y) = variance.shape

    height = np.zeros([T, X, Y])

    for t in range(T):
        for x in range(X):
            for y in range(Y):
                p = variance[t, :, x, y]
                m = surface(p)
                h = [i for i, j in enumerate(p) if j == m][0]

                height[t, x, y] = h

    height = sp.ndimage.median_filter(height, size=9)
    height = np.asarray(height, "uint8")
    tifffile.imwrite(f"dat/{filename}/surface{filename}.tif", height)

    # Height filter

    vidFile = f"dat/{filename}/3dH2{filename}.tif"
    H2 = sm.io.imread(vidFile).astype(int)

    for t in range(T):
        print(t)
        for x in range(X):
            for y in range(Y):

                z0 = height[t, x, y]

                for z in range(Z):

                    scale = heightScale(z0, z)
                    H2[t, z, x, y] = H2[t, z, x, y] * scale

    H2 = np.asarray(H2, "uint8")
    tifffile.imwrite(f"dat/{filename}/heightH2{filename}.tif", H2)

    vidFile = f"dat/{filename}/3dEcad{filename}.tif"
    Ecad = sm.io.imread(vidFile).astype(int)

    for t in range(T):
        print(t)
        for x in range(X):
            for y in range(Y):

                z0 = height[t, x, y]

                for z in range(Z):

                    scale = heightScale(z0, z)
                    Ecad[t, z, x, y] = Ecad[t, z, x, y] * scale

    Ecad = np.asarray(Ecad, "uint8")
    tifffile.imwrite(f"dat/{filename}/heightEcad{filename}.tif", Ecad)

