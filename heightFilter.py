import os
import shutil
from math import floor, log10

from collections import Counter
import cv2
import matplotlib
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy as sp
import scipy.linalg as linalg
import shapely
import skimage as sm
import skimage.io
import skimage.measure
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
import tifffile
from skimage.draw import circle_perimeter
from scipy import optimize
import xml.etree.ElementTree as et

import cellProperties as cell
import findGoodCells as fi
import commonLiberty as cl

plt.rcParams.update({"font.size": 20})

# -------------------


def heightScale(z0, z):

    # e where scaling starts from the surface and d is the cut off
    d = 10
    e = 8

    if z0 + e > z:
        scale = 1
    elif z > z0 + d:
        scale = 0
    else:
        scale = 1 - abs(z - z0 - e) / (d - e)

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


# -------------------

filenames, fileType = cl.getFilesType()

background = 4

for filename in filenames:

    # Surface height
    print(filename)

    varFile = f"dat/{filename}/varEcad{filename}.tif"
    variance = sm.io.imread(varFile).astype(int)

    (T, Z, X, Y) = variance.shape

    height = np.zeros([T, X, Y])

    for t in range(T):
        for x in range(X):
            for y in range(Y):
                p = variance[t, :, x, y]

                # p = list(p)
                # p.reverse()
                # p = np.array(p)

                m = surface(p)
                h = [i for i, j in enumerate(p) if j == m][0]

                height[t, x, y] = h

    height = sp.ndimage.median_filter(height, size=9)
    height = np.asarray(height, "uint8")
    tifffile.imwrite(f"dat/{filename}/surface{filename}.tif", height)

    # Height filter

    vidFile = f"dat/{filename}/3dH2{filename}.tif"
    H2 = sm.io.imread(vidFile).astype(int)

    for z in range(Z):
        for z0 in range(Z):
            scale = heightScale(z0, z)
            H2[:, z][height == z0] = H2[:, z][height == z0] * scale

    H2 = np.asarray(H2, "uint8")
    tifffile.imwrite(f"dat/{filename}/heightH2{filename}.tif", H2)

    vidFile = f"dat/{filename}/3dEcad{filename}.tif"
    Ecad = sm.io.imread(vidFile).astype(int)

    for z in range(Z):
        for z0 in range(Z):
            scale = heightScale(z0, z)
            Ecad[:, z][height == z0] = Ecad[:, z][height == z0] * scale

    Ecad = np.asarray(Ecad, "uint8")
    tifffile.imwrite(f"dat/{filename}/heightEcad{filename}.tif", Ecad)

