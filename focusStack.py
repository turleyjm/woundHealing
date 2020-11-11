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


f = open("pythonText.txt", "r")

filenames = f.read()
filenames = filenames.split(", ")

for filename in filenames:

    vidFile = f"dat/{filename}/Ecad3d{filename}.tif"
    varFile = f"dat/{filename}/EcadVariance{filename}.tif"

    stack = sm.io.imread(vidFile).astype(int)
    variance = sm.io.imread(varFile).astype(int)

    (T, Z, X, Y) = stack.shape

    height = np.zeros([60, 512, 512])
    vid = np.zeros([60, 512, 512])

    for t in range(T):
        for x in range(X):
            for y in range(Y):
                p = variance[t, :, x, y]
                m = p.max()
                h = [i for i, j in enumerate(p) if j == m][0]

                vid[t, x, y] = stack[t, h, x, y]
                height[t, x, y] = h

    height = sp.ndimage.median_filter(height, size=5)

    height = np.asarray(height, "uint8")
    tifffile.imwrite(f"dat/{filename}/EcadHeight{filename}.tif", height)
