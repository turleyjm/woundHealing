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

    File = f"dat/{filename}/{filename}.tif"
    vid = sm.io.imread(File).astype(int)

    (T, Z, C, X, Y) = vid.shape

    reverse = np.zeros([T, Z, C, X, Y])

    for t in range(T):
        for z in range(Z):
            for c in range(C):
                reverse[t, Z - z - 1, c] = vid[t, z, c]

    reverse = np.asarray(reverse, "uint8")
    tifffile.imwrite(f"dat/{filename}/{filename}.tif", reverse)
