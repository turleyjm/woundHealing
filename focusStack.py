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


plt.rcParams.update({"font.size": 20})
plt.ioff()
pd.set_option("display.width", 1000)

filename = "HelenH2"
vid = np.zeros([60, 512, 512])
height = np.zeros([60, 512, 512])

stack = sm.io.imread("dat_nucleus/nucleusTracksH2.tif").astype(float)
variance = sm.io.imread("dat_nucleus/mirror_HelenH2.tif").astype(float)
variance = variance[:, 2:20]

for t in range(len(stack)):
    for x in range(512):
        for y in range(512):
            p = variance[t, :, x, y]
            m = p.max()
            h = [i for i, j in enumerate(p) if j == m][0]

            vid[t, x, y] = stack[t, h, x, y]
            height[t, x, y] = h

variance = np.asarray(variance, "uint16")
tifffile.imwrite(f"results/mitosis/variance_{filename}.tif", variance)

vid = np.asarray(vid, "uint8")
tifffile.imwrite(f"results/mitosis/focus_{filename}.tif", vid)

height = np.asarray(height, "uint8")
tifffile.imwrite(f"results/mitosis/height_{filename}.tif", height)
