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


filenames = cl.getFiles()
background = 0

for filename in filenames:

    print(filename)

    vidFile = f"dat/{filename}/bleachedH2{filename}.tif"

    vid = sm.io.imread(vidFile).astype(int)

    (T, X, Y) = vid.shape

    mu0 = 50

    for t in range(T):
        mu = vid[t, 50:462, 50:462][vid[t, 50:462, 50:462] > background]
        vid[t][vid[t] <= background] = 0

        mu = np.quantile(mu, 0.75)

        ratio = mu0 / mu

        vid[t] = vid[t] * ratio
        vid[t][vid[t] > 255] = 255

    vid = np.asarray(vid, "uint8")
    tifffile.imwrite(f"dat/{filename}/focusH2{filename}.tif", vid)

    # -------------------

    vidFile = f"dat/{filename}/bleachedEcad{filename}.tif"

    vid = sm.io.imread(vidFile).astype(int)

    (T, X, Y) = vid.shape

    mu0 = 40

    for t in range(T):
        mu = vid[t, 50:462, 50:462][vid[t, 50:462, 50:462] > background]
        vid[t][vid[t] <= background] = 0

        mu = np.mean(mu)

        ratio = mu0 / mu

        vid[t] = vid[t] * ratio
        vid[t][vid[t] > 255] = 255

    vid = np.asarray(vid, "uint8")
    tifffile.imwrite(f"dat/{filename}/focusEcad{filename}.tif", vid)

# -------------------

# vidFile = f"dat/{filename}/3dH2{filename}.tif"

# vid = sm.io.imread(vidFile).astype(int)

# (T, Z, X, Y) = vid.shape

# mu0 = 25

# for t in range(T):
#     mu = vid[t, 50:462, 50:462][vid[t, 50:462, 50:462] > background]
#     vid[t][vid[t] <= background] = 0

#     mu = np.mean(mu)

#     ratio = mu0 / mu

#     vid[t] = vid[t] * ratio
#     vid[t][vid[t] > 255] = 255

# vid = np.asarray(vid, "uint8")
# tifffile.imwrite(f"dat/{filename}/migration{filename}.tif", vid)
