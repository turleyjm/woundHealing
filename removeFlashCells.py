import os
from math import floor, log10

import cv2
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

import cellProperties as cell
import findGoodCells as fi

plt.rcParams.update({"font.size": 20})
plt.ioff()
pd.set_option("display.width", 1000)


f = open("pythonText.txt", "r")

filenames = f.read()
filenames = filenames.split(", ")

for filename in filenames:

    if "Unwound" in filename:
        wound = False
    else:
        wound = True

    vidFile = f"dat/{filename}/track{filename}.tif"

    vid = sm.io.imread(vidFile).astype(int)

    img = vid[2]

    label = img[127, 187]

    img[img == label] = 0

    vid[2] = img
    vid = np.asarray(vid, "uint8")
    tifffile.imwrite(f"dat/{filename}/removeLabel{filename}.tif", vid)
