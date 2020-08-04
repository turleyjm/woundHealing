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


folder = "dat/datProbBoundary"

cwd = os.getcwd()

files = os.listdir(cwd + f"/{folder}")

for vidFile in files:

    filename = vidFile

    filename = filename.replace("probBoundary", "")
    filename = filename.replace(".tif", "")

    vid = sm.io.imread(folder + "/" + vidFile).astype(float)

    for t in range(len(vid)):
        if t > 99:
            T = f"{t}"
        elif t > 9:
            T = "0" + f"{t}"
        else:
            T = "00" + f"{t}"

        img = np.asarray(vid[t] * 255, "uint8")
        tifffile.imwrite(f"dat/imagesForSeg/probBoundary{filename}_{T}.tif", img)

