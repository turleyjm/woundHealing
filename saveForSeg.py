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

    vid = sm.io.imread(f"dat/{filename}/probBoundary{filename}.tif").astype(float)

    video = []

    for j in range(12):
        for i in range(5):
            video.append(vid[j, i])

    for t in range(len(video)):
        if t > 99:
            T = f"{t}"
        elif t > 9:
            T = "0" + f"{t}"
        else:
            T = "00" + f"{t}"

        img = np.asarray(video[t] * 255, "uint8")
        tifffile.imwrite(
            f"dat/{filename}/imagesForSeg/probBoundary{filename}_{T}.tif", img
        )

