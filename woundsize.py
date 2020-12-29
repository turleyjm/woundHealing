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

f = open("pythonText.txt", "r")

fileType = f.read()
cwd = os.getcwd()
Fullfilenames = os.listdir(cwd + "/dat")
filenames = []
for filename in Fullfilenames:
    if fileType in filename:
        filenames.append(filename)

filenames.sort()

scale = 147.91 / 512

fig = plt.figure(1, figsize=(9, 8))

for filename in filenames:

    dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")

    # area = dfWound["Area"].iloc[0] * (scale) ** 2
    # print(f"{filename} {area} {2*(area/np.pi)**0.5} {t}")

    time = np.array(dfWound["Time"])
    area = np.array(dfWound["Area"]) * (scale) ** 2
    radius = (area / np.pi) ** 0.5

    plt.plot(time, radius)

plt.xlabel("Time")
plt.ylabel(f"Radius")
fig.savefig(
    f"results/Wound Radius {fileType}", dpi=300, transparent=True,
)
plt.close("all")
