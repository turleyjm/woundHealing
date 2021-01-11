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

plt.rcParams.update({"font.size": 24})
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

T = 181

R = [[] for col in range(T)]
for filename in filenames:

    dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")

    # area = dfWound["Area"].iloc[0] * (scale) ** 2
    # print(f"{filename} {area} {2*(area/np.pi)**0.5}")

    time = np.array(dfWound["Time"])
    area = np.array(dfWound["Area"]) * (scale) ** 2
    for t in range(T):
        if pd.isnull(area[t]):
            area[t] = 0

    for t in range(T):
        R[t].append(area[t])

    plt.plot(time, area)

plt.xlabel("Time")
plt.ylabel(r"Area ($\mu m ^2$)")
fig.savefig(
    f"results/Wound Radius {fileType}", dpi=300, transparent=True,
)
plt.close("all")

err = []
for t in range(T):
    err.append(cell.sd(R[t]) / (len(R[t]) ** 0.5))
    R[t] = cell.mean(R[t])

meanFinish = []
for filename in filenames:

    dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")

    area = np.array(dfWound["Area"]) * (scale) ** 2
    t = 0
    while pd.notnull(area[t]):
        t += 1

    meanFinish.append(t - 1)

meanFinish = cell.mean(meanFinish)

fig = plt.figure(1, figsize=(9, 8))
plt.errorbar(time, R, yerr=err)
plt.gcf().subplots_adjust(left=0.15)
plt.title(f"Mean finish time = {meanFinish}")
plt.suptitle("Wound Radius")
plt.xlabel("Time (mins)")
plt.ylabel(r"Area ($\mu m ^2$)")
fig.savefig(
    f"results/Wound Radius Mean {fileType}", dpi=300, transparent=True,
)
plt.close("all")

