import os
import shutil
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
scale = 147.91 / 512


def divisonsWeight(x, y, distance, scale):

    background = np.zeros([2512, 2512])
    r0 = int(scale * distance / 10) * 10
    r1 = int(scale * distance / 10) * 10 + 10
    rr1, cc1 = sm.draw.circle(x + 1000, y + 1000, r1 / scale)
    rr0, cc0 = sm.draw.circle(x + 1000, y + 1000, r0 / scale)
    background[rr1, cc1] = 1
    background[rr0, cc0] = 0

    Astar = sum(sum(background[1000:1512, 1000:1512]))
    A = sum(sum(background))

    weight = A / Astar

    return weight


f = open("pythonText.txt", "r")

fileType = f.read()
cwd = os.getcwd()
Fullfilenames = os.listdir(cwd + "/dat")
filenames = []
for filename in Fullfilenames:
    if fileType in filename:
        filenames.append(filename)

filenames.sort()

_df2 = []

for filename in filenames:

    dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
    dfDivisions = pd.read_pickle(f"dat/{filename}/mitosisTracks{filename}.pkl")
    df = dfDivisions[dfDivisions["Chain"] == "parent"]
    n = int(len(df))

    for i in range(n):

        label = df["Label"].iloc[i]
        ori = df["Division Orientation"].iloc[i]
        T = df["Time"].iloc[i]
        t = T[-1]

        [xw, yw] = dfWound["Centroid"].iloc[t]
        [x, y] = df["Position"].iloc[i][-1]
        distance = ((x - xw) ** 2 + (y - yw) ** 2) ** 0.5
        weight = divisonsWeight(x, y, distance, scale)
        distance = distance * scale

        _df2.append(
            {
                "Filename": filename,
                "Label": label,
                "Wound Orientation": ori,
                "Distance": distance,
                "Weight": weight,
                "T": t,
            }
        )

dfDivisions = pd.DataFrame(_df2)

time = dfDivisions["T"]
orientation = dfDivisions["Wound Orientation"]

fig = plt.figure(1, figsize=(9, 8))
plt.hist(time, 18, density=True)
plt.ylabel("Number of Divisons")
plt.xlabel("Time (mins)")
plt.title(f"Division time")
plt.ylim([0, 0.01])
fig.savefig(
    f"results/Division time after wounding {fileType}", dpi=300, transparent=True,
)
plt.close("all")

fig = plt.figure(1, figsize=(9, 8))
plt.hist(orientation, 9, density=True)
plt.ylabel("Number of Divisons")
plt.xlabel("Orientation")
plt.title(f"Division Orientation")
plt.ylim([0, 0.015])
fig.savefig(
    f"results/Division Orientation {fileType}", dpi=300, transparent=True,
)
plt.close("all")

weight = dfDivisions["Weight"][dfDivisions["Distance"] < 130]
distance = dfDivisions["Distance"][dfDivisions["Distance"] < 130]

# fig = plt.figure(1, figsize=(9, 8))
# plt.hist(distance, 13, density=True, weights=weight)
# plt.ylabel("Number of Divisons")
# plt.xlabel("Distance")
# plt.title(f"Division Density")
# plt.ylim([0, 0.016])
# fig.savefig(
#     f"results/Division Density {fileType}", dpi=300, transparent=True,
# )
# plt.close("all")
