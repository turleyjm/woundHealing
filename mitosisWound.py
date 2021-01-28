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


def divisonsWeight(x, y, distance, scale, outPlane):

    background = np.zeros([2512, 2512])
    r0 = int(scale * distance / 10) * 10
    r1 = int(scale * distance / 10) * 10 + 10
    rr1, cc1 = sm.draw.circle(x + 1000, y + 1000, r1 / scale)
    rr0, cc0 = sm.draw.circle(x + 1000, y + 1000, r0 / scale)
    background[rr1, cc1] = 1
    background[rr0, cc0] = 0

    background[1000:1512, 1000:1512][outPlane == 255] = 0

    Astar = sum(sum(background[1000:1512, 1000:1512])) * (scale ** 2)

    weight = 1 / Astar

    return weight


# -------------------

filenames, fileType = cl.getFilesType()
scale = 147.91 / 512
_df2 = []

for filename in filenames:

    dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
    dfDivisions = pd.read_pickle(f"dat/{filename}/mitosisTracks{filename}.pkl")
    outPlane = sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype("uint8")
    df = dfDivisions[dfDivisions["Chain"] == "parent"]
    n = int(len(df))

    for i in range(n):

        label = df["Label"].iloc[i]
        ori = df["Division Orientation"].iloc[i]
        T = df["Time"].iloc[i]
        t = T[-1]

        [xw, yw] = dfWound["Position"].iloc[t]
        [x, y] = df["Position"].iloc[i][-1]
        distance = ((x - xw) ** 2 + (y - yw) ** 2) ** 0.5
        weight = divisonsWeight(xw, yw, distance, scale, outPlane[t])
        distance = distance * scale

        _df2.append(
            {
                "Filename": filename,
                "Label": label,
                "Wound Orientation": ori,
                "Distance": distance,
                "Weight": weight,
                "T": t,
                "X": (x - xw) * scale,
                "Y": (y - yw) * scale,
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

# -------------------

density = []
number = []
position = np.linspace(0, 120, 13)
for pos in position:
    df = dfDivisions[dfDivisions["Distance"] > pos]
    weight = df["Weight"][df["Distance"] < pos + 10]
    density.append(sum(weight))
    number.append(len(weight))


fig = plt.figure(1, figsize=(9, 8))
plt.plot(position, density)
plt.ylabel("Density of Divisons")
plt.xlabel("Wound Distance")
plt.title(f"Division Density")
fig.savefig(
    f"results/Division Density {fileType}", dpi=300, transparent=True,
)
plt.close("all")


fig = plt.figure(1, figsize=(9, 8))
plt.plot(position, number)
plt.ylabel("Number of Divisons")
plt.xlabel("Wound Distance")
plt.title(f"Division number")
fig.savefig(
    f"results/Division number {fileType}", dpi=300, transparent=True,
)
plt.close("all")

x = dfDivisions["X"]
y = dfDivisions["Y"]

heatmap = np.histogram2d(x, y, range=[[-100, 100], [-100, 100]], bins=20)[0]
x, y = np.mgrid[-100:100:10, -100:100:10]

fig, ax = plt.subplots()
c = ax.pcolor(x, y, heatmap, cmap="Reds")
fig.colorbar(c, ax=ax)
fig.savefig(
    f"results/Division heatmap {fileType}", dpi=300, transparent=True,
)
plt.close("all")
