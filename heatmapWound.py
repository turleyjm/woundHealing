import os
import shutil
from math import floor, log10

from collections import Counter
import cv2
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

scale = 147.91 / 512
grid = 32
T = 181

filenames, fileType = cl.getFilesType()

_dfCells = []


for filename in filenames:

    df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")

    for i in range(len(df)):

        t = df["Time"].iloc[i]
        polygon = df["Polygon"].iloc[i]
        x = df["RWCo x"].iloc[i] * scale
        y = df["RWCo y"].iloc[i] * scale
        orientaion = df["Orientation"].iloc[i]
        sf = df["Shape Factor"].iloc[i]

        _dfCells.append(
            {
                "Time": t,
                "Polygon": polygon,
                "X": x,
                "Y": y,
                "R": (x ** 2 + y ** 2) ** 0.5,
                "Orientation": orientaion,
                "Shape Factor": sf,
            }
        )


dfCells = pd.DataFrame(_dfCells)

heatmaps = np.zeros([T, grid, grid])

cl.createFolder("results/video/")
for t in range(40):

    heatmap = np.zeros([grid, grid])

    for i in range(grid):
        for j in range(grid):
            x = [160 / grid * j - 80, 160 / grid * j - 70]
            y = [160 / grid * i - 80, 160 / grid * i - 70]
            dfxy = cl.sortGrid(dfCells[dfCells["Time"] == t], x, y)
            a = list(dfxy["Shape Factor"])
            if a == []:
                heatmap[i][j] = 0
                heatmaps[t][i][j] = 0
            else:
                heatmap[i][j] = np.mean(a)
                heatmaps[t][i][j] = np.mean(a)

for t in range(40):
    heatmap[heatmap == 0] = np.mean(heatmap)
    heatmap = heatmap - np.mean(heatmap)

    # generate 2 2d grids for the x & y bounds
    dx, dy = 160 / grid, 160 / grid
    y, x = np.mgrid[-80:80:dy, -80:80:dx]
    z_min, z_max = heatmap.min(), heatmap.max()
    midpoint = 1 - z_max / (z_max + abs(z_min))
    orig_cmap = matplotlib.cm.seismic
    shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")

    fig, ax = plt.subplots()
    c = ax.pcolor(x, y, heatmap, cmap=shifted_cmap, vmin=z_min, vmax=z_max)
    fig.colorbar(c, ax=ax)
    fig.savefig(
        f"results/video/heatmap shape factor {t}", dpi=300, transparent=True,
    )
    plt.close("all")

heatmaps = np.asarray(heatmaps, "float")
tifffile.imwrite(f"results/heatmapShapeFactor{fileType}.tif", heatmaps)

# make video
img_array = []

for t in range(T):
    img = cv2.imread(f"results/video/heatmap shape factor {t}.png")
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)


out = cv2.VideoWriter(
    f"results/heatmap shape factor {fileType}.mp4",
    cv2.VideoWriter_fourcc(*"DIVX"),
    10,
    size,
)
for i in range(len(img_array)):
    out.write(img_array[i])

out.release()
cv2.destroyAllWindows()

shutil.rmtree("results/video")


# kymograph

heatmap = np.zeros([int(T / 5), grid])

for i in range(0, 180, 5):
    for j in range(grid):
        r = [40 / grid * j, 40 / grid * j + 5]
        t = [i, i + 5]
        dfr = cl.sortRadius(dfCells, t, r)
        if list(dfr["Shape Factor"]) == []:
            sf = 0
        else:
            sf = np.mean(list(dfr["Shape Factor"]))

        heatmap[int(i / 5), j] = sf

heatmap[heatmap == 0] = np.mean(heatmap)
heatmap = heatmap - np.mean(heatmap)

dt, dr = 5, 40 / grid
t, r = np.mgrid[0:180:dt, 0:40:dr]
z_min, z_max = heatmap.min(), heatmap.max()
midpoint = 1 - z_max / (z_max + abs(z_min))
orig_cmap = matplotlib.cm.seismic
shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")
fig, ax = plt.subplots()
c = ax.pcolor(t, r, heatmap, cmap=shifted_cmap, vmin=z_min, vmax=z_max)
fig.colorbar(c, ax=ax)
fig.savefig(
    f"results/shape factor kymograph {fileType}", dpi=300, transparent=True,
)
plt.close("all")
