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

plt.rcParams.update({"font.size": 16})

# -------------------


def inPlaneWeight(x, y, r0, r1, outPlane):

    background = np.zeros([2512, 2512])
    rr1, cc1 = sm.draw.circle(x + 1000, y + 1000, r1)
    rr0, cc0 = sm.draw.circle(x + 1000, y + 1000, r0)
    background[rr1, cc1] = 1
    background[rr0, cc0] = 0

    area = sum(sum(background))

    background[1000:1512, 1000:1512][outPlane == 255] = 0

    inPlane = sum(sum(background[1000:1512, 1000:1512]))

    weight = inPlane / area

    return weight


# -------------------

filenames, fileType = cl.getFilesType()
scale = 147.91 / 512

_df2 = []

xw = 256
yw = 256
Rbin = 5
Tbin = 10

run = False
if run:
    for filename in filenames:

        dfDivisions = pd.read_pickle(f"dat/{filename}/mitosisTracks{filename}.pkl")
        df = dfDivisions[dfDivisions["Chain"] == "parent"]
        n = int(len(df))

        for i in range(n):

            label = df["Label"].iloc[i]
            ori = df["Division Orientation"].iloc[i]
            if ori > 90:
                ori = 180 - ori
            T = df["Time"].iloc[i]
            t = T[-1]
            [x, y] = df["Position"].iloc[i][-1]

            distance = ((x - xw) ** 2 + (y - yw) ** 2) ** 0.5
            distance = distance * scale
            _df2.append(
                {
                    "Filename": filename,
                    "Label": label,
                    "Orientation": ori,
                    "Distance": distance,
                    "T": t,
                    "X": x,
                    "Y": y,
                }
            )

    dfDivisions = pd.DataFrame(_df2)

    _df2 = []
    T = 180
    R = range(0, 80, Rbin)

    for filename in filenames:

        # outPlane = sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype("uint8")
        outPlane = np.zeros([181, 512, 512])

        for t in range(T):
            for r in R:

                r0 = r
                area = np.pi * ((r0 + 10) ** 2 - r0 ** 2)
                weight = inPlaneWeight(
                    256, 256, r0 / scale, (r0 + 10) / scale, outPlane[t]
                )
                df = dfDivisions[dfDivisions["Distance"] > r0]
                df2 = df[df["Distance"] < r0 + 10]
                df3 = df2[df2["T"] == t]

                n = len(df3)
                if n == 0:
                    ori = []
                else:
                    ori = list(df3["Orientation"])

                _df2.append(
                    {
                        "Filename": filename,
                        "Orientation": ori,
                        "T": t,
                        "R": r,
                        "Number": n,
                        "Area": area * weight,
                        "Weight": weight,
                    }
                )

    dfDensity_t = pd.DataFrame(_df2)

    _df2 = []
    T = range(0, 180, Tbin)
    for t in T:
        for r in R:
            df = dfDensity_t[dfDensity_t["T"] >= t]
            df2 = df[df["T"] < t + 20]
            df3 = df2[df2["R"] >= r]
            df4 = df3[df3["R"] < r + 10]
            ori = []
            n = sum(df4["Number"])
            area = np.mean(df4["Area"])
            oriList = list(df4["Orientation"])

            for List in oriList:
                if List == []:
                    a = 0
                else:
                    for i in range(len(List)):
                        ori.append(List[i])

            _df2.append(
                {
                    "T": t,
                    "R": r,
                    "Orientation": ori,
                    "Number": n,
                    "Area": area,
                    "Weight": weight,
                }
            )

    dfDensity = pd.DataFrame(_df2)
    dfDensity.to_pickle(f"databases/dfDensity{fileType}.pkl")
else:
    dfDensity = pd.read_pickle(f"databases/dfDensity{fileType}.pkl")
    R = range(0, 80, Rbin)
    T = range(0, 180, Tbin)

# -------------------

run = False
if run:
    orientation = dfDivisions["Orientation"]

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

run = True
if run:
    density = []
    position = range(5, 85, Rbin)
    for r in R:
        area = np.mean(dfDensity["Area"][dfDensity["R"] == r])
        n = np.mean(dfDensity["Number"][dfDensity["R"] == r])

        density.append(n / area)

    fig = plt.figure(1, figsize=(9, 8))
    plt.plot(position, density)
    plt.ylabel("Density of Divisons")
    plt.xlabel("Wound Distance")
    plt.ylim([0, 0.5])
    plt.title(f"Division Density")
    fig.savefig(
        f"results/Division Density {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")

    density = []
    time = range(10, 190, Tbin)
    for t in T:
        area = np.mean(dfDensity["Area"][dfDensity["T"] == t])
        n = np.mean(dfDensity["Number"][dfDensity["T"] == t])
        density.append(n / area)

    fig = plt.figure(1, figsize=(9, 8))
    plt.plot(time, density)
    plt.ylabel("Density of Divisons")
    plt.xlabel("Time (mins)")
    plt.ylim([0, 0.6])
    plt.title(f"Division time")
    fig.savefig(
        f"results/Division time {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")


# -------------------

run = False
if run:
    x = dfDivisions["X"]
    y = dfDivisions["Y"]

    heatmap = np.histogram2d(x, y, range=[[0, 512], [0, 512]], bins=20)[0]
    x, y = np.mgrid[0 : 512 : 512 / 20, 0 : 512 : 512 / 20]

    fig, ax = plt.subplots()
    c = ax.pcolor(x, y, heatmap, cmap="Reds")
    fig.colorbar(c, ax=ax)
    fig.savefig(
        f"results/Division heatmap {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")

# -------------------

run = True
if run:
    heatmapDensity = np.zeros([len(T), len(R)])
    heatmapOrientation = np.zeros([len(T), len(R)])
    heatmapArea = np.zeros([len(T), len(R)])
    x = 0
    y = 0
    for t in T:
        for r in R:
            df = dfDensity[dfDensity["T"] == t]
            df2 = df[df["R"] == r]
            area = np.mean(df2["Area"].iloc[0])
            n = np.mean(df2["Number"].iloc[0])
            ori = df2["Orientation"].iloc[0]

            if ori == []:
                ori = np.nan
            else:
                for i in range(len(ori)):
                    theta = ori[i]
                    if theta > 0:
                        continue
                    ori[i] = 0

                if ori != []:
                    if 0 in ori:
                        ori.remove(0)
                    ori = np.mean(ori)

            heatmapDensity[x, y] = n / area
            heatmapOrientation[x, y] = ori
            heatmapArea[x, y] = area

            y += 1
        x += 1
        y = 0

    x, y = np.mgrid[0 : 180 + Tbin : Tbin, 0 : 80 + Rbin : Rbin]

    fig, ax = plt.subplots()
    c = ax.pcolor(x, y, heatmapDensity, cmap="Blues", vmin=0, vmax=0.8)
    plt.xlabel("Time (mins)")
    plt.ylabel(r"Distance from wound center $(\mu m)$")
    plt.title(f"Division Density")
    fig.colorbar(c, ax=ax)
    fig.savefig(
        f"results/Division Time Distance Heatmap {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")

    fig, ax = plt.subplots()
    c = ax.pcolor(x, y, heatmapOrientation, cmap="Blues", vmin=0, vmax=90)
    plt.xlabel("Time (mins)")
    plt.ylabel(r"Distance from wound center $(\mu m)$")
    plt.title(f"Division Orientation")
    fig.colorbar(c, ax=ax)
    fig.savefig(
        f"results/Division Orientation Heatmap {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")

    # fig, ax = plt.subplots()
    # c = ax.pcolor(x, y, heatmapArea, cmap="Blues")
    # plt.xlabel("Time (mins)")
    # plt.ylabel(r"Distance from wound center $(\mu m)$")
    # plt.title(f"Division Density")
    # fig.colorbar(c, ax=ax)
    # fig.savefig(
    #     f"results/Division Area Distance Heatmap {fileType}", dpi=300, transparent=True,
    # )
    # plt.close("all")

# -----------------------

# _df2 = []
# for i in range(10000):
#     x = int(random.uniform(0, 1) * 512)
#     y = int(random.uniform(0, 1) * 512)
#     distance = scale * ((x - xw) ** 2 + (y - yw) ** 2) ** 0.5

#     _df2.append({"X": x, "Y": y, "Distance": distance})

# df2 = pd.DataFrame(_df2)

# _df2 = []
# for r in R:

#     r0 = r
#     area = np.pi * ((r0 + 10) ** 2 - r0 ** 2)
#     weight = inPlaneWeight(256, 256, r0 / scale, (r0 + 10) / scale, outPlane[t])
#     df = df2[df2["Distance"] > r0]
#     df3 = df[df["Distance"] < r0 + 10]

#     n = len(df3)

#     _df2.append(
#         {"R": r, "Number": n, "Area": area * weight, "Weight": weight,}
#     )
# dfDensity = pd.DataFrame(_df2)

# if run:
#     density = []
#     position = range(5, 85, 10)
#     for r in R:
#         area = np.array(dfDensity["Area"][dfDensity["R"] == r])
#         n = np.array(dfDensity["Number"][dfDensity["R"] == r])

#         density.append(sum(n * area) / sum(area ** 2))

#     fig = plt.figure(1, figsize=(9, 8))
#     plt.plot(position, density)
#     plt.ylabel("Density of Divisons")
#     plt.xlabel("Wound Distance")
#     plt.title(f"Division Density")
#     fig.savefig(
#         f"results/Division Density {fileType}", dpi=300, transparent=True,
#     )
#     plt.close("all")
