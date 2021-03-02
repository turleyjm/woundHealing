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

fileType = "Unwound"
filenames = cl.getFilesOfType(fileType)
scale = 147.91 / 512

_df2 = []

xw = 256
yw = 256

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
R = range(0, 80, 10)

for filename in filenames:

    # outPlane = sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype("uint8")
    outPlane = np.zeros([181, 512, 512])

    for t in range(T):
        for r in R:

            r0 = r
            area = np.pi * ((r0 + 10) ** 2 - r0 ** 2)
            weight = inPlaneWeight(256, 256, r0 / scale, (r0 + 10) / scale, outPlane[t])
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
T = range(0, 180, 20)
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

dfDensityUnwound = pd.DataFrame(_df2)

# -------------------

fileType = "WoundL"
filenames = cl.getFilesOfType(fileType)

filenames, fileType = cl.getFilesType()
scale = 147.91 / 512
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

        [xw, yw] = dfWound["Position"].iloc[t]
        [x, y] = df["Position"].iloc[i][-1]
        distance = ((x - xw) ** 2 + (y - yw) ** 2) ** 0.5
        distance = distance * scale

        _df2.append(
            {
                "Filename": filename,
                "Label": label,
                "Wound Orientation": ori,
                "Distance": distance,
                "T": t,
                "X": (x - xw) * scale,
                "Y": (y - yw) * scale,
            }
        )

dfDivisions = pd.DataFrame(_df2)

# -------------------

_df2 = []
T = 180
R = range(0, 80, 10)

for filename in filenames:

    dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
    outPlane = sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype("uint8")

    for t in range(T):
        for r in R:

            area = list(dfWound["Area"][dfWound["Time"] == t])[0]
            [x, y] = list(dfWound["Position"][dfWound["Time"] == t])[0]
            if area > 0:
                area = area * scale ** 2
                rw = (area / np.pi) ** 0.5
            else:
                rw = 0

            r0 = r + rw
            area = np.pi * ((r0 + 10) ** 2 - r0 ** 2)
            weight = inPlaneWeight(x, y, r0 / scale, (r0 + 10) / scale, outPlane[t])
            df = dfDivisions[dfDivisions["Distance"] > r0]
            df2 = df[df["Distance"] < r0 + 10]
            df3 = df2[df2["T"] == t]

            n = len(df3)
            if n == 0:
                ori = []
            else:
                ori = list(df3["Wound Orientation"])

            _df2.append(
                {
                    "Filename": filename,
                    "Wound Orientation": ori,
                    "T": t,
                    "R": r,
                    "Number": n,
                    "Area": area * weight,
                    "Weight": weight,
                }
            )

dfDensity_t = pd.DataFrame(_df2)


_df2 = []
T = range(0, 180, 20)
for t in T:
    for r in R:
        df = dfDensity_t[dfDensity_t["T"] >= t]
        df2 = df[df["T"] < t + 20]
        df3 = df2[df2["R"] >= r]
        df4 = df3[df3["R"] < r + 10]
        ori = []
        n = sum(df4["Number"])
        area = np.mean(df4["Area"])
        oriList = list(df4["Wound Orientation"])

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
                "Wound Orientation": ori,
                "Number": n,
                "Area": area,
                "Weight": weight,
            }
        )

dfDensityWoundL = pd.DataFrame(_df2)

# -------------------

fileType = "WoundS"
filenames = cl.getFilesOfType(fileType)

scale = 147.91 / 512
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

        [xw, yw] = dfWound["Position"].iloc[t]
        [x, y] = df["Position"].iloc[i][-1]
        distance = ((x - xw) ** 2 + (y - yw) ** 2) ** 0.5
        distance = distance * scale

        _df2.append(
            {
                "Filename": filename,
                "Label": label,
                "Wound Orientation": ori,
                "Distance": distance,
                "T": t,
                "X": (x - xw) * scale,
                "Y": (y - yw) * scale,
            }
        )

dfDivisions = pd.DataFrame(_df2)

# -------------------

_df2 = []
T = 180
R = range(0, 80, 10)

for filename in filenames:

    dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
    outPlane = sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype("uint8")

    for t in range(T):
        for r in R:

            area = list(dfWound["Area"][dfWound["Time"] == t])[0]
            [x, y] = list(dfWound["Position"][dfWound["Time"] == t])[0]
            if area > 0:
                area = area * scale ** 2
                rw = (area / np.pi) ** 0.5
            else:
                rw = 0

            r0 = r + rw
            area = np.pi * ((r0 + 10) ** 2 - r0 ** 2)
            weight = inPlaneWeight(x, y, r0 / scale, (r0 + 10) / scale, outPlane[t])
            df = dfDivisions[dfDivisions["Distance"] > r0]
            df2 = df[df["Distance"] < r0 + 10]
            df3 = df2[df2["T"] == t]

            n = len(df3)
            if n == 0:
                ori = []
            else:
                ori = list(df3["Wound Orientation"])

            _df2.append(
                {
                    "Filename": filename,
                    "Wound Orientation": ori,
                    "T": t,
                    "R": r,
                    "Number": n,
                    "Area": area * weight,
                    "Weight": weight,
                }
            )

dfDensity_t = pd.DataFrame(_df2)


_df2 = []
T = range(0, 180, 20)
for t in T:
    for r in R:
        df = dfDensity_t[dfDensity_t["T"] >= t]
        df2 = df[df["T"] < t + 20]
        df3 = df2[df2["R"] >= r]
        df4 = df3[df3["R"] < r + 10]
        ori = []
        n = sum(df4["Number"])
        area = np.mean(df4["Area"])
        oriList = list(df4["Wound Orientation"])

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
                "Wound Orientation": ori,
                "Number": n,
                "Area": area,
                "Weight": weight,
            }
        )

dfDensityWoundS = pd.DataFrame(_df2)


run = True
if run:
    density = []
    position = range(5, 85, 10)

    fig = plt.figure(1, figsize=(9, 8))
    for r in R:
        area = np.mean(dfDensityUnwound["Area"][dfDensityUnwound["R"] == r])
        n = np.mean(dfDensityUnwound["Number"][dfDensityUnwound["R"] == r])

        density.append(n / area)

    plt.plot(position, density, label="Unwound")

    density = []
    for r in R:
        area = np.mean(dfDensityWoundS["Area"][dfDensityWoundS["R"] == r])
        n = np.mean(dfDensityWoundS["Number"][dfDensityWoundS["R"] == r])

        density.append(n / area)

    plt.plot(position, density, label="WoundS")

    density = []
    for r in R:
        area = np.mean(dfDensityWoundL["Area"][dfDensityWoundL["R"] == r])
        n = np.mean(dfDensityWoundL["Number"][dfDensityWoundL["R"] == r])

        density.append(n / area)

    plt.plot(position, density, label="WoundL")

    plt.ylabel("Density of Divisons")
    plt.xlabel("Wound Distance")
    plt.title(f"Division Density")
    plt.legend()
    fig.savefig(
        f"results/Division Density {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")

    density = []
    time = range(10, 190, 20)

    fig = plt.figure(1, figsize=(9, 8))
    for t in T:
        area = np.mean(dfDensityUnwound["Area"][dfDensityUnwound["T"] == t])
        n = np.mean(dfDensityUnwound["Number"][dfDensityUnwound["T"] == t])
        density.append(n / area)

    plt.plot(time, density, label="Unwound")
    density = []
    for t in T:
        area = np.mean(dfDensityWoundS["Area"][dfDensityWoundS["T"] == t])
        n = np.mean(dfDensityWoundS["Number"][dfDensityWoundS["T"] == t])
        density.append(n / area)

    plt.plot(time, density, label="WoundS")

    density = []
    for t in T:
        area = np.mean(dfDensityWoundL["Area"][dfDensityWoundL["T"] == t])
        n = np.mean(dfDensityWoundL["Number"][dfDensityWoundL["T"] == t])
        density.append(n / area)

    plt.plot(time, density, label="WoundL")

    plt.ylabel("Density of Divisons")
    plt.xlabel("Time (mins)")
    plt.title(f"Division time")
    plt.legend()
    fig.savefig(
        f"results/Division time {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")
