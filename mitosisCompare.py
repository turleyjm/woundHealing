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

scale = 147.91 / 512
fileTypes = ["Unwound", "WoundS", "WoundL"]
filenamesUnwound = cl.getFilesOfType(fileTypes[1])
_df2 = []

xw = 256
yw = 256
outPlane = np.zeros([181, 512, 512])

for filename in filenamesUnwound:

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
        weight = divisonsWeight(xw, yw, distance, scale, outPlane[t])
        distance = distance * scale
        _df2.append(
            {
                "FileType": fileTypes[0],
                "Filename": filename,
                "Label": label,
                "Orientation": ori,
                "T": t,
                "X": x,
                "Y": y,
                "Distance": distance,
                "Weight": weight,
            }
        )


# -------------------

filenamesWoundS = cl.getFilesOfType(fileTypes[1])

for filename in filenamesWoundS:

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
                "FileType": fileTypes[1],
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

# -------------------

filenamesWoundL = cl.getFilesOfType(fileTypes[2])

for filename in filenamesWoundL:

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
                "FileType": fileTypes[2],
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

# -------------------

run = True
if run:
    fig = plt.figure(1, figsize=(9, 8))
    for fileType in fileTypes:
        dfDivisionsType = dfDivisions[dfDivisions["FileType"] == fileType]

        density = []
        position = np.linspace(0, 120, 13)
        for pos in position:
            df = dfDivisionsType[dfDivisionsType["Distance"] > pos]
            weight = list(df["Weight"][df["Distance"] < pos + 10])
            if 0 in weight:
                weight.remove(0)
            density.append(sum(weight))

        plt.plot(position, density)

    plt.legend((fileTypes[0], fileTypes[1], fileTypes[2]), loc="upper right")
    plt.ylabel(r"Density of Divisons $(\mu m)^{-2}$")
    plt.xlabel(r"Distance from wound center $(\mu m)$")
    plt.title(f"Division Density by Distance")
    fig.savefig(
        f"results/Division Distance compare", dpi=300, transparent=True,
    )
    plt.close("all")

    fig = plt.figure(1, figsize=(9, 8))
    for fileType in fileTypes:
        dfDivisionsType = dfDivisions[dfDivisions["FileType"] == fileType]
        density = []
        time = np.linspace(0, 170, 9)
        for t in time:
            df = dfDivisionsType[dfDivisionsType["T"] > t]
            weight = list(df["Weight"][df["T"] < t + 20])
            if 0 in weight:
                weight.remove(0)
            density.append(sum(weight))

        plt.plot(time, density)

    plt.legend((fileTypes[0], fileTypes[1], fileTypes[2]), loc="upper right")
    plt.ylabel(r"Density of Divisons $(\mu m)^{-2}$")
    plt.xlabel("Time (mins)")
    plt.title(f"Division time unwounded")
    fig.savefig(
        f"results/Division time compare", dpi=300, transparent=True,
    )
    plt.close("all")
