import os
import shutil
from math import dist, floor, log10

from collections import Counter
import cv2
import matplotlib
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from pandas.core import frame
from PIL import Image
import scipy as sp
import scipy.linalg as linalg
import scipy.ndimage as nd
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

label = 0

fileType = "training"
filenames = [
    "Unwound18h13",
    "Unwound18h14",
    "Unwound18h15",
    "Unwound18h16",
    "Unwound18h17",
]

if False:
    for filename in filenames:

        vidFocus = sm.io.imread(f"dat/{filename}/input3h{filename}.tif").astype(float)

        [T, X, Y, rgb] = vidFocus.shape

        dfDivisions = pd.read_excel(f"dat/{filename}/dfDivisionsEdit{filename}.xlsx")
        dfDivisions = dfDivisions.sort_values(["T", "X"], ascending=[True, True])

        for k in range(len(dfDivisions)):
            t = int(dfDivisions["T"].iloc[k])
            x = int(dfDivisions["X"].iloc[k])
            y = int(dfDivisions["Y"].iloc[k])
            ori = dfDivisions["Orientation"].iloc[k]

            xMax = int(x + 30)
            xMin = int(x - 30)
            yMax = int(y + 30)
            yMin = int(y - 30)
            if xMax > 512:
                xMaxCrop = 512 - xMax
                xMax = 512
            else:
                xMaxCrop = 0
            if xMin < 0:
                xMinCrop = xMin
                xMin = 0
            else:
                xMinCrop = 0
            if yMax > 512:
                yMaxCrop = 512 - yMax
                yMax = 512
            else:
                yMaxCrop = 0
            if yMin < 0:
                yMinCrop = yMin
                yMin = 0
            else:
                yMinCrop = 0

            frame = np.zeros([60, 60, 3])
            background = np.zeros([60, 60, 3])

            frame[-yMinCrop : 60 + yMaxCrop, -xMinCrop : 60 + xMaxCrop] = vidFocus[t][
                yMin:yMax, xMin:xMax
            ]
            rr1, cc1 = sm.draw.disk((30, 30), 30)
            background[rr1, cc1] = frame[rr1, cc1]
            division = background
            division = np.asarray(division, "uint8")
            tifffile.imwrite(
                f"trainOri/Division_{str(label).zfill(4)}.tif",
                division,
            )

            division = Image.open(f"trainOri/Division_{str(label).zfill(4)}.tif")

            division = division.resize((420, 420))

            rotation = division.rotate(-ori)

            rotation = np.asarray(rotation, "uint8")
            tifffile.imwrite(
                f"trainOri/Division_rotation_{str(label).zfill(4)}.tif",
                rotation,
            )

            label += 1

label += -1
if False:
    _meanFilter = []

    for i in range(label):

        rotation = sm.io.imread(
            f"trainOri/Division_rotation_{str(i).zfill(4)}.tif"
        ).astype(float)
        rotation = 255 * rotation / np.max(rotation)

        _meanFilter.append(rotation)

    meanFilter = np.mean(_meanFilter, axis=0)

    meanFilter[0:130] = 0
    meanFilter[290:] = 0
    meanFilter[:, 0:60] = 0
    meanFilter[:, 360:] = 0

    meanFilter = 255 * meanFilter / np.max(meanFilter)

    meanFilter = np.asarray(meanFilter, "uint8")
    tifffile.imwrite(
        f"divisionFilter.tif",
        meanFilter,
    )

# error rate
if True:
    divisionFilter = sm.io.imread(f"divisionFilter.tif").astype(float)
    _df = []
    label = 0
    for filename in filenames:
        dfDivisions = pd.read_excel(f"dat/{filename}/dfDivisionsEdit{filename}.xlsx")
        dfDivisions = dfDivisions.sort_values(["T", "X"], ascending=[True, True])
        for k in range(len(dfDivisions)):
            t = int(dfDivisions["T"].iloc[k])
            x = int(dfDivisions["X"].iloc[k])
            y = int(dfDivisions["Y"].iloc[k])
            ori = dfDivisions["Orientation"].iloc[k]
            _df.append(
                {
                    "Filename": filename,
                    "Label": label,
                    "T": t,
                    "X": x,
                    "Y": y,
                    "Ori": ori,
                }
            )
            label += 1

    df = pd.DataFrame(_df)

    error = []
    rr1, cc1 = sm.draw.disk((210, 210), 210)
    for j in range(len(df)):
        ori = df["Ori"].iloc[j]
        label = df["Label"].iloc[j]
        division = Image.open(f"trainOri/Division_{str(label).zfill(4)}.tif")

        cost = []
        for i in range(180):
            background = np.ones((420, 420))
            division = division.resize((420, 420))
            rotation = division.rotate(-i)
            rotation = np.asarray(rotation, "float")
            rotation = 255 * rotation / np.max(rotation)
            background[np.all(rotation == 0, axis=2)] = 0
            cost.append(
                np.sum(
                    (divisionFilter[background == 1] - rotation[background == 1]) ** 2
                )
            )
        minCost = np.min(cost)
        theta = cost.index(minCost)
        if abs(theta - ori) > 90:
            error.append(abs(180 - theta - ori))
        else:
            error.append(abs(theta - ori))

    fig, ax = plt.subplots()
    ax.hist(error, density=False, bins=18, range=[0, 90])
    ax.set_xlim([0, 90])
    # ax.set_ylim([0, 800])
    ax.set_xlabel("error", y=0.13)
    ax.axvline(np.median(error), c="k", label="mean")
    fig.savefig(
        f"results/orientationError.png",
        dpi=200,
        transparent=True,
    )
    plt.close("all")