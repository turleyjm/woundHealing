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

plt.rcParams.update({"font.size": 16})

from math import sqrt
from skimage import data
from skimage.feature import blob, blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

import matplotlib.pyplot as plt


def sortConfusion(df, t, x, y, frameNum):

    a = df[df["T"] == t - frameNum]
    b = df[df["T"] == t]
    c = df[df["T"] == t + frameNum]

    df = pd.concat([a, b, c])

    xMax = x + 12
    xMin = x - 12
    yMax = y + 12
    yMin = y - 12
    if xMax > 511:
        xMax = 511
    if yMax > 511:
        yMax = 511
    if xMin < 0:
        xMin = 0
    if yMin < 0:
        yMin = 0

    dfxmin = df[df["X"] >= xMin]
    dfx = dfxmin[dfxmin["X"] <= xMax]

    dfymin = dfx[dfx["Y"] >= yMin]
    df = dfymin[dfymin["Y"] <= yMax]

    return df


fileType = "validation"
filenames = [
    "Unwound18h11",
    "Unwound18h12",
    "WoundL18h07",
    "WoundL18h08",
    "WoundL18h09",
    "WoundS18h10",
    "WoundS18h11",
    "WoundS18h12",
    "WoundS18h13",
]

# fileType = "training"
# filenames = [
#     "Unwound18h01",
#     "Unwound18h02",
#     "Unwound18h03",
#     "Unwound18h04",
#     "Unwound18h05",
#     "Unwound18h06",
#     "Unwound18h07",
#     "Unwound18h08",
#     "Unwound18h09",
#     "Unwound18h10",
#     "WoundL18h01",
#     "WoundL18h02",
#     "WoundL18h03",
#     "WoundL18h04",
#     "WoundL18h05",
#     "WoundL18h06",
#     "WoundS18h01",
#     "WoundS18h02",
#     "WoundS18h03",
#     "WoundS18h04",
#     "WoundS18h05",
#     "WoundS18h06",
#     "WoundS18h07",
#     "WoundS18h08",
#     "WoundS18h09",
# ]

label = 0
_meanFilter = []

if False:
    for filename in filenames:

        vidFocus = sm.io.imread(f"train/3h2f{filename}.tif").astype(float)

        [T, X, Y, rgb] = vidFocus.shape

        dfDivisions = pd.read_pickle(f"dat/{filename}/mitosisTracks{filename}.pkl")
        df = dfDivisions[dfDivisions["Chain"] == "parent"]
        df0 = dfDivisions[dfDivisions["Chain"] == "daughter0"]
        df1 = dfDivisions[dfDivisions["Chain"] == "daughter1"]
        _dfSpaceTime = []

        for i in range(len(df)):

            Label = df["Label"].iloc[i]
            t = df["Time"].iloc[i][-1]
            [x, y] = df["Position"].iloc[i][-1]
            ori = df["Division Orientation"].iloc[i]
            [x0, y0] = df0["Position"].iloc[i][0]
            [x1, y1] = df1["Position"].iloc[i][0]

            _dfSpaceTime.append(
                {
                    "Filename": filename,
                    "Label": Label,
                    "Orientation": ori,
                    "T": int(t / 2),
                    "X": x,
                    "Y": y,
                    "Daughter0": [x0, y0],
                    "Daughter1": [x1, y1],
                }
            )

        dfSpaceTime = pd.DataFrame(_dfSpaceTime)
        dfSpaceTime = dfSpaceTime[dfSpaceTime["T"] < 89]

        for k in range(len(dfSpaceTime)):
            t0 = int(dfSpaceTime["T"].iloc[k])
            x = int(dfSpaceTime["X"].iloc[k])
            y = int(dfSpaceTime["Y"].iloc[k])
            ori = dfSpaceTime["Orientation"].iloc[k]
            y = 512 - y

            xMax = int(x + 20)
            xMin = int(x - 20)
            yMax = int(y + 20)
            yMin = int(y - 20)
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

            frame = np.zeros([40, 40, 3])
            background = np.zeros([40, 40, 3])

            frame[-yMinCrop : 40 + yMaxCrop, -xMinCrop : 40 + xMaxCrop] = vidFocus[t0][
                yMin:yMax, xMin:xMax
            ]
            rr1, cc1 = sm.draw.disk((20, 20), 20)
            background[rr1, cc1] = frame[rr1, cc1]
            division = background
            division = np.asarray(division, "uint8")
            tifffile.imwrite(
                f"train/orientationTraining/Division_{str(label).zfill(4)}.tif",
                division,
            )

            division = Image.open(
                f"train/orientationTraining/Division_{str(label).zfill(4)}.tif"
            )

            division = division.resize((320, 320))

            rotation = division.rotate(-ori)

            rotation = np.asarray(rotation, "uint8")
            # tifffile.imwrite(
            #     f"train/orientationTraining/Division_{str(label).zfill(4)}rotation.tif",
            #     rotation,
            # )

            _meanFilter.append(rotation)

            label += 1

    meanFilter = np.mean(_meanFilter, axis=0)

    meanFilter[0:100] = 0
    meanFilter[220:320] = 0
    meanFilter[:, 0:35] = 0
    meanFilter[:, 285:320] = 0

    meanFilter = 255 * meanFilter / np.max(meanFilter)

    meanFilter = np.asarray(meanFilter, "uint8")
    tifffile.imwrite(
        f"meanFilter.tif",
        meanFilter,
    )


if True:
    _df = []
    for filename in filenames:

        vidFocus = sm.io.imread(f"train/3h2f{filename}.tif").astype(float)

        [T, X, Y, rgb] = vidFocus.shape

        dfDivisions = pd.read_pickle(f"dat/{filename}/mitosisTracks{filename}.pkl")
        df = dfDivisions[dfDivisions["Chain"] == "parent"]
        df0 = dfDivisions[dfDivisions["Chain"] == "daughter0"]
        df1 = dfDivisions[dfDivisions["Chain"] == "daughter1"]
        _dfSpaceTime = []

        for i in range(len(df)):

            Label = df["Label"].iloc[i]
            t = df["Time"].iloc[i][-1]
            [x, y] = df["Position"].iloc[i][-1]
            ori = df["Division Orientation"].iloc[i]
            [x0, y0] = df0["Position"].iloc[i][0]
            [x1, y1] = df1["Position"].iloc[i][0]

            _dfSpaceTime.append(
                {
                    "Filename": filename,
                    "Label": Label,
                    "Orientation": ori,
                    "T": int(t / 2),
                    "X": x,
                    "Y": y,
                    "Daughter0": [x0, y0],
                    "Daughter1": [x1, y1],
                }
            )

        dfSpaceTime = pd.DataFrame(_dfSpaceTime)
        dfSpaceTime = dfSpaceTime[dfSpaceTime["T"] < 89]

        for k in range(len(dfSpaceTime)):
            t0 = int(dfSpaceTime["T"].iloc[k])
            x = int(dfSpaceTime["X"].iloc[k])
            y = int(dfSpaceTime["Y"].iloc[k])
            y = 512 - y

            xMax = int(x + 20)
            xMin = int(x - 20)
            yMax = int(y + 20)
            yMin = int(y - 20)
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

            frame = np.zeros([40, 40, 3])
            background = np.zeros([40, 40, 3])

            frame[-yMinCrop : 40 + yMaxCrop, -xMinCrop : 40 + xMaxCrop] = vidFocus[t0][
                yMin:yMax, xMin:xMax
            ]

            rr1, cc1 = sm.draw.disk((20, 20), 20)
            background[rr1, cc1] = frame[rr1, cc1]
            division = background

            division = 255 * division / np.max(division)
            division = np.asarray(division, "uint8")
            tifffile.imwrite(
                f"train/orientationTraining/Division.tif",
                division,
            )

            division = Image.open(f"train/orientationTraining/Division.tif")

            division = division.resize((320, 320))

            meanDivision = sm.io.imread(f"meanFilter.tif").astype(float)

            cost = []
            for i in range(180):
                background = np.ones((320, 320))
                rotation = division.rotate(-i)
                rotation = np.asarray(rotation, "uint8")
                background[np.all(rotation == 0, axis=2)] = 0
                cost.append(
                    np.sum(
                        (meanDivision[background == 1] - rotation[background == 1]) ** 2
                    )
                )
                # tifffile.imwrite(
                #     f"train/orientationTraining/Rotation{i}.tif",
                #     rotation,
                # )

            rotation = np.zeros([320, 960, 3])

            minCost = np.min(cost)
            calOri = cost.index(minCost)
            ori = dfSpaceTime["Orientation"].iloc[k]
            rotation[:, :320] = np.asarray(division.rotate(-ori), "uint8")
            rotation[:, 320:640] = np.asarray(division, "uint8")
            rotation[:, 640:] = np.asarray(division.rotate(-calOri), "uint8")
            rotation = np.asarray(rotation, "uint8")
            tifffile.imwrite(
                f"train/orientationTraining/Rotation{label}.tif",
                rotation,
            )

            _df.append(
                {
                    "Label": label,
                    "T": int(dfSpaceTime["T"].iloc[k]),
                    "X": int(dfSpaceTime["X"].iloc[k]),
                    "Y": int(dfSpaceTime["Y"].iloc[k]),
                    "Filename": filename,
                    "Ori": ori,
                    "Cal Ori": calOri,
                }
            )
            label += 1

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/dfCalOrientation{fileType}.pkl")


if True:
    _dfConfusion = []
    dfnotDL = pd.read_pickle(f"databases/dfCalOrientation{fileType}.pkl")
    for filename in filenames:
        dfCal = dfnotDL[dfnotDL["Filename"] == filename]

        dfDivisions = pd.read_pickle(f"dat/{filename}/mitosisTracks{filename}.pkl")
        df = dfDivisions[dfDivisions["Chain"] == "parent"]
        _dfSpaceTime = []

        for i in range(len(df)):

            label = df["Label"].iloc[i]
            t = df["Time"].iloc[i][-1]
            [x, y] = df["Position"].iloc[i][-1]
            ori = df["Division Orientation"].iloc[i]

            _dfSpaceTime.append(
                {
                    "Filename": filename,
                    "Label": label,
                    "Orientation": ori,
                    "T": int(t / 2),
                    "X": x,
                    "Y": y,
                }
            )

        dfSpaceTime = pd.DataFrame(_dfSpaceTime)

        for i in range(len(dfSpaceTime)):
            t = int(dfSpaceTime["T"].iloc[i])
            x = dfSpaceTime["X"].iloc[i]
            y = dfSpaceTime["Y"].iloc[i]
            ori = np.pi * dfSpaceTime["Orientation"].iloc[i] / 180

            dfCon = sortConfusion(dfCal, t, x, y, 1)

            if len(dfCon) > 0:
                ori = np.pi * dfCon["Ori"].iloc[0] / 180
                calOri = np.pi * dfCon["Cal Ori"].iloc[0] / 180
                label = dfCon["Label"].iloc[0]
                error = 1 - np.dot(
                    np.array([np.cos(2 * ori), np.sin(2 * ori)]),
                    np.array([np.cos(2 * calOri), np.sin(2 * calOri)]),
                )
                dtheta = np.arccos(1 - error) / 2
                _dfConfusion.append(
                    {
                        "Filename": filename,
                        "T": int(t),
                        "X": x,
                        "Y": y,
                        "Orientation Error": error,
                        "Delta Theta": dtheta,
                        "Label": label,
                    }
                )

    dfConfusion = pd.DataFrame(_dfConfusion)
    dfConfusion.to_pickle(f"databases/dfConfusionOrientation{fileType}.pkl")

    err = abs(np.array(dfConfusion["Delta Theta"])) * 180 / np.pi

    fig, ax = plt.subplots()
    ax.hist(err, density=False, bins=18, range=(0, 90))
    ax.set_xlim([0, 90])
    ax.set_ylim([0, 800])
    ax.set_xlabel("error", y=0.13)
    ax.axvline(np.median(err), c="k", label="mean")
    fig.savefig(
        f"results/orientationError{fileType}.png",
        dpi=200,
        transparent=True,
    )
    plt.close("all")

    df = dfConfusion[dfConfusion["Delta Theta"] > np.pi / 9]
