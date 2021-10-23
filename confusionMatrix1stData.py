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


# -------------------


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


def findT(frameRate):
    if frameRate == "1f":
        T = 179
        frameNum = 1
    elif frameRate == "2f":
        T = 89
        frameNum = 2
    else:
        T = 59
        frameNum = 3
    return T, frameNum


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

# fileType = "test"
# filenames = ["Unwound18h11"]

# "1f", "2f", "3f"
frameRates = ["1f", "2f", "3f"]

# first stage Confusion Matrix
if False:
    print("")
    print("frameRates", "True Pos", "False Pos", "False Neg")
    for frameRate in frameRates:
        T, frameNum = findT(frameRate)
        _dfConfusion = []
        for filename in filenames:
            dfDL = pd.read_pickle(
                f"databases/firstStageDatabases/dfDivisionDL{frameRate}{filename}.pkl"
            )
            dfDivisions = pd.read_pickle(
                f"dat010921/{filename}/mitosisTracks{filename}.pkl"
            )
            df = dfDivisions[dfDivisions["Chain"] == "parent"]
            df0 = dfDivisions[dfDivisions["Chain"] == "daughter0"]
            df1 = dfDivisions[dfDivisions["Chain"] == "daughter1"]
            _dfSpaceTime = []

            for i in range(len(df)):

                label = df["Label"].iloc[i]
                t = df["Time"].iloc[i][-1]
                [x, y] = df["Position"].iloc[i][-1]
                ori = df["Division Orientation"].iloc[i]
                [x0, y0] = df0["Position"].iloc[i][0]
                [x1, y1] = df1["Position"].iloc[i][0]

                _dfSpaceTime.append(
                    {
                        "Filename": filename,
                        "Label": label,
                        "Orientation": ori,
                        "T": int(t / frameNum),
                        "X": x,
                        "Y": y,
                        "Daughter0": [x0, y0],
                        "Daughter1": [x1, y1],
                    }
                )

            dfSpaceTime = pd.DataFrame(_dfSpaceTime)

            for i in range(len(dfSpaceTime)):
                label = dfSpaceTime["Label"].iloc[i]
                ti = int(dfSpaceTime["T"].iloc[i])
                xi = int(dfSpaceTime["X"].iloc[i])
                yi = int(dfSpaceTime["Y"].iloc[i])
                ori = dfSpaceTime["Orientation"].iloc[i]

                dfCon = sortConfusion(dfDL, ti, xi, yi, 1)

                if len(dfCon) > 0:
                    label_DL = dfCon["Label"].iloc[0]
                    t = dfCon["T"].iloc[0]
                    x = dfCon["X"].iloc[0]
                    y = dfCon["Y"].iloc[0]
                    _dfConfusion.append(
                        {
                            "Filename": filename,
                            "Label": label,
                            "Label DL": label_DL,
                            "T": int(t),
                            "X": x,
                            "Y": y,
                            "Orientation": ori,
                        }
                    )
                elif len(dfCon) == 0:
                    _dfConfusion.append(
                        {
                            "Filename": filename,
                            "Label": label,
                            "T": int(ti),
                            "X": xi,
                            "Y": yi,
                            "Orientation": ori,
                        }
                    )

            uniqueLabel = list(set(dfDL["Label"]))
            dfConfusion_ = pd.DataFrame(_dfConfusion)
            dfConfusion_ = dfConfusion_[dfConfusion_["Filename"] == filename]
            linkedLabels = np.sort(dfConfusion_["Label DL"])

            for label_DL in uniqueLabel:
                if label_DL not in linkedLabels:
                    t = dfDL["T"][dfDL["Label"] == label_DL].iloc[0]
                    x = dfDL["X"][dfDL["Label"] == label_DL].iloc[0]
                    y = dfDL["Y"][dfDL["Label"] == label_DL].iloc[0]
                    _dfConfusion.append(
                        {
                            "Filename": filename,
                            "Label DL": label_DL,
                            "T": int(t),
                            "X": x,
                            "Y": y,
                        }
                    )

        dfConfusion = pd.DataFrame(_dfConfusion)
        dfConfusion.to_pickle(f"databases/dfConfusionFS{frameRate}{fileType}.pkl")
        falseNeg = len(dfConfusion[dfConfusion["Label DL"].isnull()])
        falsePos = len(dfConfusion[dfConfusion["Label"].isnull()])
        print(
            frameRate + " First step",
            len(dfConfusion) - falsePos - falseNeg,
            falsePos,
            falseNeg,
        )

# first stage
if False:
    for frameRate in frameRates:
        dfConfusion = pd.read_pickle(
            f"databases/dfConfusionFS{frameRate}{fileType}.pkl"
        )

        for filename in filenames:

            T, frameNum = findT(frameRate)

            vidFocus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(float)
            vidFocus = vidFocus[::frameNum]

            [T, X, Y, rgb] = vidFocus.shape

            dfConfusionf = dfConfusion[dfConfusion["Filename"] == filename]
            dfFalsePos = dfConfusionf[dfConfusionf["Label"].isnull()]
            dfTrue = dfConfusionf[dfConfusionf["Label"].notnull()]
            dfTruePos = dfTrue[dfTrue["Label DL"].notnull()]

            for k in range(len(dfFalsePos)):
                t0 = int(dfFalsePos["T"].iloc[k])
                x = dfFalsePos["X"].iloc[k]
                y = dfFalsePos["Y"].iloc[k]
                label_DL = dfFalsePos["Label DL"].iloc[k]
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

                frames9 = np.zeros([126, 126, 3])

                times = range(t0 - 4, t0 + 5)

                timeVid = []
                for t in times:
                    if t >= 0 and t <= T - 1:
                        timeVid.append(t)

                if timeVid[0] == 0:
                    start = 4 - t0
                else:
                    start = 0

                j = start // 3
                i = start - 3 * j
                for t in timeVid:
                    frames9[
                        43 * j - yMinCrop : 43 * j + 40 + yMaxCrop,
                        43 * i - xMinCrop : 43 * i + 40 + xMaxCrop,
                    ] = vidFocus[t][yMin:yMax, xMin:xMax]
                    i += 1
                    if i == 3:
                        i = 0
                        j += 1

                frames9 = np.asarray(frames9, "uint8")
                tifffile.imwrite(
                    f"train/frame9Training/{frameRate}/division{filename}_{int(label_DL)}.tif",
                    frames9,
                )

            for k in range(len(dfTruePos)):
                t0 = int(dfTruePos["T"].iloc[k])
                x = int(dfTruePos["X"].iloc[k])
                y = int(dfTruePos["Y"].iloc[k])
                label_DL = dfTruePos["Label DL"].iloc[k]
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

                frames9 = np.zeros([126, 126, 3])

                times = range(t0 - 4, t0 + 5)

                timeVid = []
                for t in times:
                    if t >= 0 and t <= T - 1:
                        timeVid.append(t)

                if timeVid[0] == 0:
                    start = 4 - t0
                else:
                    start = 0

                j = start // 3
                i = start - 3 * j
                for t in timeVid:
                    frames9[
                        43 * j - yMinCrop : 43 * j + 40 + yMaxCrop,
                        43 * i - xMinCrop : 43 * i + 40 + xMaxCrop,
                    ] = vidFocus[t][yMin:yMax, xMin:xMax]
                    i += 1
                    if i == 3:
                        i = 0
                        j += 1

                frames9 = np.asarray(frames9, "uint8")
                tifffile.imwrite(
                    f"train/frame9Training/{frameRate}/Division{filename}_{int(label_DL)}.tif",
                    frames9,
                )


# orientation training set
if False:
    for frameRate in frameRates:
        dfConfusion = pd.read_pickle(f"databases/dfConfusionFS1f{fileType}.pkl")
        label = 0

        for filename in filenames:

            T, frameNum = findT(frameRate)

            vidFocus = sm.io.imread(f"train/3h1f{filename}.tif").astype(float)

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
                        "T": int(t),
                        "X": x,
                        "Y": y,
                        "Daughter0": [x0, y0],
                        "Daughter1": [x1, y1],
                    }
                )

            dfSpaceTime = pd.DataFrame(_dfSpaceTime)
            dfSpaceTime = dfSpaceTime[dfSpaceTime["T"] < 179]

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

                frame[-yMinCrop : 40 + yMaxCrop, -xMinCrop : 40 + xMaxCrop] = vidFocus[
                    t0
                ][yMin:yMax, xMin:xMax]

                frame = np.asarray(frame, "uint8")
                tifffile.imwrite(
                    f"train/orientationTraining/Division_{str(label).zfill(5)}.tif",
                    frame,
                )

                [x0, y0] = dfSpaceTime["Daughter0"].iloc[k]
                [x1, y1] = dfSpaceTime["Daughter1"].iloc[k]

                f = open(
                    f"train/orientationTraining/Division_{str(label).zfill(5)}.txt",
                    "w+",
                )

                ori = ori * np.pi / 180

                y = 512 - y
                shiftx0 = 20 + x0 - x
                shifty0 = 20 - (y0 - y)  # flipped y
                shiftx1 = 20 + x1 - x
                shifty1 = 20 - (y1 - y)  # flipped y

                # frame[round(shifty0), round(shiftx0), 0] = 255
                # frame[round(shifty1), round(shiftx1), 0] = 255

                # tifffile.imwrite(
                #     f"train/orientationTraining/check_{str(label).zfill(5)}.tif",
                #     frame,
                # )

                if shiftx0 > shiftx1:
                    f.write(f"{shiftx0} {shifty0}")
                    f.write("\n")
                    f.write(f"{shiftx1} {shifty1}")
                else:
                    f.write(f"{shiftx1} {shifty1}")
                    f.write("\n")
                    f.write(f"{shiftx0} {shifty0}")
                f.close()

                label += 1


# Confusion Matrix
if False:
    print("")
    print("frameRates", "True Pos", "False Pos", "False Neg")
    for frameRate in frameRates:
        T, frameNum = findT(frameRate)
        _dfConfusion = []
        for filename in filenames:
            dfDL = pd.read_pickle(
                f"databases/outputDL/divisions{frameRate}{filename}.pkl"
            )
            dfDivisions = pd.read_pickle(
                f"dat010921/{filename}/mitosisTracks{filename}.pkl"
            )
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
                        "T": int(t / frameNum),
                        "X": x,
                        "Y": y,
                    }
                )

            dfSpaceTime = pd.DataFrame(_dfSpaceTime)

            for i in range(len(dfSpaceTime)):
                label = dfSpaceTime["Label"].iloc[i]
                t = int(dfSpaceTime["T"].iloc[i])
                x = dfSpaceTime["X"].iloc[i]
                y = dfSpaceTime["Y"].iloc[i]
                ori = dfSpaceTime["Orientation"].iloc[i]

                dfCon = sortConfusion(dfDL, t, x, y, 1)

                if len(dfCon) > 0:
                    label_DL = dfCon["Label"].iloc[0]
                    _dfConfusion.append(
                        {
                            "Filename": filename,
                            "Label": label,
                            "Label DL": label_DL,
                            "T": int(t),
                            "X": x,
                            "Y": y,
                            "Orientation": ori,
                        }
                    )
                elif len(dfCon) == 0:
                    _dfConfusion.append(
                        {
                            "Filename": filename,
                            "Label": label,
                            "T": int(t),
                            "X": x,
                            "Y": y,
                            "Orientation": ori,
                        }
                    )

            uniqueLabel = list(set(dfDL["Label"]))
            dfConfusion_ = pd.DataFrame(_dfConfusion)
            dfConfusion_ = dfConfusion_[dfConfusion_["Filename"] == filename]
            linkedLabels = np.sort(dfConfusion_["Label DL"])

            for label_DL in uniqueLabel:
                if label_DL not in linkedLabels:
                    t = dfDL["T"][dfDL["Label"] == label_DL].iloc[0]
                    x = dfDL["X"][dfDL["Label"] == label_DL].iloc[0]
                    y = dfDL["Y"][dfDL["Label"] == label_DL].iloc[0]
                    _dfConfusion.append(
                        {
                            "Filename": filename,
                            "Label DL": label_DL,
                            "T": int(t),
                            "X": x,
                            "Y": y,
                        }
                    )

        dfConfusion = pd.DataFrame(_dfConfusion)
        dfConfusion.to_pickle(f"databases/dfConfusion{frameRate}{fileType}.pkl")
        falseNeg = len(dfConfusion[dfConfusion["Label DL"].isnull()])
        falsePos = len(dfConfusion[dfConfusion["Label"].isnull()])
        print(frameRate, len(dfConfusion) - falsePos - falseNeg, falsePos, falseNeg)


# second stage
if False:
    for frameRate in frameRates:
        dfConfusion = pd.read_pickle(
            f"databases/dfConfusionFS{frameRate}{fileType}.pkl"
        )

        for filename in filenames:

            T, frameNum = findT(frameRate)

            vidFocus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(float)
            vidFocus = vidFocus[::frameNum]

            [T, X, Y, rgb] = vidFocus.shape

            dfConfusionf = dfConfusion[dfConfusion["Filename"] == filename]
            dfFalsePos = dfConfusionf[dfConfusionf["Label"].isnull()]
            dfTrue = dfConfusionf[dfConfusionf["Label"].notnull()]

            for k in range(len(dfFalsePos)):
                t0 = int(dfFalsePos["T"].iloc[k])
                x = dfFalsePos["X"].iloc[k]
                y = dfFalsePos["Y"].iloc[k]
                label_DL = dfFalsePos["Label DL"].iloc[k]
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

                frames9 = np.zeros([126, 126, 3])

                times = range(t0 - 4, t0 + 5)

                timeVid = []
                for t in times:
                    if t >= 0 and t <= T - 1:
                        timeVid.append(t)

                if timeVid[0] == 0:
                    start = 4 - t0
                else:
                    start = 0

                j = start // 3
                i = start - 3 * j
                for t in timeVid:
                    frames9[
                        43 * j - yMinCrop : 43 * j + 40 + yMaxCrop,
                        43 * i - xMinCrop : 43 * i + 40 + xMaxCrop,
                    ] = vidFocus[t][yMin:yMax, xMin:xMax]
                    i += 1
                    if i == 3:
                        i = 0
                        j += 1

                frames9 = np.asarray(frames9, "uint8")
                tifffile.imwrite(
                    f"train/frame9Training/{frameRate}_2/division{filename}_{int(label_DL)}.tif",
                    frames9,
                )

            for k in range(len(dfTrue)):
                t0 = int(dfTrue["T"].iloc[k])
                x = int(dfTrue["X"].iloc[k])
                y = int(dfTrue["Y"].iloc[k])
                label_DL = dfTrue["Label DL"].iloc[k]
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

                frames9 = np.zeros([126, 126, 3])

                times = range(t0 - 4, t0 + 5)

                timeVid = []
                for t in times:
                    if t >= 0 and t <= T - 1:
                        timeVid.append(t)

                if timeVid[0] == 0:
                    start = 4 - t0
                else:
                    start = 0

                j = start // 3
                i = start - 3 * j
                for t in timeVid:
                    frames9[
                        43 * j - yMinCrop : 43 * j + 40 + yMaxCrop,
                        43 * i - xMinCrop : 43 * i + 40 + xMaxCrop,
                    ] = vidFocus[t][yMin:yMax, xMin:xMax]
                    i += 1
                    if i == 3:
                        i = 0
                        j += 1

                frames9 = np.asarray(frames9, "uint8")
                tifffile.imwrite(
                    f"train/frame9Training/{frameRate}_2/Division{filename}_{int(label_DL)}.tif",
                    frames9,
                )

# show false positives and negatives
if False:
    for frameRate in frameRates:
        dfConfusion = pd.read_pickle(f"databases/dfConfusion{frameRate}{fileType}.pkl")

        for filename in filenames:

            T, frameNum = findT(frameRate)

            dfConfusionf = dfConfusion[dfConfusion["Filename"] == filename]
            vidFocus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(float)
            vidFocus = vidFocus[::frameNum]

            [T, X, Y, rgb] = vidFocus.shape

            highlightDivisions = np.zeros([T, 552, 552, 3])

            for x in range(X):
                for y in range(Y):
                    highlightDivisions[:, 20 + x, 20 + y, :] = vidFocus[:, x, y, :]

            dfFalsePos = dfConfusionf[dfConfusionf["Label"].isnull()]

            for i in range(len(dfFalsePos)):
                t, x, y = (
                    dfFalsePos["T"].iloc[i],
                    dfFalsePos["X"].iloc[i],
                    dfFalsePos["Y"].iloc[i],
                )
                x = int(x)
                y = int(y)
                t0 = int(t)

                rr0, cc0 = sm.draw.disk([551 - (y + 20), x + 20], 16)
                rr1, cc1 = sm.draw.disk([551 - (y + 20), x + 20], 11)

                times = range(t0, t0 + 2)

                timeVid = []
                for t in times:
                    if t >= 0 and t <= T - 1:
                        timeVid.append(t)

                for t in timeVid:
                    highlightDivisions[t][rr0, cc0, 2] = 250
                    highlightDivisions[t][rr1, cc1, 2] = 0

            highlightDivisions = highlightDivisions[:, 20:532, 20:532]

            highlightDivisions = np.asarray(highlightDivisions, "uint8")
            tifffile.imwrite(
                f"results/falsePositives{frameRate}{filename}.tif",
                highlightDivisions,
            )

            vidFocus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(float)
            vidFocus = vidFocus[::frameNum]

            highlightDivisions = np.zeros([T, 552, 552, 3])

            for x in range(X):
                for y in range(Y):
                    highlightDivisions[:, 20 + x, 20 + y, :] = vidFocus[:, x, y, :]

            dfFalseNeg = dfConfusionf[dfConfusionf["Label DL"].isnull()]

            for i in range(len(dfFalseNeg)):
                t, x, y = (
                    dfFalseNeg["T"].iloc[i],
                    dfFalseNeg["X"].iloc[i],
                    dfFalseNeg["Y"].iloc[i],
                )
                x = int(x)
                y = int(y)
                t0 = int(t)

                rr0, cc0 = sm.draw.disk([551 - (y + 20), x + 20], 16)
                rr1, cc1 = sm.draw.disk([551 - (y + 20), x + 20], 11)

                times = range(t0, t0 + 2)

                timeVid = []
                for t in times:
                    if t >= 0 and t <= T - 1:
                        timeVid.append(t)

                for t in timeVid:
                    highlightDivisions[t][rr0, cc0, 2] = 250
                    highlightDivisions[t][rr1, cc1, 2] = 0

            highlightDivisions = highlightDivisions[:, 20:532, 20:532]

            highlightDivisions = np.asarray(highlightDivisions, "uint8")
            tifffile.imwrite(
                f"results/falseNegatives{frameRate}{filename}.tif",
                highlightDivisions,
            )


# Old Data Confusion Matrix
if True:
    print("")
    print("Old Data", "True Pos", "False Pos", "False Neg")
    _dfConfusion = []
    for filename in filenames:
        dfDiv = pd.read_pickle(f"dat010921/{filename}/mitosisTracks{filename}.pkl")
        dfDiv = dfDiv[dfDiv["Chain"] == "parent"]
        _dfDiv = []

        for i in range(len(dfDiv)):

            label = dfDiv["Label"].iloc[i]
            t = dfDiv["Time"].iloc[i][-1]
            [x, y] = dfDiv["Position"].iloc[i][-1]
            ori = dfDiv["Division Orientation"].iloc[i]

            _dfDiv.append(
                {
                    "Filename": filename,
                    "Label": label,
                    "Orientation": ori,
                    "T": int(t),
                    "X": x,
                    "Y": y,
                }
            )

        dfDiv = pd.DataFrame(_dfDiv)

        dfOldDiv = pd.read_pickle(
            f"databases/old division datasets/mitosisTracks{filename}.pkl"
        )
        dfOldDiv = dfOldDiv[dfOldDiv["Chain"] == "parent"]
        _dfOldDiv = []

        for i in range(len(dfOldDiv)):

            label = dfOldDiv["Label"].iloc[i]
            t = dfOldDiv["Time"].iloc[i][-1]
            [x, y] = dfOldDiv["Position"].iloc[i][-1]
            ori = dfOldDiv["Division Orientation"].iloc[i]

            _dfOldDiv.append(
                {
                    "Filename": filename,
                    "Label": label,
                    "Orientation": ori,
                    "T": int(t),
                    "X": x,
                    "Y": y,
                }
            )

        dfOldDiv = pd.DataFrame(_dfOldDiv)

        for i in range(len(dfDiv)):
            label = dfDiv["Label"].iloc[i]
            t = int(dfDiv["T"].iloc[i])
            x = dfDiv["X"].iloc[i]
            y = dfDiv["Y"].iloc[i]

            dfCon = sortConfusion(dfOldDiv, t, x, y, 1)

            if len(dfCon) > 0:
                labelOld = dfCon["Label"].iloc[0]
                _dfConfusion.append(
                    {
                        "Filename": filename,
                        "Label": label,
                        "Label Old": labelOld,
                        "T": int(t),
                        "X": x,
                        "Y": y,
                    }
                )
            elif len(dfCon) == 0:
                _dfConfusion.append(
                    {
                        "Filename": filename,
                        "Label": label,
                        "T": int(t),
                        "X": x,
                        "Y": y,
                    }
                )

        uniqueLabel = list(set(dfOldDiv["Label"]))
        dfConfusion_ = pd.DataFrame(_dfConfusion)
        dfConfusion_ = dfConfusion_[dfConfusion_["Filename"] == filename]
        linkedLabels = np.sort(dfConfusion_["Label Old"])

        for labelOld in uniqueLabel:
            if labelOld not in linkedLabels:
                t = dfOldDiv["T"][dfOldDiv["Label"] == labelOld].iloc[0]
                x = dfOldDiv["X"][dfOldDiv["Label"] == labelOld].iloc[0]
                y = dfOldDiv["Y"][dfOldDiv["Label"] == labelOld].iloc[0]
                _dfConfusion.append(
                    {
                        "Filename": filename,
                        "Label Old": labelOld,
                        "T": int(t),
                        "X": x,
                        "Y": y,
                    }
                )

    dfConfusion = pd.DataFrame(_dfConfusion)
    dfConfusion.to_pickle(f"databases/dfConfusionOldData{fileType}.pkl")
    falseNeg = len(dfConfusion[dfConfusion["Label Old"].isnull()])
    falsePos = len(dfConfusion[dfConfusion["Label"].isnull()])
    print("Old Data", len(dfConfusion) - falsePos - falseNeg, falsePos, falseNeg)
    print("")
    print("Total number of divisions", len(dfConfusion) - falsePos)
    print("")


# orientation error
if False:
    for frameRate in frameRates:
        _dfConfusion = []
        T, frameNum = findT(frameRate)
        for filename in filenames:
            dfDL = pd.read_pickle(
                f"databases/outputDL/divisions{frameRate}{filename}.pkl"
            )

            dfDivisions = pd.read_pickle(
                f"dat010921/{filename}/mitosisTracks{filename}.pkl"
            )
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
                        "T": int(t / frameNum),
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

                dfCon = sortConfusion(dfDL, t, x, y, 1)

                if len(dfCon) > 0:
                    label = dfCon["Label"].iloc[0]
                    ori_DL = dfCon["Ori"].iloc[0]
                    dist = dfCon["distance"].iloc[0]
                    error = 1 - np.dot(
                        np.array([np.cos(2 * ori), np.sin(2 * ori)]),
                        np.array([np.cos(2 * ori_DL), np.sin(2 * ori_DL)]),
                    )
                    dtheta = np.arccos(1 - error) / 2
                    _dfConfusion.append(
                        {
                            "Filename": filename,
                            "Label": label,
                            "T": int(t),
                            "X": x,
                            "Y": y,
                            "Orientation Error": error,
                            "Delta Theta": dtheta,
                            "Distance": dist,
                        }
                    )

        dfConfusion = pd.DataFrame(_dfConfusion)
        dfConfusion.to_pickle(
            f"databases/dfConfusionOrientation{frameRate}{fileType}.pkl"
        )

        # err = np.array(dfConfusion["Orientation Error"])

        # fig, ax = plt.subplots()
        # ax.hist(err, density=False, bins=40)
        # ax.set_xlim([0, 2])
        # ax.set_xlabel("error", y=0.13)
        # ax.axvline(np.median(err), c="k", label="mean")
        # fig.savefig(
        #     f"results/orientationError{frameRate}{fileType}.png",
        #     dpi=200,
        #     transparent=True,
        # )
        # plt.close("all")

        err = abs(np.array(dfConfusion["Delta Theta"])) * 180 / np.pi
        dist = np.array(dfConfusion["Distance"])

        fig, ax = plt.subplots()
        ax.hist(err, density=False, bins=18, range=[0, 90])
        ax.set_xlim([0, 90])
        ax.set_ylim([0, 800])
        ax.set_xlabel("error", y=0.13)
        ax.axvline(np.median(err), c="k", label="mean")
        fig.savefig(
            f"results/orientationDiff{frameRate}{fileType}.png",
            dpi=200,
            transparent=True,
        )
        plt.close("all")

        # fig, ax = plt.subplots()
        # ax.scatter(err, dist)
        # ax.set_xlim([0, 90])
        # fig.savefig(
        #     f"results/orientationDiffErr{frameRate}{fileType}.png",
        #     dpi=200,
        #     transparent=True,
        # )
        # plt.close("all")
