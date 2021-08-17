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

import matplotlib.pyplot as plt

# -------------------


def sort1e2h_3h(df, t, x, y):

    a = df[df["T"] == t - 2]
    b = df[df["T"] == t - 1]
    c = df[df["T"] == t]
    d = df[df["T"] == t + 1]
    e = df[df["T"] == t + 2]

    df = pd.concat([a, b, c, d, e])

    xMax = x + 9
    xMin = x - 9
    yMax = y + 9
    yMin = y - 9
    if xMax > 511:
        xMax = 511
    if yMax > 511:
        yMax = 511
    if xMin < 0:
        xMin = 0
    if yMin < 0:
        yMin = 0

    dfxmin = df[df["X"] >= xMin]
    dfx = dfxmin[dfxmin["X"] < xMax]

    dfymin = dfx[dfx["Y"] >= yMin]
    df = dfymin[dfymin["Y"] < yMax]

    return df


def sortDL(df, t, x, y):

    a = df[df["T"] == t - 2]
    b = df[df["T"] == t - 1]
    c = df[df["T"] == t + 1]
    d = df[df["T"] == t + 2]

    df = pd.concat([a, b, c, d])

    xMax = x + 9
    xMin = x - 9
    yMax = y + 9
    yMin = y - 9
    if xMax > 511:
        xMax = 511
    if yMax > 511:
        yMax = 511
    if xMin < 0:
        xMin = 0
    if yMin < 0:
        yMin = 0

    dfxmin = df[df["X"] >= xMin]
    dfx = dfxmin[dfxmin["X"] < xMax]

    dfymin = dfx[dfx["Y"] >= yMin]
    df = dfymin[dfymin["Y"] < yMax]

    return df


def intensity(vid, ti, xi, yi):

    [T, X, Y] = vid.shape

    vidBoundary = np.zeros([T, 552, 552])

    for x in range(X):
        for y in range(Y):
            vidBoundary[:, 20 + x, 20 + y] = vid[:, x, y]

    rr, cc = sm.draw.disk([yi + 20, xi + 20], 9)
    div = vidBoundary[ti][rr, cc]
    div = div[div > 0]

    mu = np.mean(div)

    return mu


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


def findT(segType):
    if segType == "1e2h1f":
        T = 180
        frameNum = 1
    elif segType == "3h1f":
        T = 179
        frameNum = 1
    elif segType == "1e2h2f":
        T = 90
        frameNum = 2
    elif segType == "3h2f":
        T = 89
        frameNum = 2
    elif segType == "1e2h3f":
        T = 60
        frameNum = 3
    else:
        T = 59
        frameNum = 3
    return T, frameNum


# -------------------

# filenames, fileType = cl.getFilesType()
# "1e2h1f", "3h1f", "1e2h2f", "3h2f", "1e2h3f", "3h3f"

segTypes = ["1e2h1f", "3h1f", "1e2h2f", "3h2f", "1e2h3f", "3h3f"]

filenames = [
    "Unwound18h11",
    #     "Unwound18h12",
    #     "WoundL18h07",
    #     "WoundL18h08",
    #     "WoundL18h09",
    #     "WoundS18h10",
    #     "WoundS18h11",
    #     "WoundS18h12",
    #     "WoundS18h13",
]
# fileType = "validation"

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
# fileType = "training"

for segType in segTypes:
    if False:
        for filename in filenames:
            print(filename)

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
                        "T": t,
                        "X": x,
                        "Y": y,
                    }
                )

            dfSpaceTime = pd.DataFrame(_dfSpaceTime)

            vid = sm.io.imread(f"dat/{filename}/predict{segType}{filename}.tif").astype(
                int
            )

            [T, X, Y] = vid.shape
            vid = vid / 255
            vid = np.asarray(vid, "uint8")

            t = 0
            img = np.zeros([552, 552])
            img[20:532, 20:532] = vid[t]
            blobs = blob_log(
                img, min_sigma=10, max_sigma=25, num_sigma=25, threshold=10
            )
            blobs_logs = np.concatenate((blobs, np.zeros([len(blobs), 1])), axis=1)
            for t in range(1, T):
                img = np.zeros([552, 552])
                img[20:532, 20:532] = vid[t]
                blobs = blob_log(
                    img, min_sigma=10, max_sigma=25, num_sigma=25, threshold=10
                )
                blobs_log = np.concatenate(
                    (blobs, np.zeros([len(blobs), 1]) + t), axis=1
                )
                blobs_logs = np.concatenate((blobs_logs, blobs_log))

                if True:
                    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

                    ax.set_title(f"Laplacian of Gaussian {t}")
                    ax.imshow(img)
                    for blob in blobs_log:
                        y, x, r, _t = blob
                        c = plt.Circle((x, y), r, color="red", linewidth=2, fill=False)
                        ax.add_patch(c)
                    ax.set_axis_off()

                    fig.savefig(
                        f"results/Laplacian of Gaussian {segType} {fileType} {t}.png",
                        dpi=300,
                        transparent=True,
                    )
                    plt.close("all")

            blobs_logs[:, 2] = blobs_logs[:, 2] * sqrt(2)

            _df = []
            i = 0
            for blob in blobs_logs:
                y, x, r, t = blob
                mu = intensity(vid, int(t), int(x - 20), int(y - 20))

                _df.append(
                    {
                        "Label": i,
                        "T": int(t),
                        "X": int(x - 20),
                        "Y": 532 - int(y),  # map coords without boundary
                        "R": r,
                        "Intensity": mu,
                    }
                )
                i += 1

            df = pd.DataFrame(_df)
            df.to_pickle(f"databases/_dfDivisionDL{segType}{filename}.pkl")
            dfRemove = pd.read_pickle(f"databases/_dfDivisionDL{segType}{filename}.pkl")

            for i in range(len(df)):
                ti, xi, yi = df["T"].iloc[i], df["X"].iloc[i], df["Y"].iloc[i]
                labeli = df["Label"].iloc[i]
                dfmulti = sortDL(df, ti, xi, yi)

                if len(dfmulti) > 0:
                    mui = df["Intensity"].iloc[i]
                    for j in range(len(dfmulti)):
                        tj, xj, yj = (
                            dfmulti["T"].iloc[j],
                            dfmulti["X"].iloc[j],
                            dfmulti["Y"].iloc[j],
                        )
                        labelj = dfmulti["Label"].iloc[j]
                        muj = dfmulti["Intensity"].iloc[j]

                        if mui < muj:
                            indexNames = dfRemove[dfRemove["Label"] == labeli].index
                            dfRemove.drop(indexNames, inplace=True)
                        else:
                            indexNames = dfRemove[dfRemove["Label"] == labelj].index
                            dfRemove.drop(indexNames, inplace=True)

            for i in range(len(df)):
                ti, xi, yi = df["T"].iloc[i], df["X"].iloc[i], df["Y"].iloc[i]
                dft = df[df["T"] == ti]
                dfx = dft[dft["X"] == xi]
                dfy = dfx[dfx["Y"] == yi]
                if len(dfy) == 2:
                    label = dfy.iloc[1]
                    indexNames = dfRemove[dfRemove["Label"] == label].index
                    dfRemove.drop(indexNames, inplace=True)

            dfRemove.to_pickle(f"databases/dfDivisionDL{segType}{filename}.pkl")

    # show divisions found
    if False:
        for filename in filenames:

            T, frameNum = findT(segType)

            dfDivisionDL = pd.read_pickle(
                f"databases/dfDivisionDL{segType}{filename}.pkl"
            )
            vidFocus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(float)
            vidFocus = vidFocus[::frameNum]

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
                        "T": t,
                        "X": x,
                        "Y": y,
                    }
                )

            dfSpaceTime = pd.DataFrame(_dfSpaceTime)

            [T, X, Y, rgb] = vidFocus.shape

            highlightDivisions = np.zeros([T, 552, 552, 3])

            for x in range(X):
                for y in range(Y):
                    highlightDivisions[:, 20 + x, 20 + y, :] = vidFocus[:, x, y, :]

            for i in range(len(dfSpaceTime)):

                t0 = int(dfSpaceTime["T"].iloc[i] / frameNum)
                [x, y] = [int(dfSpaceTime["X"].iloc[i]), int(dfSpaceTime["Y"].iloc[i])]

                rr0, cc0 = sm.draw.disk([551 - (y + 20), x + 20], 14)
                rr1, cc1 = sm.draw.disk([551 - (y + 20), x + 20], 12)

                times = range(t0 - round(5 / frameNum), t0 + round(5 / frameNum))

                timeVid = []
                for t in times:
                    if t >= 0 and t <= T - 1:
                        timeVid.append(t)

                for t in timeVid:
                    highlightDivisions[t][rr0, cc0, 2] = 200
                    highlightDivisions[t][rr1, cc1, 2] = 0

            for i in range(len(dfDivisionDL)):
                t, x, y = (
                    dfDivisionDL["T"].iloc[i],
                    dfDivisionDL["X"].iloc[i],
                    dfDivisionDL["Y"].iloc[i],
                )
                x = int(x)
                y = int(y)
                t0 = int(t)

                rr0, cc0 = sm.draw.disk([551 - (y + 20), x + 20], 7)

                times = range(t0 - round(5 / frameNum), t0 + round(5 / frameNum))

                timeVid = []
                for t in times:
                    if t >= 0 and t <= T - 1:
                        timeVid.append(t)

                for t in timeVid:
                    highlightDivisions[t][rr0, cc0, 2] = 200
                    highlightDivisions[t][rr0, cc0, 0] = 200

            highlightDivisions = highlightDivisions[:, 20:532, 20:532]

            highlightDivisions = np.asarray(highlightDivisions, "uint8")
            tifffile.imwrite(
                f"results/divisionsDeepLearning{segType}{filename}.tif",
                highlightDivisions,
            )

    # confusion matrix

    if False:
        _dfConfusion = []
        for filename in filenames:

            T, frameNum = findT(segType)

            dfDivisionDL = pd.read_pickle(
                f"databases/dfDivisionDL{segType}{filename}.pkl"
            )
            vid = sm.io.imread(f"dat/{filename}/predict{segType}{filename}.tif").astype(
                int
            )

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
                        "T": t,
                        "X": x,
                        "Y": y,
                    }
                )

            dfSpaceTime = pd.DataFrame(_dfSpaceTime)
            dfSpaceTime = dfSpaceTime[dfSpaceTime["T"] < T * frameNum]

            for i in range(len(dfSpaceTime)):
                label = dfSpaceTime["Label"].iloc[i]
                t = int(dfSpaceTime["T"].iloc[i] / frameNum)
                x = dfSpaceTime["X"].iloc[i]
                y = dfSpaceTime["Y"].iloc[i]
                mu = intensity(vid, t, x, 512 - y)

                dfCon = sortConfusion(dfDivisionDL, t, x, y, frameNum)

                if len(dfCon) > 0:
                    label_DL = dfCon["Label"].iloc[0]
                    _dfConfusion.append(
                        {
                            "Filename": filename,
                            "Label": label,
                            "Label DL": label_DL,
                            "T": int(t * frameNum),
                            "X": x,
                            "Y": y,
                            "Intensity": mu,
                        }
                    )
                elif len(dfCon) == 0:
                    _dfConfusion.append(
                        {
                            "Filename": filename,
                            "Label": label,
                            "T": int(t * frameNum),
                            "X": x,
                            "Y": y,
                            "Intensity": mu,
                        }
                    )

            uniqueLabel = list(set(dfDivisionDL["Label"]))
            dfConfusion_ = pd.DataFrame(_dfConfusion)
            dfConfusion_ = dfConfusion_[dfConfusion_["Filename"] == filename]
            linkedLabels = np.sort(dfConfusion_["Label DL"])

            for label_DL in uniqueLabel:
                if label_DL not in linkedLabels:
                    t = dfDivisionDL["T"][dfDivisionDL["Label"] == label_DL].iloc[0]
                    x = dfDivisionDL["X"][dfDivisionDL["Label"] == label_DL].iloc[0]
                    y = dfDivisionDL["Y"][dfDivisionDL["Label"] == label_DL].iloc[0]
                    mu = intensity(vid, int(t), int(x), int(512 - y))
                    _dfConfusion.append(
                        {
                            "Filename": filename,
                            "Label DL": label_DL,
                            "T": int(t * frameNum),
                            "X": x,
                            "Y": y,
                            "Intensity": mu,
                        }
                    )

        dfConfusion = pd.DataFrame(_dfConfusion)
        dfConfusion.to_pickle(f"databases/dfConfusion{segType}{fileType}.pkl")

    # show false positives

    if False:
        dfConfusion = pd.read_pickle(f"databases/dfConfusion{segType}{fileType}.pkl")
        for filename in filenames:

            T, frameNum = findT(segType)

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

                times = range(t0 - round(5 / frameNum), t0 + round(5 / frameNum))

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
                f"dat/{filename}/falsePositives{segType}{filename}.tif",
                highlightDivisions,
            )

    # frame9 for DL
    if False:
        dfConfusion = pd.read_pickle(f"databases/dfConfusion{segType}{fileType}.pkl")
        for filename in filenames:

            T, frameNum = findT(segType)

            vidFocus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(float)
            vidFocus = vidFocus[::frameNum]

            [T, X, Y, rgb] = vidFocus.shape

            dfConfusionf = dfConfusion[dfConfusion["Filename"] == filename]
            dfFalsePos = dfConfusionf[dfConfusionf["Label"].isnull()]
            dfTruePos = dfConfusionf[dfConfusionf["Label"].notnull()]
            dfTruePos = dfTruePos[dfTruePos["Label DL"].notnull()]

            for k in range(len(dfFalsePos)):
                t0 = int(dfFalsePos["T"].iloc[k] / frameNum)
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

                if fileType == "training":

                    frames9 = np.asarray(frames9, "uint8")
                    tifffile.imwrite(
                        f"train/frame9 training/{segType}/division{filename}_{int(label_DL)}.tif",
                        frames9,
                    )
                else:
                    frames9 = np.asarray(frames9, "uint8")
                    tifffile.imwrite(
                        f"train/{segType}/falseDivision/{filename}_{int(label_DL)}.tif",
                        frames9,
                    )

            for k in range(len(dfTruePos)):
                t0 = int(dfTruePos["T"].iloc[k] / frameNum)
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

                if fileType == "training":

                    frames9 = np.asarray(frames9, "uint8")
                    tifffile.imwrite(
                        f"train/frame9 training/{segType}/Division{filename}_{int(label_DL)}.tif",
                        frames9,
                    )
                else:
                    frames9 = np.asarray(frames9, "uint8")
                    tifffile.imwrite(
                        f"train/{segType}/division/{filename}_{int(label_DL)}.tif",
                        frames9,
                    )


if False:
    segTypes = ["1e2h1f", "3h1f", "1e2h2f", "3h2f", "1e2h3f", "3h3f"]
    # fileType = "validation"
    # filenames = [
    #     "Unwound18h11",
    #     "Unwound18h12",
    #     "WoundL18h07",
    #     "WoundL18h08",
    #     "WoundL18h09",
    #     "WoundS18h10",
    #     "WoundS18h11",
    #     "WoundS18h12",
    #     "WoundS18h13",
    # ]

    fileType = "training"
    filenames = [
        "Unwound18h01",
        "Unwound18h02",
        "Unwound18h03",
        "Unwound18h04",
        "Unwound18h05",
        "Unwound18h06",
        "Unwound18h07",
        "Unwound18h08",
        "Unwound18h09",
        "Unwound18h10",
        "WoundL18h01",
        "WoundL18h02",
        "WoundL18h03",
        "WoundL18h04",
        "WoundL18h05",
        "WoundL18h06",
        "WoundS18h01",
        "WoundS18h02",
        "WoundS18h03",
        "WoundS18h04",
        "WoundS18h05",
        "WoundS18h06",
        "WoundS18h07",
        "WoundS18h08",
        "WoundS18h09",
    ]

    print("segType", "True Pos", "False Pos", "False Neg")
    for segType in segTypes:
        dfConfusion = pd.read_pickle(f"databases/dfConfusion{segType}{fileType}.pkl")
        falseNeg = len(dfConfusion[dfConfusion["Label DL"].isnull()])
        falsePos = len(dfConfusion[dfConfusion["Label"].isnull()])
        print(segType, len(dfConfusion) - falsePos - falseNeg, falsePos, falseNeg)

    count = 0
    for filename in filenames:
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
                    "T": t,
                    "X": x,
                    "Y": y,
                }
            )

        dfSpaceTime = pd.DataFrame(_dfSpaceTime)
        count += len(dfSpaceTime)

    print(f"Total divisons {count}")


# combine 1e2h and 3h

if False:
    for filename in filenames:

        vidFocus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(float)
        [T, X, Y, rgb] = vidFocus.shape

        highlightDivisions = np.zeros([T, 552, 552, 3])

        for x in range(X):
            for y in range(Y):
                highlightDivisions[:, 20 + x, 20 + y, :] = vidFocus[:, x, y, :]

        df1e2h1f = pd.read_pickle(f"databases/dfDivisionDL1e2h1f{filename}.pkl")
        df3h1f = pd.read_pickle(f"databases/dfDivisionDL3h1f{filename}.pkl")

        for i in range(len(df1e2h1f)):
            ti, xi, yi = (
                df1e2h1f["T"].iloc[i],
                df1e2h1f["X"].iloc[i],
                df1e2h1f["Y"].iloc[i],
            )
            labeli = df1e2h1f["Label"].iloc[i]
            dfmulti = sort1e2h_3h(df3h1f, ti, xi, yi)

            if len(dfmulti) > 0:
                for j in range(len(dfmulti)):
                    labelj = dfmulti["Label"].iloc[j]

                    indexNames = df3h1f[df3h1f["Label"] == labelj].index
                    df3h1f.drop(indexNames, inplace=True)

        df1f = pd.concat([df1e2h1f, df3h1f])

        df1f.to_pickle(f"databases/dfDivisionDL1f{filename}.pkl")

        for i in range(len(df1f)):
            t, x, y = (
                df1f["T"].iloc[i],
                df1f["X"].iloc[i],
                df1f["Y"].iloc[i],
            )
            x = int(x)
            y = int(y)
            t0 = int(t)

            rr0, cc0 = sm.draw.disk([551 - (y + 20), x + 20], 14)
            rr1, cc1 = sm.draw.disk([551 - (y + 20), x + 20], 12)

            times = range(t0, t0 + 2)

            timeVid = []
            for t in times:
                if t >= 0 and t <= T - 1:
                    timeVid.append(t)

            for t in timeVid:
                highlightDivisions[t][rr0, cc0, 2] = 200
                highlightDivisions[t][rr1, cc1, 2] = 0

        highlightDivisions = highlightDivisions[:, 20:532, 20:532]

        highlightDivisions = np.asarray(highlightDivisions, "uint8")
        tifffile.imwrite(
            f"results/divisionsDeepLearning1f{filename}.tif",
            highlightDivisions,
        )

if False:
    _dfConfusion = []
    for filename in filenames:

        dfDivisionDL = pd.read_pickle(f"databases/dfDivisionDL1f{filename}.pkl")
        vid = sm.io.imread(f"dat/{filename}/predict3h1f{filename}.tif").astype(int)

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
                    "T": t,
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
            mu = intensity(vid, t, x, 512 - y)

            dfCon = sortConfusion(dfDivisionDL, t, x, y, 1)

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
                        "Intensity": mu,
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
                        "Intensity": mu,
                    }
                )

        uniqueLabel = list(set(dfDivisionDL["Label"]))
        dfConfusion_ = pd.DataFrame(_dfConfusion)
        dfConfusion_ = dfConfusion_[dfConfusion_["Filename"] == filename]
        linkedLabels = np.sort(dfConfusion_["Label DL"])

        for label_DL in uniqueLabel:
            if label_DL not in linkedLabels:
                t = dfDivisionDL["T"][dfDivisionDL["Label"] == label_DL].iloc[0]
                x = dfDivisionDL["X"][dfDivisionDL["Label"] == label_DL].iloc[0]
                y = dfDivisionDL["Y"][dfDivisionDL["Label"] == label_DL].iloc[0]
                mu = intensity(vid, int(t), int(x), int(512 - y))
                _dfConfusion.append(
                    {
                        "Filename": filename,
                        "Label DL": label_DL,
                        "T": int(t),
                        "X": x,
                        "Y": y,
                        "Intensity": mu,
                    }
                )

    dfConfusion = pd.DataFrame(_dfConfusion)
    dfConfusion.to_pickle(f"databases/dfConfusion1f{filename}.pkl")


# Test DL


filenames = ["Unwound18h11"]
frameRates = ["1f"]
fileType = "test"
if True:
    for frameRate in frameRates:
        _dfConfusion = []
        for filename in filenames:
            dfDL = pd.read_pickle(f"dat/{filename}/divisions{frameRate}{filename}.pkl")
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
                        "T": t,
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
        dfConfusion.to_pickle(f"databases/dfConfusion{frameRate}{filename}.pkl")
        falseNeg = len(dfConfusion[dfConfusion["Label DL"].isnull()])
        falsePos = len(dfConfusion[dfConfusion["Label"].isnull()])
        print(frameRate, len(dfConfusion) - falsePos - falseNeg, falsePos, falseNeg)