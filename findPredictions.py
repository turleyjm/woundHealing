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


def sortConfusion(df, t, x, y):

    a = df[df["T"] == t - 1]
    b = df[df["T"] == t]
    c = df[df["T"] == t + 1]

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


# -------------------

filenames, fileType = cl.getFilesType()

if False:
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

        vid = sm.io.imread(f"dat/{filename}/predict{filename}.tif").astype(int)

        [T, X, Y] = vid.shape
        vid = vid / 255
        vid = np.asarray(vid, "uint8")

        t = 0
        blobs = blob_log(vid[t], max_sigma=50, num_sigma=25, threshold=0.2)
        blobs_logs = np.concatenate((blobs, np.zeros([len(blobs), 1])), axis=1)
        for t in range(1, T):
            blobs = blob_log(vid[t], max_sigma=50, num_sigma=25, threshold=0.2)
            blobs_log = np.concatenate((blobs, np.zeros([len(blobs), 1]) + t), axis=1)
            blobs_logs = np.concatenate((blobs_logs, blobs_log))

            if t == 6:
                print(y)

            if False:
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))

                ax.set_title(f"Laplacian of Gaussian {t}")
                ax.imshow(vid[t])
                for blob in blobs_log:
                    y, x, r, _t = blob
                    c = plt.Circle((x, y), r, color="red", linewidth=2, fill=False)
                    ax.add_patch(c)
                ax.set_axis_off()

                fig.savefig(
                    f"results/Laplacian of Gaussian {fileType} {t}.png",
                    dpi=300,
                    transparent=True,
                )
                plt.close("all")

        blobs_logs[:, 2] = blobs_logs[:, 2] * sqrt(2)

        _df = []
        i = 0
        for blob in blobs_logs:
            y, x, r, t = blob

            _df.append(
                {
                    "Label": i,
                    "T": int(t),
                    "X": int(x),
                    "Y": 512 - int(y),
                    "R": r,
                }
            )
            i += 1

        df = pd.DataFrame(_df)
        df.to_pickle(f"databases/_dfDivisionDL{filename}.pkl")
        dfRemove = pd.read_pickle(f"databases/_dfDivisionDL{filename}.pkl")

        for i in range(len(df)):
            ti, xi, yi = df["T"].iloc[i], df["X"].iloc[i], df["Y"].iloc[i]
            labeli = df["Label"].iloc[i]
            dfmulti = sortDL(df, ti, xi, yi)

            if len(dfmulti) > 0:
                mui = intensity(vid, ti, xi, 512 - yi)
                for j in range(len(dfmulti)):
                    tj, xj, yj = (
                        dfmulti["T"].iloc[j],
                        dfmulti["X"].iloc[j],
                        dfmulti["Y"].iloc[j],
                    )
                    labelj = dfmulti["Label"].iloc[j]
                    muj = intensity(vid, tj, xj, 512 - yj)

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

        dfRemove.to_pickle(f"databases/dfDivisionDL{filename}.pkl")

if False:
    for filename in filenames:
        dfDivisionDL = pd.read_pickle(f"databases/dfDivisionDL{filename}.pkl")
        vidFocus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(float)

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

            t0 = int(dfSpaceTime["T"].iloc[i])
            [x, y] = [int(dfSpaceTime["X"].iloc[i]), int(dfSpaceTime["Y"].iloc[i])]

            rr0, cc0 = sm.draw.disk([551 - (y + 20), x + 20], 14)
            rr1, cc1 = sm.draw.disk([551 - (y + 20), x + 20], 12)

            times = range(t0 - 5, t0 + 5)

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

            times = range(t0 - 5, t0 + 5)

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
            f"dat/{filename}/divisionsDeepLearning{filename}.tif", highlightDivisions
        )

# 3 frame

if False:
    for filename in filenames:

        vid = sm.io.imread(f"dat/{filename}/predict{filename}-3.tif").astype(int)

        [T, X, Y] = vid.shape
        vid = vid / 255
        vid = np.asarray(vid, "uint8")

        t = 0
        blobs = blob_log(vid[t], max_sigma=50, num_sigma=25, threshold=0.1)
        blobs_logs = np.concatenate((blobs, np.zeros([len(blobs), 1])), axis=1)
        for t in range(1, T):
            blobs = blob_log(vid[t], max_sigma=50, num_sigma=25, threshold=0.1)
            blobs_log = np.concatenate((blobs, np.zeros([len(blobs), 1]) + t), axis=1)
            blobs_logs = np.concatenate((blobs_logs, blobs_log))

            if False:
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))

                ax.set_title(f"Laplacian of Gaussian {t}")
                ax.imshow(vid[t])
                for blob in blobs_log:
                    y, x, r, _t = blob
                    c = plt.Circle((x, y), r, color="red", linewidth=2, fill=False)
                    ax.add_patch(c)
                ax.set_axis_off()

                fig.savefig(
                    f"results/Laplacian of Gaussian {fileType} {t}.png",
                    dpi=300,
                    transparent=True,
                )
                plt.close("all")

        blobs_logs[:, 2] = blobs_logs[:, 2] * sqrt(2)

        _df = []
        i = 0
        for blob in blobs_logs:
            y, x, r, t = blob

            _df.append(
                {
                    "Label": i,
                    "T": int(t),
                    "X": int(x),
                    "Y": 512 - int(y),
                    "R": r,
                }
            )
            i += 1

        df = pd.DataFrame(_df)
        df.to_pickle(f"databases/_dfDivisionDL-3{filename}.pkl")
        dfRemove = pd.read_pickle(f"databases/_dfDivisionDL-3{filename}.pkl")

        for i in range(len(df)):
            ti, xi, yi = df["T"].iloc[i], df["X"].iloc[i], df["Y"].iloc[i]
            labeli = df["Label"].iloc[i]
            dfmulti = sortDL(df, ti, xi, yi)

            if len(dfmulti) > 0:
                mui = intensity(vid, ti, xi, yi)
                for j in range(len(dfmulti)):
                    tj, xj, yj = (
                        dfmulti["T"].iloc[j],
                        dfmulti["X"].iloc[j],
                        dfmulti["Y"].iloc[j],
                    )
                    labelj = dfmulti["Label"].iloc[j]
                    muj = intensity(vid, tj, xj, yj)

                    if mui < muj:
                        indexNames = dfRemove[dfRemove["Label"] == labeli].index
                        dfRemove.drop(indexNames, inplace=True)
                    else:
                        indexNames = dfRemove[dfRemove["Label"] == labelj].index
                        dfRemove.drop(indexNames, inplace=True)

        dfRemove.to_pickle(f"databases/dfDivisionDL-3{filename}.pkl")

if False:
    for filename in filenames:
        dfDivisionDL = pd.read_pickle(f"databases/dfDivisionDL-3{filename}.pkl")
        vidFocus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(float)
        vidFocus = vidFocus[::3]

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

            t0 = int(dfSpaceTime["T"].iloc[i] / 3)
            [x, y] = [int(dfSpaceTime["X"].iloc[i]), int(dfSpaceTime["Y"].iloc[i])]

            rr0, cc0 = sm.draw.disk([551 - (y + 20), x + 20], 14)
            rr1, cc1 = sm.draw.disk([551 - (y + 20), x + 20], 12)

            times = range(t0 - 1, t0 + 2)

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

            times = range(t0 - 1, t0 + 2)

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
            f"dat/{filename}/divisionsDeepLearning-3{filename}.tif", highlightDivisions
        )

# confusion matrix


if True:
    _dfConfusion = []
    for filename in filenames:
        dfDivisionDL = pd.read_pickle(f"databases/dfDivisionDL{filename}.pkl")
        vidFocus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(float)

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
            t = dfSpaceTime["T"].iloc[i]
            x = dfSpaceTime["X"].iloc[i]
            y = dfSpaceTime["Y"].iloc[i]

            dfCon = sortConfusion(dfDivisionDL, t, x, y)

            if len(dfCon) > 0:
                label_DL = dfCon["Label"].iloc[0]
                _dfConfusion.append(
                    {
                        "Filename": filename,
                        "Label": label,
                        "Label DL": label_DL,
                        "T": t,
                        "X": x,
                        "Y": y,
                    }
                )
            elif len(dfCon) == 0:
                _dfConfusion.append(
                    {
                        "Filename": filename,
                        "Label": label,
                        "T": t,
                        "X": x,
                        "Y": y,
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
                _dfConfusion.append(
                    {
                        "Filename": filename,
                        "Label DL": label_DL,
                        "T": t,
                        "X": x,
                        "Y": y,
                    }
                )

    dfConfusion = pd.DataFrame(_dfConfusion)
    dfConfusion.to_pickle(f"databases/dfConfusion{fileType}.pkl")
