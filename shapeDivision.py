from ast import Break
import os
from os.path import exists
import shutil
from math import floor, log10, factorial

from collections import Counter
from trace import Trace
from turtle import position
import cv2
import matplotlib
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import random
import scipy as sp
import scipy.special as sc
import scipy.linalg as linalg
import shapely
import skimage as sm
import skimage.io
import skimage.measure
import skimage.feature
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
from mpl_toolkits.mplot3d import Axes3D
import tifffile
from skimage.draw import circle_perimeter
from scipy import optimize
import xml.etree.ElementTree as et
from scipy.optimize import leastsq
from datetime import datetime
import cellProperties as cell
import utils as util

pd.options.mode.chained_assignment = None
plt.rcParams.update({"font.size": 16})

# -------------------


def dist(polygon, polygon0):
    [x1, y1] = cell.centroid(polygon)
    [x0, y0] = cell.centroid(polygon0)
    return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5


def angleDiff(theta, phi):

    diff = theta - phi

    if abs(diff) > 90:
        if diff > 0:
            diff = 180 - diff
        else:
            diff = 180 + diff

    return abs(diff)


def findtcj(polygon, img):

    centroid = cell.centroid(polygon)
    x, y = int(centroid[0]), int(centroid[1])
    img = 1 - img / 255
    img = np.asarray(img, "uint8")

    imgLabel = sm.measure.label(img, background=0, connectivity=1)
    label = imgLabel[x, y]
    contour = sm.measure.find_contours(imgLabel == label, level=0)[0]

    # imgLabelrc = util.imgxyrc(imgLabel)
    # imgLabelrc[imgLabelrc == label] = round(1.25 * imgLabelrc.max())
    # imgLabelrc = np.asarray(imgLabelrc, "uint16")
    # tifffile.imwrite(f"results/imgLabel{filename}.tif", imgLabelrc)

    if label == 0:
        print("label == 0")

    zeros = np.zeros([512, 512])

    zeros[imgLabel == label] = 1
    for con in contour:
        zeros[int(con[0]), int(con[1])] = 1

    struct2 = sp.ndimage.generate_binary_structure(2, 2)
    dilation = sp.ndimage.morphology.binary_dilation(zeros, structure=struct2).astype(
        zeros.dtype
    )
    dilation[zeros == 1] = 0
    # dilationrc = util.imgxyrc(dilation)
    # dilationrc = np.asarray(dilationrc, "uint16")
    # tifffile.imwrite(f"results/dilation{filename}.tif", dilationrc)

    tcj = np.zeros([512, 512])
    diff = img - dilation
    tcj[diff == -1] = 1
    tcj[tcj != 1] = 0

    outerTCJ = skimage.feature.peak_local_max(tcj)
    # tcjrc = util.imgxyrc(tcj)
    # tcjrc = np.asarray(tcjrc, "uint16")
    # tifffile.imwrite(f"results/tcj{filename}.tif", tcjrc)

    tcj = []
    for coord in outerTCJ:
        tcj.append(findtcjContour(coord, contour[0:-1]))

    if "False" in tcj:
        tcj.remove("False")
        print("removed")

    return tcj


def isBoundary(contour):

    boundary = False

    for con in contour:
        if con[0] == 0:
            boundary = True
        if con[1] == 0:
            boundary = True
        if con[0] == 511:
            boundary = True
        if con[1] == 511:
            boundary = True

    return boundary


def findtcjContour(coord, contour):

    close = []
    for con in contour:
        r = ((con[0] - coord[0]) ** 2 + (con[1] - coord[1]) ** 2) ** 0.5
        if r < 1.5:
            close.append(con)

    if len(close) == 1:
        tcj = close[0]
    elif len(close) == 0:
        tcj = "False"
    else:
        tcj = np.mean(close, axis=0)

    return tcj


def getSecondColour(track, colour):
    colours = track[np.all((track - colour) != 0, axis=1)]
    colours = colours[np.all((colours - np.array([255, 255, 255])) != 0, axis=1)]

    col = []
    count = []
    while len(colours) > 0:
        col.append(colours[0])
        count.append(len(colours[np.all((colours - colours[0]) == 0, axis=1)]))
        colours = colours[np.all((colours - colours[0]) != 0, axis=1)]

    maxm = np.max(count)
    colourD = col[count.index(maxm)]

    return colourD


def maskQ(mask):
    S = np.zeros([2, 2])
    X, Y = mask.shape
    x = np.zeros([X, Y])
    y = np.zeros([X, Y])
    x += np.arange(X)
    y += (Y - 1 - np.arange(Y)).reshape(Y, 1)
    A = np.sum(mask)
    Cx = np.sum(x * mask) / A
    Cy = np.sum(y * mask) / A
    xx = (x - Cx) ** 2
    yy = (y - Cy) ** 2
    xy = (x - Cx) * (y - Cy)
    S[0, 0] = -np.sum(yy * mask) / A**2
    S[1, 0] = S[0, 1] = np.sum(xy * mask) / A**2
    S[1, 1] = -np.sum(xx * mask) / A**2
    TrS = S[0, 0] + S[1, 1]
    I = np.zeros(shape=(2, 2))
    I[0, 0] = 1
    I[1, 1] = 1
    q = S - TrS * I / 2

    return q, Cx, Cy


# -------------------

filenames, fileType = util.getFilesType()
scale = 123.26 / 512
groupTitle = "wild type"


# area of parent dividing cells
if False:
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfDivisionTrack = dfDivisionTrack[dfDivisionTrack["Type"] == "parent"]
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    T = np.max(dfShape["T"])
    time = list(
        np.linspace(
            np.min(dfDivisionTrack["Division Time"]),
            np.max(dfDivisionTrack["Division Time"]),
            int(
                np.max(dfDivisionTrack["Division Time"])
                - np.min(dfDivisionTrack["Division Time"])
                + 1
            ),
        )
    )
    area = [[] for col in range(len(time))]
    dArea = [[] for col in range(len(time))]
    for filename in filenames:
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        dfFileShape = dfShape[dfShape["Filename"] == filename]
        areaT = []
        for t in range(T):
            areaT.append(np.mean(dfFileShape["Area"][dfFileShape["T"] == t]))

        for i in range(len(df)):

            t = df["Time"].iloc[i]
            if t < T:
                divTime = df["Division Time"].iloc[i]
                index = time.index(divTime)
                area[index].append(df["Area"].iloc[i] * scale**2)
                dArea[index].append(df["Area"].iloc[i] * scale**2 - areaT[t])

    std = []
    dAreastd = []
    for i in range(len(area)):
        std.append(np.std(area[i]))
        area[i] = np.mean(area[i])
        dAreastd.append(np.std(dArea[i]))
        dArea[i] = np.mean(dArea[i])
    time = 2 * np.array(time)

    colour, mark = util.getColorLineMarker(fileType, groupTitle)
    area = np.array(area)
    std = np.array(std)
    dArea = np.array(dArea)
    dAreastd = np.array(dAreastd)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(time, area, color=colour, marker=mark, markevery=3)
    ax[0].fill_between(time, area - std, area + std, alpha=0.15, color=colour)
    ax[0].set(xlabel=r"Time (mins)", ylabel=r"$A$ $(\mu m^2$)")
    ax[0].title.set_text(r"$A$ during division")
    ax[0].set_ylim([-5, 55])

    ax[1].plot(time, dArea, color=colour, marker=mark, markevery=3)
    ax[1].fill_between(
        time, dArea - dAreastd, dArea + dAreastd, alpha=0.15, color=colour
    )
    ax[1].set(xlabel=r"Time (mins)", ylabel=r"$\delta A$ $(\mu m^2$)")
    ax[1].title.set_text(r"$\delta A$ during division")
    ax[1].set_ylim([-5, 55])

    plt.subplots_adjust(wspace=0.5)
    fig.savefig(
        f"results/Area division {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# shape (Q0) of parent dividing cells
if False:
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfDivisionTrack = dfDivisionTrack[dfDivisionTrack["Type"] == "parent"]
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    T = np.max(dfShape["T"])
    time = list(
        np.linspace(
            np.min(dfDivisionTrack["Division Time"]),
            np.max(dfDivisionTrack["Division Time"]),
            int(
                np.max(dfDivisionTrack["Division Time"])
                - np.min(dfDivisionTrack["Division Time"])
                + 1
            ),
        )
    )
    Q0 = [[] for col in range(len(time))]
    dQ0 = [[] for col in range(len(time))]
    for filename in filenames:
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        dfFileShape = dfShape[dfShape["Filename"] == filename]
        Q0_T = []
        for t in range(T):
            Q0_T.append(
                np.mean(
                    (
                        np.stack(
                            np.array(dfFileShape["q"][dfFileShape["T"] == t]), axis=0
                        )[:, 0, 0]
                        ** 2
                        + np.stack(
                            np.array(dfFileShape["q"][dfFileShape["T"] == t]), axis=0
                        )[:, 0, 1]
                        ** 2
                    )
                    ** 0.5
                )
            )

        for i in range(len(df)):

            t = df["Time"].iloc[i]
            if t < T:
                divTime = df["Division Time"].iloc[i]
                index = time.index(divTime)
                Q0[index].append(df["q0"].iloc[i])
                dQ0[index].append(df["q0"].iloc[i] - Q0_T[t])

    std = []
    dQ0std = []
    for i in range(len(Q0)):
        std.append(np.std(Q0[i]))
        Q0[i] = np.mean(Q0[i])
        dQ0std.append(np.std(dQ0[i]))
        dQ0[i] = np.mean(dQ0[i])
    time = 2 * np.array(time)

    colour, mark = util.getColorLineMarker(fileType, groupTitle)
    Q0 = np.array(Q0)
    std = np.array(std)
    dQ0 = np.array(dQ0)
    dQ0std = np.array(dQ0std)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(time, Q0, color=colour, marker=mark, markevery=3)
    ax[0].fill_between(time, Q0 - std, Q0 + std, alpha=0.15, color=colour)
    ax[0].set(xlabel=r"Time (mins)", ylabel=r"$Q^{(0)}$")
    ax[0].title.set_text(r"$Q^{(0)}$ during division")
    ax[0].set_ylim([-0.03, 0.08])

    ax[1].plot(time, dQ0, color=colour, marker=mark, markevery=3)
    ax[1].fill_between(time, dQ0 - dQ0std, dQ0 + dQ0std, alpha=0.15, color=colour)
    ax[1].set(xlabel=r"Time (mins)", ylabel=r"$\delta Q^{(0)}$")
    ax[1].title.set_text(r"$\delta Q^{(0)}$ during division")
    ax[1].set_ylim([-0.04, 0.04])

    plt.subplots_adjust(wspace=0.5)
    fig.savefig(
        f"results/Q0 division {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# shape (Q1) of parent dividing cells
if False:
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfDivisionTrack = dfDivisionTrack[dfDivisionTrack["Type"] == "parent"]
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    T = np.max(dfShape["T"])
    time = list(
        np.linspace(
            np.min(dfDivisionTrack["Division Time"]),
            np.max(dfDivisionTrack["Division Time"]),
            int(
                np.max(dfDivisionTrack["Division Time"])
                - np.min(dfDivisionTrack["Division Time"])
                + 1
            ),
        )
    )
    Q1 = [[] for col in range(len(time))]
    dQ1 = [[] for col in range(len(time))]
    for filename in filenames:
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        dfFileShape = dfShape[dfShape["Filename"] == filename]
        Q1_T = []
        for t in range(T):
            Q1_T.append(
                np.mean(
                    (
                        np.stack(
                            np.array(dfFileShape["q"][dfFileShape["T"] == t]), axis=0
                        )[:, 0, 0]
                    )
                )
            )

        for i in range(len(df)):

            t = df["Time"].iloc[i]
            if t < T:
                divTime = df["Division Time"].iloc[i]
                index = time.index(divTime)
                Q1[index].append(df["q"].iloc[i][0, 0])
                dQ1[index].append(df["q"].iloc[i][0, 0] - Q1_T[t])

    std = []
    dQ1std = []
    for i in range(len(Q1)):
        std.append(np.std(Q1[i]))
        Q1[i] = np.mean(Q1[i])
        dQ1std.append(np.std(dQ1[i]))
        dQ1[i] = np.mean(dQ1[i])
    time = 2 * np.array(time)

    colour, mark = util.getColorLineMarker(fileType, groupTitle)
    Q1 = np.array(Q1)
    std = np.array(std)
    dQ1 = np.array(dQ1)
    dQ1std = np.array(dQ1std)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(time, Q1, color=colour, marker=mark, markevery=3)
    ax[0].fill_between(time, Q1 - std, Q1 + std, alpha=0.15, color=colour)
    ax[0].set(xlabel=r"Time (mins)", ylabel=r"$Q^{(1)}$")
    ax[0].title.set_text(r"$Q^{(1)}$ during division")
    ax[0].set_ylim([-0.02, 0.05])

    ax[1].plot(time, dQ1, color=colour, marker=mark, markevery=3)
    ax[1].fill_between(time, dQ1 - dQ1std, dQ1 + dQ1std, alpha=0.15, color=colour)
    ax[1].set(xlabel=r"Time (mins)", ylabel=r"$\delta Q^{(1)}$")
    ax[1].title.set_text(r"$\delta Q^{(1)}$ during division")
    ax[1].set_ylim([-0.04, 0.04])

    plt.subplots_adjust(wspace=0.5)
    fig.savefig(
        f"results/Q1 division {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# Area of daughter cells
if False:
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfDivisionTrack = dfDivisionTrack[dfDivisionTrack["Type"] != "parent"]
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    T = np.max(dfShape["T"])
    time = list(
        np.linspace(
            np.min(dfDivisionTrack["Division Time"]),
            np.max(dfDivisionTrack["Division Time"]),
            int(
                np.max(dfDivisionTrack["Division Time"])
                - np.min(dfDivisionTrack["Division Time"])
                + 1
            ),
        )
    )
    area = [[] for col in range(len(time))]
    dArea = [[] for col in range(len(time))]
    for filename in filenames:
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        dfFileShape = dfShape[dfShape["Filename"] == filename]
        areaT = []
        for t in range(T):
            areaT.append(np.mean(dfFileShape["Area"][dfFileShape["T"] == t]))

        for i in range(len(df)):

            t = df["Time"].iloc[i]
            if t < T:
                divTime = df["Division Time"].iloc[i]
                index = time.index(divTime)
                area[index].append(df["Area"].iloc[i] * scale**2)
                dArea[index].append(df["Area"].iloc[i] * scale**2 - areaT[t])

    std = []
    dAreastd = []
    for i in range(len(area)):
        std.append(np.std(area[i]))
        area[i] = np.mean(area[i])
        dAreastd.append(np.std(dArea[i]))
        dArea[i] = np.mean(dArea[i])
    time = 2 * np.array(time)

    colour, mark = util.getColorLineMarker(fileType, groupTitle)
    area = np.array(area)
    std = np.array(std)
    dArea = np.array(dArea)
    dAreastd = np.array(dAreastd)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(time, area, color=colour, marker=mark, markevery=3)
    ax[0].fill_between(time, area - std, area + std, alpha=0.15, color=colour)
    ax[0].set(xlabel=r"Time (mins)", ylabel=r"$A$ $(\mu m^2$)")
    ax[0].title.set_text(r"$A$ after division")
    ax[0].set_ylim([-10, 55])

    ax[1].plot(time, dArea, color=colour, marker=mark, markevery=3)
    ax[1].fill_between(
        time, dArea - dAreastd, dArea + dAreastd, alpha=0.15, color=colour
    )
    ax[1].set(xlabel=r"Time (mins)", ylabel=r"$\delta A$ $(\mu m^2$)")
    ax[1].title.set_text(r"$\delta A$ after division")
    ax[1].set_ylim([-10, 55])

    plt.subplots_adjust(wspace=0.5)
    fig.savefig(
        f"results/Area Daughter Cell {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# shape (Q0) of daughter cells
if False:
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfDivisionTrack = dfDivisionTrack[dfDivisionTrack["Type"] != "parent"]
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    T = np.max(dfShape["T"])
    time = list(
        np.linspace(
            np.min(dfDivisionTrack["Division Time"]),
            np.max(dfDivisionTrack["Division Time"]),
            int(
                np.max(dfDivisionTrack["Division Time"])
                - np.min(dfDivisionTrack["Division Time"])
                + 1
            ),
        )
    )
    Q0 = [[] for col in range(len(time))]
    dQ0 = [[] for col in range(len(time))]
    for filename in filenames:
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        dfFileShape = dfShape[dfShape["Filename"] == filename]
        Q0_T = []
        for t in range(T):
            Q0_T.append(
                np.mean(
                    (
                        np.stack(
                            np.array(dfFileShape["q"][dfFileShape["T"] == t]), axis=0
                        )[:, 0, 0]
                        ** 2
                        + np.stack(
                            np.array(dfFileShape["q"][dfFileShape["T"] == t]), axis=0
                        )[:, 0, 1]
                        ** 2
                    )
                    ** 0.5
                )
            )

        for i in range(len(df)):

            t = df["Time"].iloc[i]
            if t < T:
                divTime = df["Division Time"].iloc[i]
                index = time.index(divTime)
                Q0[index].append(df["q0"].iloc[i])
                dQ0[index].append(df["q0"].iloc[i] - Q0_T[t])

    std = []
    dQ0std = []
    for i in range(len(Q0)):
        std.append(np.std(Q0[i]))
        Q0[i] = np.mean(Q0[i])
        dQ0std.append(np.std(dQ0[i]))
        dQ0[i] = np.mean(dQ0[i])
    time = 2 * np.array(time)

    colour, mark = util.getColorLineMarker(fileType, groupTitle)
    Q0 = np.array(Q0)
    std = np.array(std)
    dQ0 = np.array(dQ0)
    dQ0std = np.array(dQ0std)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(time, Q0, color=colour, marker=mark, markevery=3)
    ax[0].fill_between(time, Q0 - std, Q0 + std, alpha=0.15, color=colour)
    ax[0].set(xlabel=r"Time (mins)", ylabel=r"$Q^{(0)}$")
    ax[0].title.set_text(r"$Q^{(0)}$ after division")
    ax[0].set_ylim([-0.03, 0.08])

    ax[1].plot(time, dQ0, color=colour, marker=mark, markevery=3)
    ax[1].fill_between(time, dQ0 - dQ0std, dQ0 + dQ0std, alpha=0.15, color=colour)
    ax[1].set(xlabel=r"Time (mins)", ylabel=r"$\delta Q^{(0)}$")
    ax[1].title.set_text(r"$\delta Q^{(0)}$ after division")
    ax[1].set_ylim([-0.04, 0.04])

    plt.subplots_adjust(wspace=0.5)
    fig.savefig(
        f"results/Q0 Daughter Cell {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# shape (Q1) of daughter cells
if False:
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfDivisionTrack = dfDivisionTrack[dfDivisionTrack["Type"] != "parent"]
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    T = np.max(dfShape["T"])
    time = list(
        np.linspace(
            np.min(dfDivisionTrack["Division Time"]),
            np.max(dfDivisionTrack["Division Time"]),
            int(
                np.max(dfDivisionTrack["Division Time"])
                - np.min(dfDivisionTrack["Division Time"])
                + 1
            ),
        )
    )
    Q1 = [[] for col in range(len(time))]
    dQ1 = [[] for col in range(len(time))]
    for filename in filenames:
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        dfFileShape = dfShape[dfShape["Filename"] == filename]
        Q1_T = []
        for t in range(T):
            Q1_T.append(
                np.mean(
                    (
                        np.stack(
                            np.array(dfFileShape["q"][dfFileShape["T"] == t]), axis=0
                        )[:, 0, 0]
                    )
                )
            )

        for i in range(len(df)):

            t = df["Time"].iloc[i]
            if t < T:
                divTime = df["Division Time"].iloc[i]
                index = time.index(divTime)
                Q1[index].append(df["q"].iloc[i][0, 0])
                dQ1[index].append(df["q"].iloc[i][0, 0] - Q1_T[t])

    std = []
    dQ1std = []
    for i in range(len(Q1)):
        std.append(np.std(Q1[i]))
        Q1[i] = np.mean(Q1[i])
        dQ1std.append(np.std(dQ1[i]))
        dQ1[i] = np.mean(dQ1[i])
    time = 2 * np.array(time)

    colour, mark = util.getColorLineMarker(fileType, groupTitle)
    Q1 = np.array(Q1)
    std = np.array(std)
    dQ1 = np.array(dQ1)
    dQ1std = np.array(dQ1std)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(time, Q1, color=colour, marker=mark, markevery=3)
    ax[0].fill_between(time, Q1 - std, Q1 + std, alpha=0.15, color=colour)
    ax[0].set(xlabel=r"Time (mins)", ylabel=r"$Q^{(1)}$")
    ax[0].title.set_text(r"$Q^{(1)}$ after division")
    ax[0].set_ylim([-0.02, 0.05])

    ax[1].plot(time, dQ1, color=colour, marker=mark, markevery=3)
    ax[1].fill_between(time, dQ1 - dQ1std, dQ1 + dQ1std, alpha=0.15, color=colour)
    ax[1].set(xlabel=r"Time (mins)", ylabel=r"$\delta Q^{(1)}$")
    ax[1].title.set_text(r"$\delta Q^{(1)}$ after division")
    ax[1].set_ylim([-0.04, 0.04])

    plt.subplots_adjust(wspace=0.5)
    fig.savefig(
        f"results/Q1 Daughter Cell {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# difference in orientation of division and cell shape (whole shape/tcj shape)
if False:
    dfDivisionShape = pd.read_pickle(f"databases/dfDivisionShape{fileType}.pkl")
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfDivisionTrack = dfDivisionTrack[dfDivisionTrack["Type"] == "parent"]
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    T = np.max(dfShape["T"])
    time = list(
        np.linspace(
            np.min(dfDivisionTrack["Division Time"]),
            np.max(dfDivisionTrack["Division Time"]),
            int(
                np.max(dfDivisionTrack["Division Time"])
                - np.min(dfDivisionTrack["Division Time"])
                + 1
            ),
        )
    )
    diff = []
    sf = []
    q0 = []
    for filename in filenames:
        dfFileShape = dfDivisionShape[dfDivisionShape["Filename"] == filename]
        dfFileShape = dfFileShape[dfFileShape["Track length"] > 13]
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        labels = list(dfFileShape["Label"])
        for label in labels:
            dfDiv = df[df["Label"] == label]
            tcjs = dfDiv["TCJ"][dfDiv["Division Time"] == -10].iloc[0]
            if tcjs != False:
                ori = dfFileShape["Orientation"][dfFileShape["Label"] == label].iloc[0]
                oriPre = dfDiv["Orientation"][dfDiv["Division Time"] == -10].iloc[0]
                diff.append(angleDiff(ori, oriPre))
                sf.append(dfDiv["Shape Factor"][dfDiv["Division Time"] == -10].iloc[0])
                q0.append(dfDiv["q0"][dfDiv["Division Time"] == -10].iloc[0])

    heatmap = np.histogram2d(
        diff,
        sf,
        range=[[0, 90], [0, 1]],
        bins=(9, 10),
    )[0]
    x, y = np.mgrid[5:95:10, 0.05:1.05:0.1]

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    colour, mark = util.getColorLineMarker(fileType, groupTitle)

    ax[0].hist(diff, 9, density=True, color=colour)
    ax[0].set(xlabel=r"$|\theta_{d} - \theta_{A}|$", ylabel=r"density")
    ax[0].set_ylim([0, 0.02])

    c = ax[1].pcolor(
        x,
        y,
        heatmap / np.sum(heatmap),
        vmin=0,
        vmax=0.04,
        cmap="Reds",
    )
    fig.colorbar(c, ax=ax[1])
    ax[1].set(xlabel=r"$|\theta_{d} - \theta_{A}|$", ylabel=r"Area $S_f$")
    ax[1].set_ylim([0, 1])

    plt.subplots_adjust(
        left=0.075, bottom=0.1, right=0.95, top=0.9, wspace=0.5, hspace=0.4
    )
    fig.savefig(
        f"results/Orientation division {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

    # orientation tcj of parent dividing cells
    dfDivisionShape = pd.read_pickle(f"databases/dfDivisionShape{fileType}.pkl")
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfDivisionTrack = dfDivisionTrack[dfDivisionTrack["Type"] == "parent"]
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    T = np.max(dfShape["T"])
    time = list(
        np.linspace(
            np.min(dfDivisionTrack["Division Time"]),
            np.max(dfDivisionTrack["Division Time"]),
            int(
                np.max(dfDivisionTrack["Division Time"])
                - np.min(dfDivisionTrack["Division Time"])
                + 1
            ),
        )
    )
    diff = []
    sf_tcj = []
    q0_tcj = []
    for filename in filenames:
        dfFileShape = dfDivisionShape[dfDivisionShape["Filename"] == filename]
        dfFileShape = dfFileShape[dfFileShape["Track length"] > 13]
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        labels = list(dfFileShape["Label"])
        for label in labels:
            dfDiv = df[df["Label"] == label]
            tcjs = dfDiv["TCJ"][dfDiv["Division Time"] == -10].iloc[0]
            if tcjs != False:
                ori = dfFileShape["Orientation"][dfFileShape["Label"] == label].iloc[0]
                oriPre = dfDiv["Orientation tcj"][dfDiv["Division Time"] == -10].iloc[0]
                diff.append(angleDiff(ori, oriPre))
                sf_tcj.append(
                    dfDiv["Shape Factor tcj"][dfDiv["Division Time"] == -10].iloc[0]
                )
                q0_tcj.append(
                    dfDiv["q0_tcj"][dfDiv["Division Time"] == -10].iloc[0]
                    / (
                        dfDiv["Polygon"][dfDiv["Division Time"] == -10].iloc[0].area
                        ** 2
                    )
                )

    heatmap = np.histogram2d(
        diff,
        sf_tcj,
        range=[[0, 90], [0, 1]],
        bins=(9, 10),
    )[0]

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].hist(diff, 9, density=True, color=colour)
    ax[0].set(xlabel=r"$|\theta_{d} - \theta_{tcj}|$", ylabel=r"density")
    ax[0].set_ylim([0, 0.02])

    c = ax[1].pcolor(
        x,
        y,
        heatmap / np.sum(heatmap),
        vmin=0,
        vmax=0.04,
        cmap="Reds",
    )
    fig.colorbar(c, ax=ax[1])
    ax[1].set(xlabel=r"$|\theta_{d} - \theta_{tcj}|$", ylabel=r"tcj $S_f$")

    plt.subplots_adjust(
        left=0.075, bottom=0.1, right=0.95, top=0.9, wspace=0.5, hspace=0.4
    )
    fig.savefig(
        f"results/Orientation tcj division {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].scatter(sf, sf_tcj, color=colour)
    ax[0].set(xlabel=r"Area $S_f$", ylabel=r"tcj $S_f$")
    ax[0].set_xlim([0, 1])
    ax[0].set_ylim([0, 1])

    ax[1].scatter(q0, q0_tcj, color=colour)
    ax[1].set(xlabel=r"Area $q_0$", ylabel=r"tcj $q_0$")

    plt.subplots_adjust(
        left=0.075, bottom=0.1, right=0.95, top=0.9, wspace=0.5, hspace=0.4
    )
    fig.savefig(
        f"results/s_f and q_0 correlation tcj shape division {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# |\theta_{tcj} - \theta_{s}| > 15 compare different in orientation
if True:
    dfDivisionShape = pd.read_pickle(f"databases/dfDivisionShape{fileType}.pkl")
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfDivisionTrack = dfDivisionTrack[dfDivisionTrack["Type"] == "parent"]
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")

    oriDiff = []
    oriDiff_tcj = []
    oriDiff_sfL = []
    oriDiff_tcj_sfL = []
    oriDiff_sfH = []
    oriDiff_tcj_sfH = []
    tcj_n = []
    for filename in filenames:
        dfFileShape = dfDivisionShape[dfDivisionShape["Filename"] == filename]
        dfFileShape = dfFileShape[dfFileShape["Track length"] > 18]
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        labels = list(dfFileShape["Label"])
        for label in labels:
            dfDiv = df[df["Label"] == label]

            tcj_n.append(len(dfDiv["TCJ"][dfDiv["Division Time"] == -15].iloc[0]))
            ori = dfFileShape["Orientation"][dfFileShape["Label"] == label].iloc[0]
            oriPre = dfDiv["Orientation"][dfDiv["Division Time"] == -15].iloc[0]
            oriPre_tcj = dfDiv["Orientation tcj"][dfDiv["Division Time"] == -15].iloc[0]
            sf = dfDiv["Shape Factor tcj"][dfDiv["Division Time"] == -15].iloc[0]
            if angleDiff(oriPre, oriPre_tcj) > 15:
                oriDiff.append(angleDiff(ori, oriPre))
                oriDiff_tcj.append(angleDiff(ori, oriPre_tcj))
                if sf < 0.2:
                    oriDiff_sfL.append(angleDiff(ori, oriPre))
                    oriDiff_tcj_sfL.append(angleDiff(ori, oriPre_tcj))
                else:
                    oriDiff_sfH.append(angleDiff(ori, oriPre))
                    oriDiff_tcj_sfH.append(angleDiff(ori, oriPre_tcj))

    colour, mark = util.getColorLineMarker(fileType, groupTitle)
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    ax[0, 0].hist(oriDiff, 9, density=True, color=colour)
    ax[0, 0].axvline(np.median(oriDiff), color="b")
    ax[0, 0].set(xlabel=r"$|\theta_{d} - \theta_{A}|$", ylabel=r"Density")
    ax[0, 0].title.set_text(r"$|\theta_{tcj} - \theta_{A}| > 15$")
    ax[0, 0].set_ylim([0, 0.026])

    ax[1, 0].hist(oriDiff_tcj, 9, density=True, color=colour)
    ax[1, 0].axvline(np.median(oriDiff_tcj), color="b")
    ax[1, 0].set(xlabel=r"$|\theta_{d} - \theta_{tcj}|$", ylabel=r"Density")
    ax[1, 0].title.set_text(r"$|\theta_{tcj} - \theta_{A}| > 15$")
    ax[1, 0].set_ylim([0, 0.026])

    ax[0, 1].hist(oriDiff_sfL, 9, density=True, color=colour)
    ax[0, 1].axvline(np.median(oriDiff_sfL), color="b")
    ax[0, 1].set(xlabel=r"$|\theta_{d} - \theta_{A}|$", ylabel=r"Density")
    ax[0, 1].title.set_text(
        r"$|\theta_{tcj} - \theta_{A}| > 15$" + "\n and " + r"Area $S_f<0.2$"
    )
    ax[0, 1].set_ylim([0, 0.026])

    ax[1, 1].hist(oriDiff_tcj_sfL, 9, density=True, color=colour)
    ax[1, 1].axvline(np.median(oriDiff_tcj_sfL), color="b")
    ax[1, 1].set(xlabel=r"$|\theta_{d} - \theta_{tcj}|$", ylabel=r"Density")
    ax[1, 1].title.set_text(
        r"$|\theta_{tcj} - \theta_{A}| > 15$" + "\n and " + r"Area $S_f<0.2$"
    )
    ax[1, 1].set_ylim([0, 0.026])

    ax[0, 2].hist(oriDiff_sfH, 9, density=True, color=colour)
    ax[0, 2].axvline(np.median(oriDiff_sfH), color="b")
    ax[0, 2].set(xlabel=r"$|\theta_{d} - \theta_{A}|$", ylabel=r"Density")
    ax[0, 2].title.set_text(
        r"$|\theta_{tcj} - \theta_{A}| > 15$" + "\n and " + r"Area $S_f>0.2$"
    )
    ax[0, 2].set_ylim([0, 0.026])

    ax[1, 2].hist(oriDiff_tcj_sfH, 9, density=True, color=colour)
    ax[1, 2].axvline(np.median(oriDiff_tcj_sfH), color="b")
    ax[1, 2].set(xlabel=r"$|\theta_{d} - \theta_{tcj}|$", ylabel=r"Density")
    ax[1, 2].title.set_text(
        r"$|\theta_{tcj} - \theta_{A}| > 15$" + "\n and " + r"Area $S_f>0.2$"
    )
    ax[1, 2].set_ylim([0, 0.026])

    plt.subplots_adjust(
        left=0.075, bottom=0.1, right=0.95, top=0.9, wspace=0.4, hspace=0.5
    )
    fig.savefig(
        f"results/Orientation division diff when shape ori disagree {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# compare properties of good and poor predictions
if False:
    dfDivisionShape = pd.read_pickle(f"databases/dfDivisionShape{fileType}.pkl")
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfDivisionTrack = dfDivisionTrack[dfDivisionTrack["Type"] == "parent"]
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")

    oriShapeDiff = []
    oriTissueDiff = []
    shapeTissueDiff = []
    dQ1 = []
    area = []
    time = []
    oriShapeDiffAll = []
    oriTissueDiffAll = []
    shapeTissueDiffAll = []
    dQ1All = []
    areaAll = []
    timeAll = []
    for filename in filenames:
        dfTissue = dfShape[dfShape["Filename"] == filename]
        Q = np.mean(dfTissue["q"])
        theta0 = (0.5 * np.arctan2(Q[1, 0], Q[0, 0])) * 180 / np.pi
        dfFileShape = dfDivisionShape[dfDivisionShape["Filename"] == filename]
        dfFileShape = dfFileShape[dfFileShape["Track length"] > 18]
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        labels = list(dfFileShape["Label"])
        for label in labels:
            dfDiv = df[df["Label"] == label]
            tcjs = dfDiv["TCJ"][dfDiv["Division Time"] == -15].iloc[0]
            if tcjs != False:
                ori = dfFileShape["Orientation"][dfFileShape["Label"] == label].iloc[0]
                oriPre = dfDiv["Orientation"][dfDiv["Division Time"] == -15].iloc[0]
                if angleDiff(oriPre, ori) > 45:
                    oriShapeDiff.append(angleDiff(oriPre, ori))
                    oriTissueDiff.append(angleDiff(theta0, ori))
                    shapeTissueDiff.append(angleDiff(theta0, oriPre))
                    dQ1.append(
                        (
                            cell.qTensor(
                                dfDiv["Polygon"][dfDiv["Division Time"] == -15].iloc[0]
                            )
                            - Q
                        )[0, 0]
                    )
                    area.append(
                        dfDiv["Polygon"][dfDiv["Division Time"] == -15].iloc[0].area
                        * scale**2
                    )
                    time.append(
                        dfDiv["Time"][dfDiv["Division Time"] == -15].iloc[0] * 2
                    )
                else:
                    oriShapeDiffAll.append(angleDiff(oriPre, ori))
                    oriTissueDiffAll.append(angleDiff(theta0, ori))
                    shapeTissueDiffAll.append(angleDiff(theta0, oriPre))
                    dQ1All.append(
                        (
                            cell.qTensor(
                                dfDiv["Polygon"][dfDiv["Division Time"] == -15].iloc[0]
                            )
                            - Q
                        )[0, 0]
                    )
                    areaAll.append(
                        dfDiv["Polygon"][dfDiv["Division Time"] == -15].iloc[0].area
                        * scale**2
                    )
                    timeAll.append(
                        dfDiv["Time"][dfDiv["Division Time"] == -15].iloc[0] * 2
                    )

    fig, ax = plt.subplots(2, 6, figsize=(30, 10))

    ax[0, 0].hist(oriShapeDiff, 5)
    ax[0, 0].set(xlabel=r"$|\theta_{d} - \theta_{s}|$", ylabel=r"number")
    ax[0, 0].set_ylim([0, 220])
    ax[0, 0].set_xlim([0, 90])

    ax[0, 1].hist(oriTissueDiff, 9)
    ax[0, 1].set(xlabel=r"$|\theta_{d} - \theta_{w}|$", ylabel=r"number")
    ax[0, 1].set_ylim([0, 110])

    ax[0, 2].hist(shapeTissueDiff, 9)
    ax[0, 2].set(xlabel=r"$|\theta_{s} - \theta_{w}|$", ylabel=r"number")
    ax[0, 2].set_ylim([0, 110])

    ax[0, 3].hist(dQ1, 9)
    ax[0, 3].set(xlabel=r"$\delta Q^1$ of poor predictions", ylabel=r"number")
    ax[0, 3].set_xlim([-0.08, 0.08])
    ax[0, 3].axvline(np.median(dQ1), c="r", label="median")

    ax[0, 4].hist(area, 9)
    ax[0, 4].set(xlabel=r"Area of poor predictions", ylabel=r"number")
    ax[0, 4].set_xlim([0, 60])

    ax[0, 5].hist(time, 9)
    ax[0, 5].set(xlabel=r"time of poor predictions", ylabel=r"number")
    ax[0, 5].set_xlim([0, 180])

    ax[1, 0].hist(oriShapeDiffAll, 5)
    ax[1, 0].set(xlabel=r"$|\theta_{d} - \theta_{s}|$", ylabel=r"number")
    ax[1, 0].set_ylim([0, 220])
    ax[1, 0].set_xlim([0, 90])

    ax[1, 1].hist(oriTissueDiffAll, 9)
    ax[1, 1].set(xlabel=r"$|\theta_{d} - \theta_{w}|$", ylabel=r"number")
    ax[1, 1].set_ylim([0, 160])

    ax[1, 2].hist(shapeTissueDiffAll, 9)
    ax[1, 2].set(xlabel=r"$|\theta_{s} - \theta_{w}|$", ylabel=r"number")
    ax[1, 2].set_ylim([0, 160])

    ax[1, 3].hist(dQ1All, 9)
    ax[1, 3].set(xlabel=r"$\delta Q^1$ of good predictions", ylabel=r"number")
    ax[1, 3].set_xlim([-0.08, 0.08])
    ax[1, 3].axvline(np.median(dQ1All), c="r", label="median")

    ax[1, 4].hist(areaAll, 9)
    ax[1, 4].set(xlabel=r"Area of good predictions", ylabel=r"number")
    ax[1, 4].set_xlim([0, 60])

    ax[1, 5].hist(timeAll, 9)
    ax[1, 5].set(xlabel=r"time of good predictions", ylabel=r"number")
    ax[1, 5].set_xlim([0, 180])

    fig.savefig(
        f"results/poor predictions and tissue orientation {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# orientation of combined daughter cells relative to tissue
if False:
    dfDivisionShape = pd.read_pickle(f"databases/dfDivisionShape{fileType}.pkl")
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    T = np.max(dfShape["T"])
    time = list(np.linspace(-10, 10, 21))
    dq = [[] for col in range(len(time))]
    for filename in filenames:
        tracks = sm.io.imread(f"dat/{filename}/binaryTracks{filename}.tif").astype(int)
        dfFileShape = dfDivisionShape[dfDivisionShape["Filename"] == filename]
        Q = np.mean(dfShape["q"][dfShape["Filename"] == filename])
        theta0 = 0.5 * np.arctan2(Q[1, 0], Q[0, 0])
        R = util.rotation_matrix(-theta0)

        dfFileShape = dfFileShape[dfFileShape["Daughter length"] > 10]
        dfFileShape = dfFileShape[dfFileShape["Track length"] > 13]
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        labels = list(dfFileShape["Label"])

        for label in labels:
            dfDiv = df[df["Label"] == label]
            polygon = dfDiv["Polygon"][dfDiv["Division Time"] == time[0]].iloc[0]
            q0 = np.matmul(R, np.matmul(cell.qTensor(polygon), np.matrix.transpose(R)))
            for t in time:
                if dfDiv["Type"][dfDiv["Division Time"] == t].iloc[0] == "parent":
                    polygon = dfDiv["Polygon"][dfDiv["Division Time"] == t].iloc[0]
                    q = np.matmul(
                        R, np.matmul(cell.qTensor(polygon), np.matrix.transpose(R))
                    )
                    dq[time.index(t)].append(q - q0)
                else:
                    T = dfDiv["Time"][dfDiv["Division Time"] == t].iloc[0]
                    colour1 = dfDiv["Colour"][dfDiv["Division Time"] == t].iloc[0]
                    colour2 = dfDiv["Colour"][dfDiv["Division Time"] == t].iloc[1]
                    mask = np.zeros([512, 512])
                    mask[np.all((tracks[int(T)] - colour1) == 0, axis=2)] = 1
                    mask[np.all((tracks[int(T)] - colour2) == 0, axis=2)] = 1
                    q = np.matmul(R, np.matmul(maskQ(mask)[0], np.matrix.transpose(R)))
                    dq[time.index(t)].append(q - q0)

    dQ = []
    dQstd = []
    for i in range(len(dq)):
        dQ.append(np.mean(dq[i], axis=0))
        dQstd.append(np.std(dq[i], axis=0))

    dQ = np.array(dQ)
    dQstd = np.array(dQstd)
    time = 2 * np.array(time)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].errorbar(time, dQ[:, 0, 0], dQstd[:, 0, 0])
    ax[0].set(xlabel=r"Time (mins)", ylabel=r"$\delta Q^1$")
    ax[0].title.set_text(r"$\delta Q_1$ during division")
    ax[0].set_ylim([-0.07, 0.07])

    ax[1].errorbar(time, dQ[:, 1, 0], dQstd[:, 1, 0])
    ax[1].set(xlabel=r"Time (mins)", ylabel=r"$\delta Q^2$")
    ax[1].title.set_text(r"$\delta Q_2$ during division")
    ax[1].set_ylim([-0.07, 0.07])

    plt.subplots_adjust(wspace=0.28)
    fig.savefig(
        f"results/change in Q division relative to tissue {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# orientation of combined daughter cells relative to wound
if False:
    dfDivisionShape = pd.read_pickle(f"databases/dfDivisionShape{fileType}.pkl")
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    time = list(np.linspace(-10, 10, 21))
    dq = [[] for col in range(len(time))]
    for filename in filenames:
        tracks = sm.io.imread(f"dat/{filename}/binaryTracks{filename}.tif").astype(int)
        dfFileShape = dfDivisionShape[dfDivisionShape["Filename"] == filename]
        dist = sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)

        if "Wound" in filename:
            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")

        else:
            dfVelocityMean = pd.read_pickle(f"databases/dfVelocityMean{fileType}.pkl")
            dfFilename = dfVelocityMean[
                dfVelocityMean["Filename"] == filename
            ].reset_index()

        dfFileShape = dfFileShape[dfFileShape["Daughter length"] > 10]
        dfFileShape = dfFileShape[dfFileShape["Track length"] > 13]
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        labels = list(dfFileShape["Label"])

        for label in labels:
            dfDiv = df[df["Label"] == label]
            polygon = dfDiv["Polygon"][dfDiv["Division Time"] == time[0]].iloc[0]
            T = dfDiv["Time"][dfDiv["Division Time"] == time[0]].iloc[0]
            if "Wound" in filename:
                (xc, yc) = dfWound["Position"].iloc[T]
            else:
                mig = np.sum(
                    np.stack(np.array(dfFilename.loc[:T, "V"]), axis=0), axis=0
                )
                xc = 256 * scale - mig[0]
                yc = 256 * scale - mig[1]
            x, y = np.array(cell.centroid(polygon)) * scale
            r = dist[T, int(x / scale), int(512 - y / scale)]
            if r * scale < 30:
                phi = np.arctan2(y - yc, x - xc)
                Rw = util.rotation_matrix(-phi)
                q0 = np.matmul(
                    Rw, np.matmul(cell.qTensor(polygon), np.matrix.transpose(Rw))
                )

                for t in time:
                    if dfDiv["Type"][dfDiv["Division Time"] == t].iloc[0] == "parent":
                        polygon = dfDiv["Polygon"][dfDiv["Division Time"] == t].iloc[0]
                        T = dfDiv["Time"][dfDiv["Division Time"] == t].iloc[0]
                        if "Wound" in filename:
                            (xc, yc) = dfWound["Position"].iloc[T]
                        else:
                            mig = np.sum(
                                np.stack(np.array(dfFilename.loc[:T, "V"]), axis=0),
                                axis=0,
                            )
                            xc = 256 * scale - mig[0]
                            yc = 256 * scale - mig[1]
                        x, y = np.array(cell.centroid(polygon)) * scale
                        phi = np.arctan2(y - yc, x - xc)
                        Rw = util.rotation_matrix(-phi)

                        q = np.matmul(
                            Rw,
                            np.matmul(cell.qTensor(polygon), np.matrix.transpose(Rw)),
                        )
                        dq[time.index(t)].append(q - q0)
                    else:
                        T = dfDiv["Time"][dfDiv["Division Time"] == t].iloc[0]
                        if "Wound" in filename:
                            (xc, yc) = dfWound["Position"].iloc[T]
                        else:
                            mig = np.sum(
                                np.stack(np.array(dfFilename.loc[:T, "V"]), axis=0),
                                axis=0,
                            )
                            xc = 256 * scale - mig[0]
                            yc = 256 * scale - mig[1]

                        colour1 = dfDiv["Colour"][dfDiv["Division Time"] == t].iloc[0]
                        colour2 = dfDiv["Colour"][dfDiv["Division Time"] == t].iloc[1]
                        mask = np.zeros([512, 512])
                        mask[np.all((tracks[int(T)] - colour1) == 0, axis=2)] = 1
                        mask[np.all((tracks[int(T)] - colour2) == 0, axis=2)] = 1
                        q, x, y = maskQ(mask)
                        x, y = np.array([x, y]) * scale
                        phi = np.arctan2(y - yc, x - xc)
                        Rw = util.rotation_matrix(-phi)

                        q = np.matmul(Rw, np.matmul(q, np.matrix.transpose(Rw)))
                        dq[time.index(t)].append(q - q0)

    dQ = []
    dQstd = []
    for i in range(len(dq)):
        dQ.append(np.mean(dq[i], axis=0))
        dQstd.append(np.std(dq[i], axis=0))

    dQ = np.array(dQ)
    dQstd = np.array(dQstd)
    time = 2 * np.array(time)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].errorbar(time, dQ[:, 0, 0], dQstd[:, 0, 0])
    ax[0].set(xlabel=r"Time (mins)", ylabel=r"$\delta Q^1$")
    ax[0].title.set_text(r"$\delta Q_1$ during division")
    ax[0].set_ylim([-0.07, 0.07])

    ax[1].errorbar(time, dQ[:, 1, 0], dQstd[:, 1, 0])
    ax[1].set(xlabel=r"Time (mins)", ylabel=r"$\delta Q^2$")
    ax[1].title.set_text(r"$\delta Q_2$ during division")
    ax[1].set_ylim([-0.07, 0.07])

    plt.subplots_adjust(wspace=0.28)
    fig.savefig(
        f"results/change in Q division relative to wound {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# orientation of daughter cells relative to tissue both of combined and individual also looking at poor predictions
if False:
    dfDivisionShape = pd.read_pickle(f"databases/dfDivisionShape{fileType}.pkl")
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    time = list(np.linspace(-15, 10, 26))
    dq = [[] for col in range(len(time))]
    dq_p = [[] for col in range(len(time))]
    dq_inv = [[] for col in range(len(time))]
    dq_inv_p = [[] for col in range(len(time))]

    for filename in filenames:
        tracks = sm.io.imread(f"dat/{filename}/binaryTracks{filename}.tif").astype(int)
        dfFileShape = dfDivisionShape[dfDivisionShape["Filename"] == filename]
        Q = np.mean(dfShape["q"][dfShape["Filename"] == filename])
        theta0 = 0.5 * np.arctan2(Q[1, 0], Q[0, 0])
        R = util.rotation_matrix(-theta0)

        dfFileShape = dfFileShape[dfFileShape["Daughter length"] > 10]
        dfFileShape = dfFileShape[dfFileShape["Track length"] > 18]
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        labels = list(dfFileShape["Label"])

        for label in labels:
            dfDiv = df[df["Label"] == label]
            tcjs = dfDiv["TCJ"][dfDiv["Division Time"] == -15].iloc[0]
            if tcjs != False:
                ori = dfFileShape["Orientation"][dfFileShape["Label"] == label].iloc[0]
                oriPre = dfDiv["Orientation"][dfDiv["Division Time"] == -15].iloc[0]
                if angleDiff(oriPre, ori) > 45:

                    polygon = dfDiv["Polygon"][dfDiv["Division Time"] == time[0]].iloc[
                        0
                    ]
                    q0 = np.matmul(
                        R, np.matmul(cell.qTensor(polygon), np.matrix.transpose(R))
                    )
                    for t in time:
                        if (
                            dfDiv["Type"][dfDiv["Division Time"] == t].iloc[0]
                            == "parent"
                        ):
                            polygon = dfDiv["Polygon"][
                                dfDiv["Division Time"] == t
                            ].iloc[0]
                            q = np.matmul(
                                R,
                                np.matmul(
                                    cell.qTensor(polygon), np.matrix.transpose(R)
                                ),
                            )
                            dq_p[time.index(t)].append(q - q0)
                            dq_inv_p[time.index(t)].append(q - q0)
                        else:
                            T = dfDiv["Time"][dfDiv["Division Time"] == t].iloc[0]
                            colour1 = dfDiv["Colour"][dfDiv["Division Time"] == t].iloc[
                                0
                            ]
                            colour2 = dfDiv["Colour"][dfDiv["Division Time"] == t].iloc[
                                1
                            ]
                            mask = np.zeros([512, 512])
                            mask[np.all((tracks[int(T)] - colour1) == 0, axis=2)] = 1
                            mask[np.all((tracks[int(T)] - colour2) == 0, axis=2)] = 1
                            q = np.matmul(
                                R, np.matmul(maskQ(mask)[0], np.matrix.transpose(R))
                            )
                            dq_p[time.index(t)].append(q - q0)

                            polygon = dfDiv["Polygon"][
                                dfDiv["Division Time"] == t
                            ].iloc[0]
                            q = np.matmul(
                                R,
                                np.matmul(
                                    cell.qTensor(polygon), np.matrix.transpose(R)
                                ),
                            )
                            dq_inv_p[time.index(t)].append(q - q0)

                            polygon = dfDiv["Polygon"][
                                dfDiv["Division Time"] == t
                            ].iloc[1]
                            q = np.matmul(
                                R,
                                np.matmul(
                                    cell.qTensor(polygon), np.matrix.transpose(R)
                                ),
                            )
                            dq_inv_p[time.index(t)].append(q - q0)
                else:

                    polygon = dfDiv["Polygon"][dfDiv["Division Time"] == time[0]].iloc[
                        0
                    ]
                    q0 = np.matmul(
                        R, np.matmul(cell.qTensor(polygon), np.matrix.transpose(R))
                    )
                    for t in time:
                        if (
                            dfDiv["Type"][dfDiv["Division Time"] == t].iloc[0]
                            == "parent"
                        ):
                            polygon = dfDiv["Polygon"][
                                dfDiv["Division Time"] == t
                            ].iloc[0]
                            q = np.matmul(
                                R,
                                np.matmul(
                                    cell.qTensor(polygon), np.matrix.transpose(R)
                                ),
                            )
                            dq[time.index(t)].append(q - q0)
                            dq_inv[time.index(t)].append(q - q0)
                        else:
                            T = dfDiv["Time"][dfDiv["Division Time"] == t].iloc[0]
                            colour1 = dfDiv["Colour"][dfDiv["Division Time"] == t].iloc[
                                0
                            ]
                            colour2 = dfDiv["Colour"][dfDiv["Division Time"] == t].iloc[
                                1
                            ]
                            mask = np.zeros([512, 512])
                            mask[np.all((tracks[int(T)] - colour1) == 0, axis=2)] = 1
                            mask[np.all((tracks[int(T)] - colour2) == 0, axis=2)] = 1
                            q = np.matmul(
                                R, np.matmul(maskQ(mask)[0], np.matrix.transpose(R))
                            )
                            dq[time.index(t)].append(q - q0)

                            polygon = dfDiv["Polygon"][
                                dfDiv["Division Time"] == t
                            ].iloc[0]
                            q = np.matmul(
                                R,
                                np.matmul(
                                    cell.qTensor(polygon), np.matrix.transpose(R)
                                ),
                            )
                            dq_inv[time.index(t)].append(q - q0)

                            polygon = dfDiv["Polygon"][
                                dfDiv["Division Time"] == t
                            ].iloc[1]
                            q = np.matmul(
                                R,
                                np.matmul(
                                    cell.qTensor(polygon), np.matrix.transpose(R)
                                ),
                            )
                            dq_inv[time.index(t)].append(q - q0)

    dQ = []
    dQstd = []
    dQ_p = []
    dQstd_p = []

    dQ_inv = []
    dQstd_inv = []
    dQ_inv_p = []
    dQstd_inv_p = []
    for i in range(len(dq)):
        dQ.append(np.mean(dq[i], axis=0))
        dQstd.append(np.std(dq[i], axis=0))
        dQ_p.append(np.mean(dq_p[i], axis=0))
        dQstd_p.append(np.std(dq_p[i], axis=0))

        dQ_inv.append(np.mean(dq_inv[i], axis=0))
        dQstd_inv.append(np.std(dq_inv[i], axis=0))
        dQ_inv_p.append(np.mean(dq_inv_p[i], axis=0))
        dQstd_inv_p.append(np.std(dq_inv_p[i], axis=0))

    dQ = np.array(dQ)
    dQstd = np.array(dQstd)
    dQ_p = np.array(dQ_p)
    dQstd_p = np.array(dQstd_p)

    dQ_inv = np.array(dQ_inv)
    dQstd_inv = np.array(dQstd_inv)
    dQ_inv_p = np.array(dQ_inv_p)
    dQstd_inv_p = np.array(dQstd_inv_p)
    time = 2 * np.array(time)

    fig, ax = plt.subplots(2, 4, figsize=(24, 16))

    ax[0, 0].errorbar(time, dQ[:, 0, 0], dQstd[:, 0, 0])
    ax[0, 0].set(xlabel=r"Time (mins)", ylabel=r"$\delta Q^1$")
    ax[0, 0].title.set_text(r"combined $\delta Q_1$ during division")
    ax[0, 0].set_ylim([-0.07, 0.07])

    ax[0, 1].errorbar(time, dQ[:, 1, 0], dQstd[:, 1, 0])
    ax[0, 1].set(xlabel=r"Time (mins)", ylabel=r"$\delta Q^2$")
    ax[0, 1].title.set_text(r"combined $\delta Q_2$ during division")
    ax[0, 1].set_ylim([-0.07, 0.07])

    ax[0, 2].errorbar(time, dQ_inv[:, 0, 0], dQstd_inv[:, 0, 0])
    ax[0, 2].set(xlabel=r"Time (mins)", ylabel=r"$\delta Q^1$")
    ax[0, 2].title.set_text(r"individual $\delta Q_1$ during division")
    ax[0, 2].set_ylim([-0.07, 0.07])

    ax[0, 3].errorbar(time, dQ_inv[:, 1, 0], dQstd_inv[:, 1, 0])
    ax[0, 3].set(xlabel=r"Time (mins)", ylabel=r"$\delta Q^2$")
    ax[0, 3].title.set_text(r"individual $\delta Q_2$ during division")
    ax[0, 3].set_ylim([-0.07, 0.07])

    ax[1, 0].errorbar(time, dQ_p[:, 0, 0], dQstd_p[:, 0, 0])
    ax[1, 0].set(xlabel=r"Time (mins)", ylabel=r"$\delta Q^1$")
    ax[1, 0].title.set_text(r"combined $\delta Q_1$ during division poor predictions")
    ax[1, 0].set_ylim([-0.07, 0.07])

    ax[1, 1].errorbar(time, dQ_p[:, 1, 0], dQstd_p[:, 1, 0])
    ax[1, 1].set(xlabel=r"Time (mins)", ylabel=r"$\delta Q^2$")
    ax[1, 1].title.set_text(r"combined $\delta Q_2$ during division poor predictions")
    ax[1, 1].set_ylim([-0.07, 0.07])

    ax[1, 2].errorbar(time, dQ_inv_p[:, 0, 0], dQstd_inv_p[:, 0, 0])
    ax[1, 2].set(xlabel=r"Time (mins)", ylabel=r"$\delta Q^1$")
    ax[1, 2].title.set_text(r"individual $\delta Q_1$ during division poor predictions")
    ax[1, 2].set_ylim([-0.07, 0.07])

    ax[1, 3].errorbar(time, dQ_inv_p[:, 1, 0], dQstd_inv_p[:, 1, 0])
    ax[1, 3].set(xlabel=r"Time (mins)", ylabel=r"$\delta Q^2$")
    ax[1, 3].title.set_text(r"individual $\delta Q_2$ during division poor predictions")
    ax[1, 3].set_ylim([-0.07, 0.07])

    plt.subplots_adjust(wspace=0.28)
    fig.savefig(
        f"results/change in Q division relative to tissue poor pred {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# delta Q of daughter cells with division orientation relative to tissue
if False:
    dfDivisionShape = pd.read_pickle(f"databases/dfDivisionShape{fileType}.pkl")
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    time = list(np.linspace(-15, 10, 26))
    _df = []

    for filename in filenames:
        tracks = sm.io.imread(f"dat/{filename}/binaryTracks{filename}.tif").astype(int)
        dfFileShape = dfDivisionShape[dfDivisionShape["Filename"] == filename]
        Q = np.mean(dfShape["q"][dfShape["Filename"] == filename])
        theta0 = 0.5 * np.arctan2(Q[1, 0], Q[0, 0])
        R = util.rotation_matrix(-theta0)

        dfFileShape = dfFileShape[dfFileShape["Daughter length"] > 10]
        dfFileShape = dfFileShape[dfFileShape["Track length"] > 18]
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        labels = list(dfFileShape["Label"])

        for label in labels:
            dfDiv = df[df["Label"] == label]
            tcjs = dfDiv["TCJ"][dfDiv["Division Time"] == -15].iloc[0]
            if tcjs != False:
                ori = dfFileShape["Orientation"][dfFileShape["Label"] == label].iloc[0]

                polygon = dfDiv["Polygon"][dfDiv["Division Time"] == time[0]].iloc[0]
                q0 = np.matmul(
                    R, np.matmul(cell.qTensor(polygon), np.matrix.transpose(R))
                )
                t = 10

                T = dfDiv["Time"][dfDiv["Division Time"] == t].iloc[0]
                colour1 = dfDiv["Colour"][dfDiv["Division Time"] == t].iloc[0]
                colour2 = dfDiv["Colour"][dfDiv["Division Time"] == t].iloc[1]
                mask = np.zeros([512, 512])
                mask[np.all((tracks[int(T)] - colour1) == 0, axis=2)] = 1
                mask[np.all((tracks[int(T)] - colour2) == 0, axis=2)] = 1
                q = np.matmul(R, np.matmul(maskQ(mask)[0], np.matrix.transpose(R)))
                dq_con = q - q0

                polygon = dfDiv["Polygon"][dfDiv["Division Time"] == t].iloc[0]
                q = np.matmul(
                    R,
                    np.matmul(cell.qTensor(polygon), np.matrix.transpose(R)),
                )
                dq_inv1 = q - q0

                polygon = dfDiv["Polygon"][dfDiv["Division Time"] == t].iloc[1]
                q = np.matmul(
                    R,
                    np.matmul(cell.qTensor(polygon), np.matrix.transpose(R)),
                )
                dq_inv2 = q - q0
                if ori > 90:
                    ori = 180 - ori
                    dq_con[1, 0] = -dq_con[1, 0]
                    dq_con[0, 1] = -dq_con[0, 1]
                    dq_inv1[1, 0] = -dq_inv1[1, 0]
                    dq_inv1[0, 1] = -dq_inv1[0, 1]
                    dq_inv2[1, 0] = -dq_inv2[1, 0]
                    dq_inv2[0, 1] = -dq_inv2[0, 1]

                _df.append(
                    {
                        "Filename": filename,
                        "Label": label,
                        "Orientation": ori,
                        "dq_con": dq_con,
                        "dq_inv1": dq_inv1,
                        "dq_inv2": dq_inv2,
                    }
                )

    df = pd.DataFrame(_df)

    thetas = np.linspace(0, 80, 9)
    dQ1_con = []
    dQ2_con = []
    dQ1_inv = []
    dQ2_inv = []

    dQ1std_con = []
    dQ2std_con = []
    dQ1std_inv = []
    dQ2std_inv = []
    for theta in thetas:
        df1 = df[df["Orientation"] > theta]
        df2 = df1[df1["Orientation"] < theta + 10]
        dQ1_con.append(np.mean(df2["dq_con"], axis=0)[0, 0])
        dQ2_con.append(np.mean(df2["dq_con"], axis=0)[1, 0])
        dQ1_inv.append(
            np.mean(list(df2["dq_inv1"]) + list(df2["dq_inv2"]), axis=0)[0, 0]
        )
        dQ2_inv.append(
            np.mean(list(df2["dq_inv1"]) + list(df2["dq_inv2"]), axis=0)[1, 0]
        )

        dQ1std_con.append(np.std(list(df2["dq_con"]), axis=0)[0, 0])
        dQ2std_con.append(np.std(list(df2["dq_con"]), axis=0)[1, 0])
        dQ1std_inv.append(
            np.std(list(df2["dq_inv1"]) + list(df2["dq_inv2"]), axis=0)[0, 0]
        )
        dQ2std_inv.append(
            np.std(list(df2["dq_inv1"]) + list(df2["dq_inv2"]), axis=0)[1, 0]
        )

    colour, mark = util.getColorLineMarker(fileType, groupTitle)
    dQ1_con = np.array(dQ1_con)
    dQ2_con = np.array(dQ2_con)
    dQ1_inv = np.array(dQ1_inv)
    dQ2_inv = np.array(dQ2_inv)
    dQ1std_con = np.array(dQ1std_con)
    dQ2std_con = np.array(dQ2std_con)
    dQ1std_inv = np.array(dQ1std_inv)
    dQ2std_inv = np.array(dQ2std_inv)

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    ax[0, 0].plot(thetas + 5, dQ1_con, color=colour, marker=mark, markevery=1)
    ax[0, 0].fill_between(
        thetas + 5, dQ1_con - dQ1std_con, dQ1_con + dQ1std_con, alpha=0.15, color=colour
    )
    ax[0, 0].set(xlabel=r"Division ori. rel. to tissue", ylabel=r"$\delta Q^1$")
    ax[0, 0].title.set_text(r"comb. $\delta Q_1$ after division")
    ax[0, 0].set_ylim([-0.1, 0.1])
    ax[0, 0].set_xlim([0, 90])

    ax[0, 1].plot(thetas + 5, dQ2_con, color=colour, marker=mark, markevery=1)
    ax[0, 1].fill_between(
        thetas + 5, dQ2_con - dQ2std_con, dQ2_con + dQ2std_con, alpha=0.15, color=colour
    )
    ax[0, 1].set(xlabel=r"Division ori. rel. to tissue", ylabel=r"$\delta Q^2$")
    ax[0, 1].title.set_text(r"comb. $\delta Q_2$ after division")
    ax[0, 1].set_ylim([-0.1, 0.1])
    ax[0, 1].set_xlim([0, 90])

    ax[1, 0].plot(thetas + 5, dQ1_inv, color=colour, marker=mark, markevery=1)
    ax[1, 0].fill_between(
        thetas + 5, dQ1_inv - dQ1std_inv, dQ1_inv + dQ1std_inv, alpha=0.15, color=colour
    )
    ax[1, 0].set(xlabel=r"Division ori. rel. to tissue", ylabel=r"$\delta Q^1$")
    ax[1, 0].title.set_text(r"indiv. $\delta Q_1$ after division")
    ax[1, 0].set_ylim([-0.1, 0.1])
    ax[1, 0].set_xlim([0, 90])

    ax[1, 1].plot(thetas + 5, dQ2_inv, color=colour, marker=mark, markevery=1)
    ax[1, 1].fill_between(
        thetas + 5, dQ2_inv - dQ2std_inv, dQ2_inv + dQ2std_inv, alpha=0.15, color=colour
    )
    ax[1, 1].set(xlabel=r"Division ori. rel. to tissue", ylabel=r"$\delta Q^2$")
    ax[1, 1].title.set_text(r"indiv. $\delta Q_2$ after division")
    ax[1, 1].set_ylim([-0.1, 0.1])
    ax[1, 1].set_xlim([0, 90])

    plt.subplots_adjust(hspace=0.4, wspace=0.5)
    fig.savefig(
        f"results/deltaQ after division with theta {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# shift in orientation after division relative to tissue
if False:
    dfDivisionShape = pd.read_pickle(f"databases/dfDivisionShape{fileType}.pkl")
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    _df = []

    for filename in filenames:
        tracks = sm.io.imread(f"dat/{filename}/binaryTracks{filename}.tif").astype(int)
        dfFileShape = dfDivisionShape[dfDivisionShape["Filename"] == filename]
        Q = np.mean(dfShape["q"][dfShape["Filename"] == filename])
        theta0 = 0.5 * np.arctan2(Q[1, 0], Q[0, 0])
        R = util.rotation_matrix(-theta0)

        dfFileShape = dfFileShape[dfFileShape["Daughter length"] > 10]
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        labels = list(dfFileShape["Label"])

        for label in labels:
            dfDiv = df[df["Label"] == label]
            ori = dfFileShape["Orientation"][dfFileShape["Label"] == label].iloc[0]
            t = 10

            polygon = dfDiv["Polygon"][dfDiv["Division Time"] == t].iloc[0]
            x0, y0 = cell.centroid(polygon)
            polygon = dfDiv["Polygon"][dfDiv["Division Time"] == t].iloc[1]
            x1, y1 = cell.centroid(polygon)
            phi = (np.arctan2(y0 - y1, x0 - x1) * 180 / np.pi) % 180

            ori = ori - theta0
            phi = phi - theta0
            if ori > 90:
                ori = 180 - ori
            if phi > 90:
                phi = 180 - phi

            _df.append(
                {
                    "Filename": filename,
                    "Label": label,
                    "Orientation": ori,
                    "Shape Orientation": phi,
                    "Change Towards Tissue": ori - phi,
                }
            )

    df = pd.DataFrame(_df)

    thetas = np.linspace(0, 80, 9)
    deltaOri = []
    deltaOristd = []
    for theta in thetas:
        df1 = df[df["Orientation"] > theta]
        df2 = df1[df1["Orientation"] < theta + 10]
        deltaOri.append(np.mean(df2["Change Towards Tissue"]))
        deltaOristd.append(np.std(df2["Change Towards Tissue"]))

    deltaOri = np.array(deltaOri)
    deltaOristd = np.array(deltaOristd)

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    ax[0, 0].hist(
        df["Orientation"],
    )
    ax[0, 0].axvline(np.mean(df["Orientation"]), color="r")
    ax[0, 0].set(xlabel=r"Nuclei Division Orientation relative to Tissue")
    ax[0, 0].title.set_text(r"Distribution of Nuclei Division Orientation")
    ax[0, 0].set_ylim([0, 130])

    ax[0, 1].hist(
        df["Shape Orientation"],
    )
    ax[0, 1].axvline(np.mean(df["Shape Orientation"]), color="r")
    ax[0, 1].set(xlabel=r"Shape Division Orientation relative to Tissue")
    ax[0, 1].title.set_text(r"Distribution of Shape Division Orientation")
    ax[0, 1].set_ylim([0, 130])

    ax[1, 0].hist(
        df["Change Towards Tissue"],
    )
    ax[1, 0].set(xlabel=r"Division Orientation Change Towards Tissue")
    ax[1, 0].title.set_text(r"Change in orientation Towards Tissue")
    ax[1, 0].set_xlim([-90, 90])
    ax[1, 0].axvline(np.mean(df["Change Towards Tissue"]), color="r")

    ax[1, 1].errorbar(thetas + 5, deltaOri, deltaOristd)
    ax[1, 1].set(
        xlabel=r"Mitosis Division Orientation",
        ylabel=r"Division Orientation Change Towards Tissue",
    )
    ax[1, 1].title.set_text(r"Change in orientation Towards Tissue with theta")
    ax[1, 1].set_ylim([-40, 40])

    fig.tight_layout()
    fig.savefig(
        f"results/change in ori after division relative to tissue {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.hist(
        df["Orientation"],
        alpha=0.4,
        density=True,
        label="Mitosis orientation",
        color="g",
    )
    ax.axvline(np.median(df["Orientation"]), color="g")
    ax.hist(
        df["Shape Orientation"],
        alpha=0.4,
        color="m",
        density=True,
        label="Post shuffling orientation",
    )
    ax.axvline(np.median(df["Shape Orientation"]), color="m")

    ax.set(xlabel="Division orientation relative to wing", ylabel="Frequency")
    fileTitle = util.getFileTitle(fileType)
    boldTitle = util.getBoldTitle(fileTitle)
    if "Wound" in fileType:
        ax.title.set_text(
            f"Shift in division orientation relative \n to wing in tissue with "
            + boldTitle
        )
    else:
        ax.title.set_text(
            f"Shift in division orientation relative \n to wing in " + boldTitle
        )

    ax.set_ylim([0, 0.024])
    ax.legend(fontsize=12, loc="upper left")

    # ax[1].hist(
    #     df["Change Towards Tissue"],
    # )
    # ax[1].set(xlabel=r"Division Orientation Change Towards Tissue", ylabel="Frequency")
    # ax[1].title.set_text(r"Change in orientation Towards Tissue")
    # ax[1].set_xlim([-90, 90])
    # ax[1].axvline(np.mean(df["Change Towards Tissue"]), color="r")

    # ax[2].plot(thetas + 5, deltaOri, marker="o")
    # ax[2].fill_between(thetas + 5, deltaOri- deltaOristd, deltaOri+ deltaOristd, alpha=0.2)
    # ax[2].set(
    #     xlabel=r"Mitosis Division Orientation",
    #     ylabel=r"Division Orientation Change Towards Tissue",
    # )
    # ax[2].title.set_text(r"Change in orientation Towards Tissue with theta")
    # ax[2].set_ylim([-40, 40])

    fig.tight_layout()
    fig.savefig(
        f"results/change in ori after division relative to tissue {fileType} figure",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# shift in orientation after division relative to wound
if False:
    dfDivisionShape = pd.read_pickle(f"databases/dfDivisionShape{fileType}.pkl")
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    _df = []

    for filename in filenames:
        dfDivision = pd.read_pickle(f"dat/{filename}/dfDivision{filename}.pkl")
        tracks = sm.io.imread(f"dat/{filename}/binaryTracks{filename}.tif").astype(int)
        dfFileShape = dfDivisionShape[dfDivisionShape["Filename"] == filename]

        if "Wound" in filename:
            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")

        else:
            dfVelocityMean = pd.read_pickle(f"databases/dfVelocityMean{fileType}.pkl")
            dfFilename = dfVelocityMean[
                dfVelocityMean["Filename"] == filename
            ].reset_index()

        dfFileShape = dfFileShape[dfFileShape["Daughter length"] > 10]
        dfFileShape = dfFileShape[dfFileShape["Track length"] > 3]
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        labels = list(dfFileShape["Label"])
        dist = (
            sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int) * scale
        )
        t0 = util.findStartTime(filename)

        for label in labels:
            dfDiv = df[df["Label"] == label]
            ori = dfDivision["Orientation"][dfDivision["Label"] == label].iloc[0] % 180
            t = dfDiv["Time"][dfDiv["Division Time"] == 0].iloc[0]
            x = dfDivision["X"][dfDivision["Label"] == label].iloc[0]
            y = dfDivision["Y"][dfDivision["Label"] == label].iloc[0]
            if "Wound" in filename:
                (xc, yc) = dfWound["Position"].iloc[t]
            else:
                (xc, yc) = np.sum(
                    np.stack(np.array(dfFilename.loc[:t, "V"]), axis=0), axis=0
                )
            psi = np.arctan2(y - yc, x - xc) * 180 / np.pi
            ori = (ori - psi) % 180

            dfDiv = df[df["Label"] == label]
            T = 10
            polygon = dfDiv["Polygon"][dfDiv["Division Time"] == T].iloc[0]
            x0, y0 = cell.centroid(polygon)
            polygon = dfDiv["Polygon"][dfDiv["Division Time"] == T].iloc[1]
            x1, y1 = cell.centroid(polygon)
            phi = (np.arctan2(y0 - y1, x0 - x1) * 180 / np.pi) % 180
            xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
            psi = np.arctan2(ym - yc, xm - xc) * 180 / np.pi
            phi = (phi - psi) % 180
            r = dist[t, int(512 - ym), int(xm)]

            if ori > 90:
                ori = 180 - ori
            if phi > 90:
                phi = 180 - phi

            _df.append(
                {
                    "Filename": filename,
                    "Label": label,
                    "T": int(t0 + t * 2),
                    "R": r,
                    "Nuclei Orientation": ori,
                    "Daughter Orientation": phi,
                    "Change Towards Wound": ori - phi,
                }
            )

    df = pd.DataFrame(_df)

    thetas = np.linspace(0, 80, 9)
    deltaOri = []
    deltaOristd = []
    for theta in thetas:
        df1 = df[df["Nuclei Orientation"] > theta]
        df2 = df1[df1["Nuclei Orientation"] < theta + 10]
        deltaOri.append(np.mean(df2["Change Towards Wound"]))
        deltaOristd.append(np.std(df2["Change Towards Wound"]))

    deltaOri = np.array(deltaOri)
    deltaOristd = np.array(deltaOristd)

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    ax[0, 0].hist(
        df["Nuclei Orientation"],
    )
    ax[0, 0].axvline(np.mean(df["Nuclei Orientation"]), color="r")
    ax[0, 0].set(xlabel=r"Nuclei Division Orientation relative to Wound")
    ax[0, 0].title.set_text(r"Distribution of Nuclei Division Orientation")
    if "Wound" in filename:
        ax[0, 0].set_ylim([0, 50])
    else:
        ax[0, 0].set_ylim([0, 130])

    ax[0, 1].hist(
        df["Daughter Orientation"],
    )
    ax[0, 1].axvline(np.mean(df["Daughter Orientation"]), color="r")
    ax[0, 1].set(xlabel=r"Shape Division Orientation relative to Wound")
    ax[0, 1].title.set_text(r"Distribution of Shape Division Orientation")
    if "Wound" in filename:
        ax[0, 1].set_ylim([0, 50])
    else:
        ax[0, 1].set_ylim([0, 130])

    ax[1, 0].hist(
        df["Change Towards Wound"],
    )
    ax[1, 0].set(xlabel=r"Division Orientation Change Towards Wound")
    ax[1, 0].title.set_text(r"Change in orientation Towards Wound")
    ax[1, 0].set_xlim([-90, 90])
    ax[1, 0].axvline(np.mean(df["Change Towards Wound"]), color="r")

    ax[1, 1].errorbar(thetas + 5, deltaOri, deltaOristd)
    ax[1, 1].set(
        xlabel=r"Mitosis Division Orientation",
        ylabel=r"Division Orientation Change Towards Wound",
    )
    ax[1, 1].title.set_text(r"Change in orientation Towards Wound with theta")
    ax[1, 1].set_ylim([-40, 40])

    fig.tight_layout()
    fig.savefig(
        f"results/change in ori after division relative to wound {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax.hist(
        df["Nuclei Orientation"],
        alpha=0.4,
        density=True,
        label="Mitosis orientation",
        color="g",
    )
    ax.axvline(np.median(df["Nuclei Orientation"]), color="g")
    ax.hist(
        df["Daughter Orientation"],
        alpha=0.4,
        color="m",
        density=True,
        label="Post shuffling orientation",
    )
    ax.axvline(np.median(df["Daughter Orientation"]), color="m")

    ax.set(xlabel="Division orientation relative \n to wound", ylabel="Frequency")
    fileTitle = util.getFileTitle(fileType)
    boldTitle = util.getBoldTitle(fileTitle)
    if "Wound" in fileType:
        ax.title.set_text(f"Shift in division orientation \n relative to " + boldTitle)
    else:
        ax.title.set_text(
            f"Shift in division orientation relative \n to wing in " + boldTitle
        )
    ax.set_ylim([0, 0.024])
    ax.legend(fontsize=12, loc="upper left")

    # ax[1].hist(
    #     df["Change Towards Wound"],
    # )
    # ax[1].set(xlabel=r"Division Orientation Change Towards Wound", ylabel="Frequency")
    # ax[1].title.set_text(r"Change in orientation Towards Wound")
    # ax[1].set_xlim([-90, 90])
    # ax[1].axvline(np.mean(df["Change Towards Wound"]), color="r")

    # ax[2].plot(thetas + 5, deltaOri, marker="o")
    # ax[2].fill_between(thetas + 5, deltaOri- deltaOristd, deltaOri+ deltaOristd, alpha=0.2)
    # ax[2].set(
    #     xlabel=r"Mitosis Division Orientation",
    #     ylabel=r"Division Orientation Change Towards Wound",
    # )
    # ax[2].title.set_text(r"Change in orientation Towards Wound with theta")
    # ax[2].set_ylim([-40, 40])

    fig.tight_layout()
    fig.savefig(
        f"results/change in ori after division relative to wound {fileType} figure",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

    # during wound closer

    thetas = np.linspace(0, 80, 9)
    deltaOri = []
    deltaOristd = []
    for theta in thetas:
        df1 = df[df["T"] < 45]
        df2 = df1[df1["Nuclei Orientation"] > theta]
        df3 = df2[df2["Nuclei Orientation"] < theta + 10]
        deltaOri.append(np.mean(df3["Change Towards Wound"]))
        deltaOristd.append(np.std(df3["Change Towards Wound"]))

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    ax[0, 0].hist(
        df["Nuclei Orientation"][df["T"] < 45],
    )
    ax[0, 0].axvline(np.mean(df["Nuclei Orientation"][df["T"] < 45]), color="r")
    ax[0, 0].set(xlabel=r"Nuclei Division Orientation relative to Wound")
    ax[0, 0].title.set_text(r"Distribution of Nuclei Division Orientation")
    if "Wound" in filename:
        ax[0, 0].set_ylim([0, 50])
    else:
        ax[0, 0].set_ylim([0, 130])

    ax[0, 1].hist(
        df["Daughter Orientation"][df["T"] < 45],
    )
    ax[0, 1].axvline(np.mean(df["Daughter Orientation"][df["T"] < 45]), color="r")
    ax[0, 1].set(xlabel=r"Shape Division Orientation relative to Wound")
    ax[0, 1].title.set_text(r"Distribution of Shape Division Orientation")
    if "Wound" in filename:
        ax[0, 1].set_ylim([0, 50])
    else:
        ax[0, 1].set_ylim([0, 130])

    ax[1, 0].hist(
        df["Change Towards Wound"][df["T"] < 45],
    )
    ax[1, 0].set(xlabel=r"Division Orientation Change Towards Wound")
    ax[1, 0].title.set_text(r"Change in orientation Towards Wound")
    ax[1, 0].set_xlim([-90, 90])
    ax[1, 0].axvline(np.mean(df["Change Towards Wound"][df["T"] < 45]), color="r")

    ax[1, 1].errorbar(thetas + 5, deltaOri, deltaOristd)
    ax[1, 1].set(
        xlabel=r"Mitosis Division Orientation",
        ylabel=r"Division Orientation Change Towards Wound",
    )
    ax[1, 1].title.set_text(r"Change in orientation Towards Wound with theta")
    ax[1, 1].set_ylim([-40, 40])

    fig.tight_layout()
    fig.savefig(
        f"results/change in ori after division relative to wound {fileType} during wound closer",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

    # close to wound

    thetas = np.linspace(0, 80, 9)
    deltaOri = []
    deltaOristd = []
    for theta in thetas:
        df1 = df[(df["R"] < 30) & (df["T"] < 30)]
        df2 = df1[df1["Nuclei Orientation"] > theta]
        df3 = df2[df2["Nuclei Orientation"] < theta + 10]
        deltaOri.append(np.mean(df3["Change Towards Wound"]))
        deltaOristd.append(np.std(df3["Change Towards Wound"]))

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    ax[0, 0].hist(
        df["Nuclei Orientation"][(df["R"] < 30) & (df["T"] < 30)],
    )
    ax[0, 0].axvline(
        np.mean(df["Nuclei Orientation"][(df["R"] < 30) & (df["T"] < 30)]), color="r"
    )
    ax[0, 0].set(xlabel=r"Nuclei Division Orientation relative to Wound")
    ax[0, 0].title.set_text(r"Distribution of Nuclei Division Orientation")
    if "Wound" in filename:
        ax[0, 0].set_ylim([0, 50])
    else:
        ax[0, 0].set_ylim([0, 130])

    ax[0, 1].hist(
        df["Daughter Orientation"][(df["R"] < 30) & (df["T"] < 30)],
    )
    ax[0, 1].axvline(
        np.mean(df["Daughter Orientation"][(df["R"] < 30) & (df["T"] < 30)]), color="r"
    )
    ax[0, 1].set(xlabel=r"Shape Division Orientation relative to Wound")
    ax[0, 1].title.set_text(r"Distribution of Shape Division Orientation")
    if "Wound" in filename:
        ax[0, 1].set_ylim([0, 50])
    else:
        ax[0, 1].set_ylim([0, 130])

    ax[1, 0].hist(
        df["Change Towards Wound"][(df["R"] < 30) & (df["T"] < 30)],
    )
    ax[1, 0].set(xlabel=r"Division Orientation Change Towards Wound")
    ax[1, 0].title.set_text(r"Change in orientation Towards Wound")
    ax[1, 0].set_xlim([-90, 90])
    ax[1, 0].axvline(
        np.mean(df["Change Towards Wound"][(df["R"] < 30) & (df["T"] < 30)]), color="r"
    )

    ax[1, 1].errorbar(thetas + 5, deltaOri, deltaOristd)
    ax[1, 1].set(
        xlabel=r"Mitosis Division Orientation",
        ylabel=r"Division Orientation Change Towards Wound",
    )
    ax[1, 1].title.set_text(r"Change in orientation Towards Wound with theta")
    ax[1, 1].set_ylim([-40, 40])

    fig.tight_layout()
    fig.savefig(
        f"results/change in ori after division relative to wound {fileType} close to wound",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")
