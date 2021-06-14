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
from PIL import Image
import random
import scipy as sp
import scipy.linalg as linalg
from scipy.stats import mannwhitneyu
import shapely
import skimage as sm
import skimage.feature
import skimage.io
import skimage.measure
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
import tifffile
from skimage.draw import circle_perimeter
from scipy import optimize
import xml.etree.ElementTree as et
import plotly.graph_objects as go

import cellProperties as cell
import findGoodCells as fi
import commonLiberty as cl

# -------------------


def sphere(shape, radius, position):
    # https://stackoverflow.com/questions/46626267/how-to-generate-a-sphere-in-3d-numpy-array
    # assume shape and position are both a 3-tuple of int or float
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    semisizes = (radius,) * 3

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        arr += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below 1
    return arr <= 1.0


def correlationFunction(x, y, t, r):

    count = 0

    n = len(x)
    for i in range(n):
        for j in range(n):
            if (
                (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (t[i] - t[j]) ** 0.5
            ) ** 0.5 > r and (
                (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (t[i] - t[j]) ** 0.5
            ) ** 0.5 <= r + 5:
                count += 1

    corr = 3 * count / (2 * np.pi * (3 * r ** 2 + 3 * r + 1))
    return corr


def inPlaneShell(x, y, t, t0, t1, r0, r1, outPlane):

    if r0 == 0:
        r0 = 1

    t0 = t + t0
    t1 = t + t1

    if t1 > 180:
        t1 = 180

    background = np.zeros([181, 500 + 148, 500 + 148])

    rr1, cc1 = sm.draw.circle(250 + x, 250 + y, r0)
    rr2, cc2 = sm.draw.circle(250 + x, 250 + y, r1)

    background[t0:t1, rr2, cc2] = 1
    background[t0:t1, rr1, cc1] = 0

    inPlane = background[:, 250 : 250 + 148, 250 : 250 + 148]

    inPlane[outPlane == 255] = 0

    return inPlane


plt.rcParams.update({"font.size": 16})

filenames, fileType = cl.getFilesType()
scale = 147.91 / 512

_dfSpaceTime = []

if False:
    for filename in filenames:

        dfDivisions = pd.read_pickle(f"dat/{filename}/mitosisTracks{filename}.pkl")
        df = dfDivisions[dfDivisions["Chain"] == "parent"]

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
                    "X": x * scale,
                    "Y": y * scale,
                }
            )

    dfSpaceTime = pd.DataFrame(_dfSpaceTime)
    dfSpaceTime.to_pickle(f"databases/dfSpaceTime{fileType}.pkl")
else:
    dfSpaceTime = pd.read_pickle(f"databases/dfSpaceTime{fileType}.pkl")

if True:
    for filename in filenames:
        df = dfSpaceTime[dfSpaceTime["Filename"] == filename]
        x = np.array(df["X"])
        y = np.array(df["Y"])
        t = np.array(df["T"])
        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=t, mode="markers")])
        fig.show()
        plt.close("all")


if False:
    x = np.array(dfSpaceTime["X"])
    y = np.array(dfSpaceTime["Y"])

    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    ax[0].hist(x, bins=10)
    ax[0].set(xlabel="x")

    ax[1].hist(y, bins=10)
    ax[1].set(xlabel="y")

    fig.savefig(
        f"results/xy distributions {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


# correction of divisions weird
if False:
    volume = [
        [[[] for col in range(len(filenames))] for col in range(21)]
        for col in range(18)
    ]
    count = [
        [[[] for col in range(len(filenames))] for col in range(21)]
        for col in range(18)
    ]
    thetaCorrelation = [
        [[[] for col in range(len(filenames))] for col in range(21)]
        for col in range(18)
    ]

    for m in range(len(filenames)):

        filename = filenames[m]
        print(filename)
        divisions = np.zeros([181, 148, 148])
        orientations = np.zeros([181, 148, 148])
        outPlanePixel = sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype(
            float
        )
        outPlane = []
        for t in range(len(outPlanePixel)):
            img = Image.fromarray(outPlanePixel[t])
            outPlane.append(np.array(img.resize((148, 148))))
        outPlane = np.array(outPlane)
        outPlane[outPlane > 50] = 255
        outPlane[outPlane < 0] = 0

        df = dfSpaceTime[dfSpaceTime["Filename"] == filename]
        x = np.array(df["X"])
        y = np.array(df["Y"])
        t = np.array(df["T"])
        ori = np.array(df["Orientation"])

        for i in range(len(x)):
            divisions[int(t[i]), int(x[i]), int(y[i])] = 1
            orientations[int(t[i]), int(x[i]), int(y[i])] = ori[i]

        T = np.array(range(18)) * 10
        R = np.array(range(21)) * 10

        for i in range(len(T)):
            print(i)
            t0 = T[i]
            t1 = t0 + 10
            for j in range(len(R)):
                r0 = R[j]
                r1 = r0 + 10
                for k in range(len(x)):
                    if t[k] + t0 < 181:
                        shell = inPlaneShell(x[k], y[k], t[k], t0, t1, r0, r1, outPlane)
                        volume[i][j][m].append(np.sum(shell))
                        count[i][j][m].append(np.sum(divisions[shell == 1]))
                        thetas = orientations[shell == 1]
                        thetas = thetas[thetas != 0]
                        if len(thetas) != 0:
                            corr = []
                            v = np.array(
                                [
                                    np.cos(np.pi * ori[k] / 90),
                                    np.sin(np.pi * ori[k] / 90),
                                ]
                            )
                            for theta in thetas:
                                u = np.array(
                                    [
                                        np.cos(np.pi * theta / 90),
                                        np.sin(np.pi * theta / 90),
                                    ]
                                )
                                thetaCorrelation[i][j][m].append(np.dot(v, u))

    correlation = [[] for col in range(18)]
    oriCorrelation = [[] for col in range(18)]
    for i in range(18):
        for j in range(21):
            correlation[i].append(
                np.sum(np.sum(count[i][j])) / np.sum(np.sum(volume[i][j]))
            )
            oriCorrelation[i].append(np.mean(np.mean(thetaCorrelation[i][j])))

    volume_uni = [
        [[[] for col in range(len(filenames))] for col in range(21)]
        for col in range(18)
    ]
    count_uni = [
        [[[] for col in range(len(filenames))] for col in range(21)]
        for col in range(18)
    ]

    for m in range(len(filenames)):
        print(filenames[m])
        divisions = np.zeros([181, 148, 148])
        outPlane = np.zeros([181, 148, 148])
        df = dfSpaceTime[dfSpaceTime["Filename"] == filenames[m]]
        n = len(df)

        x = 148 * np.random.random_sample(n)
        y = 148 * np.random.random_sample(n)
        t = np.array(df["T"])

        for i in range(n):
            divisions[int(t[i]), int(x[i]), int(y[i])] = 1

        T = np.array(range(18)) * 10
        R = np.array(range(21)) * 10
        for i in range(len(T)):
            print(i)
            t0 = T[i]
            t1 = t0 + 10
            for j in range(len(R)):
                r0 = R[j]
                r1 = r0 + 10
                for k in range(len(x)):
                    if t[k] + t0 < 181:
                        shell = inPlaneShell(x[k], y[k], t[k], t0, t1, r0, r1, outPlane)
                        volume_uni[i][j][m].append(np.sum(shell))
                        count_uni[i][j][m].append(np.sum(divisions[shell == 1]))

    correlationUni = [[] for col in range(18)]
    for i in range(18):
        for j in range(21):
            correlationUni[i].append(
                np.sum(np.sum(count_uni[i][j])) / np.sum(np.sum(volume_uni[i][j]))
            )

    t, r = np.mgrid[0:160:10, 0:110:10]
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    correlation = np.array(correlation)[:15, :10]
    correlationUni = np.array(correlationUni)[:15, :10]
    oriCorrelation = np.array(oriCorrelation)[:15, :10]

    # maxCorr = max([np.max(correlation), np.max(correlationUni)])
    maxCorr = 0.0002

    c = ax[0, 0].pcolor(t, r, correlation, cmap="Reds", vmin=0, vmax=maxCorr)
    fig.colorbar(c, ax=ax[0, 0])
    ax[0, 0].set_xlabel("Time (min)")
    ax[0, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 0].title.set_text(f"Correlation {fileType}")

    c = ax[0, 1].pcolor(t, r, correlationUni, cmap="Reds", vmin=0, vmax=maxCorr)
    fig.colorbar(c, ax=ax[0, 1])
    ax[0, 1].set_xlabel("Time (min)")
    ax[0, 1].set_ylabel(r"$R (\mu m)$")
    ax[0, 1].title.set_text(f"Correlation of Uniform xy {fileType}")

    correlationDiff = correlation - correlationUni

    z_min, z_max = -0.00006, 0.00006
    midpoint = 1 - z_max / (z_max + abs(z_min))
    orig_cmap = matplotlib.cm.RdBu_r
    shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")

    c = ax[1, 0].pcolor(
        t,
        r,
        correlation - correlationUni,
        cmap=shifted_cmap,
        vmin=-0.00006,
        vmax=0.00006,
    )
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set_xlabel("Time (min)")
    ax[1, 0].set_ylabel(r"$R (\mu m)$")
    ax[1, 0].title.set_text(f"Difference in Correlation {fileType}")

    c = ax[1, 1].pcolor(
        t,
        r,
        oriCorrelation,
        cmap="RdBu_r",
        vmin=-np.max(oriCorrelation),
        vmax=np.max(oriCorrelation),
    )
    fig.colorbar(c, ax=ax[1, 1])
    ax[1, 1].set_xlabel("Time (min)")
    ax[1, 1].set_ylabel(r"$R (\mu m)$")
    ax[1, 1].title.set_text(f"Correlation Orientation {fileType}")

    fig.savefig(
        f"results/Division Correlation {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

    _df = []

    _df.append(
        {
            "volume": volume,
            "count": count,
            "volume_uni": volume_uni,
            "count_uni": count_uni,
            "correlation": correlation,
            "correlationUni": correlationUni,
            "oriCorrelation": oriCorrelation,
        }
    )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/correlationArrays{fileType}.pkl")
