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
from scipy.optimize import leastsq

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

    background = np.zeros([181, 500 + 149, 500 + 149])

    rr1, cc1 = sm.draw.disk((250 + x, 250 + y), r0)
    rr2, cc2 = sm.draw.disk((250 + x, 250 + y), r1)

    background[t0:t1, rr2, cc2] = 1
    background[t0:t1, rr1, cc1] = 0

    inPlane = background[:, 250 : 250 + 149, 250 : 250 + 149]

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

if False:
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


def exponential(x, coeffs):
    A = coeffs[0]
    c = coeffs[1]
    return A * np.exp(c * x)


def residualsExponential(coeffs, y, x):
    return y - exponential(x, coeffs)


def exponential2D(x, coeffs):
    A = coeffs[0]
    c = coeffs[1]
    d = coeffs[2]
    return A * np.exp(c * x[0] + d * x[1])


def residualsExponential2D(coeffs, y, x):
    return y - exponential2D(x, coeffs)


def model(t, r, mt, mr):
    return exponential(t, mt) * exponential(r, mr) / mt[0]


# correction of divisions V2
if False:
    _filenames = filenames
    for filename in _filenames:
        filenames = [filename]

        T = np.array(range(15)) * 10
        R = np.array(range(10)) * 10
        correlationFile = [
            [[[] for col in range(len(filenames))] for col in range(len(R))]
            for col in range(len(T))
        ]
        thetaCorrelation = [
            [[[] for col in range(len(filenames))] for col in range(len(R))]
            for col in range(len(T))
        ]

        N = []
        divisionNum = []

        for m in range(len(filenames)):

            filename = filenames[m]
            print(filename)
            divisions = np.zeros([181, 149, 149])
            orientations = np.zeros([181, 149, 149])
            outPlanePixel = sm.io.imread(
                f"dat/{filename}/outPlane{filename}.tif"
            ).astype(float)
            outPlane = []
            for t in range(len(outPlanePixel)):
                img = Image.fromarray(outPlanePixel[t])
                outPlane.append(np.array(img.resize((149, 149))))
            outPlane = np.array(outPlane)
            outPlane[outPlane > 50] = 255
            outPlane[outPlane < 0] = 0

            N.append(181 * 149 ** 2 - np.sum(outPlane) / 255)

            df = dfSpaceTime[dfSpaceTime["Filename"] == filename]
            x = np.array(df["X"])
            y = np.array(df["Y"])
            t = np.array(df["T"])
            ori = np.array(df["Orientation"])

            divisionNum.append(len(x))

            for i in range(len(x)):
                divisions[round(t[i]), round(x[i]), round(y[i])] = 1
                orientations[round(t[i]), round(x[i]), round(y[i])] = ori[i]

            for i in range(len(T)):
                print(i)
                t0 = T[i]
                t1 = t0 + 10
                for j in range(len(R)):
                    r0 = R[j]
                    r1 = r0 + 10
                    for k in range(len(x)):
                        if t[k] + t0 < 181:
                            shell = inPlaneShell(
                                round(x[k]), round(y[k]), t[k], t0, t1, r0, r1, outPlane
                            )
                            if np.sum(shell) != 0:
                                correlationFile[i][j][m].append(
                                    np.sum(divisions[shell == 1]) / np.sum(shell)
                                )
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

        correlation = [[] for col in range(len(T))]
        oriCorrelation = [[] for col in range(len(T))]
        for i in range(len(T)):
            for j in range(len(R)):
                cor = []
                ori = []
                for m in range(len(filenames)):
                    cor.append(np.sum(correlationFile[i][j][m]) / N[m])
                    ori.append(np.mean(thetaCorrelation[i][j][m]))

                correlation[i].append(np.mean(cor))
                oriCorrelation[i].append(np.mean(ori))

        # ------------

        correlation_ran = [
            [[[] for col in range(len(filenames))] for col in range(len(R))]
            for col in range(len(T))
        ]

        for m in range(len(filenames)):
            print(filenames[m])
            df = dfSpaceTime[dfSpaceTime["Filename"] == filenames[m]]
            n = 300

            x = 148 * np.random.random_sample(n)
            y = 148 * np.random.random_sample(n)
            t = 180 * np.random.random_sample(n)

            for i in range(len(T)):
                print(i)
                t0 = T[i]
                t1 = t0 + 10
                for j in range(len(R)):
                    r0 = R[j]
                    r1 = r0 + 10
                    for k in range(len(x)):
                        if t[k] + t0 < 181:
                            if outPlane[round(t[k]), round(x[k]), round(y[k])] == 0:
                                shell = inPlaneShell(
                                    round(x[k]),
                                    round(y[k]),
                                    round(t[k]),
                                    t0,
                                    t1,
                                    r0,
                                    r1,
                                    outPlane,
                                )
                                if np.sum(shell) != 0:
                                    correlation_ran[i][j][m].append(
                                        np.sum(divisions[shell == 1]) / np.sum(shell)
                                    )

        correlationRan = [[] for col in range(len(T))]
        for i in range(len(T)):
            for j in range(len(R)):
                cor = []
                for m in range(len(filenames)):
                    cor.append(
                        np.sum(correlation_ran[i][j][m]) * (divisionNum[m] / (N[m] * n))
                    )

                correlationRan[i].append(np.mean(cor))

        t, r = np.mgrid[0:160:10, 0:110:10]
        fig, ax = plt.subplots(2, 2, figsize=(20, 20))
        plt.subplots_adjust(wspace=0.3)
        plt.gcf().subplots_adjust(bottom=0.15)

        correlation = np.array(correlation)[:15, :10]
        correlationRan = np.array(correlationRan)[:15, :10]
        oriCorrelation = np.array(oriCorrelation)[:15, :10]
        oriCorrelation = np.nan_to_num(oriCorrelation)

        correlationDiff = correlation - correlationRan

        correlation = correlation / correlationDiff[0][0]
        correlationRan = correlationRan / correlationDiff[0][0]
        correlationDiff = correlationDiff / correlationDiff[0][0]

        _df = []

        _df.append(
            {
                "correlationFile": correlationFile,
                "correlation_ran": correlation_ran,
                "correlation": correlation,
                "correlationRan": correlationRan,
                "oriCorrelation": oriCorrelation,
            }
        )

        df = pd.DataFrame(_df)
        df.to_pickle(f"databases/correlationArraysV2-1{filename}.pkl")

        maxCorr = np.max(correlation)

        c = ax[0, 0].pcolor(t, r, correlation, cmap="Reds", vmin=0, vmax=maxCorr)
        fig.colorbar(c, ax=ax[0, 0])
        ax[0, 0].set_xlabel("Time (min)")
        ax[0, 0].set_ylabel(r"$R (\mu m)$ ")
        ax[0, 0].title.set_text(f"Correlation {filename}")

        c = ax[0, 1].pcolor(t, r, correlationRan, cmap="Reds", vmin=0, vmax=maxCorr)
        fig.colorbar(c, ax=ax[0, 1])
        ax[0, 1].set_xlabel("Time (min)")
        ax[0, 1].set_ylabel(r"$R (\mu m)$")
        ax[0, 1].title.set_text(f"Correlation of Random i {filename}")

        z_min, z_max = -maxCorr, maxCorr
        midpoint = 1 - z_max / (z_max + abs(z_min))
        orig_cmap = matplotlib.cm.RdBu_r
        shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")

        c = ax[1, 0].pcolor(
            t,
            r,
            correlation - correlationRan,
            cmap=shifted_cmap,
            vmin=-maxCorr,
            vmax=maxCorr,
        )
        fig.colorbar(c, ax=ax[1, 0])
        ax[1, 0].set_xlabel("Time (min)")
        ax[1, 0].set_ylabel(r"$R (\mu m)$")
        ax[1, 0].title.set_text(f"Difference in Correlation {filename}")

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
        ax[1, 1].title.set_text(f"Correlation Orientation {filename}")

        fig.savefig(
            f"results/Division Correlation V2-1 {filename}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

    filenames, fileType = cl.getFilesType()


if False:

    correlations = np.zeros([len(filenames), 15, 10])
    correlationRans = np.zeros([len(filenames), 15, 10])
    oriCorrelations = np.zeros([len(filenames), 15, 10])
    for m in range(len(filenames)):
        filename = filenames[m]

        df = pd.read_pickle(f"databases/correlationArraysV2-1{filename}.pkl")

        correlations[m] = df["correlation"][0]
        correlationRans[m] = df["correlationRan"][0]
        oriCorrelations[m] = df["oriCorrelation"][0]

    oriCorrelations = np.nan_to_num(oriCorrelations)

    correlation = np.mean(correlations, axis=0)
    correlationRan = np.mean(correlationRans, axis=0)
    oriCorrelation = np.mean(oriCorrelations, axis=0)

    t, r = np.mgrid[0:160:10, 0:110:10]
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    maxCorr = np.max(correlation)

    c = ax[0, 0].pcolor(t, r, correlation, cmap="Reds", vmin=0, vmax=maxCorr)
    fig.colorbar(c, ax=ax[0, 0])
    ax[0, 0].set_xlabel("Time (min)")
    ax[0, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 0].title.set_text(f"Correlation {fileType}")

    c = ax[0, 1].pcolor(t, r, correlationRan, cmap="Reds", vmin=0, vmax=maxCorr)
    fig.colorbar(c, ax=ax[0, 1])
    ax[0, 1].set_xlabel("Time (min)")
    ax[0, 1].set_ylabel(r"$R (\mu m)$")
    ax[0, 1].title.set_text(f"Correlation of Random i {fileType}")

    correlationDiff = correlation - correlationRan

    z_min, z_max = -maxCorr, maxCorr
    midpoint = 1 - z_max / (z_max + abs(z_min))
    orig_cmap = matplotlib.cm.RdBu_r
    shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")

    c = ax[1, 0].pcolor(
        t,
        r,
        correlation - correlationRan,
        cmap=shifted_cmap,
        vmin=-maxCorr,
        vmax=maxCorr,
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
        f"results/Division Correlation V2-1 {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


def exponential(x, coeffs):
    A = coeffs[0]
    b = coeffs[1]
    return A * np.exp(x * b)


def residualsExponential(coeffs, y, x):
    return y - exponential(x, coeffs)


if False:

    correlations = np.zeros([len(filenames), 15, 10])
    correlationRans = np.zeros([len(filenames), 15, 10])
    oriCorrelations = np.zeros([len(filenames), 15, 10])
    for m in range(len(filenames)):
        filename = filenames[m]

        df = pd.read_pickle(f"databases/correlationArraysV2-1{filename}.pkl")

        correlations[m] = df["correlation"][0]
        correlationRans[m] = df["correlationRan"][0]
        oriCorrelations[m] = df["oriCorrelation"][0]

    oriCorrelations = np.nan_to_num(oriCorrelations)

    correlation = np.mean(correlations, axis=0)
    correlationRan = np.mean(correlationRans, axis=0)
    oriCorrelation = np.mean(oriCorrelations, axis=0)

    correlationDiff = correlation - correlationRan

    R0 = correlationDiff[0]
    T0 = correlationDiff[:, 0]
    T = np.array(range(15)) * 10
    R = np.array(range(10)) * 10

    mR = leastsq(
        residualsExponential,
        x0=(1, -1),
        args=(R0, R),
    )[0]

    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    ax[0, 0].plot(R, R0)
    ax[0, 0].plot(R, exponential(R, mR))
    ax[0, 0].set(xlabel=r"Distance $(\mu m)$", ylabel="Correlation")
    ax[0, 0].title.set_text(r"$\alpha$ = " + f"{round(mR[1],3)}")

    mT = leastsq(
        residualsExponential,
        x0=(1, -1),
        args=(T0, T),
    )[0]

    ax[0, 1].plot(T, T0)
    ax[0, 1].plot(T, exponential(T, mT))
    ax[0, 1].set(xlabel="Time (mins)", ylabel="Correlation")
    ax[0, 1].title.set_text(r"$\beta$ = " + f"{round(mT[1],3)}")

    t, r = np.mgrid[0:150:10, 0:100:10]
    correlationModel = exponential(r, mR) * exponential(t, mT)
    t, r = np.mgrid[0:160:10, 0:110:10]

    maxCorr = np.max(correlationModel)

    c = ax[1, 0].pcolor(t, r, correlationModel, cmap="Reds", vmin=0, vmax=maxCorr)
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set_xlabel("Time (min)")
    ax[1, 0].set_ylabel(r"$R (\mu m)$")
    ax[1, 0].title.set_text(f"Correlation fit curve {fileType}")

    z_min, z_max = -maxCorr, maxCorr
    midpoint = 1 - z_max / (z_max + abs(z_min))
    orig_cmap = matplotlib.cm.RdBu_r
    shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")

    c = ax[1, 1].pcolor(
        t,
        r,
        correlationDiff - correlationModel,
        cmap=shifted_cmap,
        vmin=-maxCorr,
        vmax=maxCorr,
    )
    fig.colorbar(c, ax=ax[1, 1])
    ax[1, 1].set_xlabel("Time (min)")
    ax[1, 1].set_ylabel(r"$R (\mu m)$")
    ax[1, 1].title.set_text(f"Difference in Correlation and fit {fileType}")

    fig.savefig(
        f"results/Correlation Fit {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


# divisions and density cor

if False:
    _filenames = filenames
    for filename in _filenames:
        filenames = [filename]

        T = np.array(range(10)) * 2
        R = np.array(range(10)) * 2
        correlationFile = [
            [[[] for col in range(len(filenames))] for col in range(len(R))]
            for col in range(len(T))
        ]

        N = []
        divisionNum = []

        for m in range(len(filenames)):

            filename = filenames[m]
            print(filename)
            cells = np.zeros([181, 149, 149])
            outPlanePixel = sm.io.imread(
                f"dat/{filename}/outPlane{filename}.tif"
            ).astype(float)
            outPlane = []
            for t in range(len(outPlanePixel)):
                img = Image.fromarray(outPlanePixel[t])
                outPlane.append(np.array(img.resize((149, 149))))
            outPlane = np.array(outPlane)
            outPlane[outPlane > 50] = 255
            outPlane[outPlane < 0] = 0

            N.append(181 * 149 ** 2 - np.sum(outPlane) / 255)

            dfShape = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")
            for i in range(len(dfShape)):
                x = dfShape["Centroid"].iloc[i][0] * scale
                y = dfShape["Centroid"].iloc[i][1] * scale
                t = dfShape["Time"].iloc[i]
                cells[round(t), round(x), round(y)] = 1

            df = dfSpaceTime[dfSpaceTime["Filename"] == filename]
            x = np.array(df["X"])
            y = np.array(df["Y"])
            t = np.array(df["T"])
            divisionNum.append(len(x))

            for i in range(len(T)):
                print(i)
                t0 = T[i]
                t1 = t0 + 2
                for j in range(len(R)):
                    r0 = R[j]
                    r1 = r0 + 2
                    for k in range(len(x)):
                        if t[k] + t0 < 181:
                            shell = inPlaneShell(
                                round(x[k]), round(y[k]), t[k], t0, t1, r0, r1, outPlane
                            )
                            if np.sum(shell) != 0:
                                correlationFile[i][j][m].append(
                                    np.sum(cells[shell == 1]) / np.sum(shell)
                                )

        correlation = [[] for col in range(len(T))]
        for i in range(len(T)):
            for j in range(len(R)):
                cor = []
                for m in range(len(filenames)):
                    cor.append(np.sum(correlationFile[i][j][m]) / N[m])

                correlation[i].append(np.mean(cor))

        # ------------

        correlation_ran = [
            [[[] for col in range(len(filenames))] for col in range(len(R))]
            for col in range(len(T))
        ]

        for m in range(len(filenames)):
            print(filenames[m])
            n = 300

            x = 148 * np.random.random_sample(n)
            y = 148 * np.random.random_sample(n)
            t = 180 * np.random.random_sample(n)

            for i in range(len(T)):
                print(i)
                t0 = T[i]
                t1 = t0 + 2
                for j in range(len(R)):
                    r0 = R[j]
                    r1 = r0 + 2
                    for k in range(len(x)):
                        if t[k] + t0 < 181:
                            if outPlane[round(t[k]), round(x[k]), round(y[k])] == 0:
                                shell = inPlaneShell(
                                    round(x[k]),
                                    round(y[k]),
                                    round(t[k]),
                                    t0,
                                    t1,
                                    r0,
                                    r1,
                                    outPlane,
                                )
                                if np.sum(shell) != 0:
                                    correlation_ran[i][j][m].append(
                                        np.sum(cells[shell == 1]) / np.sum(shell)
                                    )

        correlationRan = [[] for col in range(len(T))]
        for i in range(len(T)):
            for j in range(len(R)):
                cor = []
                for m in range(len(filenames)):
                    cor.append(
                        np.sum(correlation_ran[i][j][m]) * (divisionNum[m] / (N[m] * n))
                    )

                correlationRan[i].append(np.mean(cor))

        t, r = np.mgrid[0:22:2, 0:22:2]
        fig, ax = plt.subplots(1, 3, figsize=(27, 7))
        plt.subplots_adjust(wspace=0.3)
        plt.gcf().subplots_adjust(bottom=0.15)

        correlation = np.array(correlation)
        correlationRan = np.array(correlationRan)

        correlationDiff = correlation - correlationRan

        correlationRan = correlationRan / correlation[0][0]
        correlation = correlation / correlation[0][0]

        _df = []

        _df.append(
            {
                "correlationFile": correlationFile,
                "correlation_ran": correlation_ran,
                "correlation": correlation,
                "correlationRan": correlationRan,
            }
        )

        df = pd.DataFrame(_df)
        df.to_pickle(f"databases/correlationArraysAreaDivision{filename}.pkl")

        maxCorr = np.max(correlation)

        c = ax[0].pcolor(t, r, correlation, cmap="Reds", vmin=0, vmax=maxCorr)
        fig.colorbar(c, ax=ax[0])
        ax[0].set_xlabel("Time (min)")
        ax[0].set_ylabel(r"$R (\mu m)$ ")
        ax[0].title.set_text(f"Correlation AreaDivision {filename}")

        c = ax[1].pcolor(t, r, correlationRan, cmap="Reds", vmin=0, vmax=maxCorr)
        fig.colorbar(c, ax=ax[1])
        ax[1].set_xlabel("Time (min)")
        ax[1].set_ylabel(r"$R (\mu m)$")
        ax[1].title.set_text(f"Correlation of Random i {filename}")

        z_min, z_max = -maxCorr, maxCorr
        midpoint = 1 - z_max / (z_max + abs(z_min))
        orig_cmap = matplotlib.cm.RdBu_r
        shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")

        c = ax[2].pcolor(
            t,
            r,
            correlation - correlationRan,
            cmap=shifted_cmap,
            vmin=-maxCorr,
            vmax=maxCorr,
        )
        fig.colorbar(c, ax=ax[2])
        ax[2].set_xlabel("Time (min)")
        ax[2].set_ylabel(r"$R (\mu m)$")
        ax[2].title.set_text(f"Difference in Correlation AreaDivision {filename}")

        fig.savefig(
            f"results/Division Correlation AreaDivision {filename}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

    filenames, fileType = cl.getFilesType()


if False:

    correlations = np.zeros([len(filenames), 10, 10])
    correlationRans = np.zeros([len(filenames), 10, 10])
    for m in range(len(filenames)):
        filename = filenames[m]

        df = pd.read_pickle(f"databases/correlationArraysAreaDivision{filename}.pkl")

        correlations[m] = df["correlation"][0]
        correlationRans[m] = df["correlationRan"][0]

    correlation = np.mean(correlations, axis=0)
    correlationRan = np.mean(correlationRans, axis=0)

    t, r = np.mgrid[0:22:2, 0:22:2]
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    maxCorr = np.max(correlation)

    c = ax[0, 0].pcolor(t, r, correlation, cmap="Reds", vmin=0, vmax=maxCorr)
    fig.colorbar(c, ax=ax[0, 0])
    ax[0, 0].set_xlabel("Time (min)")
    ax[0, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 0].title.set_text(f"Correlation {fileType}")

    c = ax[0, 1].pcolor(t, r, correlationRan, cmap="Reds", vmin=0, vmax=maxCorr)
    fig.colorbar(c, ax=ax[0, 1])
    ax[0, 1].set_xlabel("Time (min)")
    ax[0, 1].set_ylabel(r"$R (\mu m)$")
    ax[0, 1].title.set_text(f"Correlation of Random i {fileType}")

    correlationDiff = correlation - correlationRan

    z_min, z_max = -maxCorr, maxCorr
    midpoint = 1 - z_max / (z_max + abs(z_min))
    orig_cmap = matplotlib.cm.RdBu_r
    shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")

    c = ax[1, 0].pcolor(
        t,
        r,
        correlation - correlationRan,
        cmap=shifted_cmap,
        vmin=-maxCorr,
        vmax=maxCorr,
    )
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set_xlabel("Time (min)")
    ax[1, 0].set_ylabel(r"$R (\mu m)$")
    ax[1, 0].title.set_text(f"Difference in Correlation AreaDivision {fileType}")

    fig.savefig(
        f"results/Division Correlation AreaDivision {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


def exponential(x, coeffs):
    A = coeffs[0]
    b = coeffs[1]
    return A * np.exp(x * b)


def residualsExponential(coeffs, y, x):
    return y - exponential(x, coeffs)


if False:

    correlations = np.zeros([len(filenames), 15, 10])
    correlationRans = np.zeros([len(filenames), 15, 10])
    oriCorrelations = np.zeros([len(filenames), 15, 10])
    for m in range(len(filenames)):
        filename = filenames[m]

        df = pd.read_pickle(f"databases/correlationArraysAreaDivision{filename}.pkl")

        correlations[m] = df["correlation"][0]
        correlationRans[m] = df["correlationRan"][0]

    correlation = np.mean(correlations, axis=0)
    correlationRan = np.mean(correlationRans, axis=0)

    correlationDiff = correlation - correlationRan

    R0 = correlationDiff[0]
    T0 = correlationDiff[:, 0]
    T = np.array(range(15)) * 10
    R = np.array(range(10)) * 10

    mR = leastsq(
        residualsExponential,
        x0=(1, -1),
        args=(R0, R),
    )[0]

    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    ax[0, 0].plot(R, R0)
    ax[0, 0].plot(R, exponential(R, mR))
    ax[0, 0].set(xlabel=r"Distance $(\mu m)$", ylabel="Correlation")
    ax[0, 0].title.set_text(r"$\alpha$ = " + f"{round(mR[1],3)}")

    mT = leastsq(
        residualsExponential,
        x0=(1, -1),
        args=(T0, T),
    )[0]

    ax[0, 1].plot(T, T0)
    ax[0, 1].plot(T, exponential(T, mT))
    ax[0, 1].set(xlabel="Time (mins)", ylabel="Correlation")
    ax[0, 1].title.set_text(r"$\beta$ = " + f"{round(mT[1],3)}")

    t, r = np.mgrid[0:150:10, 0:100:10]
    correlationModel = exponential(r, mR) * exponential(t, mT)
    t, r = np.mgrid[0:160:10, 0:110:10]

    maxCorr = np.max(correlationModel)

    c = ax[1, 0].pcolor(t, r, correlationModel, cmap="Reds", vmin=0, vmax=maxCorr)
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set_xlabel("Time (min)")
    ax[1, 0].set_ylabel(r"$R (\mu m)$")
    ax[1, 0].title.set_text(f"Correlation fit curve AreaDivision {fileType}")

    z_min, z_max = -maxCorr, maxCorr
    midpoint = 1 - z_max / (z_max + abs(z_min))
    orig_cmap = matplotlib.cm.RdBu_r
    shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")

    c = ax[1, 1].pcolor(
        t,
        r,
        correlationDiff - correlationModel,
        cmap=shifted_cmap,
        vmin=-maxCorr,
        vmax=maxCorr,
    )
    fig.colorbar(c, ax=ax[1, 1])
    ax[1, 1].set_xlabel("Time (min)")
    ax[1, 1].set_ylabel(r"$R (\mu m)$")
    ax[1, 1].title.set_text(
        f"Difference in Correlation and fit AreaDivision {fileType}"
    )

    fig.savefig(
        f"results/Correlation Fit AreaDivision {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


# correction of divisions band from wound
if True:
    _filenames = filenames
    for filename in _filenames:
        filenames = [filename]

        T = np.array(range(15)) * 10
        R = np.array(range(10)) * 10
        correlationFileClose = [
            [[[] for col in range(len(filenames))] for col in range(len(R))]
            for col in range(len(T))
        ]
        correlationFileFar = [
            [[[] for col in range(len(filenames))] for col in range(len(R))]
            for col in range(len(T))
        ]

        N = []
        divisionNum = np.zeros([len(filenames), 2])

        for m in range(len(filenames)):

            filename = filenames[m]
            print(filename)
            divisions = np.zeros([181, 149, 149])
            outPlanePixel = sm.io.imread(
                f"dat/{filename}/outPlane{filename}.tif"
            ).astype(float)
            distance512 = sm.io.imread(
                f"dat/{filename}/distanceWound{filename}.tif"
            ).astype(float)
            outPlane = []
            distance = []
            for t in range(len(outPlanePixel)):
                img = Image.fromarray(outPlanePixel[t])
                outPlane.append(np.array(img.resize((149, 149))))
                img = Image.fromarray(distance512[t])
                distance.append(np.array(img.resize((149, 149))))

            outPlane = np.array(outPlane)
            distance = np.array(distance)
            outPlane[outPlane > 50] = 255
            outPlane[outPlane < 0] = 0

            N.append(181 * 149 ** 2 - np.sum(outPlane) / 255)

            df = dfSpaceTime[dfSpaceTime["Filename"] == filename]
            x = np.array(df["X"])
            y = np.array(df["Y"])
            t = np.array(df["T"])
            ori = np.array(df["Orientation"])

            for i in range(len(x)):
                divisions[round(t[i]), round(x[i]), round(y[i])] = 1

            for i in range(len(T)):
                print(i)
                t0 = T[i]
                t1 = t0 + 10
                for j in range(len(R)):
                    r0 = R[j]
                    r1 = r0 + 10
                    for k in range(len(x)):
                        if t[k] + t0 < 181:
                            shell = inPlaneShell(
                                round(x[k]), round(y[k]), t[k], t0, t1, r0, r1, outPlane
                            )
                            if np.sum(shell) != 0:
                                if (
                                    distance[t[k], round(x[k]), round(y[k])]
                                    < 40 / scale
                                ):
                                    correlationFileClose[i][j][m].append(
                                        np.sum(divisions[shell == 1]) / np.sum(shell)
                                    )

                                else:
                                    correlationFileFar[i][j][m].append(
                                        np.sum(divisions[shell == 1]) / np.sum(shell)
                                    )

        correlationClose = [[] for col in range(len(T))]
        correlationFar = [[] for col in range(len(T))]
        for i in range(len(T)):
            for j in range(len(R)):
                corC = []
                corF = []
                for m in range(len(filenames)):
                    corC.append(np.sum(correlationFileClose[i][j][m]) / N[m])
                    corF.append(np.sum(correlationFileFar[i][j][m]) / N[m])

                correlationClose[i].append(np.mean(corC))
                correlationFar[i].append(np.mean(corF))

        # ------------

        correlation_ranClose = [
            [[[] for col in range(len(filenames))] for col in range(len(R))]
            for col in range(len(T))
        ]
        correlation_ranFar = [
            [[[] for col in range(len(filenames))] for col in range(len(R))]
            for col in range(len(T))
        ]

        for m in range(len(filenames)):
            print(filenames[m])
            df = dfSpaceTime[dfSpaceTime["Filename"] == filenames[m]]
            n = 300

            x = 148 * np.random.random_sample(n)
            y = 148 * np.random.random_sample(n)
            t = 180 * np.random.random_sample(n)

            for i in range(len(T)):
                print(i)
                t0 = T[i]
                t1 = t0 + 10
                for j in range(len(R)):
                    r0 = R[j]
                    r1 = r0 + 10
                    for k in range(len(x)):
                        if t[k] + t0 < 181:
                            if outPlane[round(t[k]), round(x[k]), round(y[k])] == 0:
                                shell = inPlaneShell(
                                    round(x[k]),
                                    round(y[k]),
                                    round(t[k]),
                                    t0,
                                    t1,
                                    r0,
                                    r1,
                                    outPlane,
                                )
                                if np.sum(shell) != 0:
                                    if (
                                        distance[round(t[k]), round(x[k]), round(y[k])]
                                        < 40 / scale
                                    ):
                                        correlation_ranClose[i][j][m].append(
                                            np.sum(divisions[shell == 1])
                                            / np.sum(shell)
                                        )
                                        divisionNum[m, 0] += 1
                                    else:
                                        correlation_ranFar[i][j][m].append(
                                            np.sum(divisions[shell == 1])
                                            / np.sum(shell)
                                        )
                                        divisionNum[m, 1] += 1

        correlationRanClose = [[] for col in range(len(T))]
        correlationRanFar = [[] for col in range(len(T))]
        for i in range(len(T)):
            for j in range(len(R)):
                corC = []
                corF = []
                for m in range(len(filenames)):
                    corC.append(
                        np.sum(correlation_ranClose[i][j][m])
                        * (divisionNum[m, 0] / (N[m] * n))
                    )
                    corF.append(
                        np.sum(correlation_ranFar[i][j][m])
                        * (divisionNum[m, 1] / (N[m] * n))
                    )

                correlationRanClose[i].append(np.mean(corC))
                correlationRanFar[i].append(np.mean(corF))

        correlationClose = np.nan_to_num(np.array(correlationClose)[:15, :10])
        correlationFar = np.nan_to_num(np.array(correlationFar)[:15, :10])
        correlationRanClose = np.nan_to_num(np.array(correlationRanClose)[:15, :10])
        correlationRanFar = np.nan_to_num(np.array(correlationRanFar)[:15, :10])

        correlationDiffClose = correlationClose - correlationRanClose
        correlationDiffFar = correlationFar - correlationRanFar

        # correlationClose = correlationClose / correlationDiffClose[0][0]
        # correlationRanClose = correlationRanClose / correlationDiffClose[0][0]
        # correlationDiffClose = correlationDiffClose / correlationDiffClose[0][0]
        # correlationFar = correlationFar / correlationDiffFar[0][0]
        # correlationRanFar = correlationRanFar / correlationDiffFar[0][0]
        # correlationDiffFar = correlationDiffFar / correlationDiffFar[0][0]

        _df = []

        _df.append(
            {
                "correlationFileClose": correlationFileClose,
                "correlationFileFar": correlationFileFar,
                "correlation_ranClose": correlation_ranClose,
                "correlation_ranFar": correlation_ranFar,
                "correlation": correlationClose,
                "correlation": correlationFar,
                "correlationRanClose": correlationRanClose,
                "correlationRanFar": correlationRanFar,
            }
        )

        df = pd.DataFrame(_df)
        df.to_pickle(f"databases/correlationArraysCloseFar{filename}.pkl")

        t, r = np.mgrid[0:160:10, 0:110:10]
        fig, ax = plt.subplots(2, 3, figsize=(30, 20))
        plt.subplots_adjust(wspace=0.3)
        plt.gcf().subplots_adjust(bottom=0.15)

        maxCorr = np.max(correlationClose)

        c = ax[0, 0].pcolor(t, r, correlationClose, cmap="Reds", vmin=0, vmax=maxCorr)
        fig.colorbar(c, ax=ax[0, 0])
        ax[0, 0].set_xlabel("Time (min)")
        ax[0, 0].set_ylabel(r"$R (\mu m)$ ")
        ax[0, 0].title.set_text(f"Correlation Close {filename}")

        c = ax[0, 1].pcolor(
            t, r, correlationRanClose, cmap="Reds", vmin=0, vmax=maxCorr
        )
        fig.colorbar(c, ax=ax[0, 1])
        ax[0, 1].set_xlabel("Time (min)")
        ax[0, 1].set_ylabel(r"$R (\mu m)$")
        ax[0, 1].title.set_text(f"Correlation Close of Random i {filename}")

        z_min, z_max = -maxCorr, maxCorr
        midpoint = 1 - z_max / (z_max + abs(z_min))
        orig_cmap = matplotlib.cm.RdBu_r
        shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")

        c = ax[0, 2].pcolor(
            t,
            r,
            correlationClose - correlationRanClose,
            cmap=shifted_cmap,
            vmin=-maxCorr,
            vmax=maxCorr,
        )
        fig.colorbar(c, ax=ax[0, 2])
        ax[0, 2].set_xlabel("Time (min)")
        ax[0, 2].set_ylabel(r"$R (\mu m)$")
        ax[0, 2].title.set_text(f"Difference in Correlation Close {filename}")

        # -----------

        maxCorr = np.max(correlationFar)

        c = ax[1, 0].pcolor(t, r, correlationFar, cmap="Reds")
        fig.colorbar(c, ax=ax[1, 0])
        ax[1, 0].set_xlabel("Time (min)")
        ax[1, 0].set_ylabel(r"$R (\mu m)$ ")
        ax[1, 0].title.set_text(f"Correlation Far {filename}")

        c = ax[1, 1].pcolor(t, r, correlationRanFar, cmap="Reds")
        fig.colorbar(c, ax=ax[1, 1])
        ax[1, 1].set_xlabel("Time (min)")
        ax[1, 1].set_ylabel(r"$R (\mu m)$")
        ax[1, 1].title.set_text(f"Correlation Far of Random i {filename}")

        # z_min, z_max = -maxCorr, maxCorr
        # midpoint = 1 - z_max / (z_max + abs(z_min))
        # orig_cmap = matplotlib.cm.RdBu_r
        # shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")

        c = ax[1, 2].pcolor(
            t,
            r,
            correlationDiffFar,
            cmap="RdBu_r",
        )
        fig.colorbar(c, ax=ax[1, 2])
        ax[1, 2].set_xlabel("Time (min)")
        ax[1, 2].set_ylabel(r"$R (\mu m)$")
        ax[1, 2].title.set_text(f"Difference in Correlation Far {filename}")

        fig.savefig(
            f"results/Division Correlation CloseFar {filename}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

    filenames, fileType = cl.getFilesType()


if True:

    correlations = np.zeros([len(filenames), 15, 10])
    correlationRans = np.zeros([len(filenames), 15, 10])
    oriCorrelations = np.zeros([len(filenames), 15, 10])
    for m in range(len(filenames)):
        filename = filenames[m]

        df = pd.read_pickle(f"databases/correlationArraysV2-1{filename}.pkl")

        correlations[m] = df["correlation"][0]
        correlationRans[m] = df["correlationRan"][0]
        oriCorrelations[m] = df["oriCorrelation"][0]

    oriCorrelations = np.nan_to_num(oriCorrelations)

    correlation = np.mean(correlations, axis=0)
    correlationRan = np.mean(correlationRans, axis=0)
    oriCorrelation = np.mean(oriCorrelations, axis=0)

    t, r = np.mgrid[0:160:10, 0:110:10]
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    maxCorr = np.max(correlation)

    c = ax[0, 0].pcolor(t, r, correlation, cmap="Reds", vmin=0, vmax=maxCorr)
    fig.colorbar(c, ax=ax[0, 0])
    ax[0, 0].set_xlabel("Time (min)")
    ax[0, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 0].title.set_text(f"Correlation {fileType}")

    c = ax[0, 1].pcolor(t, r, correlationRan, cmap="Reds", vmin=0, vmax=maxCorr)
    fig.colorbar(c, ax=ax[0, 1])
    ax[0, 1].set_xlabel("Time (min)")
    ax[0, 1].set_ylabel(r"$R (\mu m)$")
    ax[0, 1].title.set_text(f"Correlation of Random i {fileType}")

    correlationDiff = correlation - correlationRan

    z_min, z_max = -maxCorr, maxCorr
    midpoint = 1 - z_max / (z_max + abs(z_min))
    orig_cmap = matplotlib.cm.RdBu_r
    shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")

    c = ax[1, 0].pcolor(
        t,
        r,
        correlation - correlationRan,
        cmap=shifted_cmap,
        vmin=-maxCorr,
        vmax=maxCorr,
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
        f"results/Division Correlation V2-1 {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


if True:

    correlations = np.zeros([len(filenames), 15, 10])
    correlationRans = np.zeros([len(filenames), 15, 10])
    oriCorrelations = np.zeros([len(filenames), 15, 10])
    for m in range(len(filenames)):
        filename = filenames[m]

        df = pd.read_pickle(f"databases/correlationArraysV2-1{filename}.pkl")

        correlations[m] = df["correlation"][0]
        correlationRans[m] = df["correlationRan"][0]
        oriCorrelations[m] = df["oriCorrelation"][0]

    oriCorrelations = np.nan_to_num(oriCorrelations)

    correlation = np.mean(correlations, axis=0)
    correlationRan = np.mean(correlationRans, axis=0)
    oriCorrelation = np.mean(oriCorrelations, axis=0)

    correlationDiff = correlation - correlationRan

    R0 = correlationDiff[0]
    T0 = correlationDiff[:, 0]
    T = np.array(range(15)) * 10
    R = np.array(range(10)) * 10

    mR = leastsq(
        residualsExponential,
        x0=(1, -1),
        args=(R0, R),
    )[0]

    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    ax[0, 0].plot(R, R0)
    ax[0, 0].plot(R, exponential(R, mR))
    ax[0, 0].set(xlabel=r"Distance $(\mu m)$", ylabel="Correlation")
    ax[0, 0].title.set_text(r"$\alpha$ = " + f"{round(mR[1],3)}")

    mT = leastsq(
        residualsExponential,
        x0=(1, -1),
        args=(T0, T),
    )[0]

    ax[0, 1].plot(T, T0)
    ax[0, 1].plot(T, exponential(T, mT))
    ax[0, 1].set(xlabel="Time (mins)", ylabel="Correlation")
    ax[0, 1].title.set_text(r"$\beta$ = " + f"{round(mT[1],3)}")

    t, r = np.mgrid[0:150:10, 0:100:10]
    correlationModel = exponential(r, mR) * exponential(t, mT)
    t, r = np.mgrid[0:160:10, 0:110:10]

    maxCorr = np.max(correlationModel)

    c = ax[1, 0].pcolor(t, r, correlationModel, cmap="Reds", vmin=0, vmax=maxCorr)
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set_xlabel("Time (min)")
    ax[1, 0].set_ylabel(r"$R (\mu m)$")
    ax[1, 0].title.set_text(f"Correlation fit curve {fileType}")

    z_min, z_max = -maxCorr, maxCorr
    midpoint = 1 - z_max / (z_max + abs(z_min))
    orig_cmap = matplotlib.cm.RdBu_r
    shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")

    c = ax[1, 1].pcolor(
        t,
        r,
        correlationDiff - correlationModel,
        cmap=shifted_cmap,
        vmin=-maxCorr,
        vmax=maxCorr,
    )
    fig.colorbar(c, ax=ax[1, 1])
    ax[1, 1].set_xlabel("Time (min)")
    ax[1, 1].set_ylabel(r"$R (\mu m)$")
    ax[1, 1].title.set_text(f"Difference in Correlation and fit {fileType}")

    fig.savefig(
        f"results/Correlation Fit {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")