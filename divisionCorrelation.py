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

    T = outPlane.shape[0]

    if t1 > T - 1:
        t1 = T - 1

    background = np.zeros([T, 500 + 148, 500 + 148])

    rr1, cc1 = sm.draw.disk((250 + x, 250 + y), r0)
    rr2, cc2 = sm.draw.disk((250 + x, 250 + y), r1)

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
            divisions = np.zeros([181, 148, 148])
            orientations = np.zeros([181, 148, 148])
            outPlanePixel = sm.io.imread(
                f"dat/{filename}/outPlane{filename}.tif"
            ).astype(float)
            outPlane = []
            for t in range(len(outPlanePixel)):
                img = Image.fromarray(outPlanePixel[t])
                outPlane.append(np.array(img.resize((148, 148))))
            outPlane = np.array(outPlane)
            outPlane[outPlane > 50] = 255
            outPlane[outPlane < 0] = 0

            N.append(181 * 148 ** 2 - np.sum(outPlane) / 255)

            df = dfSpaceTime[dfSpaceTime["Filename"] == filename]
            x = np.array(df["X"])
            y = np.array(df["Y"])
            t = np.array(df["T"])
            ori = np.array(df["Orientation"])

            divisionNum.append(len(x))

            for i in range(len(x)):
                divisions[int(t[i]), int(x[i]), int(y[i])] = 1
                orientations[int(t[i]), int(x[i]), int(y[i])] = ori[i]

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
                                int(x[k]), int(y[k]), t[k], t0, t1, r0, r1, outPlane
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
                            if outPlane[int(t[k]), int(x[k]), int(y[k])] == 0:
                                shell = inPlaneShell(
                                    int(x[k]),
                                    int(y[k]),
                                    int(t[k]),
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
    ax[0, 0].plot(R + 10, R0)
    ax[0, 0].plot(R + 10, exponential(R, mR))
    ax[0, 0].set(xlabel=r"Distance $(\mu m)$", ylabel="Correlation")
    ax[0, 0].title.set_text(r"$\alpha$ = " + f"{round(mR[1],3)}")

    mT = leastsq(
        residualsExponential,
        x0=(1, -1),
        args=(T0, T),
    )[0]

    ax[0, 1].plot(T + 10, T0)
    ax[0, 1].plot(T + 10, exponential(T, mT))
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
            cells = np.zeros([181, 148, 148])
            outPlanePixel = sm.io.imread(
                f"dat/{filename}/outPlane{filename}.tif"
            ).astype(float)
            outPlane = []
            for t in range(len(outPlanePixel)):
                img = Image.fromarray(outPlanePixel[t])
                outPlane.append(np.array(img.resize((148, 148))))
            outPlane = np.array(outPlane)
            outPlane[outPlane > 50] = 255
            outPlane[outPlane < 0] = 0

            N.append(181 * 148 ** 2 - np.sum(outPlane) / 255)

            dfShape = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")
            for i in range(len(dfShape)):
                x = dfShape["Centroid"].iloc[i][0] * scale
                y = dfShape["Centroid"].iloc[i][1] * scale
                t = dfShape["Time"].iloc[i]
                cells[int(t), int(x), int(y)] = 1

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
                                int(x[k]), int(y[k]), t[k], t0, t1, r0, r1, outPlane
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


# Division Velocity Correlation
if True:
    _filenames = filenames
    dfVelocity = pd.read_pickle(f"databases/dfVelocity{fileType}.pkl")
    for filename in _filenames:
        filenames = [filename]

        T = np.array(range(15)) * 10
        R = np.array(range(10)) * 10
        velCorrelation = [
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
            radialVel = np.zeros([180, 148, 148])
            thetaVel = np.zeros([180, 148, 148])

            outPlanePixel = sm.io.imread(
                f"dat/{filename}/outPlane{filename}.tif"
            ).astype(float)
            outPlane = []
            for t in range(180):
                img = Image.fromarray(outPlanePixel[t])
                outPlane.append(np.array(img.resize((148, 148))))
            outPlane = np.array(outPlane)
            outPlane[outPlane > 50] = 255
            outPlane[outPlane < 0] = 0

            N.append(180 * 148 ** 2 - np.sum(outPlane) / 255)

            df = dfSpaceTime[dfSpaceTime["Filename"] == filename]
            x = np.array(df["X"])
            y = np.array(df["Y"])
            t = np.array(df["T"])
            ori = np.array(df["Orientation"])

            divisionNum.append(len(x))

            dfVelFilename = dfVelocity[dfVelocity["Filename"] == filename]
            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")

            x_v = np.array(dfVelFilename["X"]) * scale
            y_v = np.array(dfVelFilename["Y"]) * scale
            t_v = np.array(dfVelFilename["Time"])
            V = np.array(dfVelFilename["Velocity"]) * scale

            Vmu = []
            for tau in range(180):
                dft = dfVelFilename[dfVelFilename["Time"] == tau]
                Vmu.append(np.mean(list(dft["Velocity"]), axis=0) * scale)

            for i in range(len(x_v)):
                wx = dfWound["Position"].iloc[int(t_v[i])][0] * scale
                wy = dfWound["Position"].iloc[int(t_v[i])][1] * scale
                theta = np.arctan2(y_v[i] - wy, x_v[i] - wx)
                rotation = cl.rotation_matrix(-theta)
                v = -np.matmul(rotation, V[i] - Vmu[int(t_v[i])])
                radialVel[int(t_v[i]), int(x_v[i]), int(y_v[i])] = v[0]
                thetaVel[int(t_v[i]), int(x_v[i]), int(y_v[i])] = np.arctan2(v[1], v[0])

            for i in range(len(T)):
                print(i)
                t0 = T[i]
                t1 = t0 + 10
                for j in range(len(R)):
                    r0 = R[j]
                    r1 = r0 + 10
                    for k in range(len(x)):
                        if t[k] + t0 < 180:
                            shell = inPlaneShell(
                                int(x[k]), int(y[k]), t[k], t0, t1, r0, r1, outPlane
                            )
                            velocities = radialVel[shell == 1]
                            velocities = velocities[velocities != 0]
                            if np.sum(velocities) != 0:
                                velCorrelation[i][j][m].append(np.mean(velocities))
                                if np.isnan(np.mean(velocities)):
                                    print(0)
                            thetas = thetaVel[shell == 1]
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
                                            np.cos(np.pi * theta),
                                            np.sin(np.pi * theta),
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
                    cor.append(np.sum(velCorrelation[i][j][m]) / N[m])
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
                        if t[k] + t0 < 180:
                            if outPlane[int(t[k]), int(x[k]), int(y[k])] == 0:
                                shell = inPlaneShell(
                                    int(x[k]),
                                    int(y[k]),
                                    int(t[k]),
                                    t0,
                                    t1,
                                    r0,
                                    r1,
                                    outPlane,
                                )
                                velocities = radialVel[shell == 1]
                                velocities = velocities[velocities != 0]
                                if np.sum(velocities) != 0:
                                    correlation_ran[i][j][m].append(np.mean(velocities))

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
        correlation = np.nan_to_num(correlation)
        correlationRan = np.nan_to_num(correlationRan)
        oriCorrelation = np.nan_to_num(oriCorrelation)

        correlationDiff = correlation - correlationRan

        # correlation = correlation / correlationDiff[0][0]
        # correlationRan = correlationRan / correlationDiff[0][0]
        # correlationDiff = correlationDiff / correlationDiff[0][0]

        _df = []

        _df.append(
            {
                "correlationFile": velCorrelation,
                "correlation_ran": correlation_ran,
                "correlation": correlation,
                "correlationRan": correlationRan,
                "oriCorrelation": oriCorrelation,
            }
        )

        df = pd.DataFrame(_df)
        df.to_pickle(f"databases/divisionVelocityCorrelation{filename}.pkl")

        maxCorr = np.max(correlation)

        c = ax[0, 0].pcolor(t, r, correlation, cmap="Reds", vmin=0, vmax=maxCorr)
        fig.colorbar(c, ax=ax[0, 0])
        ax[0, 0].set_xlabel("Time (min)")
        ax[0, 0].set_ylabel(r"$R (\mu m)$ ")
        ax[0, 0].title.set_text(r"$\frac{1}{N}M^i V^i$")

        c = ax[0, 1].pcolor(t, r, correlationRan, cmap="Reds", vmin=0, vmax=maxCorr)
        fig.colorbar(c, ax=ax[0, 1])
        ax[0, 1].set_xlabel("Time (min)")
        ax[0, 1].set_ylabel(r"$R (\mu m)$")
        ax[0, 1].title.set_text(r"$(\frac{N_d}{N})(\frac{1}{N}M^i)$")

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
        ax[1, 0].title.set_text(f"Correlation Division-Velocity {filename}")

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
        ax[1, 1].title.set_text(f"Correlation Division-Velocity Orientation {filename}")

        fig.savefig(
            f"results/Division-Velocity Correlation {filename}",
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

        df = pd.read_pickle(f"databases/divisionVelocityCorrelation{filename}.pkl")

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
        f"results/Division Velocity Correlation {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")