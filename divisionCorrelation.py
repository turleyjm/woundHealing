import os
from os.path import exists
import shutil
from math import floor, log10, factorial

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
import scipy.special as sc
import scipy.linalg as linalg
import shapely
import skimage as sm
import skimage.io
import skimage.measure
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

filenames, fileType = util.getFilesType()
scale = 123.26 / 512
T = 160
timeStep = 10
R = 110
rStep = 10


def inPlaneShell(x, y, t, t0, t1, r0, r1, outPlane):

    if r0 == 0:
        r0 = 1

    t0 = t + t0
    t1 = t + t1

    T = outPlane.shape[0]

    if t1 > T - 1:
        t1 = T - 1

    background = np.zeros([T, 500 + 124, 500 + 124])

    rr1, cc1 = sm.draw.disk((250 + x, 250 + y), r0)
    rr2, cc2 = sm.draw.disk((250 + x, 250 + y), r1)

    background[t0:t1, rr2, cc2] = 1
    background[t0:t1, rr1, cc1] = 0

    inPlane = background[:, 250 : 250 + 124, 250 : 250 + 124]

    inPlane[outPlane == 255] = 0

    return inPlane


# -------------------


if False:
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    for filename in filenames:
        df = dfDivisions[dfDivisions["Filename"] == filename]
        x = np.array(df["X"])
        y = np.array(df["Y"])
        t = np.array(df["T"])
        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=t, mode="markers")])
        fig.show()
        plt.close("all")


if False:
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    x = np.array(dfDivisions["X"])
    y = np.array(dfDivisions["Y"])

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

if False:
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    ori = np.array(dfDivisions["Orientation"])

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    ax.hist(ori, bins=10)
    ax.set(xlabel="Orientation")

    fig.savefig(
        f"results/ori distributions {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

if False:
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    for filename in filenames:
        df = dfDivisions[dfDivisions["Filename"] == filename]
        ori = np.array(df["Orientation"])

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.gcf().subplots_adjust(bottom=0.15)

        ax.hist(ori, bins=10)
        ax.set(xlabel="Orientation")

        fig.savefig(
            f"results/ori distributions {filename}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")


# correction of divisions
if False:
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    expectedXY = np.zeros([int(T / timeStep), int(R / rStep)])
    oriCorr = np.zeros([int(T / timeStep), int(R / rStep)])
    ExXExY = np.zeros([int(T / timeStep), int(R / rStep)])
    N = []
    divisionNum = []
    time = np.array(range(int(T / timeStep))) * timeStep
    rad = np.array(range(int(R / rStep))) * rStep

    expectedXYFile = [
        [[[] for col in range(len(filenames))] for col in range(int(R / rStep))]
        for col in range(int(T / timeStep))
    ]
    thetaCorr = [
        [[[] for col in range(len(filenames))] for col in range(int(R / rStep))]
        for col in range(int(T / timeStep))
    ]
    ExXExYFile = [
        [[[] for col in range(len(filenames))] for col in range(int(R / rStep))]
        for col in range(int(T / timeStep))
    ]
    n = 1000
    for m in range(len(filenames)):
        filename = filenames[m]
        print(filename)
        df = dfDivisions[dfDivisions["Filename"] == filename]
        t0 = util.findStartTime(filename)

        divisions = np.zeros([90, 124, 124])
        orientations = np.zeros([90, 124, 124])
        outPlanePixel = sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype(
            float
        )[:90]
        outPlane = []
        for t in range(len(outPlanePixel)):
            img = Image.fromarray(outPlanePixel[t])
            outPlane.append(np.array(img.resize((124, 124))))
        outPlane = np.array(outPlane)
        outPlane[outPlane > 50] = 255
        outPlane[outPlane < 0] = 0

        N.append(90 * 124 ** 2 - np.sum(outPlane) / 255)

        x = np.array(df["X"])
        y = np.array(df["Y"])
        t = np.array((df["T"] - t0) / 2)
        ori = np.array(df["Orientation"])

        count = 0
        for i in range(len(x)):
            if outPlane[int(t[i]), int(x[i]), int(y[i])] == 0:
                divisions[int(t[i]), int(x[i]), int(y[i])] = 1
                orientations[int(t[i]), int(x[i]), int(y[i])] = ori[i]
                count += 1

        divisionNum.append(count)

        for i in range(len(time)):
            print(i)
            t0 = int(time[i] / 2)
            t1 = int(t0 + timeStep / 2)
            for j in range(len(rad)):
                r0 = rad[j]
                r1 = r0 + 10
                for k in range(len(x)):
                    if t[k] + t0 < 91:
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
                                expectedXYFile[i][j][m].append(
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
                                    thetaCorr[i][j][m].append(np.dot(v, u))

        x = 124 * np.random.random_sample(n)
        y = 124 * np.random.random_sample(n)
        t = 90 * np.random.random_sample(n)

        for i in range(len(time)):
            print(i)
            t0 = int(time[i] / 2)
            t1 = int(t0 + timeStep / 2)
            for j in range(len(rad)):
                r0 = rad[j]
                r1 = r0 + 10
                for k in range(len(x)):
                    if t[k] + t0 < 90:
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
                                ExXExYFile[i][j][m].append(
                                    np.sum(divisions[shell == 1]) / np.sum(shell)
                                )

    for i in range(len(time)):
        for j in range(len(rad)):
            cor = []
            ori = []
            corRan = []
            for m in range(len(filenames)):
                cor.append(np.sum(expectedXYFile[i][j][m]) / N[m])
                ori.append(np.mean(thetaCorr[i][j][m]))
                corRan.append(
                    np.sum(ExXExYFile[i][j][m]) * (divisionNum[m] / (N[m] * n))
                )

            expectedXY[i, j] = np.mean(cor)
            oriCorr[i, j] = np.mean(ori)
            ExXExY[i, j] = np.mean(corRan)

    expectedXY = np.nan_to_num(expectedXY)
    ExXExY = np.nan_to_num(ExXExY)
    oriCorrelation = np.nan_to_num(oriCorr)

    divCorr = expectedXY - ExXExY

    _df = []

    _df.append(
        {
            "expectedXYFile": expectedXYFile,
            "ExXExYFile": ExXExYFile,
            "expectedXY": expectedXY,
            "ExXExY": ExXExY,
            "divCorr": divCorr,
            "thetaCorr": thetaCorr,
            "oriCorr": oriCorr,
        }
    )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/divCorr{fileType}.pkl")

if True:
    df = pd.read_pickle(f"databases/divCorr{fileType}.pkl")
    expectedXY = df["expectedXY"].iloc[0]
    ExXExY = df["ExXExY"].iloc[0]
    divCorr = df["divCorr"].iloc[0]
    oriCorr = df["oriCorr"].iloc[0]
    df = 0
    maxCorr = np.max(expectedXY)

    t, r = np.mgrid[0:160:10, 0:110:10]
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    c = ax[0, 0].pcolor(t, r, expectedXY, cmap="Reds", vmin=0, vmax=maxCorr)
    fig.colorbar(c, ax=ax[0, 0])
    ax[0, 0].set_xlabel("Time (min)")
    ax[0, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 0].title.set_text(f"expectedXY")

    c = ax[0, 1].pcolor(t, r, ExXExY, cmap="Reds", vmin=0, vmax=maxCorr)
    fig.colorbar(c, ax=ax[0, 1])
    ax[0, 1].set_xlabel("Time (min)")
    ax[0, 1].set_ylabel(r"$R (\mu m)$")
    ax[0, 1].title.set_text(f"ExXExY")

    c = ax[1, 0].pcolor(
        t,
        r,
        divCorr,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
    )
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set_xlabel("Time (min)")
    ax[1, 0].set_ylabel(r"$R (\mu m)$")
    ax[1, 0].title.set_text(f"Correlation")

    c = ax[1, 1].pcolor(
        t,
        r,
        oriCorr,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )
    fig.colorbar(c, ax=ax[1, 1])
    ax[1, 1].set_xlabel("Time (min)")
    ax[1, 1].set_ylabel(r"$R (\mu m)$")
    ax[1, 1].title.set_text(f"Correlation Orientation")

    fig.savefig(
        f"results/Division Correlation {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

    t, r = np.mgrid[0:100:10, 0:110:10]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    maxCorr = np.max(divCorr[:10])
    c = ax.pcolor(
        t,
        r,
        divCorr[:10] * 10000 ** 2,
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
    )
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"$R (\mu m)$")
    ax.title.set_text(f"Division density correlation")

    fig.savefig(
        f"results/Division Correlation figure {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")


if False:
    df = pd.read_pickle(f"databases/divCorr{fileType}.pkl")
    thetaCorr = df["thetaCorr"].iloc[0]
    df = 0
    time = np.array(range(int(T / timeStep))) * timeStep
    rad = np.array(range(int(R / rStep))) * rStep

    t, r = np.mgrid[0:160:10, 0:110:10]

    for m in range(len(filenames)):
        oriCorr = np.zeros([int(T / timeStep), int(R / rStep)])

        for i in range(len(time)):
            for j in range(len(rad)):
                oriCorr[i][j] = np.mean(thetaCorr[i][j][m])

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.subplots_adjust(wspace=0.3)
        plt.gcf().subplots_adjust(bottom=0.15)

        c = ax.pcolor(
            t,
            r,
            oriCorr,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        fig.colorbar(c, ax=ax)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(r"$R (\mu m)$")
        ax.title.set_text(f"Correlation Orientation {filenames[m]}")

        fig.savefig(
            f"results/Correlation Orientation {filenames[m]}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")


# correction of divisions and
if False:
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    expectedXY = np.zeros([int(T / timeStep), int(R / rStep)])
    ExXExY = np.zeros([int(T / timeStep), int(R / rStep)])
    N = []
    divisionNum = []
    time = np.array(range(int(T / timeStep))) * timeStep
    rad = np.array(range(int(R / rStep))) * rStep

    expectedXYFile = [
        [[[] for col in range(len(filenames))] for col in range(int(R / rStep))]
        for col in range(int(T / timeStep))
    ]
    ExXExYFile = [
        [[[] for col in range(len(filenames))] for col in range(int(R / rStep))]
        for col in range(int(T / timeStep))
    ]
    n = 1000
    for m in range(len(filenames)):
        filename = filenames[m]
        print(filename)
        df = dfDivisions[dfDivisions["Filename"] == filename]
        dfFileShape = dfShape[dfShape["Filename"] == filename]
        t0 = util.findStartTime(filename)

        cells = np.zeros([90, 124, 124])
        outPlanePixel = sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype(
            float
        )[:90]
        outPlane = []
        for t in range(len(outPlanePixel)):
            img = Image.fromarray(outPlanePixel[t])
            outPlane.append(np.array(img.resize((124, 124))))
        outPlane = np.array(outPlane)
        outPlane[outPlane > 50] = 255
        outPlane[outPlane < 0] = 0

        N.append(90 * 124 ** 2 - np.sum(outPlane) / 255)

        x = np.array(df["X"])
        y = np.array(df["Y"])
        t = np.array((df["T"] - t0) / 2)
        X = np.stack(np.array(dfFileShape.loc[:, "Centroid"]), axis=0)[:, 0]
        Y = np.stack(np.array(dfFileShape.loc[:, "Centroid"]), axis=0)[:, 1]
        T = np.array(dfFileShape["T"])

        count = 0
        for i in range(len(x)):
            if outPlane[int(t[i]), int(x[i]), int(y[i])] == 0:
                count += 1
        divisionNum.append(count)

        for i in range(len(X)):
            if outPlane[int(T[i]), int(X[i]), int(Y[i])] == 0:
                cells[int(T[i]), int(X[i]), int(Y[i])] = 1

        for i in range(len(time)):
            print(i)
            t0 = int(time[i] / 2)
            t1 = int(t0 + timeStep / 2)
            for j in range(len(rad)):
                r0 = rad[j]
                r1 = r0 + 10
                for k in range(len(x)):
                    if t[k] + t0 < 91:
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
                                expectedXYFile[i][j][m].append(
                                    np.sum(cells[shell == 1]) / np.sum(shell)
                                )

        x = 124 * np.random.random_sample(n)
        y = 124 * np.random.random_sample(n)
        t = 90 * np.random.random_sample(n)

        for i in range(len(time)):
            print(i)
            t0 = int(time[i] / 2)
            t1 = int(t0 + timeStep / 2)
            for j in range(len(rad)):
                r0 = rad[j]
                r1 = r0 + 10
                for k in range(len(x)):
                    if t[k] + t0 < 90:
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
                                ExXExYFile[i][j][m].append(
                                    np.sum(cells[shell == 1]) / np.sum(shell)
                                )

    for i in range(len(time)):
        for j in range(len(rad)):
            cor = []
            corRan = []
            for m in range(len(filenames)):
                cor.append(np.sum(expectedXYFile[i][j][m]) / N[m])
                corRan.append(
                    np.sum(ExXExYFile[i][j][m]) * (divisionNum[m] / (N[m] * n))
                )

            expectedXY[i, j] = np.mean(cor)
            ExXExY[i, j] = np.mean(corRan)

    expectedXY = np.nan_to_num(expectedXY)
    ExXExY = np.nan_to_num(ExXExY)

    divRhoCorr = expectedXY - ExXExY

    _df = []

    _df.append(
        {
            "expectedXYFile": expectedXYFile,
            "ExXExYFile": ExXExYFile,
            "expectedXY": expectedXY,
            "ExXExY": ExXExY,
            "divRhoCorr": divRhoCorr,
        }
    )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/divRhoCorr{fileType}.pkl")


if False:
    df = pd.read_pickle(f"databases/divRhoCorr{fileType}.pkl")
    expectedXY = df["expectedXY"].iloc[0]
    ExXExY = df["ExXExY"].iloc[0]
    divRhoCorr = df["divRhoCorr"].iloc[0]
    df = 0
    maxCorr = np.max(expectedXY)

    t, r = np.mgrid[0:160:10, 0:110:10]
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    c = ax[0, 0].pcolor(t, r, expectedXY, cmap="Reds", vmin=0, vmax=maxCorr)
    fig.colorbar(c, ax=ax[0, 0])
    ax[0, 0].set_xlabel("Time (min)")
    ax[0, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 0].title.set_text(f"expectedXY")

    c = ax[0, 1].pcolor(t, r, ExXExY, cmap="Reds", vmin=0, vmax=maxCorr)
    fig.colorbar(c, ax=ax[0, 1])
    ax[0, 1].set_xlabel("Time (min)")
    ax[0, 1].set_ylabel(r"$R (\mu m)$")
    ax[0, 1].title.set_text(f"ExXExY")

    c = ax[1, 0].pcolor(
        t,
        r,
        divRhoCorr,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
    )
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set_xlabel("Time (min)")
    ax[1, 0].set_ylabel(r"$R (\mu m)$")
    ax[1, 0].title.set_text(f"Correlation")

    fig.savefig(
        f"results/Division Rho Correlation {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")
