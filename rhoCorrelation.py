import os
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
plt.rcParams.update({"font.size": 14})


# -------------------

filenames, fileType = util.getFilesType()
T = 90
scale = 123.26 / 512


# -------------------


if False:
    _df2 = []
    _df = []
    for filename in filenames:

        df = pd.read_pickle(f"dat/{filename}/shape{filename}.pkl")
        Q = np.mean(df["q"])
        theta0 = np.arccos(Q[0, 0] / (Q[0, 0] ** 2 + Q[0, 1] ** 2) ** 0.5) / 2
        R = util.rotation_matrix(-theta0)

        df = pd.read_pickle(f"dat/{filename}/nucleusVelocity{filename}.pkl")
        mig = np.zeros(2)

        for t in range(T):
            dft = df[df["T"] == t]
            v = np.mean(dft["Velocity"]) * scale
            v = np.matmul(R, v)
            _df.append(
                {
                    "Filename": filename,
                    "T": t,
                    "v": v,
                }
            )

            for i in range(len(dft)):
                x = dft["X"].iloc[i] * scale
                y = dft["Y"].iloc[i] * scale
                dv = np.matmul(R, dft["Velocity"].iloc[i] * scale) - v
                [x, y] = np.matmul(R, np.array([x, y]))

                _df2.append(
                    {
                        "Filename": filename,
                        "T": t,
                        "X": x - mig[0],
                        "Y": y - mig[1],
                        "dv": dv,
                    }
                )
            mig += v

    dfVelocityMean = pd.DataFrame(_df)
    dfVelocityMean.to_pickle(f"databases/dfVelocityMean{fileType}.pkl")
    dfVelocity = pd.DataFrame(_df2)
    dfVelocity.to_pickle(f"databases/dfVelocity{fileType}.pkl")

if False:
    _df2 = []
    dfVelocity = pd.read_pickle(f"databases/dfVelocity{fileType}.pkl")
    dfVelocityMean = pd.read_pickle(f"databases/dfVelocityMean{fileType}.pkl")
    for filename in filenames:

        df = pd.read_pickle(f"dat/{filename}/shape{filename}.pkl")
        dfFilename = dfVelocityMean[dfVelocityMean["Filename"] == filename]
        mig = np.zeros(2)
        Q = np.mean(df["q"])
        theta0 = np.arctan2(Q[0, 1], Q[0, 0]) / 2
        R = util.rotation_matrix(-theta0)

        for t in range(T):
            dft = df[df["Time"] == t]
            Q = np.matmul(R, np.matmul(np.mean(dft["q"]), np.matrix.transpose(R)))
            P = np.matmul(R, np.mean(dft["Polar"]))

            for i in range(len(dft)):
                [x, y] = [
                    dft["Centroid"].iloc[i][0] * scale,
                    dft["Centroid"].iloc[i][1] * scale,
                ]
                q = np.matmul(R, np.matmul(dft["q"].iloc[i], np.matrix.transpose(R)))
                dq = q - Q
                A = dft["Area"].iloc[i] * scale ** 2
                TrQdq = np.trace(np.matmul(Q, dq))
                dp = np.matmul(R, dft["Polar"].iloc[i]) - P
                [x, y] = np.matmul(R, np.array([x, y]))
                p = np.matmul(R, dft["Polar"].iloc[i])

                _df2.append(
                    {
                        "Filename": filename,
                        "T": t,
                        "X": x - mig[0],
                        "Y": y - mig[1],
                        "dq": dq,
                        "q": q,
                        "TrQdq": TrQdq,
                        "Area": A,
                        "dp": dp,
                        "Polar": p,
                    }
                )

            mig += np.array(dfFilename["v"][dfFilename["T"] == t])[0]

    dfShape = pd.DataFrame(_df2)
    dfShape.to_pickle(f"databases/dfShape{fileType}.pkl")

# space time correlation
grid = 8
timeGrid = 18
gridSize = 10
gridSizeT = 5
if False:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")

    xMax = np.max(dfShape["X"])
    xMin = np.min(dfShape["X"])
    yMax = np.max(dfShape["Y"])
    yMin = np.min(dfShape["Y"])
    xGrid = int(1 + (xMax - xMin) // gridSize)
    yGrid = int(1 + (yMax - yMin) // gridSize)

    T = np.linspace(0, gridSizeT * (timeGrid - 1), timeGrid)
    R = np.linspace(0, gridSize * (grid - 1), grid)
    rho = [[[] for col in range(len(R))] for col in range(len(T))]
    for filename in filenames:
        print(filename + datetime.now().strftime(" %H:%M:%S"))

        df = dfShape[dfShape["Filename"] == filename]
        heatmapdrho = np.zeros([90, xGrid, yGrid])
        inPlaneEcad = np.zeros([90, xGrid, yGrid])

        for t in range(90):

            dft = df[df["T"] == t]
            for i in range(xGrid):
                for j in range(yGrid):
                    x = [
                        xMin + i * gridSize,
                        xMin + (i + 1) * gridSize,
                    ]
                    y = [
                        yMin + j * gridSize,
                        yMin + (j + 1) * gridSize,
                    ]

                    dfg = util.sortGrid(dft, x, y)
                    if list(dfg["Area"]) != []:
                        heatmapdrho[t, i, j] = len(dfg["Area"]) / np.sum(dfg["Area"])
                        inPlaneEcad[t, i, j] = 1

            heatmapdrho[t] = heatmapdrho[t] - np.mean(
                heatmapdrho[t][inPlaneEcad[t] == 1]
            )

        for i in range(xGrid):
            for j in range(yGrid):
                for t in T:
                    t = int(t)
                    deltarho = np.mean(heatmapdrho[t : t + gridSizeT, i, j])
                    if np.sum(inPlaneEcad[t : t + gridSizeT, i, j]) > 0:
                        for idash in range(xGrid):
                            for jdash in range(yGrid):
                                for tdash in T:
                                    tdash = int(tdash)
                                    deltaT = int((tdash - t) / gridSizeT)
                                    deltaR = int(
                                        ((i - idash) ** 2 + (j - jdash) ** 2) ** 0.5
                                    )
                                    if deltaR < grid:
                                        if deltaT >= 0 and deltaT < timeGrid:
                                            if (
                                                np.sum(
                                                    inPlaneEcad[
                                                        tdash : tdash + gridSizeT,
                                                        idash,
                                                        jdash,
                                                    ]
                                                )
                                                > 0
                                            ):

                                                rho[deltaT][deltaR].append(
                                                    deltarho
                                                    * np.mean(
                                                        heatmapdrho[
                                                            tdash : tdash + gridSizeT,
                                                            idash,
                                                            jdash,
                                                        ]
                                                    )
                                                )

    rhoCorrelation = [[] for col in range(len(T))]

    for i in range(len(T)):
        for j in range(len(R)):
            rhoCorrelation[i].append(np.mean(rho[i][j]))

    rhoCorrelation = np.array(rhoCorrelation)
    rhoCorrelation = np.nan_to_num(rhoCorrelation)

    deltarhoVar = np.mean(heatmapdrho[inPlaneEcad == 1] ** 2)

    _df = []

    _df.append(
        {
            "rho": rho,
            "rhoCorrelation": rhoCorrelation,
            "deltarhoVar": deltarhoVar,
        }
    )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/continuumCorrelation{fileType}.pkl")

if True:
    df = pd.read_pickle(f"databases/continuumCorrelation{fileType}.pkl")
    rhoCorrelation = df["rhoCorrelation"].iloc[0]

    deltarhoVar = df["deltarhoVar"].iloc[0]

    t, r = np.mgrid[0:180:10, 0:80:10]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    lims = np.max([np.max(rhoCorrelation), abs(np.min(rhoCorrelation))])

    rhoCorrelation = rhoCorrelation - rhoCorrelation[-1]

    c = ax.pcolor(
        t,
        r,
        rhoCorrelation,
        cmap="RdBu_r",
        vmin=-lims,
        vmax=lims,
        shading="auto",
    )
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"$R (\mu m)$ ")
    ax.title.set_text(r"Correlation of $\delta \rho$ str" + f" {fileType}")

    fig.savefig(
        f"results/Correlation Rho str {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

if True:
    df = pd.read_pickle(f"databases/continuumCorrelation{fileType}.pkl")
    rhoCorrelation = df["rhoCorrelation"].iloc[0]

    deltarhoVar = df["deltarhoVar"].iloc[0]

    t, r = np.mgrid[0:180:10, 0:80:10]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    lims = np.max([np.max(rhoCorrelation), abs(np.min(rhoCorrelation))])

    c = ax.pcolor(
        t,
        r,
        rhoCorrelation,
        cmap="RdBu_r",
        vmin=-lims,
        vmax=lims,
        shading="auto",
    )
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"$R (\mu m)$ ")
    ax.title.set_text(r"Correlation of $\delta \rho$" + f" {fileType}")

    fig.savefig(
        f"results/Correlation Rho {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


def corRho_T(T, C):
    return C / T


def corRho_R(R, C, D):
    T = 5
    return C / T * np.exp(-(R ** 2) / (4 * D * T))


if True:
    df = pd.read_pickle(f"databases/continuumCorrelation{fileType}.pkl")
    rhoCorrelation = df["rhoCorrelation"].iloc[0]
    T = np.linspace(0, gridSizeT * (timeGrid - 1), timeGrid)
    R = np.linspace(0, gridSize * (grid - 1), grid)

    m = sp.optimize.curve_fit(
        f=corRho_T,
        xdata=T[1:],
        ydata=rhoCorrelation[:, 0][1:],
        p0=0.003,
    )[0]

    fig, ax = plt.subplots(1, 2, figsize=(14, 8))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    ax[0].plot(T[1:], rhoCorrelation[:, 0][1:])
    ax[0].plot(T[1:], corRho_T(T, m)[1:])
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel(r"$\delta\rho$ Correlation")
    ax[0].set_ylim([-0.00001, 0.0007])
    ax[0].set_xlim([0, gridSizeT * timeGrid])
    ax[0].title.set_text(r"Correlation of $\delta \rho$" + f" {fileType}")

    m = sp.optimize.curve_fit(
        f=corRho_R,
        xdata=R,
        ydata=rhoCorrelation[0],
        p0=(0.003, 10),
    )[0]

    ax[1].plot(R, rhoCorrelation[0])
    ax[1].plot(R, corRho_R(R, m[0], m[1]))
    ax[1].set_xlabel(r"$R (\mu m)$")
    ax[1].set_ylabel(r"$\delta\rho$ Correlation")
    ax[0].set_ylim([-0.00001, 0.0007])
    ax[1].title.set_text(r"Correlation of $\delta \rho$" + f" {fileType}")
    fig.savefig(
        f"results/Correlation rho in T and R {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")