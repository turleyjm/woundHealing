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
from scipy.optimize import leastsq

import cellProperties as cell
import findGoodCells as fi
import commonLiberty as cl

plt.rcParams.update({"font.size": 16})

# -------------------

filenames, fileType = cl.getFilesType()

T = 93
scale = 123.26 / 512
L = 123.26
grid = 11
timeGrid = 9

# -------------------


def exponential(x, coeffs):
    A = coeffs[0]
    c = coeffs[1]
    return A * np.exp(c * x)


def residualsExponential(coeffs, y, x):
    return y - exponential(x, coeffs)


# -------------------

_df2 = []
if False:
    for filename in filenames:

        df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")

        for t in range(T):
            dft = df[df["Time"] == t]
            Q = np.mean(dft["q"])
            for i in range(len(dft)):
                [x, y] = [
                    dft["Centroid"].iloc[i][0] * scale,
                    dft["Centroid"].iloc[i][1] * scale,
                ]
                dQ = dft["q"].iloc[i] - Q
                A = dft["Area"].iloc[i] * scale ** 2
                TrdQ = np.trace(np.matmul(Q, dQ))
                Pol = dft["Polar"].iloc[i]

                _df2.append(
                    {
                        "Filename": filename,
                        "T": t,
                        "X": x,
                        "Y": y,
                        "dQ": dQ,
                        "Q": Q,
                        "TrdQ": TrdQ,
                        "Area": A,
                        "Polar": Pol,
                    }
                )

    dfShape = pd.DataFrame(_df2)
    dfShape.to_pickle(f"databases/dfContinuum{fileType}.pkl")

else:
    dfShape = pd.read_pickle(f"databases/dfContinuum{fileType}.pkl")

# delta rho space time correlation
if False:
    k = 0
    T = np.array(range(timeGrid)) * 10
    R = np.array(range(grid - 3)) * 10
    rho = [
        [[[] for col in range(len(filenames))] for col in range(len(R))]
        for col in range(len(T))
    ]
    deltaQ = [
        [[[] for col in range(len(filenames))] for col in range(len(R))]
        for col in range(len(T))
    ]
    for filename in filenames:

        df = dfShape[dfShape["Filename"] == filename]
        heatmapdrho = np.zeros([90, grid, grid])
        heatmapdQ1 = np.zeros([90, grid, grid])
        heatmapdQ2 = np.zeros([90, grid, grid])

        # outPlanePixel = sm.io.imread(
        #         f"dat/{filename}/outPlane{filename}.tif"
        #     ).astype(float)
        # outPlane = []
        # for t in range(90):
        #     img = Image.fromarray(outPlanePixel[t])
        #     outPlane.append(np.array(img.resize((124, 124)))[7:117, 7:117])
        # outPlane = np.array(outPlane)
        # outPlane[outPlane > 50] = 255
        # outPlane[outPlane < 0] = 0
        # outPlane[outPlane == 255] = 1

        outPlane = np.ones([90, 124, 124])

        for t in range(90):
            dft = df[df["T"] == t]
            for i in range(grid):
                for j in range(grid):
                    x = [
                        (L - 110) / 2 + i * 110 / grid,
                        (L - 110) / 2 + (i + 1) * 110 / grid,
                    ]
                    y = [
                        (L - 110) / 2 + j * 110 / grid,
                        (L - 110) / 2 + (j + 1) * 110 / grid,
                    ]
                    area = np.sum(
                        outPlane[
                            t, round(x[0]) : round(x[1]), round(y[0]) : round(y[1])
                        ]
                    )
                    dfg = cl.sortGrid(dft, x, y)
                    if list(dfg["Area"]) != []:
                        heatmapdrho[t, i, j] = len(dfg["Area"]) / area
                        heatmapdQ1[t, i, j] = np.mean(dfg["dQ"], axis=0)[0, 0]
                        heatmapdQ2[t, i, j] = np.mean(dfg["dQ"], axis=0)[1, 0]

            heatmapdrho[t] = heatmapdrho[t] - np.mean(heatmapdrho[t])

            if False:
                dx, dy = 110 / grid, 110 / grid
                xdash, ydash = np.mgrid[0:110:dx, 0:110:dy]

                fig, ax = plt.subplots()
                c = ax.pcolor(
                    xdash,
                    ydash,
                    heatmapdrho[t],
                    cmap="RdBu_r",
                    vmax=0.1,
                    vmin=-0.1,
                    shading="auto",
                )
                fig.colorbar(c, ax=ax)
                plt.xlabel(r"x $(\mu m)$")
                plt.ylabel(r"y $(\mu m)$")
                plt.title(r"$\delta \rho_0$ " + f"{filename}")
                fig.savefig(
                    f"results/P0 heatmap {t*2} {filename}",
                    dpi=300,
                    transparent=True,
                )
                plt.close("all")

        for i in range(grid):
            for j in range(grid):
                for t in T:
                    deltarho = np.mean(heatmapdrho[t : t + 10, i, j])
                    dQ1 = np.mean(heatmapdQ1[t : t + 10, i, j])
                    dQ2 = np.mean(heatmapdQ2[t : t + 10, i, j])
                    for idash in range(grid):
                        for jdash in range(grid):
                            for tdash in T:
                                deltaT = int((tdash - t) / 10)
                                deltaR = int(
                                    ((i - idash) ** 2 + (j - jdash) ** 2) ** 0.5
                                )
                                if deltaR < 8:
                                    if deltaT >= 0 and deltaT < 9:
                                        rho[deltaT][deltaR][k].append(
                                            deltarho
                                            * np.mean(
                                                heatmapdrho[
                                                    tdash : tdash + 10, idash, jdash
                                                ]
                                            )
                                        )
                                        dQ1dash = np.mean(
                                            heatmapdQ1[tdash : tdash + 10, idash, jdash]
                                        )
                                        dQ2dash = np.mean(
                                            heatmapdQ2[tdash : tdash + 10, idash, jdash]
                                        )
                                        deltaQ[deltaT][deltaR][k].append(
                                            2 * (dQ1 * dQ1dash) + 2 * (dQ2 * dQ2dash)
                                        )
        k += 1

    rhoCorrelation = [[] for col in range(len(T))]
    deltaQCorrelation = [[] for col in range(len(T))]
    for i in range(len(T)):
        for j in range(len(R)):
            rhoCor = []
            deltaQCor = []
            for m in range(len(filenames)):
                rhoCor.append(np.mean(rho[i][j]))
                deltaQCor.append(np.mean(deltaQ[i][j]))

            rhoCorrelation[i].append(np.mean(rhoCor))
            deltaQCorrelation[i].append(np.mean(deltaQCor))

    rhoCorrelation = np.array(rhoCorrelation)
    deltaQCorrelation = np.array(deltaQCorrelation)
    rhoCorrelation = np.nan_to_num(rhoCorrelation)
    deltaQCorrelation = np.nan_to_num(deltaQCorrelation)

    _df = []

    _df.append(
        {
            "rho": rho,
            "deltaQ": deltaQ,
            "rhoCorrelation": rhoCorrelation,
            "deltaQCorrelation": deltaQCorrelation,
        }
    )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/continuumCorrelation{fileType}.pkl")

    t, r = np.mgrid[0:180:20, 0:80:10]
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    maxCorr = np.max([rhoCorrelation, -rhoCorrelation])

    c = ax[0].pcolor(t, r, rhoCorrelation, cmap="RdBu_r", vmin=-maxCorr, vmax=maxCorr)
    fig.colorbar(c, ax=ax[0])
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel(r"$R (\mu m)$ ")
    ax[0].title.set_text(r"Correlation of $\delta \rho$" + f" {fileType}")

    maxCorr = np.max([deltaQCorrelation, -deltaQCorrelation])

    c = ax[1].pcolor(
        t, r, deltaQCorrelation, cmap="RdBu_r", vmin=-maxCorr, vmax=maxCorr
    )
    fig.colorbar(c, ax=ax[1])
    ax[1].set_xlabel("Time (min)")
    ax[1].set_ylabel(r"$R (\mu m)$")
    ax[1].title.set_text(r"Correlation of $\delta Q$" + f" {fileType}")

    fig.savefig(
        f"results/Correlation delta rho and Q {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

if True:
    df = pd.read_pickle(f"databases/continuumCorrelation{fileType}.pkl")

    rhoCorrelation = df["rhoCorrelation"].iloc[0]
    deltaQCorrelation = df["deltaQCorrelation"].iloc[0]

    R0 = rhoCorrelation[0]
    T0 = rhoCorrelation[:, 0]
    T = np.array(range(timeGrid)) * 20
    R = np.array(range(grid - 3)) * 10

    mR = leastsq(
        residualsExponential,
        x0=(1, -1),
        args=(R0, R),
    )[0]

    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    plt.subplots_adjust(wspace=0.3)
    ax[0, 0].plot(R + 10, R0)
    ax[0, 0].plot(R + 10, exponential(R, mR))
    ax[0, 0].set(xlabel=r"Distance $(\mu m)$", ylabel="Correlation")
    ax[0, 0].title.set_text(r"$\delta \rho$, $\alpha$ = " + f"{round(mR[1],3)}")

    mT = leastsq(
        residualsExponential,
        x0=(1, -1),
        args=(T0, T),
    )[0]

    ax[0, 1].plot(T + 10, T0)
    ax[0, 1].plot(T + 10, exponential(T, mT))
    ax[0, 1].set(xlabel="Time (mins)", ylabel="Correlation")
    ax[0, 1].title.set_text(r"$\delta \rho$, $\beta$ = " + f"{round(mT[1],3)}")

    R0 = deltaQCorrelation[0]
    T0 = deltaQCorrelation[:, 0]
    T = np.array(range(timeGrid)) * 20
    R = np.array(range(grid - 3)) * 10

    mR = leastsq(
        residualsExponential,
        x0=(1, -1),
        args=(R0, R),
    )[0]

    ax[1, 0].plot(R + 10, R0)
    ax[1, 0].plot(R + 10, exponential(R, mR))
    ax[1, 0].set(xlabel=r"Distance $(\mu m)$", ylabel="Correlation")
    ax[1, 0].title.set_text(r"$\delta Q$, $\alpha$ = " + f"{round(mR[1],3)}")

    mT = leastsq(
        residualsExponential,
        x0=(1, -1),
        args=(T0, T),
    )[0]

    ax[1, 1].plot(T + 10, T0)
    ax[1, 1].plot(T + 10, exponential(T, mT))
    ax[1, 1].set(xlabel="Time (mins)", ylabel="Correlation")
    ax[1, 1].title.set_text(r"$\delta Q$, $\beta$ = " + f"{round(mT[1],3)}")

    fig.savefig(
        f"results/continuumCorrelation Fit {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")