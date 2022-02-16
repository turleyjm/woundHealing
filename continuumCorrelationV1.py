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
from mpl_toolkits.mplot3d import Axes3D
import tifffile
from skimage.draw import circle_perimeter
from scipy import optimize
import xml.etree.ElementTree as et
from scipy.optimize import leastsq
from datetime import datetime

import cellProperties as cell
import findGoodCells as fi
import utils as util

plt.rcParams.update({"font.size": 16})

# -------------------

filenames, fileType = util.getFilesType()

T = 90
scale = 123.26 / 512

# -------------------


def fillGaps(Correlation):

    for t in range(9):
        Correlation[t, :, 0] = Correlation[t, 0, 0]

    Correlation[:, 1, 1] = (Correlation[:, 0, 1] + Correlation[:, 2, 1]) / 2
    Correlation[:, 6, 1] = (Correlation[:, 5, 1] + Correlation[:, 7, 1]) / 2
    Correlation[:, 11, 1] = (Correlation[:, 10, 1] + Correlation[:, 12, 1]) / 2
    Correlation[:, 16, 1] = (Correlation[:, 15, 1] + Correlation[:, 17, 1]) / 2

    Correlation[:, 3, 1] = (2 * Correlation[:, 2, 1] + Correlation[:, 5, 1]) / 3
    Correlation[:, 4, 1] = (Correlation[:, 2, 1] + 2 * Correlation[:, 5, 1]) / 3

    Correlation[:, 8, 1] = (2 * Correlation[:, 7, 1] + Correlation[:, 10, 1]) / 3
    Correlation[:, 9, 1] = (Correlation[:, 7, 1] + 2 * Correlation[:, 10, 1]) / 3

    Correlation[:, 13, 1] = (2 * Correlation[:, 12, 1] + Correlation[:, 15, 1]) / 3
    Correlation[:, 14, 1] = (Correlation[:, 12, 1] + 2 * Correlation[:, 15, 1]) / 3

    Correlation[:, 18, 1] = (2 * Correlation[:, 17, 1] + Correlation[:, 20, 1]) / 3
    Correlation[:, 19, 1] = (Correlation[:, 17, 1] + 2 * Correlation[:, 0, 1]) / 3

    Correlation[:, 4, 2] = (Correlation[:, 3, 2] + Correlation[:, 5, 2]) / 2
    Correlation[:, 9, 2] = (Correlation[:, 8, 2] + Correlation[:, 10, 2]) / 2
    Correlation[:, 14, 2] = (Correlation[:, 13, 2] + Correlation[:, 15, 2]) / 2
    Correlation[:, 19, 2] = (Correlation[:, 18, 2] + Correlation[:, 0, 2]) / 2

    Correlation[:, 2, 3] = (Correlation[:, 1, 3] + Correlation[:, 3, 3]) / 2
    Correlation[:, 4, 3] = (Correlation[:, 3, 3] + Correlation[:, 5, 3]) / 2
    Correlation[:, 7, 3] = (Correlation[:, 6, 3] + Correlation[:, 8, 3]) / 2
    Correlation[:, 9, 3] = (Correlation[:, 8, 3] + Correlation[:, 10, 3]) / 2
    Correlation[:, 12, 3] = (Correlation[:, 11, 3] + Correlation[:, 13, 3]) / 2
    Correlation[:, 14, 3] = (Correlation[:, 13, 3] + Correlation[:, 15, 3]) / 2
    Correlation[:, 17, 3] = (Correlation[:, 16, 3] + Correlation[:, 18, 3]) / 2
    Correlation[:, 19, 3] = (Correlation[:, 18, 3] + Correlation[:, 0, 3]) / 2

    return Correlation


def exponential(x, coeffs):
    A = coeffs[0]
    c = coeffs[1]
    return A * np.exp(c * x)


def residualsExponential(coeffs, y, x):
    return y - exponential(x, coeffs)


# -------------------

if False:
    _df2 = []
    _df = []
    for filename in filenames:

        df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")
        Q = np.mean(df["q"])
        theta0 = np.arccos(Q[0, 0] / (Q[0, 0] ** 2 + Q[0, 1] ** 2) ** 0.5) / 2
        R = cl.rotation_matrix(-theta0)

        df = pd.read_pickle(f"dat/{filename}/nucleusTracks{filename}.pkl")
        mig = np.zeros(2)

        for t in range(T):
            dft = df[df["Time"] == t]
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
    dfVelocityMean.to_pickle(f"databases/dfContinuumVelocityMean{fileType}.pkl")
    dfVelocity = pd.DataFrame(_df2)
    dfVelocity.to_pickle(f"databases/dfContinuumVelocity{fileType}.pkl")

else:
    dfVelocity = pd.read_pickle(f"databases/dfContinuumVelocity{fileType}.pkl")
    dfVelocityMean = pd.read_pickle(f"databases/dfContinuumVelocityMean{fileType}.pkl")

if False:
    _df2 = []
    for filename in filenames:

        df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")
        dfFilename = dfVelocityMean[dfVelocityMean["Filename"] == filename]
        mig = np.zeros(2)
        Q = np.mean(df["q"])
        theta0 = np.arctan2(Q[0, 1], Q[0, 0]) / 2
        R = cl.rotation_matrix(-theta0)

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
    dfShape.to_pickle(f"databases/dfContinuum{fileType}.pkl")

else:
    dfShape = pd.read_pickle(f"databases/dfContinuum{fileType}.pkl")


# space time correlation
if False:

    grid = 11
    timeGrid = 9
    xMax = np.max(dfShape["X"])
    xMin = np.min(dfShape["X"])
    yMax = np.max(dfShape["Y"])
    yMin = np.min(dfShape["Y"])
    xGrid = int(1 + (xMax - xMin) // 10)
    yGrid = int(1 + (yMax - yMin) // 10)

    T = np.linspace(0, 10 * (timeGrid - 1), timeGrid)
    R = np.linspace(0, 10 * (grid - 1), grid)
    deltaQ = [[[] for col in range(len(R))] for col in range(len(T))]
    deltaQ1 = [[[] for col in range(len(R))] for col in range(len(T))]
    deltaQ2 = [[[] for col in range(len(R))] for col in range(len(T))]
    rho = [[[] for col in range(len(R))] for col in range(len(T))]
    v = [[[] for col in range(len(R))] for col in range(len(T))]
    deltaP = [[[] for col in range(len(R))] for col in range(len(T))]
    for filename in filenames:

        dfVel = dfVelocity[dfVelocity["Filename"] == filename]
        df = dfShape[dfShape["Filename"] == filename]
        heatmapdv1 = np.zeros([90, xGrid, yGrid])
        heatmapdv2 = np.zeros([90, xGrid, yGrid])
        heatmapdrho = np.zeros([90, xGrid, yGrid])
        heatmapdQ1 = np.zeros([90, xGrid, yGrid])
        heatmapdQ2 = np.zeros([90, xGrid, yGrid])
        heatmapP1 = np.zeros([90, xGrid, yGrid])
        heatmapP2 = np.zeros([90, xGrid, yGrid])
        inPlaneEcad = np.zeros([90, xGrid, yGrid])
        inPlaneH2 = np.zeros([90, xGrid, yGrid])

        for t in range(90):
            dftVel = dfVel[dfVel["T"] == t]
            dft = df[df["T"] == t]
            for i in range(xGrid):
                for j in range(yGrid):
                    x = [
                        xMin + i * 10,
                        xMin + (i + 1) * 10,
                    ]
                    y = [
                        yMin + j * 10,
                        yMin + (j + 1) * 10,
                    ]
                    dfgVel = cl.sortGrid(dftVel, x, y)
                    if list(dfgVel["dv"]) != []:
                        heatmapdv1[t, i, j] = np.mean(dfgVel["dv"], axis=0)[0]
                        heatmapdv2[t, i, j] = np.mean(dfgVel["dv"], axis=0)[1]
                        inPlaneH2[t, i, j] = 1

                    dfg = cl.sortGrid(dft, x, y)
                    if list(dfg["Area"]) != []:
                        heatmapdrho[t, i, j] = len(dfg["Area"]) / np.sum(dfg["Area"])
                        heatmapdQ1[t, i, j] = np.mean(dfg["dQ"], axis=0)[0, 0]
                        heatmapdQ2[t, i, j] = np.mean(dfg["dQ"], axis=0)[1, 0]
                        heatmapP1[t, i, j] = np.mean(dfg["dP"], axis=0)[0]
                        heatmapP2[t, i, j] = np.mean(dfg["dP"], axis=0)[1]
                        inPlaneEcad[t, i, j] = 1

            heatmapdrho[t] = heatmapdrho[t] - np.mean(
                heatmapdrho[t][inPlaneEcad[t] == 1]
            )

        for i in range(xGrid):
            for j in range(yGrid):
                for t in T:
                    t = int(t)
                    deltarho = np.mean(heatmapdrho[t : t + 10, i, j])
                    dQ1 = np.mean(heatmapdQ1[t : t + 10, i, j])
                    dQ2 = np.mean(heatmapdQ2[t : t + 10, i, j])
                    dP1 = np.mean(heatmapP1[t : t + 10, i, j])
                    dP2 = np.mean(heatmapP2[t : t + 10, i, j])
                    if np.sum(inPlaneEcad[t : t + 10, i, j]) > 0:
                        for idash in range(xGrid):
                            for jdash in range(yGrid):
                                for tdash in T:
                                    tdash = int(tdash)
                                    deltaT = int((tdash - t) / 10)
                                    deltaR = int(
                                        ((i - idash) ** 2 + (j - jdash) ** 2) ** 0.5
                                    )
                                    if deltaR < grid:
                                        if deltaT >= 0 and deltaT < timeGrid:
                                            if (
                                                np.sum(
                                                    inPlaneEcad[
                                                        tdash : tdash + 10, idash, jdash
                                                    ]
                                                )
                                                > 0
                                            ):

                                                dQ1dash = np.mean(
                                                    heatmapdQ1[
                                                        tdash : tdash + 10, idash, jdash
                                                    ]
                                                )
                                                dQ2dash = np.mean(
                                                    heatmapdQ2[
                                                        tdash : tdash + 10, idash, jdash
                                                    ]
                                                )
                                                deltaQ[deltaT][deltaR].append(
                                                    2 * (dQ1 * dQ1dash)
                                                    + 2 * (dQ2 * dQ2dash)
                                                )
                                                deltaQ1[deltaT][deltaR].append(
                                                    dQ1 * dQ1dash
                                                )
                                                deltaQ2[deltaT][deltaR].append(
                                                    dQ2 * dQ2dash
                                                )

                                                rho[deltaT][deltaR].append(
                                                    deltarho
                                                    * np.mean(
                                                        heatmapdrho[
                                                            tdash : tdash + 10,
                                                            idash,
                                                            jdash,
                                                        ]
                                                    )
                                                )

                                                dP1dash = np.mean(
                                                    heatmapP1[
                                                        tdash : tdash + 10, idash, jdash
                                                    ]
                                                )
                                                dP2dash = np.mean(
                                                    heatmapP2[
                                                        tdash : tdash + 10, idash, jdash
                                                    ]
                                                )
                                                deltaP[deltaT][deltaR].append(
                                                    (dP1 * dP1dash) + (dP2 * dP2dash)
                                                )

        for i in range(xGrid):
            for j in range(yGrid):
                for t in T:
                    t = int(t)
                    dv1 = np.mean(heatmapdv1[t : t + 10, i, j])
                    dv2 = np.mean(heatmapdv2[t : t + 10, i, j])
                    if np.sum(inPlaneH2[t : t + 10, i, j]) > 0:
                        for idash in range(xGrid):
                            for jdash in range(yGrid):
                                for tdash in T:
                                    tdash = int(tdash)
                                    deltaT = int((tdash - t) / 10)
                                    deltaR = int(
                                        ((i - idash) ** 2 + (j - jdash) ** 2) ** 0.5
                                    )
                                    if deltaR < grid:
                                        if deltaT >= 0 and deltaT < timeGrid:
                                            if (
                                                np.sum(
                                                    inPlaneH2[
                                                        tdash : tdash + 10, idash, jdash
                                                    ]
                                                )
                                                > 0
                                            ):

                                                dv1dash = np.mean(
                                                    heatmapdv1[
                                                        tdash : tdash + 10, idash, jdash
                                                    ]
                                                )
                                                dv2dash = np.mean(
                                                    heatmapdv2[
                                                        tdash : tdash + 10, idash, jdash
                                                    ]
                                                )
                                                v[deltaT][deltaR].append(
                                                    (dv1 * dv1dash) + (dv2 * dv2dash)
                                                )

    deltaQCorrelation = [[] for col in range(len(T))]
    deltaQ1Correlation = [[] for col in range(len(T))]
    deltaQ2Correlation = [[] for col in range(len(T))]
    rhoCorrelation = [[] for col in range(len(T))]
    vCorrelation = [[] for col in range(len(T))]
    deltaPCorrelation = [[] for col in range(len(T))]

    for i in range(len(T)):
        for j in range(len(R)):
            deltaQCorrelation[i].append(np.mean(deltaQ[i][j]))
            deltaQ1Correlation[i].append(np.mean(deltaQ1[i][j]))
            deltaQ2Correlation[i].append(np.mean(deltaQ2[i][j]))
            rhoCorrelation[i].append(np.mean(rho[i][j]))
            vCorrelation[i].append(np.mean(v[i][j]))
            deltaPCorrelation[i].append(np.mean(deltaP[i][j]))

    deltaQCorrelation = np.array(deltaQCorrelation)
    deltaQCorrelation = np.nan_to_num(deltaQCorrelation)
    deltaQ1Correlation = np.array(deltaQ1Correlation)
    deltaQ1Correlation = np.nan_to_num(deltaQ1Correlation)
    deltaQ2Correlation = np.array(deltaQ2Correlation)
    deltaQ2Correlation = np.nan_to_num(deltaQ2Correlation)
    rhoCorrelation = np.array(rhoCorrelation)
    rhoCorrelation = np.nan_to_num(rhoCorrelation)
    vCorrelation = np.array(vCorrelation)
    vCorrelation = np.nan_to_num(vCorrelation)
    deltaPCorrelation = np.array(deltaPCorrelation)
    deltaPCorrelation = np.nan_to_num(deltaPCorrelation)

    deltaQVar = np.mean(
        2 * heatmapdQ1[inPlaneEcad == 1] ** 2 + 2 * heatmapdQ2[inPlaneEcad == 1] ** 2
    )
    deltaQ1Var = np.mean(heatmapdQ1[inPlaneEcad == 1] ** 2)
    deltaQ2Var = np.mean(heatmapdQ2[inPlaneEcad == 1] ** 2)
    deltarhoVar = np.mean(heatmapdrho[inPlaneEcad == 1] ** 2)
    deltavVar = np.mean(
        heatmapdv1[inPlaneH2 == 1] ** 2 + heatmapdv2[inPlaneH2 == 1] ** 2
    )
    deltaPVar = np.mean(
        heatmapP1[inPlaneEcad == 1] ** 2 + heatmapP2[inPlaneEcad == 1] ** 2
    )

    _df = []

    _df.append(
        {
            "deltaQ": deltaQ,
            "deltaQCorrelation": deltaQCorrelation,
            "deltaQVar": deltaQVar,
            "deltaQ1": deltaQ1,
            "deltaQ1Correlation": deltaQ1Correlation,
            "deltaQ1Var": deltaQ1Var,
            "deltaQ2": deltaQ2,
            "deltaQ2Correlation": deltaQ2Correlation,
            "deltaQ2Var": deltaQ2Var,
            "rho": rho,
            "rhoCorrelation": rhoCorrelation,
            "deltarhoVar": deltarhoVar,
            "v": v,
            "vCorrelation": vCorrelation,
            "deltavVar": deltavVar,
            "deltaP": deltaP,
            "deltaPCorrelation": deltaPCorrelation,
            "deltaPVar": deltaPVar,
        }
    )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/continuumCorrelation{fileType}.pkl")


# short range space time correlation
if True:
    grid = 21
    timeGrid = 11
    xMax = np.max(dfShape["X"])
    xMin = np.min(dfShape["X"])
    yMax = np.max(dfShape["Y"])
    yMin = np.min(dfShape["Y"])

    v1 = [[[] for col in range(grid)] for col in range(timeGrid)]
    v2 = [[[] for col in range(grid)] for col in range(timeGrid)]
    v = [[[] for col in range(grid)] for col in range(timeGrid)]
    deltaP1 = [[[] for col in range(grid)] for col in range(timeGrid)]
    deltaP2 = [[[] for col in range(grid)] for col in range(timeGrid)]

    for filename in filenames:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time)

        dfVel = dfVelocity[dfVelocity["Filename"] == filename]
        df = dfShape[dfShape["Filename"] == filename]

        Ns = len(df)

        for i in range(Ns):

            x = df["X"].iloc[i]
            y = df["Y"].iloc[i]
            t = df["T"].iloc[i]
            dP1 = df["dp"].iloc[i][0]
            dP2 = df["dp"].iloc[i][1]

            dfi = util.sortVolume(df, [x, x + grid], [y, y + grid], [t, t + timeGrid])

            R = np.array(((dfi["X"] - x) ** 2 + (dfi["Y"] - y) ** 2) ** 0.5)
            T = np.array(dfi["T"] - t)

            for j in range(len(dfi)):
                if R[j] < grid:
                    deltaP1[T[j]][int(R[j])].append(dP1 * dfi["dp"].iloc[j][0])
                    deltaP2[T[j]][int(R[j])].append(dP2 * dfi["dp"].iloc[j][1])

        Nv = len(dfVel)

        for i in range(Nv):

            x = dfVel["X"].iloc[i]
            y = dfVel["Y"].iloc[i]
            t = dfVel["T"].iloc[i]
            dv1 = dfVel["dv"].iloc[i][0]
            dv2 = dfVel["dv"].iloc[i][1]

            dfVeli = util.sortVolume(
                dfVel, [x, x + grid], [y, y + grid], [t, t + timeGrid]
            )

            R = np.array(((dfVeli["X"] - x) ** 2 + (dfVeli["Y"] - y) ** 2) ** 0.5)
            T = np.array(dfVeli["T"] - t)

            for j in range(len(dfVeli)):
                if R[j] < grid:
                    v1[T[j]][int(R[j])].append(dv1 * dfVeli["dv"].iloc[j][0])
                    v2[T[j]][int(R[j])].append(dv2 * dfVeli["dv"].iloc[j][1])
                    v[T[j]][int(R[j])].append(
                        dv1 * dfVeli["dv"].iloc[j][0] + dv2 * dfVeli["dv"].iloc[j][1]
                    )

    T = np.linspace(0, (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)

    v1Correlation = [[] for col in range(len(T))]
    v2Correlation = [[] for col in range(len(T))]
    vCorrelation = [[] for col in range(len(T))]
    deltaP1Correlation = [[] for col in range(len(T))]
    deltaP2Correlation = [[] for col in range(len(T))]

    for i in range(len(T)):
        for j in range(len(R)):
            v1Correlation[i].append(np.mean(v1[i][j]))
            v2Correlation[i].append(np.mean(v2[i][j]))
            vCorrelation[i].append(np.mean(v[i][j]))
            deltaP1Correlation[i].append(np.mean(deltaP1[i][j]))
            deltaP2Correlation[i].append(np.mean(deltaP2[i][j]))

    v1Correlation = np.array(v1Correlation)
    v1Correlation = np.nan_to_num(v1Correlation)
    v2Correlation = np.array(v2Correlation)
    v2Correlation = np.nan_to_num(v2Correlation)
    vCorrelation = np.array(vCorrelation)
    vCorrelation = np.nan_to_num(vCorrelation)
    deltaP1Correlation = np.array(deltaP1Correlation)
    deltaP1Correlation = np.nan_to_num(deltaP1Correlation)
    deltaP2Correlation = np.array(deltaP2Correlation)
    deltaP2Correlation = np.nan_to_num(deltaP2Correlation)

    _dfShort = []

    _dfShort.append(
        {
            "v1": v1,
            "v1Correlation": v1Correlation,
            "v2": v2,
            "v2Correlation": v2Correlation,
            "v": v,
            "vCorrelation": vCorrelation,
            "deltaP1": deltaP1,
            "deltaP1Correlation": deltaP1Correlation,
            "deltaP2": deltaP2,
            "deltaP2Correlation": deltaP2Correlation,
        }
    )

    dfShort = pd.DataFrame(_dfShort)
    dfShort.to_pickle(f"databases/continuumCorrelationShort{fileType}.pkl")

# display correlation
if False:
    df = pd.read_pickle(f"databases/continuumCorrelation{fileType}.pkl")
    deltaQCorrelation = df["deltaQCorrelation"].iloc[0]
    deltaQ1Correlation = df["deltaQ1Correlation"].iloc[0]
    deltaQ2Correlation = df["deltaQ2Correlation"].iloc[0]
    rhoCorrelation = df["rhoCorrelation"].iloc[0]
    vCorrelation = df["vCorrelation"].iloc[0]
    deltaPCorrelation = df["deltaPCorrelation"].iloc[0]

    t, r = np.mgrid[0:180:20, 0:110:10]
    fig, ax = plt.subplots(2, 3, figsize=(30, 16))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    maxCorr = np.max([deltaQCorrelation, -deltaQCorrelation])

    c = ax[0, 0].pcolor(
        t,
        r,
        deltaQCorrelation,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 0])
    ax[0, 0].set_xlabel("Time (min)")
    ax[0, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 0].title.set_text(r"Correlation of $\delta Q$" + f" {fileType}")

    maxCorr = np.max([deltaQ1Correlation, -deltaQ1Correlation])

    c = ax[0, 1].pcolor(
        t,
        r,
        deltaQ1Correlation,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 1])
    ax[0, 1].set_xlabel("Time (min)")
    ax[0, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 1].title.set_text(r"Correlation of $\delta Q_1$" + f" {fileType}")

    maxCorr = np.max([deltaQ2Correlation, -deltaQ2Correlation])

    c = ax[0, 2].pcolor(
        t,
        r,
        deltaQ2Correlation,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 2])
    ax[0, 2].set_xlabel("Time (min)")
    ax[0, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 2].title.set_text(r"Correlation of $\delta Q_2$" + f" {fileType}")

    maxCorr = np.max([rhoCorrelation, -rhoCorrelation])

    c = ax[1, 0].pcolor(
        t, r, rhoCorrelation, cmap="RdBu_r", vmin=-maxCorr, vmax=maxCorr, shading="auto"
    )
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set_xlabel("Time (min)")
    ax[1, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 0].title.set_text(r"Correlation of $\delta \rho$" + f" {fileType}")

    maxCorr = np.max([vCorrelation, -vCorrelation])

    c = ax[1, 1].pcolor(
        t, r, vCorrelation, cmap="RdBu_r", vmin=-maxCorr, vmax=maxCorr, shading="auto"
    )
    fig.colorbar(c, ax=ax[1, 1])
    ax[1, 1].set_xlabel("Time (min)")
    ax[1, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 1].title.set_text(r"Correlation of $\delta v$" + f" {fileType}")

    maxCorr = np.max([deltaPCorrelation, -deltaPCorrelation])

    c = ax[1, 2].pcolor(
        t,
        r,
        deltaPCorrelation,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 2])
    ax[1, 2].set_xlabel("Time (min)")
    ax[1, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 2].title.set_text(r"Correlation of $\delta P$" + f" {fileType}")

    fig.savefig(
        f"results/Correlation {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

# display short range correlation
if True:
    df = pd.read_pickle(f"databases/continuumCorrelationShort{fileType}.pkl")
    v1Correlation = df["v1Correlation"].iloc[0]
    v2Correlation = df["v2Correlation"].iloc[0]
    deltaP1Correlation = df["deltaP1Correlation"].iloc[0]
    deltaP2Correlation = df["deltaP2Correlation"].iloc[0]

    t, r = np.mgrid[0:22:2, 0:21:1]
    fig, ax = plt.subplots(2, 2, figsize=(16, 14))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    maxCorr = np.max([deltaP1Correlation, -deltaP1Correlation])

    c = ax[0, 0].pcolor(
        t,
        r,
        deltaP1Correlation,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 0])
    ax[0, 0].set_xlabel("Time (min)")
    ax[0, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 0].title.set_text(r"Correlation of $\delta P_1$" + f" {fileType}")

    maxCorr = np.max([deltaP2Correlation, -deltaP2Correlation])

    c = ax[0, 1].pcolor(
        t,
        r,
        deltaP2Correlation,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 1])
    ax[0, 1].set_xlabel("Time (min)")
    ax[0, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 1].title.set_text(r"Correlation of $\delta P_2$" + f" {fileType}")

    maxCorr = np.max([v1Correlation, -v1Correlation])

    c = ax[1, 0].pcolor(
        t,
        r,
        v1Correlation,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set_xlabel("Time (min)")
    ax[1, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 0].title.set_text(r"Correlation of $v_1$" + f" {fileType}")

    maxCorr = np.max([v2Correlation, -v2Correlation])

    c = ax[1, 1].pcolor(
        t,
        r,
        v2Correlation,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 1])
    ax[1, 1].set_xlabel("Time (min)")
    ax[1, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 1].title.set_text(r"Correlation of $v_1$" + f" {fileType}")

    fig.savefig(
        f"results/Correlation Short Range {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

    vCorrelation = df["vCorrelation"].iloc[0]

    t, r = np.mgrid[0:22:2, 0:21:1]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    maxCorr = np.max([vCorrelation, -vCorrelation])

    c = ax.pcolor(
        t,
        r,
        vCorrelation,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"$R (\mu m)$ ")
    ax.title.set_text(r"Correlation of $v$" + f" {fileType}")

    fig.savefig(
        f"results/Correlation Short Range v {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

    # -------------------

if False:
    df = pd.read_pickle(f"databases/continuumCorrelation{fileType}.pkl")
    deltaQCorrelation = df["deltaQCorrelation"].iloc[0]
    deltaQ1Correlation = df["deltaQ1Correlation"].iloc[0]
    deltaQ2Correlation = df["deltaQ2Correlation"].iloc[0]
    rhoCorrelation = df["rhoCorrelation"].iloc[0]
    vCorrelation = df["vCorrelation"].iloc[0]
    deltaPCorrelation = df["deltaPCorrelation"].iloc[0]

    deltaQVar = df["deltaQVar"].iloc[0]
    deltaQ1Var = df["deltaQ1Var"].iloc[0]
    deltaQ2Var = df["deltaQ2Var"].iloc[0]
    deltarhoVar = df["deltarhoVar"].iloc[0]
    deltavVar = df["deltavVar"].iloc[0]
    deltaPVar = df["deltaPVar"].iloc[0]

    t, r = np.mgrid[0:180:20, 0:110:10]
    fig, ax = plt.subplots(2, 3, figsize=(30, 16))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    c = ax[0, 0].pcolor(
        t,
        r,
        deltaQCorrelation / deltaQVar,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 0])
    ax[0, 0].set_xlabel("Time (min)")
    ax[0, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 0].title.set_text(r"Correlation of $\delta Q$ Norm" + f" {fileType}")

    c = ax[0, 1].pcolor(
        t,
        r,
        deltaQ1Correlation / deltaQ1Var,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 1])
    ax[0, 1].set_xlabel("Time (min)")
    ax[0, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 1].title.set_text(r"Correlation of $\delta Q_1$ Norm" + f" {fileType}")

    c = ax[0, 2].pcolor(
        t,
        r,
        deltaQ2Correlation / deltaQ2Var,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 2])
    ax[0, 2].set_xlabel("Time (min)")
    ax[0, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 2].title.set_text(r"Correlation of $\delta Q_2$ Norm" + f" {fileType}")

    c = ax[1, 0].pcolor(
        t,
        r,
        rhoCorrelation / deltarhoVar,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set_xlabel("Time (min)")
    ax[1, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 0].title.set_text(r"Correlation of $\delta \rho$ Norm" + f" {fileType}")

    c = ax[1, 1].pcolor(
        t, r, vCorrelation / deltavVar, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto"
    )
    fig.colorbar(c, ax=ax[1, 1])
    ax[1, 1].set_xlabel("Time (min)")
    ax[1, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 1].title.set_text(r"Correlation of $\delta v$ Norm" + f" {fileType}")

    c = ax[1, 2].pcolor(
        t,
        r,
        deltaPCorrelation / deltaPVar,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 2])
    ax[1, 2].set_xlabel("Time (min)")
    ax[1, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 2].title.set_text(r"Correlation of $\delta P$ Norm" + f" {fileType}")

    fig.savefig(
        f"results/Correlation {fileType} Norm",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


if False:
    df = pd.read_pickle(f"databases/continuumCorrelation{fileType}.pkl")

    rhoCorrelation = df["rhoCorrelation"].iloc[0]
    deltaQCorrelation = df["deltaQCorrelation"].iloc[0]

    R0 = rhoCorrelation[0]
    T0 = rhoCorrelation[:, 0]
    T = np.array(range(timeGrid)) * 20
    R = np.array(range(grid)) * 10

    mR = leastsq(
        residualsExponential,
        x0=(1, -1),
        args=(R0, R),
    )[0]

    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    plt.subplots_adjust(wspace=0.3)
    ax[0, 0].plot(R + 10, R0, label="Data")
    ax[0, 0].plot(R + 10, exponential(R, mR), label="Fit Curve")
    ax[0, 0].set(xlabel=r"Distance $(\mu m)$", ylabel="Correlation")
    ax[0, 0].title.set_text(r"$\delta \rho$, $\alpha$ = " + f"{round(mR[1],3)}")
    ax[0, 0].legend()

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
    R = np.array(range(grid)) * 10

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


# delta rho Q and P direction space time correlation
if False:

    grid = 11
    timeGrid = 9
    xMax = np.max(dfShape["X"])
    xMin = np.min(dfShape["X"])
    yMax = np.max(dfShape["Y"])
    yMin = np.min(dfShape["Y"])
    xGrid = int(1 + (xMax - xMin) // 10)
    yGrid = int(1 + (yMax - yMin) // 10)

    T = np.array(range(timeGrid)) * 10
    R = np.linspace(0, 10 * (grid - 1), grid)
    theta = np.linspace(0, 2 * np.pi, 21)
    deltaQ = [
        [[[] for col in range(len(theta))] for col in range(len(R))]
        for col in range(len(T))
    ]
    deltaQ1 = [
        [[[] for col in range(len(theta))] for col in range(len(R))]
        for col in range(len(T))
    ]
    deltaQ2 = [
        [[[] for col in range(len(theta))] for col in range(len(R))]
        for col in range(len(T))
    ]
    rho = [
        [[[] for col in range(len(theta))] for col in range(len(R))]
        for col in range(len(T))
    ]
    v = [
        [[[] for col in range(len(theta))] for col in range(len(R))]
        for col in range(len(T))
    ]
    deltaP = [
        [[[] for col in range(len(theta))] for col in range(len(R))]
        for col in range(len(T))
    ]
    for filename in filenames:

        df = dfShape[dfShape["Filename"] == filename]
        dfVel = dfVelocity[dfVelocity["Filename"] == filename]
        heatmapdrho = np.zeros([90, xGrid, yGrid])
        heatmapdQ1 = np.zeros([90, xGrid, yGrid])
        heatmapdv2 = np.zeros([90, xGrid, yGrid])
        heatmapdv1 = np.zeros([90, xGrid, yGrid])
        heatmapdQ2 = np.zeros([90, xGrid, yGrid])
        heatmapP1 = np.zeros([90, xGrid, yGrid])
        heatmapP2 = np.zeros([90, xGrid, yGrid])
        inPlaneEcad = np.zeros([90, xGrid, yGrid])
        inPlaneH2 = np.zeros([90, xGrid, yGrid])

        for t in range(90):
            dft = df[df["T"] == t]
            dftVel = dfVel[dfVel["T"] == t]
            for i in range(xGrid):
                for j in range(yGrid):
                    x = [
                        xMin + i * 10,
                        xMin + (i + 1) * 10,
                    ]
                    y = [
                        yMin + j * 10,
                        yMin + (j + 1) * 10,
                    ]
                    dfgVel = cl.sortGrid(dftVel, x, y)
                    if list(dfgVel["dv"]) != []:
                        heatmapdv1[t, i, j] = np.mean(dfgVel["dv"], axis=0)[0]
                        heatmapdv2[t, i, j] = np.mean(dfgVel["dv"], axis=0)[1]
                        inPlaneH2[t, i, j] = 1

                    dfg = cl.sortGrid(dft, x, y)
                    if list(dfg["Area"]) != []:
                        heatmapdrho[t, i, j] = len(dfg["Area"]) / np.sum(dfg["Area"])
                        heatmapdQ1[t, i, j] = np.mean(dfg["dQ"], axis=0)[0, 0]
                        heatmapdQ2[t, i, j] = np.mean(dfg["dQ"], axis=0)[1, 0]
                        heatmapP1[t, i, j] = np.mean(dfg["dP"], axis=0)[0]
                        heatmapP2[t, i, j] = np.mean(dfg["dP"], axis=0)[1]
                        inPlaneEcad[t, i, j] = 1

            heatmapdrho[t] = heatmapdrho[t] - np.mean(
                heatmapdrho[t][inPlaneEcad[t] == 1]
            )

        for i in range(xGrid):
            for j in range(yGrid):
                for t in T:
                    deltarho = np.mean(heatmapdrho[t : t + 10, i, j])
                    dQ1 = np.mean(heatmapdQ1[t : t + 10, i, j])
                    dQ2 = np.mean(heatmapdQ2[t : t + 10, i, j])
                    dP1 = np.mean(heatmapP1[t : t + 10, i, j])
                    dP2 = np.mean(heatmapP2[t : t + 10, i, j])
                    if np.sum(inPlaneEcad[t : t + 10, i, j]) > 0:
                        for idash in range(xGrid):
                            for jdash in range(yGrid):
                                for tdash in T:
                                    deltaT = int((tdash - t) / 10)
                                    deltaR = int(
                                        ((i - idash) ** 2 + (j - jdash) ** 2) ** 0.5
                                    )
                                    deltatheta = int(
                                        (
                                            20
                                            * np.arctan2(jdash - j, idash - i)
                                            / (2 * np.pi)
                                        )
                                        % 20
                                    )
                                    if deltaR < grid:
                                        if deltaT >= 0 and deltaT < timeGrid:
                                            if (
                                                np.sum(
                                                    inPlaneEcad[
                                                        tdash : tdash + 10, idash, jdash
                                                    ]
                                                )
                                                > 0
                                            ):

                                                dQ1dash = np.mean(
                                                    heatmapdQ1[
                                                        tdash : tdash + 10, idash, jdash
                                                    ]
                                                )
                                                dQ2dash = np.mean(
                                                    heatmapdQ2[
                                                        tdash : tdash + 10, idash, jdash
                                                    ]
                                                )
                                                deltaQ[deltaT][deltaR][
                                                    deltatheta
                                                ].append(
                                                    2 * (dQ1 * dQ1dash)
                                                    + 2 * (dQ2 * dQ2dash)
                                                )
                                                deltaQ1[deltaT][deltaR][
                                                    deltatheta
                                                ].append(dQ1 * dQ1dash)
                                                deltaQ2[deltaT][deltaR][
                                                    deltatheta
                                                ].append(dQ2 * dQ2dash)

                                                rho[deltaT][deltaR][deltatheta].append(
                                                    deltarho
                                                    * np.mean(
                                                        heatmapdrho[
                                                            tdash : tdash + 10,
                                                            idash,
                                                            jdash,
                                                        ]
                                                    )
                                                )

                                                dP1dash = np.mean(
                                                    heatmapP1[
                                                        tdash : tdash + 10, idash, jdash
                                                    ]
                                                )
                                                dP2dash = np.mean(
                                                    heatmapP2[
                                                        tdash : tdash + 10, idash, jdash
                                                    ]
                                                )
                                                deltaP[deltaT][deltaR][
                                                    deltatheta
                                                ].append(
                                                    (dP1 * dP1dash) + (dP2 * dP2dash)
                                                )

        for i in range(xGrid):
            for j in range(yGrid):
                for t in T:
                    dv1 = np.mean(heatmapdv1[t : t + 10, i, j])
                    dv2 = np.mean(heatmapdv2[t : t + 10, i, j])
                    if np.sum(inPlaneH2[t : t + 10, i, j]) > 0:
                        for idash in range(xGrid):
                            for jdash in range(yGrid):
                                for tdash in T:
                                    deltaT = int((tdash - t) / 10)
                                    deltaR = int(
                                        ((i - idash) ** 2 + (j - jdash) ** 2) ** 0.5
                                    )
                                    deltatheta = int(
                                        (
                                            20
                                            * np.arctan2(jdash - j, idash - i)
                                            / (2 * np.pi)
                                        )
                                        % 20
                                    )
                                    if deltaR < grid:
                                        if deltaT >= 0 and deltaT < timeGrid:
                                            if (
                                                np.sum(
                                                    inPlaneH2[
                                                        tdash : tdash + 10, idash, jdash
                                                    ]
                                                )
                                                > 0
                                            ):

                                                dv1dash = np.mean(
                                                    heatmapdv1[
                                                        tdash : tdash + 10, idash, jdash
                                                    ]
                                                )
                                                dv2dash = np.mean(
                                                    heatmapdv2[
                                                        tdash : tdash + 10, idash, jdash
                                                    ]
                                                )
                                                v[deltaT][deltaR][deltatheta].append(
                                                    (dv1 * dv1dash) + (dv2 * dv2dash)
                                                )

    deltaQCorrelation = [[[] for col in range(len(theta))] for col in range(len(T))]
    deltaQ1Correlation = [[[] for col in range(len(theta))] for col in range(len(T))]
    deltaQ2Correlation = [[[] for col in range(len(theta))] for col in range(len(T))]
    rhoCorrelation = [[[] for col in range(len(theta))] for col in range(len(T))]
    vCorrelation = [[[] for col in range(len(theta))] for col in range(len(T))]
    deltaPCorrelation = [[[] for col in range(len(theta))] for col in range(len(T))]
    for i in range(len(T)):
        for j in range(len(R)):
            for th in range(len(theta)):
                deltaQCorrelation[i][th].append(np.mean(deltaQ[i][j][th]))
                deltaQ1Correlation[i][th].append(np.mean(deltaQ1[i][j][th]))
                deltaQ2Correlation[i][th].append(np.mean(deltaQ2[i][j][th]))
                rhoCorrelation[i][th].append(np.mean(rho[i][j][th]))
                vCorrelation[i][th].append(np.mean(v[i][j][th]))
                deltaPCorrelation[i][th].append(np.mean(deltaP[i][j][th]))

    deltaQCorrelation = np.array(deltaQCorrelation)
    deltaQ1Correlation = np.array(deltaQ1Correlation)
    deltaQ2Correlation = np.array(deltaQ2Correlation)
    rhoCorrelation = np.array(rhoCorrelation)
    deltaQCorrelation = np.array(deltaQCorrelation)
    deltaPCorrelation = np.array(deltaPCorrelation)

    deltaQCorrelation = np.nan_to_num(deltaQCorrelation)
    deltaQ1Correlation = np.nan_to_num(deltaQ1Correlation)
    deltaQ2Correlation = np.nan_to_num(deltaQ2Correlation)
    rhoCorrelation = np.nan_to_num(rhoCorrelation)
    vCorrelation = np.nan_to_num(vCorrelation)
    deltaPCorrelation = np.nan_to_num(deltaPCorrelation)

    _df = []

    _df.append(
        {
            "deltaQ": deltaQ,
            "deltaQ1": deltaQ1,
            "deltaQ2": deltaQ2,
            "rho": rho,
            "v": v,
            "deltaP": deltaP,
            "deltaQCorrelation": deltaQCorrelation,
            "deltaQ1Correlation": deltaQ1Correlation,
            "deltaQ2Correlation": deltaQ2Correlation,
            "rhoCorrelation": rhoCorrelation,
            "vCorrelation": vCorrelation,
            "deltaPCorrelation": deltaPCorrelation,
        }
    )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/continuumDirectionCorrelation{fileType}.pkl")
else:
    df = pd.read_pickle(f"databases/continuumDirectionCorrelation{fileType}.pkl")
    deltaQCorrelation = df["deltaQCorrelation"].iloc[0]
    deltaQ1Correlation = df["deltaQ1Correlation"].iloc[0]
    deltaQ2Correlation = df["deltaQ2Correlation"].iloc[0]
    rhoCorrelation = df["rhoCorrelation"].iloc[0]
    vCorrelation = df["vCorrelation"].iloc[0]
    deltaPCorrelation = df["deltaPCorrelation"].iloc[0]


if False:
    T = np.array(range(timeGrid)) * 10
    R = np.linspace(0, 10 * (grid - 1), grid)
    theta = np.linspace(0, 2 * np.pi, 21)
    rad = np.linspace(0, 10 * (grid - 1), grid)
    azm = np.linspace(0, 2 * np.pi, 21)

    # -----------------------------------------------------

    cl.createFolder("results/video/")
    maxCorr = np.max([deltaQCorrelation, -deltaQCorrelation])

    deltaQCorrelation = fillGaps(deltaQCorrelation)

    for t in range(len(T)):
        ra2, th2 = np.meshgrid(R, theta)

        fig = plt.figure()
        ax = Axes3D(fig)
        fig.add_axes(ax)

        plt.subplot(projection="polar")
        plt.title(f"Time= {t*20}")

        pc = plt.pcolormesh(
            th2,
            ra2,
            deltaQCorrelation[t],
            cmap="RdBu_r",
            vmin=-maxCorr,
            vmax=maxCorr,
        )

        plt.colorbar(pc)
        plt.grid()
        fig.savefig(
            "results/video/"
            + f"Directional correlation deltaQ {fileType} at T={t}.png",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

    # make video
    img_array = []

    for t in range(len(T)):
        img = cv2.imread(
            f"results/video/Directional correlation deltaQ {fileType} at T={t}.png"
        )
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(
        f"results/Directional correlation deltaQ {fileType}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        3,
        size,
    )
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
    cv2.destroyAllWindows()

    shutil.rmtree("results/video")

    # -----------------------------------------------------

    cl.createFolder("results/video/")
    maxCorr = np.max([deltaQ1Correlation, -deltaQ1Correlation])

    deltaQ1Correlation = fillGaps(deltaQ1Correlation)

    for t in range(len(T)):
        ra2, th2 = np.meshgrid(R, theta)

        fig = plt.figure()
        ax = Axes3D(fig)
        fig.add_axes(ax)

        plt.subplot(projection="polar")
        plt.title(f"Time= {t*20}")

        pc = plt.pcolormesh(
            th2,
            ra2,
            deltaQ1Correlation[t],
            cmap="RdBu_r",
            vmin=-maxCorr,
            vmax=maxCorr,
        )

        plt.colorbar(pc)
        plt.grid()
        fig.savefig(
            "results/video/"
            + f"Directional correlation deltaQ1 {fileType} at T={t}.png",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

    # make video
    img_array = []

    for t in range(len(T)):
        img = cv2.imread(
            f"results/video/Directional correlation deltaQ1 {fileType} at T={t}.png"
        )
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(
        f"results/Directional correlation deltaQ1 {fileType}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        3,
        size,
    )
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
    cv2.destroyAllWindows()

    shutil.rmtree("results/video")

    # -----------------------------------------------------

    cl.createFolder("results/video/")
    maxCorr = np.max([deltaQ2Correlation, -deltaQ2Correlation])

    deltaQ2Correlation = fillGaps(deltaQ2Correlation)

    for t in range(len(T)):
        ra2, th2 = np.meshgrid(R, theta)

        fig = plt.figure()
        ax = Axes3D(fig)
        fig.add_axes(ax)

        plt.subplot(projection="polar")
        plt.title(f"Time= {t*20}")

        pc = plt.pcolormesh(
            th2,
            ra2,
            deltaQ2Correlation[t],
            cmap="RdBu_r",
            vmin=-maxCorr,
            vmax=maxCorr,
        )

        plt.colorbar(pc)
        plt.grid()
        fig.savefig(
            "results/video/"
            + f"Directional correlation deltaQ2 {fileType} at T={t}.png",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

    # make video
    img_array = []

    for t in range(len(T)):
        img = cv2.imread(
            f"results/video/Directional correlation deltaQ2 {fileType} at T={t}.png"
        )
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(
        f"results/Directional correlation deltaQ2 {fileType}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        3,
        size,
    )
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
    cv2.destroyAllWindows()

    shutil.rmtree("results/video")

    # -----------------------------------------------------

    cl.createFolder("results/video/")
    maxCorr = np.max([rhoCorrelation, -rhoCorrelation])

    rhoCorrelation = fillGaps(rhoCorrelation)

    for t in range(len(T)):
        ra2, th2 = np.meshgrid(R, theta)

        fig = plt.figure()
        ax = Axes3D(fig)
        fig.add_axes(ax)

        plt.subplot(projection="polar")
        plt.title(f"Time= {t*20}")

        pc = plt.pcolormesh(
            th2,
            ra2,
            rhoCorrelation[t],
            cmap="RdBu_r",
            vmin=-maxCorr,
            vmax=maxCorr,
        )

        plt.colorbar(pc)
        plt.grid()
        fig.savefig(
            "results/video/" + f"Directional correlation rho {fileType} at T={t}.png",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

    # make video
    img_array = []

    for t in range(len(T)):
        img = cv2.imread(
            f"results/video/Directional correlation rho {fileType} at T={t}.png"
        )
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(
        f"results/Directional correlation rho {fileType}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        3,
        size,
    )
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
    cv2.destroyAllWindows()

    shutil.rmtree("results/video")

    # -----------------------------------------------------

    cl.createFolder("results/video/")
    maxCorr = np.max([vCorrelation, -vCorrelation])

    vCorrelation = fillGaps(vCorrelation)

    for t in range(len(T)):
        ra2, th2 = np.meshgrid(R, theta)

        fig = plt.figure()
        ax = Axes3D(fig)
        fig.add_axes(ax)

        plt.subplot(projection="polar")
        plt.title(f"Time= {t*20}")

        pc = plt.pcolormesh(
            th2,
            ra2,
            vCorrelation[t],
            cmap="RdBu_r",
            vmin=-maxCorr,
            vmax=maxCorr,
        )

        plt.colorbar(pc)
        plt.grid()
        fig.savefig(
            "results/video/" + f"Directional correlation v {fileType} at T={t}.png",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

    # make video
    img_array = []

    for t in range(len(T)):
        img = cv2.imread(
            f"results/video/Directional correlation v {fileType} at T={t}.png"
        )
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(
        f"results/Directional correlation v {fileType}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        3,
        size,
    )
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
    cv2.destroyAllWindows()

    shutil.rmtree("results/video")

    # -----------------------------------------------------

    cl.createFolder("results/video/")
    maxCorr = np.max([deltaPCorrelation, -deltaPCorrelation])

    deltaPCorrelation = fillGaps(deltaPCorrelation)

    for t in range(len(T)):
        ra2, th2 = np.meshgrid(R, theta)

        fig = plt.figure()
        ax = Axes3D(fig)
        fig.add_axes(ax)

        plt.subplot(projection="polar")
        plt.title(f"Time= {t*20}")

        pc = plt.pcolormesh(
            th2,
            ra2,
            deltaPCorrelation[t],
            cmap="RdBu_r",
            vmin=-maxCorr,
            vmax=maxCorr,
        )

        plt.colorbar(pc)
        plt.grid()
        fig.savefig(
            "results/video/"
            + f"Directional correlation deltaP {fileType} at T={t}.png",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

    # make video
    img_array = []

    for t in range(len(T)):
        img = cv2.imread(
            f"results/video/Directional correlation deltaP {fileType} at T={t}.png"
        )
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(
        f"results/Directional correlation deltaP {fileType}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        3,
        size,
    )
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
    cv2.destroyAllWindows()

    shutil.rmtree("results/video")
