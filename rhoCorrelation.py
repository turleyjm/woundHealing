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

# space time density correlation
grid = 9
timeGrid = 18
gridSize = 10
gridSizeT = 5
theta = np.linspace(0, 2 * np.pi, 17)
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
    for filename in filenames:
        print(filename + datetime.now().strftime(" %H:%M:%S"))
        drhodrhoij = [
            [[[] for col in range(17)] for col in range(len(R))]
            for col in range(len(T))
        ]
        dRhodRhoCorrelation = np.zeros([len(T), len(R), len(theta)])
        dRhodRhoCorrelation_std = np.zeros([len(T), len(R), len(theta)])
        total = np.zeros([len(T), len(R), len(theta)])

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
                    if np.sum(inPlaneEcad[t : t + gridSizeT, i, j]) > 0:
                        deltarho = np.mean(
                            heatmapdrho[t : t + gridSizeT, i, j][
                                inPlaneEcad[t : t + gridSizeT, i, j] > 0
                            ]
                        )
                        for idash in range(xGrid):
                            for jdash in range(yGrid):
                                for tdash in T:
                                    tdash = int(tdash)
                                    deltaT = int((tdash - t) / gridSizeT)
                                    deltaR = int(
                                        ((i - idash) ** 2 + (j - jdash) ** 2) ** 0.5
                                    )
                                    deltaTheta = int(
                                        np.arctan2((j - jdash), (i - idash)) * 8 / np.pi
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

                                                drhodrhoij[deltaT][deltaR][
                                                    deltaTheta
                                                ].append(
                                                    deltarho
                                                    * np.mean(
                                                        heatmapdrho[
                                                            tdash : tdash + gridSizeT,
                                                            idash,
                                                            jdash,
                                                        ][
                                                            inPlaneEcad[
                                                                t : t + gridSizeT, i, j
                                                            ]
                                                            > 0
                                                        ]
                                                    )
                                                )

        for i in range(len(T)):
            for j in range(len(R)):
                for th in range(len(theta)):
                    dRhodRhoCorrelation[i][j][th] = np.mean(drhodrhoij[i][j][th])
                    dRhodRhoCorrelation_std[i][j][th] = np.std(drhodrhoij[i][j][th])
                    total[i][j][th] = len(drhodrhoij[i][j][th])

        _df = []

        _df.append(
            {
                "Filename": filename,
                "dRhodRhoCorrelation": dRhodRhoCorrelation,
                "dRhodRhoCorrelation_std": dRhodRhoCorrelation_std,
                "Count": total,
            }
        )

        df = pd.DataFrame(_df)
        df.to_pickle(f"databases/dfCorRho{filename}.pkl")

# space time density correlation
grid = 9
timeGrid = 18
gridSize = 10
gridSizeT = 5
theta = np.linspace(0, 2 * np.pi, 17)
if True:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    dfShape["dR"] = list(np.zeros([len(dfShape)]))
    dfShape["dT"] = list(np.zeros([len(dfShape)]))
    dfShape["dtheta"] = list(np.zeros([len(dfShape)]))

    dfShape["drhodq1i"] = list(np.zeros([len(dfShape)]))
    dfShape["drhodq2i"] = list(np.zeros([len(dfShape)]))

    xMax = np.max(dfShape["X"])
    xMin = np.min(dfShape["X"])
    yMax = np.max(dfShape["Y"])
    yMin = np.min(dfShape["Y"])
    xGrid = int(1 + (xMax - xMin) // gridSize)
    yGrid = int(1 + (yMax - yMin) // gridSize)

    T = np.linspace(0, gridSizeT * (timeGrid - 1), timeGrid)
    R = np.linspace(0, gridSize * (grid - 1), grid)
    for filename in filenames:
        path_to_file = f"databases/dfCorRhoQ{filename}.pkl"
        if False == exists(path_to_file):
            print(datetime.now().strftime("%H:%M:%S ") + filename)

            dfShapeF = dfShape[dfShape["Filename"] == filename]
            heatmapdrho = np.zeros([90, xGrid, yGrid])
            inPlaneEcad = np.zeros([90, xGrid, yGrid])

            for t in range(90):

                dft = dfShapeF[dfShapeF["T"] == t]
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
                            heatmapdrho[t, i, j] = len(dfg["Area"]) / np.sum(
                                dfg["Area"]
                            )
                            inPlaneEcad[t, i, j] = 1

                heatmapdrho[t] = heatmapdrho[t] - np.mean(
                    heatmapdrho[t][inPlaneEcad[t] == 1]
                )

            dRhodQ1Correlation = np.zeros([len(T), len(R), len(theta)])
            dRhodQ2Correlation = np.zeros([len(T), len(R), len(theta)])
            dRhodQ1Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dRhodQ2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            total = np.zeros([len(T), len(R), len(theta)])
            drhodq1ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]
            drhodq2ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]
            count = 0
            percent_count = 0
            for i in range(xGrid):
                for j in range(yGrid):
                    if count % int((xGrid * yGrid) / 10) == 0:
                        print(
                            datetime.now().strftime("%H:%M:%S")
                            + f" {10*percent_count}%"
                        )
                        percent_count += 1
                    count += 1
                    for t in T:
                        t = int(t)
                        if np.sum(inPlaneEcad[t : int(t + gridSizeT), i, j]) > 0:
                            drho = np.mean(
                                heatmapdrho[t : int(t + gridSizeT), i, j][
                                    inPlaneEcad[t : int(t + gridSizeT), i, j] > 0
                                ]
                            )
                            x = xMin + (i + 0.5) * gridSize
                            y = yMin + (j + 0.5) * gridSize

                            dfShapeF["dR"] = (
                                (dfShapeF.loc[:, "X"] - x) ** 2
                                + (dfShapeF.loc[:, "Y"] - y) ** 2
                            ) ** 0.5
                            df = dfShapeF[
                                [
                                    "X",
                                    "Y",
                                    "T",
                                    "dp",
                                    "dq",
                                    "dR",
                                    "dT",
                                    "dtheta",
                                    "drhodq1i",
                                    "drhodq2i",
                                ]
                            ]
                            df = df[
                                np.array(df["dR"] < R[-1] + gridSize)
                                & np.array(df["dR"] >= 0)
                            ]

                            df["dT"] = df.loc[:, "T"] - t
                            df = df[
                                np.array(df["dT"] < T[-1] + gridSizeT)
                                & np.array(df["dT"] >= 0)
                            ]
                            if len(df) != 0:
                                phi = np.arctan2(df.loc[:, "Y"] - y, df.loc[:, "X"] - x)
                                df["dtheta"] = np.where(phi < 0, 2 * np.pi + phi, phi)
                                df["drhodq1i"] = list(
                                    np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 0]
                                    * drho
                                )
                                df["drhodq2i"] = list(
                                    np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 1]
                                    * drho
                                )
                                for k in range(len(df)):
                                    drhodq1ij[int(df["dT"].iloc[k] / gridSizeT)][
                                        int(df["dR"].iloc[k] / gridSize)
                                    ][int(8 * df["dtheta"].iloc[k] / np.pi)].append(
                                        df["drhodq1i"].iloc[k]
                                    )
                                    drhodq2ij[int(df["dT"].iloc[k] / gridSizeT)][
                                        int(df["dR"].iloc[k] / gridSize)
                                    ][int(8 * df["dtheta"].iloc[k] / np.pi)].append(
                                        df["drhodq2i"].iloc[k]
                                    )

            for i in range(len(T)):
                for j in range(len(R)):
                    for th in range(len(theta)):
                        dRhodQ1Correlation[i][j][th] = np.mean(drhodq1ij[i][j][th])
                        dRhodQ2Correlation[i][j][th] = np.mean(drhodq2ij[i][j][th])
                        dRhodQ1Correlation_std[i][j][th] = np.std(drhodq1ij[i][j][th])
                        dRhodQ2Correlation_std[i][j][th] = np.std(drhodq2ij[i][j][th])
                        total[i][j][th] = len(drhodq1ij[i][j][th])

            _df = []
            _df.append(
                {
                    "Filename": filename,
                    "dRhodQ1Correlation": dRhodQ1Correlation,
                    "dRhodQ2Correlation": dRhodQ2Correlation,
                    "dRhodQ1Correlation_std": dRhodQ1Correlation_std,
                    "dRhodQ2Correlation_std": dRhodQ2Correlation_std,
                    "Count": total,
                }
            )

            df = pd.DataFrame(_df)
            df.to_pickle(f"databases/dfCorRhoQ{filename}.pkl")

if False:
    df = pd.read_pickle(f"databases/dfCorRho{fileType}.pkl")
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

if False:
    df = pd.read_pickle(f"databases/dfCorRho{fileType}.pkl")
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


if False:
    df = pd.read_pickle(f"databases/dfCorRho{fileType}.pkl")
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