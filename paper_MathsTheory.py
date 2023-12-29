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
import shapely
import skimage as sm
import skimage.measure
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
from mpl_toolkits.mplot3d import Axes3D
import tifffile
from skimage.draw import circle_perimeter
import xml.etree.ElementTree as et
from scipy.optimize import leastsq
from datetime import datetime
import cellProperties as cell
import utils as util

pd.options.mode.chained_assignment = None  # default='warn'

plt.rcParams.update({"font.size": 16})


# -------------------

filenames, fileType = util.getFilesType()
scale = 123.26 / 512


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


def forIntegral(k, R, T, B, mQ, L):
    k, R, T = np.meshgrid(k, R, T, indexing="ij")
    return mQ * k * np.exp(-(B + L * k**2) * T) * sc.jv(0, R * k) / (B + L * k**2)

fileTypes = "Unwound18h"
T = np.linspace(0, 89, 90)

# compare: Mean rho
if False:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    filenames = util.getFilesType(fileType)[0]
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    rho = np.zeros([len(filenames), len(T)])
    for i in range(len(filenames)):
        filename = filenames[i]
        df = dfShape[dfShape["Filename"] == filename]
        for j in range(len(T)):
            t = T[j]
            rho[i, j] = 1 / np.mean(df["Area"][df["T"] == t])

    time = 2 * T

    std = np.std(rho, axis=0)
    rho = np.mean(rho, axis=0)
    colour, mark = util.getColorLineMarker(fileType, "18h")
    fileTitle = util.getFileTitle(fileType)
    ax.plot(time, rho, color=colour, marker=mark, markevery=10)
    ax.fill_between(time, rho - std, rho + std, alpha=0.15, color=colour)

    ax.set_ylim([0.055, 0.09])
    ax.set(xlabel="Time (mins)", ylabel=r"$\bar{\rho}$")
    ax.set_title(r"Mean $\rho$" + " with \n time unwounded")
    fig.savefig(
        f"results/mathPostWoundPaper/mean rho {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# compare: Mean Q1 tensor
if False:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    filenames = util.getFilesType(fileType)[0]
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    q1 = np.zeros([len(filenames), len(T)])
    for i in range(len(filenames)):
        filename = filenames[i]
        df = dfShape[dfShape["Filename"] == filename]
        for j in range(len(T)):
            t = T[j]
            q1[i, j] = np.mean(df["q"][df["T"] == t])[0, 0]

    time = 2 * T

    std = np.std(q1, axis=0)
    Q1 = np.mean(q1, axis=0)
    colour, mark = util.getColorLineMarker(fileType, "18h")
    fileTitle = util.getFileTitle(fileType)
    ax.plot(time, Q1, label=fileTitle, color=colour, marker=mark, markevery=10)
    ax.fill_between(time, Q1 - std, Q1 + std, alpha=0.15, color=colour)

    ax.set_ylim([0.002, 0.035])
    ax.set(xlabel="Time (mins)", ylabel=r"$\bar{Q}^{(1)}$")
    ax.set_title(r"Mean $Q^{(1)}$" + " with \n time unwounded")
    fig.savefig(
        f"results/mathPostWoundPaper/mean Q1 {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


# --------- Unwounded wt ----------

filenames, fileType = util.getFilesType("Unwound18h")
T = 90

# space time cell-cell shape correlation
if False:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    grid = 20
    timeGrid = 30

    T = np.linspace(0, (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)
    theta = np.linspace(0, 2 * np.pi, 17)

    dfShape["dR"] = list(np.zeros([len(dfShape)]))
    dfShape["dT"] = list(np.zeros([len(dfShape)]))
    dfShape["dtheta"] = list(np.zeros([len(dfShape)]))

    dfShape["dq1dq1i"] = list(np.zeros([len(dfShape)]))
    dfShape["dq2dq2i"] = list(np.zeros([len(dfShape)]))
    dfShape["dq1dq2i"] = list(np.zeros([len(dfShape)]))

    for k in range(len(filenames)):
        filename = filenames[k]
        path_to_file = f"databases/postWoundPaperCorrelations/dfCorMidway{filename}.pkl"
        if False == exists(path_to_file):
            _df = []

            dQ1dQ1Correlation = np.zeros([len(T), len(R), len(theta)])
            dQ2dQ2Correlation = np.zeros([len(T), len(R), len(theta)])
            dQ1dQ2Correlation = np.zeros([len(T), len(R), len(theta)])

            dQ1dQ1Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dQ2dQ2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dQ1dQ2Correlation_std = np.zeros([len(T), len(R), len(theta)])

            dQ1dQ1total = np.zeros([len(T), len(R), len(theta)])
            dQ2dQ2total = np.zeros([len(T), len(R), len(theta)])
            dQ1dQ2total = np.zeros([len(T), len(R), len(theta)])

            dq1dq1ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]
            dq2dq2ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]
            dq1dq2ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]  # t, r, theta

            print(datetime.now().strftime("%H:%M:%S ") + filename)
            dfShapef = dfShape[dfShape["Filename"] == filename].copy()
            dfPostWound = dfShapef[
                np.array(dfShapef["T"] >= 45) & np.array(dfShapef["T"] < 75)
            ]
            n = int(len(dfPostWound) / 5)
            random.seed(10)
            count = 0
            Is = []
            for i0 in range(n):
                i = int(random.random() * n)
                while i in Is:
                    i = int(random.random() * n)
                Is.append(i)
                if i0 % int((n) / 10) == 0:
                    print(datetime.now().strftime("%H:%M:%S") + f" {10*count}%")
                    count += 1

                x = dfPostWound["X"].iloc[i]
                y = dfPostWound["Y"].iloc[i]
                t = dfPostWound["T"].iloc[i]
                dq1 = dfPostWound["dq"].iloc[i][0, 0]
                dq2 = dfPostWound["dq"].iloc[i][0, 1]
                dfPostWound.loc[:, "dR"] = (
                    (
                        (dfPostWound.loc[:, "X"] - x) ** 2
                        + (dfPostWound.loc[:, "Y"] - y) ** 2
                    )
                    ** 0.5
                ).copy()
                df = dfPostWound[
                    [
                        "X",
                        "Y",
                        "T",
                        "dq",
                        "dR",
                        "dT",
                        "dtheta",
                        "dq1dq1i",
                        "dq2dq2i",
                        "dq1dq2i",
                    ]
                ]
                df = df[np.array(df["dR"] < R[-1]) & np.array(df["dR"] >= 0)]

                df["dT"] = df.loc[:, "T"] - t
                df = df[np.array(df["dT"] < timeGrid) & np.array(df["dT"] >= 0)]
                if len(df) != 0:
                    theta = np.arctan2(df.loc[:, "Y"] - y, df.loc[:, "X"] - x)
                    df["dtheta"] = np.where(theta < 0, 2 * np.pi + theta, theta)
                    df["dq1dq1i"] = list(
                        dq1 * np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 0]
                    )
                    df["dq2dq2i"] = list(
                        dq2 * np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 1]
                    )
                    df["dq1dq2i"] = list(
                        dq1 * np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 1]
                    )

                    for j in range(len(df)):
                        dq1dq1ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dq1dq1i"].iloc[j])
                        dq2dq2ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dq2dq2i"].iloc[j])
                        dq1dq2ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dq1dq2i"].iloc[j])

            T = np.linspace(0, (timeGrid - 1), timeGrid)
            R = np.linspace(0, 2 * (grid - 1), grid)
            theta = np.linspace(0, 2 * np.pi, 17)
            for i in range(len(T)):
                for j in range(len(R)):
                    for th in range(len(theta)):
                        dQ1dQ1Correlation[i][j][th] = np.mean(dq1dq1ij[i][j][th])
                        dQ2dQ2Correlation[i][j][th] = np.mean(dq2dq2ij[i][j][th])
                        dQ1dQ2Correlation[i][j][th] = np.mean(dq1dq2ij[i][j][th])

                        dQ1dQ1Correlation_std[i][j][th] = np.std(dq1dq1ij[i][j][th])
                        dQ2dQ2Correlation_std[i][j][th] = np.std(dq2dq2ij[i][j][th])
                        dQ1dQ2Correlation_std[i][j][th] = np.std(dq1dq2ij[i][j][th])

                        dQ1dQ1total[i][j][th] = len(dq1dq1ij[i][j][th])
                        dQ2dQ2total[i][j][th] = len(dq2dq2ij[i][j][th])
                        dQ1dQ2total[i][j][th] = len(dq1dq2ij[i][j][th])

            _df.append(
                {
                    "Filename": filename,
                    "dQ1dQ1Correlation": dQ1dQ1Correlation,
                    "dQ2dQ2Correlation": dQ2dQ2Correlation,
                    "dQ1dQ2Correlation": dQ1dQ2Correlation,
                    "dQ1dQ1Correlation_std": dQ1dQ1Correlation_std,
                    "dQ2dQ2Correlation_std": dQ2dQ2Correlation_std,
                    "dQ1dQ2Correlation_std": dQ1dQ2Correlation_std,
                    "dQ1dQ1Count": dQ1dQ1total,
                    "dQ2dQ2Count": dQ2dQ2total,
                    "dQ1dQ2Count": dQ1dQ2total,
                }
            )
            dfCorrelation = pd.DataFrame(_df)
            dfCorrelation.to_pickle(
                f"databases/postWoundPaperCorrelations/dfCorMidway{filename}.pkl"
            )

# space time cell density correlation
if False:
    grid = 5
    timeGrid = 6
    gridSize = 10
    gridSizeT = 5
    theta = np.linspace(0, 2 * np.pi, 17)

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
        heatmapdrho = np.zeros([30, xGrid, yGrid])
        inPlaneEcad = np.zeros([30, xGrid, yGrid])

        for t in range(30):
            dft = df[df["T"] == t + 45]
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
        df.to_pickle(f"databases/postWoundPaperCorrelations/dfCorRho{filename}.pkl")

# space time cell density-shape correlation
if False:
    grid = 5
    timeGrid = 6
    gridSize = 10
    gridSizeT = 5
    theta = np.linspace(0, 2 * np.pi, 17)

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
            dfShapef = dfShape[dfShape["Filename"] == filename].copy()
            dfPostWound = dfShapef[
                np.array(dfShapef["T"] >= 45) & np.array(dfShapef["T"] < 75)
            ]

            heatmapdrho = np.zeros([30, xGrid, yGrid])
            inPlaneEcad = np.zeros([30, xGrid, yGrid])

            for t in range(30):
                dft = dfPostWound[dfPostWound["T"] == t + 45]
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

                            dfPostWound.loc[:, "dR"] = (
                                (
                                    (dfPostWound.loc[:, "X"] - x) ** 2
                                    + (dfPostWound.loc[:, "Y"] - y) ** 2
                                )
                                ** 0.5
                            ).copy()

                            df = dfPostWound[
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

                            df["dT"] = df.loc[:, "T"] - (t + 45)
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
            df.to_pickle(
                f"databases/postWoundPaperCorrelations/dfCorRhoQ{filename}.pkl"
            )

# collect all correlations
if False:
    _df = []
    for filename in filenames:
        dfCorMid = pd.read_pickle(
            f"databases/postWoundPaperCorrelations/dfCorMidway{filename}.pkl"
        )
        dfCorRho = pd.read_pickle(
            f"databases/postWoundPaperCorrelations/dfCorRho{filename}.pkl"
        )
        dfCorRhoQ = pd.read_pickle(
            f"databases/postWoundPaperCorrelations/dfCorRhoQ{filename}.pkl"
        )

        dQ1dQ1 = np.nan_to_num(dfCorMid["dQ1dQ1Correlation"].iloc[0])[:, :27]
        dQ1dQ1_std = np.nan_to_num(dfCorMid["dQ1dQ1Correlation_std"].iloc[0])[:, :27]
        dQ1dQ1total = np.nan_to_num(dfCorMid["dQ1dQ1Count"].iloc[0])[:, :27]
        if np.sum(dQ1dQ1) == 0:
            print("dQ1dQ1")

        dQ2dQ2 = np.nan_to_num(dfCorMid["dQ2dQ2Correlation"].iloc[0])[:, :27]
        dQ2dQ2_std = np.nan_to_num(dfCorMid["dQ2dQ2Correlation_std"].iloc[0])[:, :27]
        dQ2dQ2total = np.nan_to_num(dfCorMid["dQ2dQ2Count"].iloc[0])[:, :27]
        if np.sum(dQ2dQ2) == 0:
            print("dQ2dQ2")

        dQ1dQ2 = np.nan_to_num(dfCorMid["dQ1dQ2Correlation"].iloc[0])
        dQ1dQ2_std = np.nan_to_num(dfCorMid["dQ1dQ2Correlation_std"].iloc[0])
        dQ1dQ2total = np.nan_to_num(dfCorMid["dQ1dQ2Count"].iloc[0])
        if np.sum(dQ1dQ2) == 0:
            print("dQ1dQ2")

        dRhodRho = np.nan_to_num(dfCorRho["dRhodRhoCorrelation"].iloc[0])
        dRhodRho_std = np.nan_to_num(dfCorRho["dRhodRhoCorrelation_std"].iloc[0])
        count_Rho = np.nan_to_num(dfCorRho["Count"].iloc[0])

        dQ1dRho = np.nan_to_num(dfCorRhoQ["dRhodQ1Correlation"].iloc[0])
        dQ1dRho_std = np.nan_to_num(dfCorRhoQ["dRhodQ1Correlation_std"].iloc[0])
        dQ2dRho = np.nan_to_num(dfCorRhoQ["dRhodQ2Correlation"].iloc[0])
        dQ2dRho_std = np.nan_to_num(dfCorRhoQ["dRhodQ2Correlation_std"].iloc[0])
        count_RhoQ = np.nan_to_num(dfCorRhoQ["Count"].iloc[0])

        _df.append(
            {
                "Filename": filename,
                "dQ1dQ1Correlation": dQ1dQ1,
                "dQ1dQ1Correlation_std": dQ1dQ1_std,
                "dQ1dQ1Count": dQ1dQ1total,
                "dQ2dQ2Correlation": dQ2dQ2,
                "dQ2dQ2Correlation_std": dQ2dQ2_std,
                "dQ2dQ2Count": dQ2dQ2total,
                "dQ1dQ2Correlation": dQ1dQ2,
                "dQ1dQ2Correlation_std": dQ1dQ2_std,
                "dQ1dQ2Count": dQ1dQ2total,
                "dRho_SdRho_S": dRhodRho,
                "dRho_SdRho_S_std": dRhodRho_std,
                "Count Rho_S": count_Rho,
                "dQ1dRho_S": dQ1dRho,
                "dQ1dRho_S_std": dQ1dRho_std,
                "dQ2dRho_S": dQ2dRho,
                "dQ2dRho_S_std": dQ2dRho_std,
                "Count Rho_S Q": count_RhoQ,
            }
        )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/postWoundPaperCorrelations/dfCorrelations{fileType}.pkl")


# --------- Unwounded JNK ----------

filenames, fileType = util.getFilesType("UnwoundJNK")

# space time cell-cell shape correlation
if False:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    grid = 20
    timeGrid = 30

    T = np.linspace(0, (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)
    theta = np.linspace(0, 2 * np.pi, 17)

    dfShape["dR"] = list(np.zeros([len(dfShape)]))
    dfShape["dT"] = list(np.zeros([len(dfShape)]))
    dfShape["dtheta"] = list(np.zeros([len(dfShape)]))

    dfShape["dq1dq1i"] = list(np.zeros([len(dfShape)]))
    dfShape["dq2dq2i"] = list(np.zeros([len(dfShape)]))

    for k in range(len(filenames)):
        filename = filenames[k]
        path_to_file = f"databases/postWoundPaperCorrelations/dfCorMidway{filename}.pkl"
        if False == exists(path_to_file):
            _df = []

            dQ1dQ1Correlation = np.zeros([len(T), len(R), len(theta)])
            dQ2dQ2Correlation = np.zeros([len(T), len(R), len(theta)])

            dQ1dQ1Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dQ2dQ2Correlation_std = np.zeros([len(T), len(R), len(theta)])

            dQ1dQ1total = np.zeros([len(T), len(R), len(theta)])
            dQ2dQ2total = np.zeros([len(T), len(R), len(theta)])

            dq1dq1ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]
            dq2dq2ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]  # t, r, theta

            print(datetime.now().strftime("%H:%M:%S ") + filename)
            dfShapef = dfShape[dfShape["Filename"] == filename].copy()
            dfPostWound = dfShapef[
                np.array(dfShapef["T"] >= 45) & np.array(dfShapef["T"] < 75)
            ]
            n = int(len(dfPostWound) / 10)
            random.seed(10)
            count = 0
            Is = []
            for i0 in range(n):
                i = int(random.random() * n)
                while i in Is:
                    i = int(random.random() * n)
                Is.append(i)
                if i0 % int((n) / 10) == 0:
                    print(datetime.now().strftime("%H:%M:%S") + f" {10*count}%")
                    count += 1

                x = dfPostWound["X"].iloc[i]
                y = dfPostWound["Y"].iloc[i]
                t = dfPostWound["T"].iloc[i]
                dq1 = dfPostWound["dq"].iloc[i][0, 0]
                dq2 = dfPostWound["dq"].iloc[i][0, 1]
                dfPostWound.loc[:, "dR"] = (
                    (
                        (dfPostWound.loc[:, "X"] - x) ** 2
                        + (dfPostWound.loc[:, "Y"] - y) ** 2
                    )
                    ** 0.5
                ).copy()
                df = dfPostWound[
                    [
                        "X",
                        "Y",
                        "T",
                        "dq",
                        "dR",
                        "dT",
                        "dtheta",
                        "dq1dq1i",
                        "dq2dq2i",
                    ]
                ]
                df = df[np.array(df["dR"] < R[-1]) & np.array(df["dR"] >= 0)]

                df["dT"] = df.loc[:, "T"] - t
                df = df[np.array(df["dT"] < timeGrid) & np.array(df["dT"] >= 0)]
                if len(df) != 0:
                    theta = np.arctan2(df.loc[:, "Y"] - y, df.loc[:, "X"] - x)
                    df["dtheta"] = np.where(theta < 0, 2 * np.pi + theta, theta)
                    df["dq1dq1i"] = list(
                        dq1 * np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 0]
                    )
                    df["dq2dq2i"] = list(
                        dq2 * np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 1]
                    )

                    for j in range(len(df)):
                        dq1dq1ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dq1dq1i"].iloc[j])
                        dq2dq2ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dq2dq2i"].iloc[j])

            T = np.linspace(0, (timeGrid - 1), timeGrid)
            R = np.linspace(0, 2 * (grid - 1), grid)
            theta = np.linspace(0, 2 * np.pi, 17)
            for i in range(len(T)):
                for j in range(len(R)):
                    for th in range(len(theta)):
                        dQ1dQ1Correlation[i][j][th] = np.mean(dq1dq1ij[i][j][th])
                        dQ2dQ2Correlation[i][j][th] = np.mean(dq2dq2ij[i][j][th])

                        dQ1dQ1Correlation_std[i][j][th] = np.std(dq1dq1ij[i][j][th])
                        dQ2dQ2Correlation_std[i][j][th] = np.std(dq2dq2ij[i][j][th])

                        dQ1dQ1total[i][j][th] = len(dq1dq1ij[i][j][th])
                        dQ2dQ2total[i][j][th] = len(dq2dq2ij[i][j][th])

            _df.append(
                {
                    "Filename": filename,
                    "dQ1dQ1Correlation": dQ1dQ1Correlation,
                    "dQ2dQ2Correlation": dQ2dQ2Correlation,
                    "dQ1dQ1Correlation_std": dQ1dQ1Correlation_std,
                    "dQ2dQ2Correlation_std": dQ2dQ2Correlation_std,
                    "dQ1dQ1Count": dQ1dQ1total,
                    "dQ2dQ2Count": dQ2dQ2total,
                }
            )
            dfCorrelation = pd.DataFrame(_df)
            dfCorrelation.to_pickle(
                f"databases/postWoundPaperCorrelations/dfCorMidway{filename}.pkl"
            )

# space time cell density correlation
if False:
    grid = 5
    timeGrid = 6
    gridSize = 10
    gridSizeT = 5
    theta = np.linspace(0, 2 * np.pi, 17)

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
        heatmapdrho = np.zeros([30, xGrid, yGrid])
        inPlaneEcad = np.zeros([30, xGrid, yGrid])

        for t in range(30):
            dft = df[df["T"] == t + 45]
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
        df.to_pickle(f"databases/postWoundPaperCorrelations/dfCorRho{filename}.pkl")

# collect all correlations
if False:
    _df = []
    for filename in filenames:
        dfCorMid = pd.read_pickle(
            f"databases/postWoundPaperCorrelations/dfCorMidway{filename}.pkl"
        )
        dfCorRho = pd.read_pickle(
            f"databases/postWoundPaperCorrelations/dfCorRho{filename}.pkl"
        )

        dQ1dQ1 = np.nan_to_num(dfCorMid["dQ1dQ1Correlation"].iloc[0])[:, :27]
        dQ1dQ1_std = np.nan_to_num(dfCorMid["dQ1dQ1Correlation_std"].iloc[0])[:, :27]
        dQ1dQ1total = np.nan_to_num(dfCorMid["dQ1dQ1Count"].iloc[0])[:, :27]
        if np.sum(dQ1dQ1) == 0:
            print("dQ1dQ1")

        dQ2dQ2 = np.nan_to_num(dfCorMid["dQ2dQ2Correlation"].iloc[0])[:, :27]
        dQ2dQ2_std = np.nan_to_num(dfCorMid["dQ2dQ2Correlation_std"].iloc[0])[:, :27]
        dQ2dQ2total = np.nan_to_num(dfCorMid["dQ2dQ2Count"].iloc[0])[:, :27]
        if np.sum(dQ2dQ2) == 0:
            print("dQ2dQ2")

        dRhodRho = np.nan_to_num(dfCorRho["dRhodRhoCorrelation"].iloc[0])
        dRhodRho_std = np.nan_to_num(dfCorRho["dRhodRhoCorrelation_std"].iloc[0])
        count_Rho = np.nan_to_num(dfCorRho["Count"].iloc[0])

        _df.append(
            {
                "Filename": filename,
                "dQ1dQ1Correlation": dQ1dQ1,
                "dQ1dQ1Correlation_std": dQ1dQ1_std,
                "dQ1dQ1Count": dQ1dQ1total,
                "dQ2dQ2Correlation": dQ2dQ2,
                "dQ2dQ2Correlation_std": dQ2dQ2_std,
                "dQ2dQ2Count": dQ2dQ2total,
                "dRho_SdRho_S": dRhodRho,
                "dRho_SdRho_S_std": dRhodRho_std,
                "Count Rho_S": count_Rho,
            }
        )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/postWoundPaperCorrelations/dfCorrelations{fileType}.pkl")


# --------- Wounded wt ----------

filenames, fileType = util.getFilesType("WoundL18h")

# space time cell density correlation close to woundsite
if False:
    grid = 5
    timeGrid = 6
    gridSize = 10
    gridSizeT = 5
    theta = np.linspace(0, 2 * np.pi, 17)

    dfShape = pd.read_pickle(f"databases/dfShapeWound{fileType}.pkl")

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
        heatmapdrho = np.zeros([30, xGrid, yGrid])
        inPlaneEcad = np.zeros([30, xGrid, yGrid])
        inNearWound = np.zeros([30, xGrid, yGrid])

        for t in range(30):
            dft = df[(df["T"] == 2 * t + 90) | (df["T"] == 2 * t + 91)]
            if list(dft["Area"]) != []:
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
                            if (np.min(dfg["R"]) < 30) & (t <= 30):
                                inNearWound[t, i, j] = 1

                heatmapdrho[t] = heatmapdrho[t] - np.mean(
                    heatmapdrho[t][inPlaneEcad[t] == 1]
                )

        for i in range(xGrid):
            for j in range(yGrid):
                for t in T:
                    t = int(t)
                    if np.sum(inNearWound[t : t + gridSizeT, i, j]) > 0:
                        deltarho = np.mean(
                            heatmapdrho[t : t + gridSizeT, i, j][
                                inNearWound[t : t + gridSizeT, i, j] > 0
                            ]
                        )
                        for idash in range(xGrid):
                            for jdash in range(yGrid):
                                for tdash in T:
                                    tdash = int(tdash)
                                    if (
                                        np.sum(
                                            inNearWound[
                                                tdash : tdash + gridSizeT, idash, jdash
                                            ]
                                        )
                                        > 0
                                    ):
                                        deltaT = int((tdash - t) / gridSizeT)
                                        deltaR = int(
                                            ((i - idash) ** 2 + (j - jdash) ** 2) ** 0.5
                                        )
                                        deltaTheta = int(
                                            np.arctan2((j - jdash), (i - idash))
                                            * 8
                                            / np.pi
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
                                                                tdash : tdash
                                                                + gridSizeT,
                                                                idash,
                                                                jdash,
                                                            ][
                                                                inPlaneEcad[
                                                                    t : t + gridSizeT,
                                                                    i,
                                                                    j,
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
        df.to_pickle(
            f"databases/postWoundPaperCorrelations/dfCorRhoClose{filename}.pkl"
        )

# space time cell density correlation far from to wound
if False:
    grid = 5
    timeGrid = 6
    gridSize = 10
    gridSizeT = 5
    theta = np.linspace(0, 2 * np.pi, 17)

    dfShape = pd.read_pickle(f"databases/dfShapeWound{fileType}.pkl")

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
        heatmapdrho = np.zeros([30, xGrid, yGrid])
        inPlaneEcad = np.zeros([30, xGrid, yGrid])
        inFarWound = np.zeros([30, xGrid, yGrid])

        for t in range(30):
            dft = df[(df["T"] == 2 * t + 90) | (df["T"] == 2 * t + 91)]
            if list(dft["Area"]) != []:
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
                            if (np.min(dfg["R"]) >= 30) & (t <= 30):
                                inFarWound[t, i, j] = 1

                heatmapdrho[t] = heatmapdrho[t] - np.mean(
                    heatmapdrho[t][inPlaneEcad[t] == 1]
                )

        for i in range(xGrid):
            for j in range(yGrid):
                for t in T:
                    t = int(t)
                    if np.sum(inFarWound[t : t + gridSizeT, i, j]) > 0:
                        deltarho = np.mean(
                            heatmapdrho[t : t + gridSizeT, i, j][
                                inFarWound[t : t + gridSizeT, i, j] > 0
                            ]
                        )
                        for idash in range(xGrid):
                            for jdash in range(yGrid):
                                for tdash in T:
                                    tdash = int(tdash)
                                    if (
                                        np.sum(
                                            inFarWound[
                                                tdash : tdash + gridSizeT, idash, jdash
                                            ]
                                        )
                                        > 0
                                    ):
                                        deltaT = int((tdash - t) / gridSizeT)
                                        deltaR = int(
                                            ((i - idash) ** 2 + (j - jdash) ** 2) ** 0.5
                                        )
                                        deltaTheta = int(
                                            np.arctan2((j - jdash), (i - idash))
                                            * 8
                                            / np.pi
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
                                                                tdash : tdash
                                                                + gridSizeT,
                                                                idash,
                                                                jdash,
                                                            ][
                                                                inPlaneEcad[
                                                                    t : t + gridSizeT,
                                                                    i,
                                                                    j,
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
        df.to_pickle(f"databases/postWoundPaperCorrelations/dfCorRhoFar{filename}.pkl")

# space time cell-cell shape correlation close to wound
if False:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    grid = 20
    timeGrid = 30

    T = np.linspace(0, (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)
    theta = np.linspace(0, 2 * np.pi, 17)

    dfShape["dR"] = list(np.zeros([len(dfShape)]))
    dfShape["dT"] = list(np.zeros([len(dfShape)]))
    dfShape["dtheta"] = list(np.zeros([len(dfShape)]))

    dfShape["dq1dq1i"] = list(np.zeros([len(dfShape)]))
    dfShape["dq2dq2i"] = list(np.zeros([len(dfShape)]))

    for k in range(len(filenames)):
        filename = filenames[k]
        dist = sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)
        path_to_file = (
            f"databases/postWoundPaperCorrelations/dfCorCloseWound{filename}.pkl"
        )
        if False == exists(path_to_file):
            _df = []
            dQ1dQ1Correlation = np.zeros([len(T), len(R), len(theta)])
            dQ2dQ2Correlation = np.zeros([len(T), len(R), len(theta)])
            dQ1dQ1Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dQ2dQ2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dQ1dQ1total = np.zeros([len(T), len(R), len(theta)])
            dQ2dQ2total = np.zeros([len(T), len(R), len(theta)])

            dq1dq1ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]
            dq2dq2ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]

            print(datetime.now().strftime("%H:%M:%S ") + filename)
            dfShapeF = dfShape[dfShape["Filename"] == filename].copy()
            dfClose = dfShapeF[
                np.array(dfShapeF["T"] > 45)
                & np.array(dfShapeF["T"] < 75)
                & np.array(dfShapeF["R"] < 30 / scale)
            ]
            n = len(dfClose)
            random.seed(10)
            count = 0
            Is = []
            for i in range(n):
                if i % int((n) / 10) == 0:
                    print(datetime.now().strftime("%H:%M:%S ") + f"{10*count}%")
                    count += 1

                x = dfClose["X"].iloc[i]
                y = dfClose["Y"].iloc[i]
                t = dfClose["T"].iloc[i]
                dp1 = dfClose["dp"].iloc[i][0]
                dp2 = dfClose["dp"].iloc[i][1]
                dq1 = dfClose["dq"].iloc[i][0, 0]
                dq2 = dfClose["dq"].iloc[i][0, 1]
                dfClose.loc[:, "dR"] = (
                    ((dfClose.loc[:, "X"] - x) ** 2 + (dfClose.loc[:, "Y"] - y) ** 2)
                    ** 0.5
                ).copy()
                df = dfClose[
                    [
                        "X",
                        "Y",
                        "T",
                        "dp",
                        "dq",
                        "dR",
                        "dT",
                        "dtheta",
                        "dq1dq1i",
                        "dq2dq2i",
                    ]
                ]
                df = df[np.array(df["dR"] < R[-1]) & np.array(df["dR"] >= 0)]

                df["dT"] = df.loc[:, "T"] - t
                df = df[np.array(df["dT"] < timeGrid) & np.array(df["dT"] >= 0)]
                if len(df) != 0:
                    theta = np.arctan2(df.loc[:, "Y"] - y, df.loc[:, "X"] - x)
                    df["dtheta"] = np.where(theta < 0, 2 * np.pi + theta, theta)
                    df["dq1dq1i"] = list(
                        dq1 * np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 0]
                    )
                    df["dq2dq2i"] = list(
                        dq2 * np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 1]
                    )

                    for j in range(len(df)):
                        dq1dq1ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dq1dq1i"].iloc[j])
                        dq2dq2ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dq2dq2i"].iloc[j])

            T = np.linspace(0, (timeGrid - 1), timeGrid)
            R = np.linspace(0, 2 * (grid - 1), grid)
            theta = np.linspace(0, 2 * np.pi, 17)
            for i in range(len(T)):
                for j in range(len(R)):
                    for th in range(len(theta)):
                        dQ1dQ1Correlation[i][j][th] = np.mean(dq1dq1ij[i][j][th])
                        dQ2dQ2Correlation[i][j][th] = np.mean(dq2dq2ij[i][j][th])
                        dQ1dQ1Correlation_std[i][j][th] = np.std(dq1dq1ij[i][j][th])
                        dQ2dQ2Correlation_std[i][j][th] = np.std(dq2dq2ij[i][j][th])
                        dQ1dQ1total[i][j][th] = len(dq1dq1ij[i][j][th])
                        dQ2dQ2total[i][j][th] = len(dq2dq2ij[i][j][th])

            _df.append(
                {
                    "Filename": filename,
                    "dQ1dQ1Correlation": dQ1dQ1Correlation,
                    "dQ2dQ2Correlation": dQ2dQ2Correlation,
                    "dQ1dQ1Correlation_std": dQ1dQ1Correlation_std,
                    "dQ2dQ2Correlation_std": dQ2dQ2Correlation_std,
                    "dQ1dQ1Count": dQ1dQ1total,
                    "dQ2dQ2Count": dQ2dQ2total,
                }
            )
            dfCorrelation = pd.DataFrame(_df)
            dfCorrelation.to_pickle(
                f"databases/postWoundPaperCorrelations/dfCorCloseWound{filename}.pkl"
            )

# space time cell-cell shape correlation far from wound
if False:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    grid = 20
    timeGrid = 30

    T = np.linspace(0, (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)
    theta = np.linspace(0, 2 * np.pi, 17)

    dfShape["dR"] = list(np.zeros([len(dfShape)]))
    dfShape["dT"] = list(np.zeros([len(dfShape)]))
    dfShape["dtheta"] = list(np.zeros([len(dfShape)]))

    dfShape["dq1dq1i"] = list(np.zeros([len(dfShape)]))
    dfShape["dq2dq2i"] = list(np.zeros([len(dfShape)]))

    for k in range(len(filenames)):
        filename = filenames[k]
        path_to_file = (
            f"databases/postWoundPaperCorrelations/dfCorFarWound{filename}.pkl"
        )
        if False == exists(path_to_file):
            _df = []
            dQ1dQ1Correlation = np.zeros([len(T), len(R), len(theta)])
            dQ2dQ2Correlation = np.zeros([len(T), len(R), len(theta)])
            dQ1dQ1Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dQ2dQ2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dQ1dQ1total = np.zeros([len(T), len(R), len(theta)])
            dQ2dQ2total = np.zeros([len(T), len(R), len(theta)])

            dq1dq1ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]
            dq2dq2ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]

            print(datetime.now().strftime("%H:%M:%S ") + filename)
            dfShapeF = dfShape[dfShape["Filename"] == filename].copy()
            dfFar = dfShapeF[
                np.array(dfShapeF["T"] > 45)
                & np.array(dfShapeF["T"] < 75)
                & np.array(dfShapeF["R"] >= 30 / scale)
            ]
            n = int(len(dfFar) / 10)
            random.seed(10)
            count = 0
            Is = []
            for i in range(n):
                if i % int((n) / 10) == 0:
                    print(datetime.now().strftime("%H:%M:%S") + f" {10*count}%")
                    count += 1

                x = dfFar["X"].iloc[i]
                y = dfFar["Y"].iloc[i]
                t = dfFar["T"].iloc[i]
                dp1 = dfFar["dp"].iloc[i][0]
                dp2 = dfFar["dp"].iloc[i][1]
                dq1 = dfFar["dq"].iloc[i][0, 0]
                dq2 = dfFar["dq"].iloc[i][0, 1]
                dfFar.loc[:, "dR"] = (
                    ((dfFar.loc[:, "X"] - x) ** 2 + (dfFar.loc[:, "Y"] - y) ** 2) ** 0.5
                ).copy()
                df = dfFar[
                    [
                        "X",
                        "Y",
                        "T",
                        "dp",
                        "dq",
                        "dR",
                        "dT",
                        "dtheta",
                        "dq1dq1i",
                        "dq2dq2i",
                    ]
                ]
                df = df[np.array(df["dR"] < R[-1]) & np.array(df["dR"] >= 0)]

                df["dT"] = df.loc[:, "T"] - t
                df = df[np.array(df["dT"] < timeGrid) & np.array(df["dT"] >= 0)]
                if len(df) != 0:
                    theta = np.arctan2(df.loc[:, "Y"] - y, df.loc[:, "X"] - x)
                    df["dtheta"] = np.where(theta < 0, 2 * np.pi + theta, theta)
                    df["dq1dq1i"] = list(
                        dq1 * np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 0]
                    )
                    df["dq2dq2i"] = list(
                        dq2 * np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 1]
                    )

                    for j in range(len(df)):
                        dq1dq1ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dq1dq1i"].iloc[j])
                        dq2dq2ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dq2dq2i"].iloc[j])

            T = np.linspace(0, (timeGrid - 1), timeGrid)
            R = np.linspace(0, 2 * (grid - 1), grid)
            theta = np.linspace(0, 2 * np.pi, 17)
            for i in range(len(T)):
                for j in range(len(R)):
                    for th in range(len(theta)):
                        dQ1dQ1Correlation[i][j][th] = np.mean(dq1dq1ij[i][j][th])
                        dQ2dQ2Correlation[i][j][th] = np.mean(dq2dq2ij[i][j][th])
                        dQ1dQ1Correlation_std[i][j][th] = np.std(dq1dq1ij[i][j][th])
                        dQ2dQ2Correlation_std[i][j][th] = np.std(dq2dq2ij[i][j][th])
                        dQ1dQ1total[i][j][th] = len(dq1dq1ij[i][j][th])
                        dQ2dQ2total[i][j][th] = len(dq2dq2ij[i][j][th])

            _df.append(
                {
                    "Filename": filename,
                    "dQ1dQ1Correlation": dQ1dQ1Correlation,
                    "dQ2dQ2Correlation": dQ2dQ2Correlation,
                    "dQ1dQ1Correlation_std": dQ1dQ1Correlation_std,
                    "dQ2dQ2Correlation_std": dQ2dQ2Correlation_std,
                    "dQ1dQ1Count": dQ1dQ1total,
                    "dQ2dQ2Count": dQ2dQ2total,
                }
            )
            dfCorrelation = pd.DataFrame(_df)
            dfCorrelation.to_pickle(
                f"databases/postWoundPaperCorrelations/dfCorFarWound{filename}.pkl"
            )

# collect all correlations
if False:
    _df = []
    for filename in filenames:
        print(filename)

        dfCorRhoClose = pd.read_pickle(
            f"databases/postWoundPaperCorrelations/dfCorRhoClose{filename}.pkl"
        )
        dfCorRhoFar = pd.read_pickle(
            f"databases/postWoundPaperCorrelations/dfCorRhoFar{filename}.pkl"
        )
        dfCorCloseWound = pd.read_pickle(
            f"databases/postWoundPaperCorrelations/dfCorCloseWound{filename}.pkl"
        )
        dfCorFarWound = pd.read_pickle(
            f"databases/postWoundPaperCorrelations/dfCorFarWound{filename}.pkl"
        )

        dRhodRhoClose = np.nan_to_num(dfCorRhoClose["dRhodRhoCorrelation"].iloc[0])
        dRhodRhoClose_std = np.nan_to_num(
            dfCorRhoClose["dRhodRhoCorrelation_std"].iloc[0]
        )
        dRhodRhoClosetotal = np.nan_to_num(dfCorRhoClose["Count"].iloc[0])
        if np.sum(dRhodRhoClosetotal) == 0:
            print("dRhodRhoClosetotal")

        dRhodRhoFar = np.nan_to_num(dfCorRhoFar["dRhodRhoCorrelation"].iloc[0])
        dRhodRhoFar_std = np.nan_to_num(dfCorRhoFar["dRhodRhoCorrelation_std"].iloc[0])
        dRhodRhoFartotal = np.nan_to_num(dfCorRhoFar["Count"].iloc[0])
        if np.sum(dRhodRhoFartotal) == 0:
            print("dRhodRhoFartotal")

        dQ1dQ1Close = np.nan_to_num(dfCorCloseWound["dQ1dQ1Correlation"].iloc[0])
        dQ1dQ1Close_std = np.nan_to_num(
            dfCorCloseWound["dQ1dQ1Correlation_std"].iloc[0]
        )
        dQ1dQ1Closetotal = np.nan_to_num(dfCorCloseWound["dQ1dQ1Count"].iloc[0])
        if np.sum(dQ1dQ1Closetotal) == 0:
            print("dQ1dQ1Closetotal")

        dQ2dQ2Close = np.nan_to_num(dfCorCloseWound["dQ2dQ2Correlation"].iloc[0])
        dQ2dQ2Close_std = np.nan_to_num(
            dfCorCloseWound["dQ2dQ2Correlation_std"].iloc[0]
        )
        dQ2dQ2Closetotal = np.nan_to_num(dfCorCloseWound["dQ2dQ2Count"].iloc[0])
        if np.sum(dQ2dQ2Closetotal) == 0:
            print("dQ2dQ2Closetotal")

        dQ1dQ1Far = np.nan_to_num(dfCorFarWound["dQ1dQ1Correlation"].iloc[0])
        dQ1dQ1Far_std = np.nan_to_num(dfCorFarWound["dQ1dQ1Correlation_std"].iloc[0])
        dQ1dQ1Fartotal = np.nan_to_num(dfCorFarWound["dQ1dQ1Count"].iloc[0])
        if np.sum(dQ1dQ1Fartotal) == 0:
            print("dQ1dQ1Fartotal")

        dQ2dQ2Far = np.nan_to_num(dfCorFarWound["dQ2dQ2Correlation"].iloc[0])
        dQ2dQ2Far_std = np.nan_to_num(dfCorFarWound["dQ2dQ2Correlation_std"].iloc[0])
        dQ2dQ2Fartotal = np.nan_to_num(dfCorFarWound["dQ2dQ2Count"].iloc[0])
        if np.sum(dQ2dQ2Fartotal) == 0:
            print("dQ2dQ2Fartotal")

        _df.append(
            {
                "Filename": filename,
                "dRhodRhoClose": dRhodRhoClose,
                "dRhodRhoClose_std": dRhodRhoClose_std,
                "dRhodRhoClosetotal": dRhodRhoClosetotal,
                "dRhodRhoFar": dRhodRhoFar,
                "dRhodRhoFar_std": dRhodRhoFar_std,
                "dRhodRhoFartotal": dRhodRhoFartotal,
                "dQ1dQ1Close": dQ1dQ1Close,
                "dQ1dQ1Close_std": dQ1dQ1Close_std,
                "dQ1dQ1Closetotal": dQ1dQ1Closetotal,
                "dQ2dQ2Close": dQ2dQ2Close,
                "dQ2dQ2Close_std": dQ2dQ2Close_std,
                "dQ2dQ2Closetotal": dQ2dQ2Closetotal,
                "dQ1dQ1Far": dQ1dQ1Far,
                "dQ1dQ1Far_std": dQ1dQ1Far_std,
                "dQ1dQ1Fartotal": dQ1dQ1Fartotal,
                "dQ2dQ2Far": dQ2dQ2Far,
                "dQ2dQ2Far_std": dQ2dQ2Far_std,
                "dQ2dQ2Fartotal": dQ2dQ2Fartotal,
            }
        )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/postWoundPaperCorrelations/dfCorrelations{fileType}.pkl")

# --------- Wounded JNK ----------

filenames, fileType = util.getFilesType("WoundLJNK")

# space time cell density correlation close to woundsite
if False:
    grid = 5
    timeGrid = 6
    gridSize = 10
    gridSizeT = 5
    theta = np.linspace(0, 2 * np.pi, 17)

    dfShape = pd.read_pickle(f"databases/dfShapeWound{fileType}.pkl")

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
        heatmapdrho = np.zeros([30, xGrid, yGrid])
        inPlaneEcad = np.zeros([30, xGrid, yGrid])
        inNearWound = np.zeros([30, xGrid, yGrid])

        for t in range(30):
            dft = df[(df["T"] == 2 * t + 90) | (df["T"] == 2 * t + 91)]
            if list(dft["Area"]) != []:
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
                            if (np.min(dfg["R"]) < 30) & (t <= 30):
                                inNearWound[t, i, j] = 1

                heatmapdrho[t] = heatmapdrho[t] - np.mean(
                    heatmapdrho[t][inPlaneEcad[t] == 1]
                )

        for i in range(xGrid):
            for j in range(yGrid):
                for t in T:
                    t = int(t)
                    if np.sum(inNearWound[t : t + gridSizeT, i, j]) > 0:
                        deltarho = np.mean(
                            heatmapdrho[t : t + gridSizeT, i, j][
                                inNearWound[t : t + gridSizeT, i, j] > 0
                            ]
                        )
                        for idash in range(xGrid):
                            for jdash in range(yGrid):
                                for tdash in T:
                                    tdash = int(tdash)
                                    if (
                                        np.sum(
                                            inNearWound[
                                                tdash : tdash + gridSizeT, idash, jdash
                                            ]
                                        )
                                        > 0
                                    ):
                                        deltaT = int((tdash - t) / gridSizeT)
                                        deltaR = int(
                                            ((i - idash) ** 2 + (j - jdash) ** 2) ** 0.5
                                        )
                                        deltaTheta = int(
                                            np.arctan2((j - jdash), (i - idash))
                                            * 8
                                            / np.pi
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
                                                                tdash : tdash
                                                                + gridSizeT,
                                                                idash,
                                                                jdash,
                                                            ][
                                                                inPlaneEcad[
                                                                    t : t + gridSizeT,
                                                                    i,
                                                                    j,
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
        df.to_pickle(
            f"databases/postWoundPaperCorrelations/dfCorRhoClose{filename}.pkl"
        )

# space time cell density correlation far from to wound
if False:
    grid = 5
    timeGrid = 6
    gridSize = 10
    gridSizeT = 5
    theta = np.linspace(0, 2 * np.pi, 17)

    dfShape = pd.read_pickle(f"databases/dfShapeWound{fileType}.pkl")

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
        heatmapdrho = np.zeros([30, xGrid, yGrid])
        inPlaneEcad = np.zeros([30, xGrid, yGrid])
        inFarWound = np.zeros([30, xGrid, yGrid])

        for t in range(30):
            dft = df[(df["T"] == 2 * t + 90) | (df["T"] == 2 * t + 91)]
            if list(dft["Area"]) != []:
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
                            if (np.min(dfg["R"]) >= 30) & (t <= 30):
                                inFarWound[t, i, j] = 1

                heatmapdrho[t] = heatmapdrho[t] - np.mean(
                    heatmapdrho[t][inPlaneEcad[t] == 1]
                )

        for i in range(xGrid):
            for j in range(yGrid):
                for t in T:
                    t = int(t)
                    if np.sum(inFarWound[t : t + gridSizeT, i, j]) > 0:
                        deltarho = np.mean(
                            heatmapdrho[t : t + gridSizeT, i, j][
                                inFarWound[t : t + gridSizeT, i, j] > 0
                            ]
                        )
                        for idash in range(xGrid):
                            for jdash in range(yGrid):
                                for tdash in T:
                                    tdash = int(tdash)
                                    if (
                                        np.sum(
                                            inFarWound[
                                                tdash : tdash + gridSizeT, idash, jdash
                                            ]
                                        )
                                        > 0
                                    ):
                                        deltaT = int((tdash - t) / gridSizeT)
                                        deltaR = int(
                                            ((i - idash) ** 2 + (j - jdash) ** 2) ** 0.5
                                        )
                                        deltaTheta = int(
                                            np.arctan2((j - jdash), (i - idash))
                                            * 8
                                            / np.pi
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
                                                                tdash : tdash
                                                                + gridSizeT,
                                                                idash,
                                                                jdash,
                                                            ][
                                                                inPlaneEcad[
                                                                    t : t + gridSizeT,
                                                                    i,
                                                                    j,
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
        df.to_pickle(f"databases/postWoundPaperCorrelations/dfCorRhoFar{filename}.pkl")

# space time cell-cell shape correlation close to wound
if False:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    grid = 20
    timeGrid = 30

    T = np.linspace(0, (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)
    theta = np.linspace(0, 2 * np.pi, 17)

    dfShape["dR"] = list(np.zeros([len(dfShape)]))
    dfShape["dT"] = list(np.zeros([len(dfShape)]))
    dfShape["dtheta"] = list(np.zeros([len(dfShape)]))

    dfShape["dq1dq1i"] = list(np.zeros([len(dfShape)]))
    dfShape["dq2dq2i"] = list(np.zeros([len(dfShape)]))

    for k in range(len(filenames)):
        filename = filenames[k]
        dist = sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)
        path_to_file = (
            f"databases/postWoundPaperCorrelations/dfCorCloseWound{filename}.pkl"
        )
        if False == exists(path_to_file):
            _df = []
            dQ1dQ1Correlation = np.zeros([len(T), len(R), len(theta)])
            dQ2dQ2Correlation = np.zeros([len(T), len(R), len(theta)])
            dQ1dQ1Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dQ2dQ2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dQ1dQ1total = np.zeros([len(T), len(R), len(theta)])
            dQ2dQ2total = np.zeros([len(T), len(R), len(theta)])

            dq1dq1ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]
            dq2dq2ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]

            print(datetime.now().strftime("%H:%M:%S ") + filename)
            dfShapeF = dfShape[dfShape["Filename"] == filename].copy()
            dfClose = dfShapeF[
                np.array(dfShapeF["T"] > 45)
                & np.array(dfShapeF["T"] < 75)
                & np.array(dfShapeF["R"] < 30 / scale)
            ]
            n = len(dfClose)
            random.seed(10)
            count = 0
            Is = []
            for i in range(n):
                if i % int((n) / 10) == 0:
                    print(datetime.now().strftime("%H:%M:%S ") + f"{10*count}%")
                    count += 1

                x = dfClose["X"].iloc[i]
                y = dfClose["Y"].iloc[i]
                t = dfClose["T"].iloc[i]
                dp1 = dfClose["dp"].iloc[i][0]
                dp2 = dfClose["dp"].iloc[i][1]
                dq1 = dfClose["dq"].iloc[i][0, 0]
                dq2 = dfClose["dq"].iloc[i][0, 1]
                dfClose.loc[:, "dR"] = (
                    ((dfClose.loc[:, "X"] - x) ** 2 + (dfClose.loc[:, "Y"] - y) ** 2)
                    ** 0.5
                ).copy()
                df = dfClose[
                    [
                        "X",
                        "Y",
                        "T",
                        "dp",
                        "dq",
                        "dR",
                        "dT",
                        "dtheta",
                        "dq1dq1i",
                        "dq2dq2i",
                    ]
                ]
                df = df[np.array(df["dR"] < R[-1]) & np.array(df["dR"] >= 0)]

                df["dT"] = df.loc[:, "T"] - t
                df = df[np.array(df["dT"] < timeGrid) & np.array(df["dT"] >= 0)]
                if len(df) != 0:
                    theta = np.arctan2(df.loc[:, "Y"] - y, df.loc[:, "X"] - x)
                    df["dtheta"] = np.where(theta < 0, 2 * np.pi + theta, theta)
                    df["dq1dq1i"] = list(
                        dq1 * np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 0]
                    )
                    df["dq2dq2i"] = list(
                        dq2 * np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 1]
                    )

                    for j in range(len(df)):
                        dq1dq1ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dq1dq1i"].iloc[j])
                        dq2dq2ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dq2dq2i"].iloc[j])

            T = np.linspace(0, (timeGrid - 1), timeGrid)
            R = np.linspace(0, 2 * (grid - 1), grid)
            theta = np.linspace(0, 2 * np.pi, 17)
            for i in range(len(T)):
                for j in range(len(R)):
                    for th in range(len(theta)):
                        dQ1dQ1Correlation[i][j][th] = np.mean(dq1dq1ij[i][j][th])
                        dQ2dQ2Correlation[i][j][th] = np.mean(dq2dq2ij[i][j][th])
                        dQ1dQ1Correlation_std[i][j][th] = np.std(dq1dq1ij[i][j][th])
                        dQ2dQ2Correlation_std[i][j][th] = np.std(dq2dq2ij[i][j][th])
                        dQ1dQ1total[i][j][th] = len(dq1dq1ij[i][j][th])
                        dQ2dQ2total[i][j][th] = len(dq2dq2ij[i][j][th])

            _df.append(
                {
                    "Filename": filename,
                    "dQ1dQ1Correlation": dQ1dQ1Correlation,
                    "dQ2dQ2Correlation": dQ2dQ2Correlation,
                    "dQ1dQ1Correlation_std": dQ1dQ1Correlation_std,
                    "dQ2dQ2Correlation_std": dQ2dQ2Correlation_std,
                    "dQ1dQ1Count": dQ1dQ1total,
                    "dQ2dQ2Count": dQ2dQ2total,
                }
            )
            dfCorrelation = pd.DataFrame(_df)
            dfCorrelation.to_pickle(
                f"databases/postWoundPaperCorrelations/dfCorCloseWound{filename}.pkl"
            )

# space time cell-cell shape correlation far from wound
if False:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    grid = 20
    timeGrid = 30

    T = np.linspace(0, (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)
    theta = np.linspace(0, 2 * np.pi, 17)

    dfShape["dR"] = list(np.zeros([len(dfShape)]))
    dfShape["dT"] = list(np.zeros([len(dfShape)]))
    dfShape["dtheta"] = list(np.zeros([len(dfShape)]))

    dfShape["dq1dq1i"] = list(np.zeros([len(dfShape)]))
    dfShape["dq2dq2i"] = list(np.zeros([len(dfShape)]))

    for k in range(len(filenames)):
        filename = filenames[k]
        path_to_file = (
            f"databases/postWoundPaperCorrelations/dfCorFarWound{filename}.pkl"
        )
        if False == exists(path_to_file):
            _df = []
            dQ1dQ1Correlation = np.zeros([len(T), len(R), len(theta)])
            dQ2dQ2Correlation = np.zeros([len(T), len(R), len(theta)])
            dQ1dQ1Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dQ2dQ2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dQ1dQ1total = np.zeros([len(T), len(R), len(theta)])
            dQ2dQ2total = np.zeros([len(T), len(R), len(theta)])

            dq1dq1ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]
            dq2dq2ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]

            print(datetime.now().strftime("%H:%M:%S ") + filename)
            dfShapeF = dfShape[dfShape["Filename"] == filename].copy()
            dfFar = dfShapeF[
                np.array(dfShapeF["T"] > 45)
                & np.array(dfShapeF["T"] < 75)
                & np.array(dfShapeF["R"] >= 30 / scale)
            ]
            n = int(len(dfFar) / 10)
            random.seed(10)
            count = 0
            Is = []
            for i in range(n):
                if i % int((n) / 10) == 0:
                    print(datetime.now().strftime("%H:%M:%S") + f" {10*count}%")
                    count += 1

                x = dfFar["X"].iloc[i]
                y = dfFar["Y"].iloc[i]
                t = dfFar["T"].iloc[i]
                dp1 = dfFar["dp"].iloc[i][0]
                dp2 = dfFar["dp"].iloc[i][1]
                dq1 = dfFar["dq"].iloc[i][0, 0]
                dq2 = dfFar["dq"].iloc[i][0, 1]
                dfFar.loc[:, "dR"] = (
                    ((dfFar.loc[:, "X"] - x) ** 2 + (dfFar.loc[:, "Y"] - y) ** 2) ** 0.5
                ).copy()
                df = dfFar[
                    [
                        "X",
                        "Y",
                        "T",
                        "dp",
                        "dq",
                        "dR",
                        "dT",
                        "dtheta",
                        "dq1dq1i",
                        "dq2dq2i",
                    ]
                ]
                df = df[np.array(df["dR"] < R[-1]) & np.array(df["dR"] >= 0)]

                df["dT"] = df.loc[:, "T"] - t
                df = df[np.array(df["dT"] < timeGrid) & np.array(df["dT"] >= 0)]
                if len(df) != 0:
                    theta = np.arctan2(df.loc[:, "Y"] - y, df.loc[:, "X"] - x)
                    df["dtheta"] = np.where(theta < 0, 2 * np.pi + theta, theta)
                    df["dq1dq1i"] = list(
                        dq1 * np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 0]
                    )
                    df["dq2dq2i"] = list(
                        dq2 * np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 1]
                    )

                    for j in range(len(df)):
                        dq1dq1ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dq1dq1i"].iloc[j])
                        dq2dq2ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dq2dq2i"].iloc[j])

            T = np.linspace(0, (timeGrid - 1), timeGrid)
            R = np.linspace(0, 2 * (grid - 1), grid)
            theta = np.linspace(0, 2 * np.pi, 17)
            for i in range(len(T)):
                for j in range(len(R)):
                    for th in range(len(theta)):
                        dQ1dQ1Correlation[i][j][th] = np.mean(dq1dq1ij[i][j][th])
                        dQ2dQ2Correlation[i][j][th] = np.mean(dq2dq2ij[i][j][th])
                        dQ1dQ1Correlation_std[i][j][th] = np.std(dq1dq1ij[i][j][th])
                        dQ2dQ2Correlation_std[i][j][th] = np.std(dq2dq2ij[i][j][th])
                        dQ1dQ1total[i][j][th] = len(dq1dq1ij[i][j][th])
                        dQ2dQ2total[i][j][th] = len(dq2dq2ij[i][j][th])

            _df.append(
                {
                    "Filename": filename,
                    "dQ1dQ1Correlation": dQ1dQ1Correlation,
                    "dQ2dQ2Correlation": dQ2dQ2Correlation,
                    "dQ1dQ1Correlation_std": dQ1dQ1Correlation_std,
                    "dQ2dQ2Correlation_std": dQ2dQ2Correlation_std,
                    "dQ1dQ1Count": dQ1dQ1total,
                    "dQ2dQ2Count": dQ2dQ2total,
                }
            )
            dfCorrelation = pd.DataFrame(_df)
            dfCorrelation.to_pickle(
                f"databases/postWoundPaperCorrelations/dfCorFarWound{filename}.pkl"
            )

# collect all correlations
if False:
    _df = []
    for filename in filenames:
        print(filename)

        dfCorRhoClose = pd.read_pickle(
            f"databases/postWoundPaperCorrelations/dfCorRhoClose{filename}.pkl"
        )
        dfCorRhoFar = pd.read_pickle(
            f"databases/postWoundPaperCorrelations/dfCorRhoFar{filename}.pkl"
        )
        dfCorCloseWound = pd.read_pickle(
            f"databases/postWoundPaperCorrelations/dfCorCloseWound{filename}.pkl"
        )
        dfCorFarWound = pd.read_pickle(
            f"databases/postWoundPaperCorrelations/dfCorFarWound{filename}.pkl"
        )

        dRhodRhoClose = np.nan_to_num(dfCorRhoClose["dRhodRhoCorrelation"].iloc[0])
        dRhodRhoClose_std = np.nan_to_num(
            dfCorRhoClose["dRhodRhoCorrelation_std"].iloc[0]
        )
        dRhodRhoClosetotal = np.nan_to_num(dfCorRhoClose["Count"].iloc[0])
        if np.sum(dRhodRhoClosetotal) == 0:
            print("dRhodRhoClosetotal")

        dRhodRhoFar = np.nan_to_num(dfCorRhoFar["dRhodRhoCorrelation"].iloc[0])
        dRhodRhoFar_std = np.nan_to_num(dfCorRhoFar["dRhodRhoCorrelation_std"].iloc[0])
        dRhodRhoFartotal = np.nan_to_num(dfCorRhoFar["Count"].iloc[0])
        if np.sum(dRhodRhoFartotal) == 0:
            print("dRhodRhoFartotal")

        dQ1dQ1Close = np.nan_to_num(dfCorCloseWound["dQ1dQ1Correlation"].iloc[0])
        dQ1dQ1Close_std = np.nan_to_num(
            dfCorCloseWound["dQ1dQ1Correlation_std"].iloc[0]
        )
        dQ1dQ1Closetotal = np.nan_to_num(dfCorCloseWound["dQ1dQ1Count"].iloc[0])
        if np.sum(dQ1dQ1Closetotal) == 0:
            print("dQ1dQ1Closetotal")

        dQ2dQ2Close = np.nan_to_num(dfCorCloseWound["dQ2dQ2Correlation"].iloc[0])
        dQ2dQ2Close_std = np.nan_to_num(
            dfCorCloseWound["dQ2dQ2Correlation_std"].iloc[0]
        )
        dQ2dQ2Closetotal = np.nan_to_num(dfCorCloseWound["dQ2dQ2Count"].iloc[0])
        if np.sum(dQ2dQ2Closetotal) == 0:
            print("dQ2dQ2Closetotal")

        dQ1dQ1Far = np.nan_to_num(dfCorFarWound["dQ1dQ1Correlation"].iloc[0])
        dQ1dQ1Far_std = np.nan_to_num(dfCorFarWound["dQ1dQ1Correlation_std"].iloc[0])
        dQ1dQ1Fartotal = np.nan_to_num(dfCorFarWound["dQ1dQ1Count"].iloc[0])
        if np.sum(dQ1dQ1Fartotal) == 0:
            print("dQ1dQ1Fartotal")

        dQ2dQ2Far = np.nan_to_num(dfCorFarWound["dQ2dQ2Correlation"].iloc[0])
        dQ2dQ2Far_std = np.nan_to_num(dfCorFarWound["dQ2dQ2Correlation_std"].iloc[0])
        dQ2dQ2Fartotal = np.nan_to_num(dfCorFarWound["dQ2dQ2Count"].iloc[0])
        if np.sum(dQ2dQ2Fartotal) == 0:
            print("dQ2dQ2Fartotal")

        _df.append(
            {
                "Filename": filename,
                "dRhodRhoClose": dRhodRhoClose,
                "dRhodRhoClose_std": dRhodRhoClose_std,
                "dRhodRhoClosetotal": dRhodRhoClosetotal,
                "dRhodRhoFar": dRhodRhoFar,
                "dRhodRhoFar_std": dRhodRhoFar_std,
                "dRhodRhoFartotal": dRhodRhoFartotal,
                "dQ1dQ1Close": dQ1dQ1Close,
                "dQ1dQ1Close_std": dQ1dQ1Close_std,
                "dQ1dQ1Closetotal": dQ1dQ1Closetotal,
                "dQ2dQ2Close": dQ2dQ2Close,
                "dQ2dQ2Close_std": dQ2dQ2Close_std,
                "dQ2dQ2Closetotal": dQ2dQ2Closetotal,
                "dQ1dQ1Far": dQ1dQ1Far,
                "dQ1dQ1Far_std": dQ1dQ1Far_std,
                "dQ1dQ1Fartotal": dQ1dQ1Fartotal,
                "dQ2dQ2Far": dQ2dQ2Far,
                "dQ2dQ2Far_std": dQ2dQ2Far_std,
                "dQ2dQ2Fartotal": dQ2dQ2Fartotal,
            }
        )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/postWoundPaperCorrelations/dfCorrelations{fileType}.pkl")


# --------- Long time unwounded ----------

grid = 26
timeGrid = 51
filenames, fileType = util.getFilesType("Unwound18h")

# display all correlations shape
if False:
    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    fig, ax = plt.subplots(3, 4, figsize=(16, 12))

    T, R, Theta = dfCor["dRho_SdRho_S"].iloc[0].shape

    dRhodRho = np.zeros([len(filenames), T, R])
    dQ1dRho = np.zeros([len(filenames), T, R])
    dQ2dRho = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):
        RhoCount = dfCor["Count Rho_S"].iloc[i][:, :, :-1]
        dRhodRho[i] = np.sum(
            dfCor["dRho_SdRho_S"].iloc[i][:, :, :-1] * RhoCount, axis=2
        ) / np.sum(RhoCount, axis=2)
        RhoQCount = dfCor["Count Rho_S Q"].iloc[i][:, :, :-1]
        dQ1dRho[i] = np.sum(
            dfCor["dQ1dRho_S"].iloc[i][:, :, :-1] * RhoQCount, axis=2
        ) / np.sum(RhoQCount, axis=2)
        dQ2dRho[i] = np.sum(
            dfCor["dQ2dRho_S"].iloc[i][:, :, :-1] * RhoQCount, axis=2
        ) / np.sum(RhoQCount, axis=2)

    dRhodRho = np.mean(dRhodRho, axis=0)
    dQ1dRho = np.mean(dQ1dRho, axis=0)
    dQ2dRho = np.mean(dQ2dRho, axis=0)

    maxCorr = np.max([dRhodRho, -dRhodRho])
    t, r = np.mgrid[0:180:10, 0:70:10]
    c = ax[0, 0].pcolor(
        t,
        r,
        dRhodRho,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[0, 0])
    cbar.formatter.set_powerlimits((0, 0))
    ax[0, 0].set_xlabel("Time apart $T$ (min)")
    ax[0, 0].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[0, 0].set_title(r"$C_{\rho\rho}(R,T) = C^{ss}_{\rho\rho}(R,T) + C^{ss}_{\rho\rho}(R,T)$", y=1.1)

    c = ax[0, 1].pcolor(
        t,
        r,
        dRhodRho - np.mean(dRhodRho[10:-1], axis=0),
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[0, 1])
    cbar.formatter.set_powerlimits((0, 0))
    ax[0, 1].set_xlabel("Time apart $T$ (min)")
    ax[0, 1].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[0, 1].set_title(r"$C^{nn}_{\rho\rho}(R,T)$", y=1.1)

    maxCorr = np.max([dQ1dRho, -dQ1dRho])
    c = ax[0, 2].pcolor(
        t,
        r,
        dQ1dRho,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[0, 2])
    cbar.formatter.set_powerlimits((0, 0))
    ax[0, 2].set_xlabel("Time apart $T$ (min)")
    ax[0, 2].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[0, 2].set_title(r"$C^{1}_{q\rho}(R,T)$", y=1.1)

    maxCorr = np.max([dQ2dRho, -dQ2dRho])
    c = ax[0, 3].pcolor(
        t,
        r,
        dQ2dRho,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[0, 3])
    cbar.formatter.set_powerlimits((0, 0))
    ax[0, 3].set_xlabel("Time apart $T$ (min)")
    ax[0, 3].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[0, 3].set_title(r"$C^{2}_{q\rho}(R,T)$", y=1.1)

    T, R, Theta = dfCor["dQ1dQ1Correlation"].iloc[0].shape

    dP1dP1 = np.zeros([len(filenames), T, R - 1])
    dP2dP2 = np.zeros([len(filenames), T, R - 1])
    dQ1dQ1 = np.zeros([len(filenames), T, R - 1])
    dQ2dQ2 = np.zeros([len(filenames), T, R - 1])
    dQ1dQ2 = np.zeros([len(filenames), T, R - 1])
    dP1dQ1 = np.zeros([len(filenames), T, R - 1])
    dP1dQ2 = np.zeros([len(filenames), T, R - 1])
    dP2dQ2 = np.zeros([len(filenames), T, R - 1])
    for i in range(len(filenames)):
        dP1dP1total = dfCor["dP1dP1Count"].iloc[i][:, :-1, :-1]
        dP1dP1[i] = np.sum(
            dfCor["dP1dP1Correlation"].iloc[i][:, :-1, :-1] * dP1dP1total, axis=2
        ) / np.sum(dP1dP1total, axis=2)

        dP2dP2total = dfCor["dP2dP2Count"].iloc[i][:, :-1, :-1]
        dP2dP2[i] = np.sum(
            dfCor["dP2dP2Correlation"].iloc[i][:, :-1, :-1] * dP2dP2total, axis=2
        ) / np.sum(dP2dP2total, axis=2)

        dQ1dQ1total = dfCor["dQ1dQ1Count"].iloc[i][:, :-1, :-1]
        dQ1dQ1[i] = np.sum(
            dfCor["dQ1dQ1Correlation"].iloc[i][:, :-1, :-1] * dQ1dQ1total, axis=2
        ) / np.sum(dQ1dQ1total, axis=2)

        dQ2dQ2total = dfCor["dQ2dQ2Count"].iloc[i][:, :-1, :-1]
        dQ2dQ2[i] = np.sum(
            dfCor["dQ2dQ2Correlation"].iloc[i][:, :-1, :-1] * dQ2dQ2total, axis=2
        ) / np.sum(dQ2dQ2total, axis=2)

        dQ1dQ2total = dfCor["dQ1dQ2Count"].iloc[i][:, :-1, :-1]
        dQ1dQ2[i] = np.sum(
            dfCor["dQ1dQ2Correlation"].iloc[i][:, :-1, :-1] * dQ1dQ2total, axis=2
        ) / np.sum(dQ1dQ2total, axis=2)

        dP1dQ1total = dfCor["dP1dQ1Count"].iloc[i][:, :-1, :-1]
        dP1dQ1[i] = np.sum(
            dfCor["dP1dQ1Correlation"].iloc[i][:, :-1, :-1] * dP1dQ1total, axis=2
        ) / np.sum(dP1dQ1total, axis=2)

        dP1dQ2total = dfCor["dP1dQ2Count"].iloc[i][:, :-1, :-1]
        dP1dQ2[i] = np.sum(
            dfCor["dP1dQ2Correlation"].iloc[i][:, :-1, :-1] * dP1dQ2total, axis=2
        ) / np.sum(dP1dQ2total, axis=2)

        dP2dQ2total = dfCor["dP2dQ2Count"].iloc[i][:, :-1, :-1]
        dP2dQ2[i] = np.sum(
            dfCor["dP2dQ2Correlation"].iloc[i][:, :-1, :-1] * dP2dQ2total, axis=2
        ) / np.sum(dP2dQ2total, axis=2)

    dP1dP1 = np.mean(dP1dP1, axis=0)
    dP2dP2 = np.mean(dP2dP2, axis=0)
    dQ1dQ1 = np.mean(dQ1dQ1, axis=0)
    dQ2dQ2 = np.mean(dQ2dQ2, axis=0)
    dQ1dQ2 = np.mean(dQ1dQ2, axis=0)
    dP1dQ1 = np.mean(dP1dQ1, axis=0)
    dP1dQ2 = np.mean(dP1dQ2, axis=0)
    dP2dQ2 = np.mean(dP2dQ2, axis=0)

    t, r = np.mgrid[0:102:2, 0:52:2]
    maxCorr = np.max([dP1dP1, -dP1dP1])
    c = ax[1, 0].pcolor(
        t,
        r,
        dP1dP1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[1, 0])
    cbar.formatter.set_powerlimits((0, 0))
    ax[1, 0].set_xlabel("Time apart $T$ (min)")
    ax[1, 0].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[1, 0].set_title(r"$C^{11}_{pp}(R,T)$", y=1.1)

    maxCorr = np.max([dP2dP2, -dP2dP2])
    c = ax[1, 1].pcolor(
        t,
        r,
        dP2dP2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[1, 1])
    cbar.formatter.set_powerlimits((0, 0))
    ax[1, 1].set_xlabel("Time apart $T$ (min)")
    ax[1, 1].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[1, 1].set_title(r"$C^{22}_{pp}(R,T)$", y=1.1)

    maxCorr = np.max([dQ1dQ1, -dQ1dQ1])
    c = ax[1, 2].pcolor(
        t,
        r,
        dQ1dQ1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[1, 2])
    cbar.formatter.set_powerlimits((0, 0))
    ax[1, 2].set_xlabel("Time apart $T$ (min)")
    ax[1, 2].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[1, 2].set_title(r"$C^{11}_{qq}(R,T)$", y=1.1)

    maxCorr = np.max([dQ2dQ2, -dQ2dQ2])
    c = ax[1, 3].pcolor(
        t,
        r,
        dQ2dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[1, 3])
    cbar.formatter.set_powerlimits((0, 0))
    ax[1, 3].set_xlabel("Time apart $T$ (min)")
    ax[1, 3].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[1, 3].set_title(r"$C^{22}_{qq}(R,T)$", y=1.1)

    maxCorr = np.max([dQ1dQ2, -dQ1dQ2])
    c = ax[2, 0].pcolor(
        t,
        r,
        dQ1dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[2, 0])
    cbar.formatter.set_powerlimits((0, 0))
    ax[2, 0].set_xlabel("Time apart $T$ (min)")
    ax[2, 0].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[2, 0].set_title(r"$C^{12}_{qq}(R,T)$", y=1.1)

    maxCorr = np.max([dP1dQ1, -dP1dQ1])
    c = ax[2, 1].pcolor(
        t,
        r,
        dP1dQ1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[2, 1])
    cbar.formatter.set_powerlimits((0, 0))
    ax[2, 1].set_xlabel("Time apart $T$ (min)")
    ax[2, 1].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[2, 1].set_title(r"$C^{11}_{pq}(R,T)$", y=1.1)

    maxCorr = np.max([dP1dQ2, -dP1dQ2])
    c = ax[2, 2].pcolor(
        t,
        r,
        dP1dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[2, 2])
    cbar.formatter.set_powerlimits((0, 0))
    ax[2, 2].set_xlabel("Time apart $T$ (min)")
    ax[2, 2].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[2, 2].set_title(r"$C^{12}_{pq}(R,T)$", y=1.1)

    maxCorr = np.max([dP2dQ2, -dP2dQ2])
    c = ax[2, 3].pcolor(
        t,
        r,
        dP2dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[2, 3])
    cbar.formatter.set_powerlimits((0, 0))
    ax[2, 3].set_xlabel("Time apart $T$ (min)")
    ax[2, 3].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[2, 3].set_title(r"$C^{22}_{pq}(R,T)$", y=1.1)

    # plt.subplot_tool()
    plt.subplots_adjust(
        left=0.08, bottom=0.1, right=0.92, top=0.9, wspace=0.4, hspace=0.50
    )

    fig.savefig(
        f"results/mathPostWoundPaper/Correlations {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# display all norm correlations shape
if False:
    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")
    df = pd.read_pickle(f"databases/dfShape{fileType}.pkl")

    fig, ax = plt.subplots(3, 4, figsize=(16, 12))

    T, R, Theta = dfCor["dRho_SdRho_S"].iloc[0].shape

    dRhodRho = np.zeros([len(filenames), T, R])
    dQ1dRho = np.zeros([len(filenames), T, R])
    dQ2dRho = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):
        RhoCount = dfCor["Count Rho_S"].iloc[i][:, :, :-1]
        dRhodRho[i] = np.sum(
            dfCor["dRho_SdRho_S"].iloc[i][:, :, :-1] * RhoCount, axis=2
        ) / np.sum(RhoCount, axis=2)
        RhoQCount = dfCor["Count Rho_S Q"].iloc[i][:, :, :-1]
        dQ1dRho[i] = np.sum(
            dfCor["dQ1dRho_S"].iloc[i][:, :, :-1] * RhoQCount, axis=2
        ) / np.sum(RhoQCount, axis=2)
        dQ2dRho[i] = np.sum(
            dfCor["dQ2dRho_S"].iloc[i][:, :, :-1] * RhoQCount, axis=2
        ) / np.sum(RhoQCount, axis=2)

    std_dq = np.std(np.stack(np.array(df.loc[:, "dq"]), axis=0), axis=0)

    dRhodRho = np.mean(dRhodRho, axis=0)
    std_rho = dRhodRho[0, 0] ** 0.5
    dRhodRho = dRhodRho / dRhodRho[0, 0]
    dQ1dRho = np.mean(dQ1dRho, axis=0)
    dQ1dRho = dQ1dRho / (std_dq[0, 0] * std_rho)
    dQ2dRho = np.mean(dQ2dRho, axis=0)
    dQ2dRho = dQ2dRho / (std_dq[0, 1] * std_rho)

    maxCorr = np.max([1, -1])
    t, r = np.mgrid[0:180:10, 0:70:10]
    c = ax[0, 0].pcolor(
        t,
        r,
        dRhodRho,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[0, 0])
    cbar.formatter.set_powerlimits((0, 0))
    ax[0, 0].set_xlabel("Time apart $T$ (min)")
    ax[0, 0].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[0, 0].set_title(r"$C_{\rho\rho}(R,T) = C^{ss}_{\rho\rho}(R,T) + C^{ss}_{\rho\rho}(R,T)$", y=1.1)

    c = ax[0, 1].pcolor(
        t,
        r,
        dRhodRho - np.mean(dRhodRho[10:-1], axis=0),
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[0, 1])
    cbar.formatter.set_powerlimits((0, 0))
    ax[0, 1].set_xlabel("Time apart $T$ (min)")
    ax[0, 1].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[0, 1].set_title(r"$C^{nn}_{\rho\rho}(R,T)$", y=1.1)

    c = ax[0, 2].pcolor(
        t,
        r,
        dQ1dRho,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[0, 2])
    cbar.formatter.set_powerlimits((0, 0))
    ax[0, 2].set_xlabel("Time apart $T$ (min)")
    ax[0, 2].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[0, 2].set_title(r"$C^{1}_{q\rho}(R,T)$", y=1.1)

    c = ax[0, 3].pcolor(
        t,
        r,
        dQ2dRho,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[0, 3])
    cbar.formatter.set_powerlimits((0, 0))
    ax[0, 3].set_xlabel("Time apart $T$ (min)")
    ax[0, 3].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[0, 3].set_title(r"$C^{1}_{q\rho}(R,T)$", y=1.1)
    print(np.max([np.max(dQ2dRho), np.max(-dQ2dRho)]), "dQ2dRho")

    T, R, Theta = dfCor["dQ1dQ1Correlation"].iloc[0].shape

    dP1dP1 = np.zeros([len(filenames), T, R - 1])
    dP2dP2 = np.zeros([len(filenames), T, R - 1])
    dQ1dQ1 = np.zeros([len(filenames), T, R - 1])
    dQ2dQ2 = np.zeros([len(filenames), T, R - 1])
    dQ1dQ2 = np.zeros([len(filenames), T, R - 1])
    dP1dQ1 = np.zeros([len(filenames), T, R - 1])
    dP1dQ2 = np.zeros([len(filenames), T, R - 1])
    dP2dQ2 = np.zeros([len(filenames), T, R - 1])
    for i in range(len(filenames)):
        dP1dP1total = dfCor["dP1dP1Count"].iloc[i][:, :-1, :-1]
        dP1dP1[i] = np.sum(
            dfCor["dP1dP1Correlation"].iloc[i][:, :-1, :-1] * dP1dP1total, axis=2
        ) / np.sum(dP1dP1total, axis=2)

        dP2dP2total = dfCor["dP2dP2Count"].iloc[i][:, :-1, :-1]
        dP2dP2[i] = np.sum(
            dfCor["dP2dP2Correlation"].iloc[i][:, :-1, :-1] * dP2dP2total, axis=2
        ) / np.sum(dP2dP2total, axis=2)

        dQ1dQ1total = dfCor["dQ1dQ1Count"].iloc[i][:, :-1, :-1]
        dQ1dQ1[i] = np.sum(
            dfCor["dQ1dQ1Correlation"].iloc[i][:, :-1, :-1] * dQ1dQ1total, axis=2
        ) / np.sum(dQ1dQ1total, axis=2)

        dQ2dQ2total = dfCor["dQ2dQ2Count"].iloc[i][:, :-1, :-1]
        dQ2dQ2[i] = np.sum(
            dfCor["dQ2dQ2Correlation"].iloc[i][:, :-1, :-1] * dQ2dQ2total, axis=2
        ) / np.sum(dQ2dQ2total, axis=2)

        dQ1dQ2total = dfCor["dQ1dQ2Count"].iloc[i][:, :-1, :-1]
        dQ1dQ2[i] = np.sum(
            dfCor["dQ1dQ2Correlation"].iloc[i][:, :-1, :-1] * dQ1dQ2total, axis=2
        ) / np.sum(dQ1dQ2total, axis=2)

        dP1dQ1total = dfCor["dP1dQ1Count"].iloc[i][:, :-1, :-1]
        dP1dQ1[i] = np.sum(
            dfCor["dP1dQ1Correlation"].iloc[i][:, :-1, :-1] * dP1dQ1total, axis=2
        ) / np.sum(dP1dQ1total, axis=2)

        dP1dQ2total = dfCor["dP1dQ2Count"].iloc[i][:, :-1, :-1]
        dP1dQ2[i] = np.sum(
            dfCor["dP1dQ2Correlation"].iloc[i][:, :-1, :-1] * dP1dQ2total, axis=2
        ) / np.sum(dP1dQ2total, axis=2)

        dP2dQ2total = dfCor["dP2dQ2Count"].iloc[i][:, :-1, :-1]
        dP2dQ2[i] = np.sum(
            dfCor["dP2dQ2Correlation"].iloc[i][:, :-1, :-1] * dP2dQ2total, axis=2
        ) / np.sum(dP2dQ2total, axis=2)

    dP1dP1 = np.mean(dP1dP1, axis=0)
    dP2dP2 = np.mean(dP2dP2, axis=0)
    dQ1dQ1 = np.mean(dQ1dQ1, axis=0)
    dQ2dQ2 = np.mean(dQ2dQ2, axis=0)
    dQ1dQ2 = np.mean(dQ1dQ2, axis=0)
    dP1dQ1 = np.mean(dP1dQ1, axis=0)
    dP1dQ2 = np.mean(dP1dQ2, axis=0)
    dP2dQ2 = np.mean(dP2dQ2, axis=0)

    std_dp = np.std(np.stack(np.array(df.loc[:, "dp"]), axis=0), axis=0)
    dP1dP1 = dP1dP1 / (std_dp[0] * std_dp[0])
    dP2dP2 = dP2dP2 / (std_dp[1] * std_dp[1])
    dQ1dQ1 = dQ1dQ1 / (std_dq[0, 0] * std_dq[0, 0])
    dQ2dQ2 = dQ2dQ2 / (std_dq[0, 1] * std_dq[0, 1])
    dQ1dQ2 = dQ1dQ2 / (std_dq[0, 0] * std_dq[0, 1])
    dP1dQ1 = dP1dQ1 / (std_dp[0] * std_dq[0, 0])
    dP1dQ2 = dP1dQ2 / (std_dp[0] * std_dq[0, 1])
    dP2dQ2 = dP2dQ2 / (std_dp[1] * std_dq[0, 1])

    t, r = np.mgrid[0:102:2, 0:52:2]
    c = ax[1, 0].pcolor(
        t,
        r,
        dP1dP1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[1, 0])
    cbar.formatter.set_powerlimits((0, 0))
    ax[1, 0].set_xlabel("Time apart $T$ (min)")
    ax[1, 0].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[1, 0].set_title(r"$C^{11}_{pp}(R,T)$", y=1.1)

    c = ax[1, 1].pcolor(
        t,
        r,
        dP2dP2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[1, 1])
    cbar.formatter.set_powerlimits((0, 0))
    ax[1, 1].set_xlabel("Time apart $T$ (min)")
    ax[1, 1].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[1, 1].set_title(r"$C^{22}_{pp}(R,T)$", y=1.1)

    c = ax[1, 2].pcolor(
        t,
        r,
        dQ1dQ1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[1, 2])
    cbar.formatter.set_powerlimits((0, 0))
    ax[1, 2].set_xlabel("Time apart $T$ (min)")
    ax[1, 2].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[1, 2].set_title(r"$C^{11}_{qq}(R,T)$", y=1.1)

    c = ax[1, 3].pcolor(
        t,
        r,
        dQ2dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[1, 3])
    cbar.formatter.set_powerlimits((0, 0))
    ax[1, 3].set_xlabel("Time apart $T$ (min)")
    ax[1, 3].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[1, 3].set_title(r"$C^{22}_{qq}(R,T)$", y=1.1)

    c = ax[2, 0].pcolor(
        t,
        r,
        dQ1dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[2, 0])
    cbar.formatter.set_powerlimits((0, 0))
    ax[2, 0].set_xlabel("Time apart $T$ (min)")
    ax[2, 0].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[2, 0].set_title(r"$C^{12}_{qq}(R,T)$", y=1.1)

    c = ax[2, 1].pcolor(
        t,
        r,
        dP1dQ1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[2, 1])
    cbar.formatter.set_powerlimits((0, 0))
    ax[2, 1].set_xlabel("Time apart $T$ (min)")
    ax[2, 1].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[2, 1].set_title(r"$C^{11}_{pq}(R,T)$", y=1.1)
    print(np.max([np.max(dP1dQ1), np.max(-dP1dQ1)]))

    c = ax[2, 2].pcolor(
        t,
        r,
        dP1dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[2, 2])
    cbar.formatter.set_powerlimits((0, 0))
    ax[2, 2].set_xlabel("Time apart $T$ (min)")
    ax[2, 2].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[2, 2].set_title(r"$C^{12}_{pq}(R,T)$", y=1.1)
    print(np.max([np.max(dP1dQ2), np.max(-dP1dQ2)]), "dP1dQ2")

    c = ax[2, 3].pcolor(
        t,
        r,
        dP2dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[2, 3])
    cbar.formatter.set_powerlimits((0, 0))
    ax[2, 3].set_xlabel("Time apart $T$ (min)")
    ax[2, 3].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[2, 3].set_title(r"$C^{22}_{pq}(R,T)$", y=1.1)
    print(np.max([np.max(dP2dQ2), np.max(-dP2dQ2)]), "dP2dQ2")

    # plt.subplot_tool()
    plt.subplots_adjust(
        left=0.08, bottom=0.1, right=0.92, top=0.9, wspace=0.4, hspace=0.50
    )

    fig.savefig(
        f"results/mathPostWoundPaper/Correlations Norm {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# deltaQ1 (model)
if True:

    def Corr_dQ1_Integral_T(R, B, L):
        mQ = 0.0005
        T = 0
        k = np.linspace(0, 20, 100000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, mQ, L) * h, axis=0)[:, 0]

    def Corr_dQ1_Integral_R(T, B, L):
        mQ = 0.0005
        R = 10
        k = np.linspace(0, 20, 100000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, mQ, L) * h, axis=0)[0]

    def Corr_dQ1(R, T):
        mQ = 0.0005
        B = 0.0038
        L = 4.8

        k = np.linspace(0, 20, 100000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, mQ, L) * h, axis=0)

    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    T, R, Theta = dfCor["dQ1dQ1Correlation"].iloc[0][:, :-1, :-1].shape

    dQ1dQ1 = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):

        dQ1dQ1total = dfCor["dQ1dQ1Count"].iloc[i][:, :-1, :-1]
        dQ1dQ1[i] = np.sum(
            dfCor["dQ1dQ1Correlation"].iloc[i][:, :-1, :-1] * dQ1dQ1total, axis=2
        ) / np.sum(dQ1dQ1total, axis=2)

    dfCor = 0

    dQ1dQ1 = np.mean(dQ1dQ1, axis=0)

    T = np.linspace(0, 2 * (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    m = sp.optimize.curve_fit(
        f=Corr_dQ1_Integral_R,
        xdata=T[1:],
        ydata=dQ1dQ1[:, 5][1:],
        p0=(0.001, 1),
        bounds=([0, 0], [np.inf, np.inf]),
    )[0]
    print(float(m[0]), float(m[1]))

    ax[0, 0].plot(T[1:], dQ1dQ1[:, 5][1:], label="Data")
    ax[0, 0].plot(T[1:], Corr_dQ1(5, T[1:])[0], label="Model")
    ax[0, 0].plot(T[1:], Corr_dQ1_Integral_R(T[1:], m[0], m[1]), label="Fit")
    ax[0, 0].set_xlabel("Time apart $T$ (min)")
    ax[0, 0].set_ylabel(r"$\delta q^{(1)}$ Correlation")
    ax[0, 0].set_ylim([0, 1.8e-04])
    ax[0, 0].set_title(r"$C^{11}_{qq}(10,T)$")
    ax[0, 0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax[0, 0].legend()

    m = sp.optimize.curve_fit(
        f=Corr_dQ1_Integral_T,
        xdata=R[5:],
        ydata=dQ1dQ1[0][5:26],
        p0=(0.001, 1),
        method="lm",
    )[0]
    print(float(m[0]), float(m[1]))

    ax[0, 1].plot(R[5:], dQ1dQ1[0][5:26], label="Data")
    ax[0, 1].plot(R[5:], Corr_dQ1(R[5:], 0), label="Model")
    ax[0, 1].plot(R[5:], Corr_dQ1_Integral_T(R, m[0], m[1])[5:], label="Fit")
    ax[0, 1].set_xlabel(r"Distance apart $R$ $(\mu m)$")
    ax[0, 1].set_ylabel(r"$\delta q^{(1)}$ Correlation")
    ax[0, 1].set_ylim([0, 1.8e-04])
    ax[0, 1].set_title(r"$C^{11}_{qq}(R,0)$")
    ax[0, 1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax[0, 1].legend()

    R, T = np.meshgrid(R, T)
    maxCorr = np.max([dQ1dQ1[:, 5:], -dQ1dQ1[:, 5:]])
    c = ax[1, 0].pcolor(
        T[:, 5:],
        R[:, 5:],
        dQ1dQ1[:, 5:],
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[1, 0])
    cbar.formatter.set_powerlimits((0, 0))
    ax[1, 0].set_xlabel("Time apart $T$ (min)")
    ax[1, 0].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[1, 0].set_title(r"Experiment $C^{11}_{qq}(R,T)$", y=1.1)

    T = np.linspace(0, 2 * (timeGrid - 1), timeGrid * 2)
    R = np.linspace(10, 2 * (grid - 1), grid * 2)
    Corr_dQ1dQ1 = np.swapaxes(Corr_dQ1(R, T), 0, 1)
    R, T = np.meshgrid(R, T)

    c = ax[1, 1].pcolor(
        T,
        R,
        Corr_dQ1dQ1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto"
        
    )
    cbar = fig.colorbar(c, ax=ax[1, 1])
    cbar.formatter.set_powerlimits((0, 0))
    ax[1, 1].set_xlabel("Time apart $T$ (min)")
    ax[1, 1].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[1, 1].set_title(r"Model $C^{11}_{qq}(R,T)$", y=1.1)

    plt.subplots_adjust(
        left=0.08, bottom=0.1, right=0.92, top=0.9, wspace=0.35, hspace=0.50
    )
    fig.savefig(
        f"results/mathPostWoundPaper/Correlation dQ1 in T and R {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# deltaQ2 (model)
if True:

    def Corr_dQ2_Integral_T(R, B, L):
        mQ = 0.0002
        T = 0
        k = np.linspace(0, 20, 100000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, mQ, L) * h, axis=0)[:, 0]

    def Corr_dQ2_Integral_R(T, B, L):
        mQ = 0.0002
        R = 10
        k = np.linspace(0, 20, 100000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, mQ, L) * h, axis=0)[0]

    def Corr_dQ2(R, T):
        mQ = 0.0002
        B = 0.0038
        L = 4.8

        k = np.linspace(0, 20, 100000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, mQ, L) * h, axis=0)

    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    T, R, Theta = dfCor["dQ2dQ2Correlation"].iloc[0][:, :-1, :-1].shape

    dQ2dQ2 = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):

        dQ2dQ2total = dfCor["dQ2dQ2Count"].iloc[i][:, :-1, :-1]
        dQ2dQ2[i] = np.sum(
            dfCor["dQ2dQ2Correlation"].iloc[i][:, :-1, :-1] * dQ2dQ2total, axis=2
        ) / np.sum(dQ2dQ2total, axis=2)

    dfCor = 0

    dQ2dQ2 = np.mean(dQ2dQ2, axis=0)

    T = np.linspace(0, 2 * (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    plt.subplots_adjust(wspace=0.4)
    plt.gcf().subplots_adjust(bottom=0.15)

    m = sp.optimize.curve_fit(
        f=Corr_dQ2_Integral_R,
        xdata=T[1:],
        ydata=dQ2dQ2[:, 5][1:],
        p0=(0.001, 1),
    )[0]
    print(float(m[0]), float(m[1]))

    ax[0, 0].plot(T[1:], dQ2dQ2[:, 5][1:], label="Data")
    ax[0, 0].plot(T[1:], Corr_dQ2(5, T[1:])[0], label="Model")
    ax[0, 0].plot(T[1:], Corr_dQ2_Integral_R(T[1:], m[0], m[1]), label="Fit")
    ax[0, 0].set_xlabel("Time apart $T$ (min)")
    ax[0, 0].set_ylabel(r"$\delta q^{(2)}$ Correlation")
    ax[0, 0].set_ylim([0, 8e-05])
    ax[0, 0].set_title(r"$C^{22}_{qq}(10,T)$")
    ax[0, 0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax[0, 0].legend()

    m = sp.optimize.curve_fit(
        f=Corr_dQ2_Integral_T,
        xdata=R[5:],
        ydata=dQ2dQ2[1][5:26],
        p0=(0.001, 1),
        method="lm",
    )[0]
    print(float(m[0]), float(m[1]))

    ax[0, 1].plot(R[5:], dQ2dQ2[0][5:26], label="Data")
    ax[0, 1].plot(R[5:], Corr_dQ2(R[5:], 0), label="Model")
    ax[0, 1].plot(R[5:], Corr_dQ2_Integral_T(R, m[0], m[1])[5:], label="Fit")
    ax[0, 1].set_xlabel(r"Distance apart $R$ $(\mu m)$")
    ax[0, 1].set_ylabel(r"$\delta q^{(2)}$ Correlation")
    ax[0, 1].set_ylim([0, 8e-05])
    ax[0, 1].set_title(r"$C^{22}_{qq}(R,0)$")
    ax[0, 1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax[0, 1].legend()

    R, T = np.meshgrid(R, T)
    maxCorr = np.max([dQ2dQ2[:, 5:], -dQ2dQ2[:, 5:]])
    c = ax[1, 0].pcolor(
        T[:, 5:],
        R[:, 5:],
        dQ2dQ2[:, 5:],
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[1, 0])
    cbar.formatter.set_powerlimits((0, 0))
    ax[1, 0].set_xlabel("Time apart $T$ (min)")
    ax[1, 0].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[1, 0].set_title(r"Experiment $C^{22}_{qq}(R,T)$", y=1.1)

    T = np.linspace(0, 2 * (timeGrid - 1), timeGrid * 2)
    R = np.linspace(10, 2 * (grid - 1), grid * 2)
    Corr_dQ2dQ2 = np.swapaxes(Corr_dQ2(R, T), 0, 1)
    R, T = np.meshgrid(R, T)

    c = ax[1, 1].pcolor(
        T,
        R,
        Corr_dQ2dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[1, 1])
    cbar.formatter.set_powerlimits((0, 0))
    ax[1, 1].set_xlabel("Time apart $T$ (min)")
    ax[1, 1].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[1, 1].set_title(r"Model $C^{22}_{qq}(R,T)$", y=1.1)

    plt.subplots_adjust(
        left=0.08, bottom=0.1, right=0.92, top=0.9, wspace=0.35, hspace=0.50
    )

    fig.savefig(
        f"results/mathPostWoundPaper/Correlation dQ2 in T and R {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# deltaRho_n (model)
if False:

    def Corr_Rho_fit(M, mrho, D):
        R, T = M
        return mrho / (32 * np.pi**3 * T * D**2) * np.exp(-(R**2) / (4 * D * T))
    
    def Corr_Rho(R, T, mrho, D):
        return mrho / (32 * np.pi**3 * T * D**2) * np.exp(-(R**2) / (4 * D * T))

    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    T, R, Theta = dfCor["dRho_SdRho_S"].iloc[0].shape

    dRho_SdRho_S = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):
        Rho_SCount = dfCor["Count Rho_S"].iloc[i][:, :, :-1]
        dRho_SdRho_S[i] = np.sum(
            dfCor["dRho_SdRho_S"].iloc[i][:, :, :-1] * Rho_SCount, axis=2
        ) / np.sum(Rho_SCount, axis=2)

    dfCor = 0

    dRho_SdRho_S = np.mean(dRho_SdRho_S, axis=0)

    dRhodRho = dRho_SdRho_S - np.mean(dRho_SdRho_S[10:-1], axis=0)

    R = np.linspace(10, 60, 6)
    R_ = np.linspace(10, 60, 501)
    T = np.linspace(10, 170, 17)
    T_ = np.linspace(10, 170, 511)

    R, T = np.meshgrid(R, T)
    xdata = np.vstack((R.ravel(), T.ravel()))

    popt = sp.optimize.curve_fit(Corr_Rho_fit, xdata, dRhodRho[1:, 1:].ravel(), [0.003, 4])[0]

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    maxCorr = np.max([dRhodRho[1:, 1:], -dRhodRho[1:, 1:]])
    c = ax[0].pcolor(
        T,
        R,
        dRhodRho[1:, 1:],
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[0])
    cbar.formatter.set_powerlimits((0, 0))
    ax[0].set_xlabel("Time apart $T$ (min)")
    ax[0].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[0].set_title(r"Experiment $C^{nn}_{\rho\rho}(R,T)$", y=1.1)

    R_, T_ = np.meshgrid(R_, T_)
    c = ax[1].pcolor(
        T_,
        R_,
        Corr_Rho(R_, T_, *popt),
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    cbar = fig.colorbar(c, ax=ax[1])
    cbar.formatter.set_powerlimits((0, 0))
    ax[1].set_xlabel("Time apart $T$ (min)")
    ax[1].set_ylabel(r"Distance apart $R$ $(\mu m)$")
    ax[1].set_title(r"Model $C^{nn}_{\rho\rho}(R,T)$", y=1.1)

    print(popt)

    plt.subplots_adjust(
        left=0.08, bottom=0.1, right=0.92, top=0.9, wspace=0.4, hspace=0.50
    )

    fig.savefig(
        f"results/mathPostWoundPaper/Correlation dRho_n in T and R model",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# deltaRho_s (model)
if False:

    def Corr_dRho_S(r, C, lamdba):
        return C * np.exp(- r/lamdba)

    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    T, R, Theta = dfCor["dRho_SdRho_S"].iloc[0].shape

    dRho_SdRho_S = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):
        Rho_SCount = dfCor["Count Rho_S"].iloc[i][:, :, :-1]
        dRho_SdRho_S[i] = np.sum(
            dfCor["dRho_SdRho_S"].iloc[i][:, :, :-1] * Rho_SCount, axis=2
        ) / np.sum(Rho_SCount, axis=2)

    dfCor = 0

    dRho_SdRho_S = np.mean(dRho_SdRho_S, axis=0)

    dRho_SdRho_S = np.mean(dRho_SdRho_S[10:-1], axis=0)

    R = np.linspace(0, 60, 7)
    R_ = np.linspace(0, 60, 61)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    m = sp.optimize.curve_fit(
        f=Corr_dRho_S,
        xdata=R,
        ydata=dRho_SdRho_S,
        p0=(0.0003, 20),
    )[0]
    print(m)

    ax.plot(R, dRho_SdRho_S, label="Data")
    ax.plot(R_, Corr_dRho_S(R_, m[0], m[1]), label="Model")
    ax.set_xlabel(r"Distance apart $R$ $(\mu m)$")
    ax.set_ylabel(r"$\delta \rho_s$ Correlation")
    ax.set_ylim([0, 4.5e-04])
    ax.set_title(r"$C^{ss}_{\rho\rho}(R)$")
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.legend()

    plt.subplots_adjust(
        left=0.08, bottom=0.1, right=0.92, top=0.9, wspace=0.35, hspace=0.50
    )
    fig.savefig(
        f"results/mathPostWoundPaper/Correlation rho_s in T and R {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

