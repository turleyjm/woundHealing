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

plt.rcParams.update({"font.size": 8})


# -------------------

filenames, fileType = util.getFilesType()
T = 90
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


# --------- density ----------

# space time cell density correlation close to wound
if False:
    grid = 7
    timeGrid = 18
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
        heatmapdrho = np.zeros([90, xGrid, yGrid])
        inPlaneEcad = np.zeros([90, xGrid, yGrid])
        inNearWound = np.zeros([90, xGrid, yGrid])

        for t in range(90):

            dft = df[(df["T"] == 2 * t) | (df["T"] == 2 * t + 1)]
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
        df.to_pickle(f"databases/correlationsWound/dfCorRhoClose{filename}.pkl")

# space time cell density correlation far from to wound
if False:
    grid = 7
    timeGrid = 18
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
        heatmapdrho = np.zeros([90, xGrid, yGrid])
        inPlaneEcad = np.zeros([90, xGrid, yGrid])
        inFarWound = np.zeros([90, xGrid, yGrid])

        for t in range(90):

            dft = df[(df["T"] == 2 * t) | (df["T"] == 2 * t + 1)]
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
                            if (np.min(dfg["R"]) > 30) & (t >= 45):
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
        df.to_pickle(f"databases/correlationsWound/dfCorRhoFar{filename}.pkl")

# --------- shape ----------

# space time cell-cell shape correlation close to wound
if False:
    dfShape = pd.read_pickle(f"databases/dfShapeWound{fileType}.pkl")
    grid = 27
    timeGrid = 51

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
        path_to_file = f"databases/correlationsWound/dfCorCloseWound{filename}.pkl"
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
                np.array(dfShapeF["T"] < 60) & np.array(dfShapeF["R"] < 30)
            ]
            n = int(len(dfClose) / 2)
            random.seed(10)
            count = 0
            Is = []
            for i0 in range(n):
                i = int(random.random() * n)
                while i in Is:
                    i = int(random.random() * n)
                Is.append(i)
                if i0 % int((n) / 10) == 0:
                    print(datetime.now().strftime("%H:%M:%S ") + f"{10*count}%")
                    count += 1

                x = dfClose["X"].iloc[i]
                y = dfClose["Y"].iloc[i]
                t = dfClose["T"].iloc[i]
                dp1 = dfClose["dp"].iloc[i][0]
                dp2 = dfClose["dp"].iloc[i][1]
                dq1 = dfClose["dq"].iloc[i][0, 0]
                dq2 = dfClose["dq"].iloc[i][0, 1]
                dfShapeF.loc[:, "dR"] = (
                    ((dfShapeF.loc[:, "X"] - x) ** 2 + (dfShapeF.loc[:, "Y"] - y) ** 2)
                    ** 0.5
                ).copy()
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
                        "dq1dq1i",
                        "dq2dq2i",
                    ]
                ]
                df = df[np.array(df["dR"] < R[-1]) & np.array(df["dR"] >= 0)]

                df["dT"] = (df.loc[:, "T"] - t) / 2
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
                f"databases/correlationsWound/dfCorCloseWound{filename}.pkl"
            )

# space time cell-cell shape correlation far from wound
if False:
    dfShape = pd.read_pickle(f"databases/dfShapeWound{fileType}.pkl")
    grid = 27
    timeGrid = 51

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
        path_to_file = f"databases/correlationsWound/dfCorFarWound{filename}.pkl"
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
                np.array(dfShapeF["T"] >= 90) & np.array(dfShapeF["R"] >= 30)
            ]
            n = int(len(dfFar) / 10)
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

                x = dfFar["X"].iloc[i]
                y = dfFar["Y"].iloc[i]
                t = dfFar["T"].iloc[i]
                dp1 = dfFar["dp"].iloc[i][0]
                dp2 = dfFar["dp"].iloc[i][1]
                dq1 = dfFar["dq"].iloc[i][0, 0]
                dq2 = dfFar["dq"].iloc[i][0, 1]
                dfShapeF.loc[:, "dR"] = (
                    ((dfShapeF.loc[:, "X"] - x) ** 2 + (dfShapeF.loc[:, "Y"] - y) ** 2)
                    ** 0.5
                ).copy()
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
                        "dq1dq1i",
                        "dq2dq2i",
                    ]
                ]
                df = df[np.array(df["dR"] < R[-1]) & np.array(df["dR"] >= 0)]

                df["dT"] = (df.loc[:, "T"] - t) / 2
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
                f"databases/correlationsWound/dfCorFarWound{filename}.pkl"
            )

# --------- velocity ----------

# space time velocity-velocity correlation Close to wound
if False:
    dfVelocity = pd.read_pickle(f"databases/dfVelocityWound{fileType}.pkl")
    grid = 27
    timeGrid = 51

    T = np.linspace(0, (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)
    theta = np.linspace(0, 2 * np.pi, 17)

    dfVelocity["dR"] = list(np.zeros([len(dfVelocity)]))
    dfVelocity["dT"] = list(np.zeros([len(dfVelocity)]))
    dfVelocity["dtheta"] = list(np.zeros([len(dfVelocity)]))

    dfVelocity["dv1dv1i"] = list(np.zeros([len(dfVelocity)]))
    dfVelocity["dv2dv2i"] = list(np.zeros([len(dfVelocity)]))

    for k in range(len(filenames)):
        filename = filenames[k]
        path_to_file = f"databases/correlationsWound/dfCorVelClose{filename}.pkl"
        if False == exists(path_to_file):
            _df = []
            dV1dV1Correlation = np.zeros([len(T), len(R), len(theta)])
            dV2dV2Correlation = np.zeros([len(T), len(R), len(theta)])
            dV1dV1Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dV2dV2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dV1dV1total = np.zeros([len(T), len(R), len(theta)])
            dV2dV2total = np.zeros([len(T), len(R), len(theta)])

            dv1dv1ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]  # t, r, theta
            dv2dv2ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]

            print(datetime.now().strftime("%H:%M:%S ") + filename)
            dfVelocityF = dfVelocity[dfVelocity["Filename"] == filename].copy()
            dfFar = dfVelocityF[
                np.array(dfVelocityF["T"] < 60) & np.array(dfVelocityF["R"] < 30)
            ]
            n = int(len(dfFar) / 2)
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

                x = dfFar["X"].iloc[i]
                y = dfFar["Y"].iloc[i]
                t = dfFar["T"].iloc[i]
                dv1 = dfFar["dv"].iloc[i][0]
                dv2 = dfFar["dv"].iloc[i][1]
                dfVelocityF.loc[:, "dR"] = (
                    (
                        (dfVelocityF.loc[:, "X"] - x) ** 2
                        + (dfVelocityF.loc[:, "Y"] - y) ** 2
                    )
                    ** 0.5
                ).copy()
                df = dfVelocityF[
                    [
                        "X",
                        "Y",
                        "T",
                        "dv",
                        "dR",
                        "dT",
                        "dtheta",
                        "dv1dv1i",
                        "dv2dv2i",
                    ]
                ]
                df = df[np.array(df["dR"] < R[-1]) & np.array(df["dR"] >= 0)]

                df["dT"] = (df.loc[:, "T"] - t) / 2
                df = df[np.array(df["dT"] < timeGrid) & np.array(df["dT"] >= 0)]
                if len(df) != 0:
                    theta = np.arctan2(df.loc[:, "Y"] - y, df.loc[:, "X"] - x)
                    df["dtheta"] = np.where(theta < 0, 2 * np.pi + theta, theta)
                    df["dv1dv1i"] = list(
                        dv1 * np.stack(np.array(df.loc[:, "dv"]), axis=0)[:, 0]
                    )
                    df["dv2dv2i"] = list(
                        dv2 * np.stack(np.array(df.loc[:, "dv"]), axis=0)[:, 1]
                    )

                    for j in range(len(df)):
                        dv1dv1ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dv1dv1i"].iloc[j])
                        dv2dv2ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dv2dv2i"].iloc[j])

            T = np.linspace(0, (timeGrid - 1), timeGrid)
            R = np.linspace(0, 2 * (grid - 1), grid)
            theta = np.linspace(0, 2 * np.pi, 17)
            for i in range(len(T)):
                for j in range(len(R)):
                    for th in range(len(theta)):
                        dV1dV1Correlation[i][j][th] = np.mean(dv1dv1ij[i][j][th])
                        dV2dV2Correlation[i][j][th] = np.mean(dv2dv2ij[i][j][th])

                        dV1dV1Correlation_std[i][j][th] = np.std(dv1dv1ij[i][j][th])
                        dV2dV2Correlation_std[i][j][th] = np.std(dv2dv2ij[i][j][th])

                        dV1dV1total[i][j][th] = len(dv1dv1ij[i][j][th])
                        dV2dV2total[i][j][th] = len(dv2dv2ij[i][j][th])

            _df.append(
                {
                    "Filename": filename,
                    "dV1dV1Correlation": dV1dV1Correlation,
                    "dV2dV2Correlation": dV2dV2Correlation,
                    "dV1dV1Correlation_std": dV1dV1Correlation_std,
                    "dV2dV2Correlation_std": dV2dV2Correlation_std,
                    "dV1dV1Count": dV1dV1total,
                    "dV2dV2Count": dV2dV2total,
                }
            )
            dfCorrelation = pd.DataFrame(_df)
            dfCorrelation.to_pickle(
                f"databases/correlationsWound/dfCorVelClose{filename}.pkl"
            )

# space time velocity-velocity correlation far from wound
if False:
    dfVelocity = pd.read_pickle(f"databases/dfVelocityWound{fileType}.pkl")
    grid = 27
    timeGrid = 51

    T = np.linspace(0, (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)
    theta = np.linspace(0, 2 * np.pi, 17)

    dfVelocity["dR"] = list(np.zeros([len(dfVelocity)]))
    dfVelocity["dT"] = list(np.zeros([len(dfVelocity)]))
    dfVelocity["dtheta"] = list(np.zeros([len(dfVelocity)]))

    dfVelocity["dv1dv1i"] = list(np.zeros([len(dfVelocity)]))
    dfVelocity["dv2dv2i"] = list(np.zeros([len(dfVelocity)]))

    for k in range(len(filenames)):
        filename = filenames[k]
        path_to_file = f"databases/correlationsWound/dfCorVelFar{filename}.pkl"
        if False == exists(path_to_file):
            _df = []
            dV1dV1Correlation = np.zeros([len(T), len(R), len(theta)])
            dV2dV2Correlation = np.zeros([len(T), len(R), len(theta)])
            dV1dV1Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dV2dV2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dV1dV1total = np.zeros([len(T), len(R), len(theta)])
            dV2dV2total = np.zeros([len(T), len(R), len(theta)])

            dv1dv1ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]  # t, r, theta
            dv2dv2ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]

            print(datetime.now().strftime("%H:%M:%S ") + filename)
            dfVelocityF = dfVelocity[dfVelocity["Filename"] == filename].copy()
            dfFar = dfVelocityF[
                np.array(dfVelocityF["T"] >= 90) & np.array(dfVelocityF["R"] >= 30)
            ]
            n = int(len(dfFar) / 10)
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

                x = dfFar["X"].iloc[i]
                y = dfFar["Y"].iloc[i]
                t = dfFar["T"].iloc[i]
                dv1 = dfFar["dv"].iloc[i][0]
                dv2 = dfFar["dv"].iloc[i][1]
                dfVelocityF.loc[:, "dR"] = (
                    (
                        (dfVelocityF.loc[:, "X"] - x) ** 2
                        + (dfVelocityF.loc[:, "Y"] - y) ** 2
                    )
                    ** 0.5
                ).copy()
                df = dfVelocityF[
                    [
                        "X",
                        "Y",
                        "T",
                        "dv",
                        "dR",
                        "dT",
                        "dtheta",
                        "dv1dv1i",
                        "dv2dv2i",
                    ]
                ]
                df = df[np.array(df["dR"] < R[-1]) & np.array(df["dR"] >= 0)]

                df["dT"] = (df.loc[:, "T"] - t) / 2
                df = df[np.array(df["dT"] < timeGrid) & np.array(df["dT"] >= 0)]
                if len(df) != 0:
                    theta = np.arctan2(df.loc[:, "Y"] - y, df.loc[:, "X"] - x)
                    df["dtheta"] = np.where(theta < 0, 2 * np.pi + theta, theta)
                    df["dv1dv1i"] = list(
                        dv1 * np.stack(np.array(df.loc[:, "dv"]), axis=0)[:, 0]
                    )
                    df["dv2dv2i"] = list(
                        dv2 * np.stack(np.array(df.loc[:, "dv"]), axis=0)[:, 1]
                    )

                    for j in range(len(df)):
                        dv1dv1ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dv1dv1i"].iloc[j])
                        dv2dv2ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dv2dv2i"].iloc[j])

            T = np.linspace(0, (timeGrid - 1), timeGrid)
            R = np.linspace(0, 2 * (grid - 1), grid)
            theta = np.linspace(0, 2 * np.pi, 17)
            for i in range(len(T)):
                for j in range(len(R)):
                    for th in range(len(theta)):
                        dV1dV1Correlation[i][j][th] = np.mean(dv1dv1ij[i][j][th])
                        dV2dV2Correlation[i][j][th] = np.mean(dv2dv2ij[i][j][th])

                        dV1dV1Correlation_std[i][j][th] = np.std(dv1dv1ij[i][j][th])
                        dV2dV2Correlation_std[i][j][th] = np.std(dv2dv2ij[i][j][th])

                        dV1dV1total[i][j][th] = len(dv1dv1ij[i][j][th])
                        dV2dV2total[i][j][th] = len(dv2dv2ij[i][j][th])

            _df.append(
                {
                    "Filename": filename,
                    "dV1dV1Correlation": dV1dV1Correlation,
                    "dV2dV2Correlation": dV2dV2Correlation,
                    "dV1dV1Correlation_std": dV1dV1Correlation_std,
                    "dV2dV2Correlation_std": dV2dV2Correlation_std,
                    "dV1dV1Count": dV1dV1total,
                    "dV2dV2Count": dV2dV2total,
                }
            )
            dfCorrelation = pd.DataFrame(_df)
            dfCorrelation.to_pickle(
                f"databases/correlationsWound/dfCorVelFar{filename}.pkl"
            )

# --------- collect all ----------

# collect all correlations
if True:
    _df = []
    for filename in filenames:

        dfCorRhoClose = pd.read_pickle(
            f"databases/correlationsWound/dfCorRhoClose{filename}.pkl"
        )
        dfCorRhoFar = pd.read_pickle(
            f"databases/correlationsWound/dfCorRhoFar{filename}.pkl"
        )
        dfCorCloseWound = pd.read_pickle(
            f"databases/correlationsWound/dfCorCloseWound{filename}.pkl"
        )
        dfCorFarWound = pd.read_pickle(
            f"databases/correlationsWound/dfCorFarWound{filename}.pkl"
        )
        dfCorVelClose = pd.read_pickle(
            f"databases/correlationsWound/dfCorVelClose{filename}.pkl"
        )
        dfCorVelFar = pd.read_pickle(
            f"databases/correlationsWound/dfCorVelFar{filename}.pkl"
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

        dV1dV1Close = np.nan_to_num(dfCorVelClose["dV1dV1Correlation"].iloc[0])
        dV1dV1Close_std = np.nan_to_num(dfCorVelClose["dV1dV1Correlation_std"].iloc[0])
        dV1dV1Closetotal = np.nan_to_num(dfCorVelClose["dV1dV1Count"].iloc[0])
        if np.sum(dV1dV1Closetotal) == 0:
            print("dV1dV1Closetotal")

        dV2dV2Close = np.nan_to_num(dfCorVelClose["dV2dV2Correlation"].iloc[0])
        dV2dV2Close_std = np.nan_to_num(dfCorVelClose["dV2dV2Correlation_std"].iloc[0])
        dV2dV2Closetotal = np.nan_to_num(dfCorVelClose["dV2dV2Count"].iloc[0])
        if np.sum(dV2dV2Closetotal) == 0:
            print("dV2dV2Closetotal")

        dV1dV1Far = np.nan_to_num(dfCorVelFar["dV1dV1Correlation"].iloc[0])
        dV1dV1Far_std = np.nan_to_num(dfCorVelFar["dV1dV1Correlation_std"].iloc[0])
        dV1dV1Fartotal = np.nan_to_num(dfCorVelFar["dV1dV1Count"].iloc[0])
        if np.sum(dV1dV1Fartotal) == 0:
            print("dV1dV1Fartotal")

        dV2dV2Far = np.nan_to_num(dfCorVelFar["dV2dV2Correlation"].iloc[0])
        dV2dV2Far_std = np.nan_to_num(dfCorVelFar["dV2dV2Correlation_std"].iloc[0])
        dV2dV2Fartotal = np.nan_to_num(dfCorVelFar["dV2dV2Count"].iloc[0])
        if np.sum(dV2dV2Fartotal) == 0:
            print("dV2dV2Fartotal")

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
                "dV1dV1Close": dV1dV1Close,
                "dV1dV1Close_std": dV1dV1Close_std,
                "dV1dV1Closetotal": dV1dV1Closetotal,
                "dV2dV2Close": dV2dV2Close,
                "dV2dV2Close_std": dV2dV2Close_std,
                "dV2dV2Closetotal": dV2dV2Closetotal,
                "dV1dV1Far": dV1dV1Far,
                "dV1dV1Far_std": dV1dV1Far_std,
                "dV1dV1Fartotal": dV1dV1Fartotal,
                "dV2dV2Far": dV2dV2Far,
                "dV2dV2Far_std": dV2dV2Far_std,
                "dV2dV2Fartotal": dV2dV2Fartotal,
            }
        )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/dfCorrelationWound{fileType}.pkl")
