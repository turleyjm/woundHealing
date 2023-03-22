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


# --------- divisions ----------

# correlation of division
if False:
    T = 160
    timeStep = 10
    R = 110
    rStep = 10

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
        dfDivision = pd.read_pickle(f"dat/{filename}/dfDivision{filename}.pkl")
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

        N.append(90 * 124**2 - np.sum(outPlane) / 255)

        x = np.array(dfDivision["X"]) * scale
        y = np.array(dfDivision["Y"]) * scale
        t = np.array(dfDivision["T"])
        ori = np.array(dfDivision["Orientation"])

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
                                        np.cos(2 * np.pi * ori[k] / 180),
                                        np.sin(2 * np.pi * ori[k] / 180),
                                    ]
                                )
                                for theta in thetas:
                                    u = np.array(
                                        [
                                            np.cos(2 * np.pi * theta / 180),
                                            np.sin(2 * np.pi * theta / 180),
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

# correlation of division-density
if False:
    T = 160
    timeStep = 10
    R = 110
    rStep = 10

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

        N.append(90 * 124**2 - np.sum(outPlane) / 255)

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

# --------- density ----------

# space time cell density correlation
if False:
    grid = 7
    timeGrid = 18
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
        df.to_pickle(f"databases/correlations/dfCorRho{filename}.pkl")

# space time cell density-shape correlation
if False:
    grid = 7
    timeGrid = 18
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

                            dfShapeF.loc[:, "dR"] = (
                                (
                                    (dfShapeF.loc[:, "X"] - x) ** 2
                                    + (dfShapeF.loc[:, "Y"] - y) ** 2
                                )
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
            df.to_pickle(f"databases/correlations/dfCorRhoQ{filename}.pkl")

# --------- shape ----------

# space time cell-cell shape correlation
if False:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    grid = 27
    timeGrid = 51

    T = np.linspace(0, (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)
    theta = np.linspace(0, 2 * np.pi, 17)

    dfShape["dR"] = list(np.zeros([len(dfShape)]))
    dfShape["dT"] = list(np.zeros([len(dfShape)]))
    dfShape["dtheta"] = list(np.zeros([len(dfShape)]))

    # dfShape["dp1dp1i"] = list(np.zeros([len(dfShape)]))
    # dfShape["dp2dp2i"] = list(np.zeros([len(dfShape)]))
    # dfShape["dq1dq1i"] = list(np.zeros([len(dfShape)]))
    # dfShape["dq2dq2i"] = list(np.zeros([len(dfShape)]))
    dfShape["dq1dq2i"] = list(np.zeros([len(dfShape)]))
    dfShape["dp1dq1i"] = list(np.zeros([len(dfShape)]))
    dfShape["dp1dq2i"] = list(np.zeros([len(dfShape)]))
    dfShape["dp2dq2i"] = list(np.zeros([len(dfShape)]))

    for k in range(len(filenames)):
        filename = filenames[k]
        path_to_file = f"databases/correlations/dfCorMidway{filename}_5-8.pkl"
        if False == exists(path_to_file):
            _df = []
            # dP1dP1Correlation = np.zeros([len(T), len(R), len(theta)])
            # dP2dP2Correlation = np.zeros([len(T), len(R), len(theta)])
            # dQ1dQ1Correlation = np.zeros([len(T), len(R), len(theta)])
            # dQ2dQ2Correlation = np.zeros([len(T), len(R), len(theta)])
            dQ1dQ2Correlation = np.zeros([len(T), len(R), len(theta)])
            dP1dQ1Correlation = np.zeros([len(T), len(R), len(theta)])
            dP1dQ2Correlation = np.zeros([len(T), len(R), len(theta)])
            dP2dQ2Correlation = np.zeros([len(T), len(R), len(theta)])
            # dP1dP1Correlation_std = np.zeros([len(T), len(R), len(theta)])
            # dP2dP2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            # dQ1dQ1Correlation_std = np.zeros([len(T), len(R), len(theta)])
            # dQ2dQ2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dQ1dQ2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dP1dQ1Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dP1dQ2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dP2dQ2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            # dP1dP1total = np.zeros([len(T), len(R), len(theta)])
            # dP2dP2total = np.zeros([len(T), len(R), len(theta)])
            # dQ1dQ1total = np.zeros([len(T), len(R), len(theta)])
            # dQ2dQ2total = np.zeros([len(T), len(R), len(theta)])
            dQ1dQ2total = np.zeros([len(T), len(R), len(theta)])
            dP1dQ1total = np.zeros([len(T), len(R), len(theta)])
            dP1dQ2total = np.zeros([len(T), len(R), len(theta)])
            dP2dQ2total = np.zeros([len(T), len(R), len(theta)])

            # dp1dp1ij = [
            #     [[[] for col in range(17)] for col in range(grid)]
            #     for col in range(timeGrid)
            # ]  # t, r, theta
            # dp2dp2ij = [
            #     [[[] for col in range(17)] for col in range(grid)]
            #     for col in range(timeGrid)
            # ]
            # dq1dq1ij = [
            #     [[[] for col in range(17)] for col in range(grid)]
            #     for col in range(timeGrid)
            # ]
            # dq2dq2ij = [
            #     [[[] for col in range(17)] for col in range(grid)]
            #     for col in range(timeGrid)
            # ]
            dq1dq2ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]  # t, r, theta
            dp1dq1ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]
            dp1dq2ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]
            dp2dq2ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]

            print(datetime.now().strftime("%H:%M:%S ") + filename)
            dfShapeF = dfShape[dfShape["Filename"] == filename].copy()
            n = int(len(dfShapeF) / 20)
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

                x = dfShapeF["X"].iloc[i]
                y = dfShapeF["Y"].iloc[i]
                t = dfShapeF["T"].iloc[i]
                dp1 = dfShapeF["dp"].iloc[i][0]
                dp2 = dfShapeF["dp"].iloc[i][1]
                dq1 = dfShapeF["dq"].iloc[i][0, 0]
                dq2 = dfShapeF["dq"].iloc[i][0, 1]
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
                        # "dp1dp1i",
                        # "dp2dp2i",
                        # "dq1dq1i",
                        # "dq2dq2i",
                        "dq1dq2i",
                        "dp1dq1i",
                        "dp1dq2i",
                        "dp2dq2i",
                    ]
                ]
                df = df[np.array(df["dR"] < R[-1]) & np.array(df["dR"] >= 0)]

                df["dT"] = df.loc[:, "T"] - t
                df = df[np.array(df["dT"] < timeGrid) & np.array(df["dT"] >= 0)]
                if len(df) != 0:
                    theta = np.arctan2(df.loc[:, "Y"] - y, df.loc[:, "X"] - x)
                    df["dtheta"] = np.where(theta < 0, 2 * np.pi + theta, theta)
                    # df["dp1dp1i"] = list(
                    #     dp1 * np.stack(np.array(df.loc[:, "dp"]), axis=0)[:, 0]
                    # )
                    # df["dp2dp2i"] = list(
                    #     dp2 * np.stack(np.array(df.loc[:, "dp"]), axis=0)[:, 1]
                    # )
                    # df["dq1dq1i"] = list(
                    #     dq1 * np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 0]
                    # )
                    # df["dq2dq2i"] = list(
                    #     dq2 * np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 1]
                    # )
                    df["dq1dq2i"] = list(
                        dq1 * np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 1]
                    )
                    df["dp1dq1i"] = list(
                        dp1 * np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 0]
                    )
                    df["dp1dq2i"] = list(
                        dp1 * np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 1]
                    )
                    df["dp2dq2i"] = list(
                        dp2 * np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 1]
                    )

                    for j in range(len(df)):
                        # dp1dp1ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                        #     int(8 * df["dtheta"].iloc[j] / np.pi)
                        # ].append(df["dp1dp1i"].iloc[j])
                        # dp2dp2ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                        #     int(8 * df["dtheta"].iloc[j] / np.pi)
                        # ].append(df["dp2dp2i"].iloc[j])
                        # dq1dq1ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                        #     int(8 * df["dtheta"].iloc[j] / np.pi)
                        # ].append(df["dq1dq1i"].iloc[j])
                        # dq2dq2ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                        #     int(8 * df["dtheta"].iloc[j] / np.pi)
                        # ].append(df["dq2dq2i"].iloc[j])
                        dq1dq2ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dq1dq2i"].iloc[j])
                        dp1dq1ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dp1dq1i"].iloc[j])
                        dp1dq2ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dp1dq2i"].iloc[j])
                        dp2dq2ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dp2dq2i"].iloc[j])

            T = np.linspace(0, (timeGrid - 1), timeGrid)
            R = np.linspace(0, 2 * (grid - 1), grid)
            theta = np.linspace(0, 2 * np.pi, 17)
            for i in range(len(T)):
                for j in range(len(R)):
                    for th in range(len(theta)):
                        # dP1dP1Correlation[i][j][th] = np.mean(dp1dp1ij[i][j][th])
                        # dP2dP2Correlation[i][j][th] = np.mean(dp2dp2ij[i][j][th])
                        # dQ1dQ1Correlation[i][j][th] = np.mean(dq1dq1ij[i][j][th])
                        # dQ2dQ2Correlation[i][j][th] = np.mean(dq2dq2ij[i][j][th])
                        dQ1dQ2Correlation[i][j][th] = np.mean(dq1dq2ij[i][j][th])
                        dP1dQ1Correlation[i][j][th] = np.mean(dp1dq1ij[i][j][th])
                        dP1dQ2Correlation[i][j][th] = np.mean(dp1dq2ij[i][j][th])
                        dP2dQ2Correlation[i][j][th] = np.mean(dp2dq2ij[i][j][th])

                        # dP1dP1Correlation_std[i][j][th] = np.std(dp1dp1ij[i][j][th])
                        # dP2dP2Correlation_std[i][j][th] = np.std(dp2dp2ij[i][j][th])
                        # dQ1dQ1Correlation_std[i][j][th] = np.std(dq1dq1ij[i][j][th])
                        # dQ2dQ2Correlation_std[i][j][th] = np.std(dq2dq2ij[i][j][th])
                        dQ1dQ2Correlation_std[i][j][th] = np.std(dq1dq2ij[i][j][th])
                        dP1dQ1Correlation_std[i][j][th] = np.std(dp1dq1ij[i][j][th])
                        dP1dQ2Correlation_std[i][j][th] = np.std(dp1dq2ij[i][j][th])
                        dP2dQ2Correlation_std[i][j][th] = np.std(dp2dq2ij[i][j][th])
                        # dP1dP1total[i][j][th] = len(dp1dp1ij[i][j][th])
                        # dP2dP2total[i][j][th] = len(dp2dp2ij[i][j][th])
                        # dQ1dQ1total[i][j][th] = len(dq1dq1ij[i][j][th])
                        # dQ2dQ2total[i][j][th] = len(dq2dq2ij[i][j][th])
                        dQ1dQ2total[i][j][th] = len(dq1dq2ij[i][j][th])
                        dP1dQ1total[i][j][th] = len(dp1dq1ij[i][j][th])
                        dP1dQ2total[i][j][th] = len(dp1dq2ij[i][j][th])
                        dP2dQ2total[i][j][th] = len(dp2dq2ij[i][j][th])

            _df.append(
                {
                    "Filename": filename,
                    # "dP1dP1Correlation": dP1dP1Correlation,
                    # "dP2dP2Correlation": dP2dP2Correlation,
                    # "dQ1dQ1Correlation": dQ1dQ1Correlation,
                    # "dQ2dQ2Correlation": dQ2dQ2Correlation,
                    "dQ1dQ2Correlation": dQ1dQ2Correlation,
                    "dP1dQ1Correlation": dP1dQ1Correlation,
                    "dP1dQ2Correlation": dP1dQ2Correlation,
                    "dP2dQ2Correlation": dP2dQ2Correlation,
                    # "dP1dP1Correlation_std": dP1dP1Correlation_std,
                    # "dP2dP2Correlation_std": dP2dP2Correlation_std,
                    # "dQ1dQ1Correlation_std": dQ1dQ1Correlation_std,
                    # "dQ2dQ2Correlation_std": dQ2dQ2Correlation_std,
                    "dQ1dQ2Correlation_std": dQ1dQ2Correlation_std,
                    "dP1dQ1Correlation_std": dP1dQ1Correlation_std,
                    "dP1dQ2Correlation_std": dP1dQ2Correlation_std,
                    "dP2dQ2Correlation_std": dP2dQ2Correlation_std,
                    # "dP1dP1Count": dP1dP1total,
                    # "dP2dP2Count": dP2dP2total,
                    # "dQ1dQ1Count": dQ1dQ1total,
                    # "dQ2dQ2Count": dQ2dQ2total,
                    "dQ1dQ2Count": dQ1dQ2total,
                    "dP1dQ1Count": dP1dQ1total,
                    "dP1dQ2Count": dP1dQ2total,
                    "dP2dQ2Count": dP2dQ2total,
                }
            )
            dfCorrelation = pd.DataFrame(_df)
            dfCorrelation.to_pickle(
                f"databases/correlations/dfCorMidway{filename}_5-8.pkl"
            )

# --------- velocity ----------

# space time d_r denstiy-velocity correlation
if False:
    grid = 9
    timeGrid = 18
    gridSize = 10
    gridSizeT = 5
    grid_df = 27
    timeGrid_df = 51

    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    dfVelocity = pd.read_pickle(f"databases/dfVelocity{fileType}.pkl")

    dfVelocity["dR"] = list(np.zeros([len(dfVelocity)]))
    dfVelocity["dT"] = list(np.zeros([len(dfVelocity)]))
    dfVelocity["dtheta"] = list(np.zeros([len(dfVelocity)]))

    dfVelocity["dr1drhodv1i"] = list(np.zeros([len(dfVelocity)]))
    dfVelocity["dr1drhodv2i"] = list(np.zeros([len(dfVelocity)]))
    dfVelocity["dr2drhodv1i"] = list(np.zeros([len(dfVelocity)]))
    dfVelocity["dr2drhodv2i"] = list(np.zeros([len(dfVelocity)]))

    xMax = np.max(dfShape["X"])
    xMin = np.min(dfShape["X"])
    yMax = np.max(dfShape["Y"])
    yMin = np.min(dfShape["Y"])
    xGrid = int(1 + (xMax - xMin) // gridSize)
    yGrid = int(1 + (yMax - yMin) // gridSize)

    T = np.linspace(0, gridSizeT * (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid_df - 1), grid_df)
    theta = np.linspace(0, 2 * np.pi, 17)
    for filename in filenames:
        path_to_file = f"databases/correlations/dfCorRhoV{filename}_1-4.pkl"
        if False == exists(path_to_file):
            print(filename + datetime.now().strftime(" %H:%M:%S"))

            dfShapeF = dfShape[dfShape["Filename"] == filename].copy()
            dfVelocityF = dfVelocity[dfVelocity["Filename"] == filename].copy()
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

            hm_dr1drho = (heatmapdrho[:, 1:] - heatmapdrho[:, :-1]) / gridSize
            hm_dr2drho = (heatmapdrho[:, :, 1:] - heatmapdrho[:, :, :-1]) / gridSize
            inPlaneEcadr1 = (inPlaneEcad[:, 1:] + inPlaneEcad[:, :-1]) / 2
            inPlaneEcadr2 = (inPlaneEcad[:, :, 1:] + inPlaneEcad[:, :, :-1]) / 2
            inPlaneEcadr1[inPlaneEcadr1 < 1] = 0
            inPlaneEcadr2[inPlaneEcadr2 < 1] = 0

            dr1dRhodV1Correlation = np.zeros([len(T), len(R), len(theta)])
            dr1dRhodV2Correlation = np.zeros([len(T), len(R), len(theta)])
            dr2dRhodV1Correlation = np.zeros([len(T), len(R), len(theta)])
            dr2dRhodV2Correlation = np.zeros([len(T), len(R), len(theta)])
            dr1dRhodV1Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dr1dRhodV2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dr2dRhodV1Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dr2dRhodV2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            total_dr1 = np.zeros([len(T), len(R), len(theta)])
            total_dr2 = np.zeros([len(T), len(R), len(theta)])
            dr1drhodv1ij = [
                [[[] for col in range(17)] for col in range(grid_df)]
                for col in range(timeGrid)
            ]
            dr1drhodv2ij = [
                [[[] for col in range(17)] for col in range(grid_df)]
                for col in range(timeGrid)
            ]
            dr2drhodv1ij = [
                [[[] for col in range(17)] for col in range(grid_df)]
                for col in range(timeGrid)
            ]
            dr2drhodv2ij = [
                [[[] for col in range(17)] for col in range(grid_df)]
                for col in range(timeGrid)
            ]

            count = 0
            percent_count = 0
            for i in range(xGrid - 1):
                for j in range(yGrid):
                    if count % int(((xGrid - 1) * yGrid) / 10) == 0:
                        print(
                            datetime.now().strftime("%H:%M:%S")
                            + f" {10*percent_count}%"
                        )
                        percent_count += 1
                    count += 1
                    for t in T:
                        t = int(t)
                        if np.sum(inPlaneEcadr1[t : int(t + gridSizeT), i, j]) > 0:
                            dr1drho = np.mean(
                                hm_dr1drho[t : int(t + gridSizeT), i, j][
                                    inPlaneEcadr1[t : int(t + gridSizeT), i, j] > 0
                                ]
                            )
                            x = xMin + (i + 1) * gridSize
                            y = yMin + (j + 0.5) * gridSize

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
                                    "dr1drhodv1i",
                                    "dr1drhodv2i",
                                ]
                            ]
                            df = df[
                                np.array(df["dR"] < R[-1]) & np.array(df["dR"] >= 0)
                            ]

                            df["dT"] = df.loc[:, "T"] - t
                            df = df[
                                np.array(df["dT"] < T[-1]) & np.array(df["dT"] >= 0)
                            ]
                            if len(df) != 0:
                                theta = np.arctan2(
                                    df.loc[:, "Y"] - y, df.loc[:, "X"] - x
                                )
                                df["dtheta"] = np.where(
                                    theta < 0, 2 * np.pi + theta, theta
                                )
                                df["dr1drhodv1i"] = list(
                                    dr1drho
                                    * np.stack(np.array(df.loc[:, "dv"]), axis=0)[:, 0]
                                )
                                df["dr1drhodv2i"] = list(
                                    dr1drho
                                    * np.stack(np.array(df.loc[:, "dv"]), axis=0)[:, 1]
                                )

                                for p in range(len(df)):
                                    dr1drhodv1ij[int(df["dT"].iloc[p] / gridSizeT)][
                                        int(df["dR"].iloc[p] / 2)
                                    ][int(8 * df["dtheta"].iloc[p] / np.pi)].append(
                                        df["dr1drhodv1i"].iloc[p]
                                    )
                                    dr1drhodv2ij[int(df["dT"].iloc[p] / gridSizeT)][
                                        int(df["dR"].iloc[p] / 2)
                                    ][int(8 * df["dtheta"].iloc[p] / np.pi)].append(
                                        df["dr1drhodv2i"].iloc[p]
                                    )

            count = 0
            percent_count = 0
            for i in range(xGrid):
                for j in range(yGrid - 1):
                    if count % int(((xGrid) * yGrid - 1) / 10) == 0:
                        print(
                            datetime.now().strftime("%H:%M:%S")
                            + f" {10*percent_count}%"
                        )
                        percent_count += 1
                    count += 1
                    for t in T:
                        t = int(t)
                        if np.sum(inPlaneEcadr2[t : int(t + gridSizeT), i, j]) > 0:
                            dr2drho = np.mean(
                                hm_dr2drho[t : int(t + gridSizeT), i, j][
                                    inPlaneEcadr2[t : int(t + gridSizeT), i, j] > 0
                                ]
                            )
                            x = xMin + (i + 0.5) * gridSize
                            y = yMin + (j + 1) * gridSize

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
                                    "dr2drhodv1i",
                                    "dr2drhodv2i",
                                ]
                            ]
                            df = df[
                                np.array(df["dR"] < R[-1]) & np.array(df["dR"] >= 0)
                            ]

                            df["dT"] = df.loc[:, "T"] - t
                            df = df[
                                np.array(df["dT"] < T[-1]) & np.array(df["dT"] >= 0)
                            ]
                            if len(df) != 0:
                                theta = np.arctan2(
                                    df.loc[:, "Y"] - y, df.loc[:, "X"] - x
                                )
                                df["dtheta"] = np.where(
                                    theta < 0, 2 * np.pi + theta, theta
                                )
                                df["dr2drhodv1i"] = list(
                                    dr2drho
                                    * np.stack(np.array(df.loc[:, "dv"]), axis=0)[:, 0]
                                )
                                df["dr2drhodv2i"] = list(
                                    dr2drho
                                    * np.stack(np.array(df.loc[:, "dv"]), axis=0)[:, 1]
                                )

                                for p in range(len(df)):
                                    dr2drhodv1ij[int(df["dT"].iloc[p] / gridSizeT)][
                                        int(df["dR"].iloc[p] / 2)
                                    ][int(8 * df["dtheta"].iloc[p] / np.pi)].append(
                                        df["dr2drhodv1i"].iloc[p]
                                    )
                                    dr2drhodv2ij[int(df["dT"].iloc[p] / gridSizeT)][
                                        int(df["dR"].iloc[p] / 2)
                                    ][int(8 * df["dtheta"].iloc[p] / np.pi)].append(
                                        df["dr2drhodv2i"].iloc[p]
                                    )

            T = np.linspace(0, gridSizeT * (timeGrid - 1), timeGrid)
            R = np.linspace(0, 2 * (grid_df - 1), grid_df)
            theta = np.linspace(0, 2 * np.pi, 17)
            for i in range(len(T)):
                for j in range(len(R)):
                    for th in range(len(theta)):
                        dr1dRhodV1Correlation[i][j][th] = np.mean(
                            dr1drhodv1ij[i][j][th]
                        )
                        dr1dRhodV2Correlation[i][j][th] = np.mean(
                            dr1drhodv2ij[i][j][th]
                        )
                        dr2dRhodV1Correlation[i][j][th] = np.mean(
                            dr2drhodv1ij[i][j][th]
                        )
                        dr2dRhodV2Correlation[i][j][th] = np.mean(
                            dr2drhodv2ij[i][j][th]
                        )
                        dr1dRhodV1Correlation_std[i][j][th] = np.std(
                            dr1drhodv1ij[i][j][th]
                        )
                        dr1dRhodV2Correlation_std[i][j][th] = np.std(
                            dr1drhodv2ij[i][j][th]
                        )
                        dr2dRhodV1Correlation_std[i][j][th] = np.std(
                            dr2drhodv1ij[i][j][th]
                        )
                        dr2dRhodV2Correlation_std[i][j][th] = np.std(
                            dr2drhodv2ij[i][j][th]
                        )
                        total_dr1[i][j][th] = len(dr1drhodv1ij[i][j][th])
                        total_dr2[i][j][th] = len(dr2drhodv1ij[i][j][th])

            _df = []
            _df.append(
                {
                    "Filename": filename,
                    "dr1dRhodV1Correlation": dr1dRhodV1Correlation,
                    "dr1dRhodV2Correlation": dr1dRhodV2Correlation,
                    "dr2dRhodV1Correlation": dr2dRhodV1Correlation,
                    "dr2dRhodV2Correlation": dr2dRhodV2Correlation,
                    "dr1dRhodV1Correlation_std": dr1dRhodV1Correlation_std,
                    "dr1dRhodV2Correlation_std": dr1dRhodV2Correlation_std,
                    "dr2dRhodV1Correlation_std": dr2dRhodV1Correlation_std,
                    "dr2dRhodV2Correlation_std": dr2dRhodV2Correlation_std,
                    "dr1Count": total_dr1,
                    "dr2Count": total_dr2,
                }
            )

            df = pd.DataFrame(_df)
            df.to_pickle(f"databases/correlations/dfCorRhoV{filename}_1-4.pkl")

# space time d_r dQ^1-velocity correlation
if False:
    grid = 9
    timeGrid = 18
    gridSize = 10
    gridSizeT = 5
    grid_df = 27
    timeGrid_df = 51
    theta = np.linspace(0, 2 * np.pi, 17)

    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    dfVelocity = pd.read_pickle(f"databases/dfVelocity{fileType}.pkl")

    dfVelocity["dR"] = list(np.zeros([len(dfVelocity)]))
    dfVelocity["dT"] = list(np.zeros([len(dfVelocity)]))
    dfVelocity["dtheta"] = list(np.zeros([len(dfVelocity)]))

    dfVelocity["dr1dq1dv1i"] = list(np.zeros([len(dfVelocity)]))
    dfVelocity["dr1dq1dv2i"] = list(np.zeros([len(dfVelocity)]))
    dfVelocity["dr2dq1dv1i"] = list(np.zeros([len(dfVelocity)]))
    dfVelocity["dr2dq1dv2i"] = list(np.zeros([len(dfVelocity)]))

    xMax = np.max(dfShape["X"])
    xMin = np.min(dfShape["X"])
    yMax = np.max(dfShape["Y"])
    yMin = np.min(dfShape["Y"])
    xGrid = int(1 + (xMax - xMin) // gridSize)
    yGrid = int(1 + (yMax - yMin) // gridSize)

    T = np.linspace(0, gridSizeT * (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid_df - 1), grid_df)
    for filename in filenames:
        path_to_file = f"databases/correlations/dfCordrdQ1V{filename}_1-4.pkl"
        if False == exists(path_to_file):
            print(filename + datetime.now().strftime(" %H:%M:%S"))

            dfShapeF = dfShape[dfShape["Filename"] == filename].copy()
            dfVelocityF = dfVelocity[dfVelocity["Filename"] == filename].copy()
            heatmapdQ1 = np.zeros([90, xGrid, yGrid])
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
                        if list(dfg["dq"]) != []:
                            heatmapdQ1[t, i, j] = np.mean(
                                np.stack(np.array(dfg.loc[:, "dq"]), axis=0)[:, 0, 0]
                            )
                            inPlaneEcad[t, i, j] = 1

                heatmapdQ1[t] = heatmapdQ1[t] - np.mean(
                    heatmapdQ1[t][inPlaneEcad[t] == 1]
                )

            hm_dr1dQ1 = (heatmapdQ1[:, 1:] - heatmapdQ1[:, :-1]) / gridSize
            hm_dr2dQ1 = (heatmapdQ1[:, :, 1:] - heatmapdQ1[:, :, :-1]) / gridSize
            inPlaneEcadr1 = (inPlaneEcad[:, 1:] + inPlaneEcad[:, :-1]) / 2
            inPlaneEcadr2 = (inPlaneEcad[:, :, 1:] + inPlaneEcad[:, :, :-1]) / 2
            inPlaneEcadr1[inPlaneEcadr1 < 1] = 0
            inPlaneEcadr2[inPlaneEcadr2 < 1] = 0

            dr1dQ1dV1Correlation = np.zeros([len(T), len(R), len(theta)])
            dr1dQ1dV2Correlation = np.zeros([len(T), len(R), len(theta)])
            dr2dQ1dV1Correlation = np.zeros([len(T), len(R), len(theta)])
            dr2dQ1dV2Correlation = np.zeros([len(T), len(R), len(theta)])
            dr1dQ1dV1Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dr1dQ1dV2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dr2dQ1dV1Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dr2dQ1dV2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            total_dr1 = np.zeros([len(T), len(R), len(theta)])
            total_dr2 = np.zeros([len(T), len(R), len(theta)])
            dr1dq1dv1ij = [
                [[[] for col in range(17)] for col in range(grid_df)]
                for col in range(timeGrid)
            ]
            dr1dq1dv2ij = [
                [[[] for col in range(17)] for col in range(grid_df)]
                for col in range(timeGrid)
            ]
            dr2dq1dv1ij = [
                [[[] for col in range(17)] for col in range(grid_df)]
                for col in range(timeGrid)
            ]
            dr2dq1dv2ij = [
                [[[] for col in range(17)] for col in range(grid_df)]
                for col in range(timeGrid)
            ]

            count = 0
            percent_count = 0
            for i in range(xGrid - 1):
                for j in range(yGrid):
                    if count % int(((xGrid - 1) * yGrid) / 10) == 0:
                        print(
                            datetime.now().strftime("%H:%M:%S")
                            + f" {10*percent_count}%"
                        )
                        percent_count += 1
                    count += 1
                    for t in T:
                        t = int(t)
                        if np.sum(inPlaneEcadr1[t : int(t + gridSizeT), i, j]) > 0:
                            dr1dq1 = np.mean(
                                hm_dr1dQ1[t : int(t + gridSizeT), i, j][
                                    inPlaneEcadr1[t : int(t + gridSizeT), i, j] > 0
                                ]
                            )
                            x = xMin + (i + 1) * gridSize
                            y = yMin + (j + 0.5) * gridSize

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
                                    "dr1dq1dv1i",
                                    "dr1dq1dv2i",
                                ]
                            ]
                            df = df[
                                np.array(df["dR"] < R[-1]) & np.array(df["dR"] >= 0)
                            ]

                            df["dT"] = df.loc[:, "T"] - t
                            df = df[
                                np.array(df["dT"] < T[-1]) & np.array(df["dT"] >= 0)
                            ]
                            if len(df) != 0:
                                theta = np.arctan2(
                                    df.loc[:, "Y"] - y, df.loc[:, "X"] - x
                                )
                                df["dtheta"] = np.where(
                                    theta < 0, 2 * np.pi + theta, theta
                                )
                                df["dr1dq1dv1i"] = list(
                                    dr1dq1
                                    * np.stack(np.array(df.loc[:, "dv"]), axis=0)[:, 0]
                                )
                                df["dr1dq1dv2i"] = list(
                                    dr1dq1
                                    * np.stack(np.array(df.loc[:, "dv"]), axis=0)[:, 1]
                                )

                                for p in range(len(df)):
                                    dr1dq1dv1ij[int(df["dT"].iloc[p] / gridSizeT)][
                                        int(df["dR"].iloc[p] / 2)
                                    ][int(8 * df["dtheta"].iloc[p] / np.pi)].append(
                                        df["dr1dq1dv1i"].iloc[p]
                                    )
                                    dr1dq1dv2ij[int(df["dT"].iloc[p] / gridSizeT)][
                                        int(df["dR"].iloc[p] / 2)
                                    ][int(8 * df["dtheta"].iloc[p] / np.pi)].append(
                                        df["dr1dq1dv2i"].iloc[p]
                                    )

            count = 0
            percent_count = 0
            for i in range(xGrid):
                for j in range(yGrid - 1):
                    if count % int(((xGrid) * yGrid - 1) / 10) == 0:
                        print(
                            datetime.now().strftime("%H:%M:%S")
                            + f" {10*percent_count}%"
                        )
                        percent_count += 1
                    count += 1
                    for t in T:
                        t = int(t)
                        if np.sum(inPlaneEcadr2[t : int(t + gridSizeT), i, j]) > 0:
                            dr2dq1 = np.mean(
                                hm_dr2dQ1[t : int(t + gridSizeT), i, j][
                                    inPlaneEcadr2[t : int(t + gridSizeT), i, j] > 0
                                ]
                            )
                            x = xMin + (i + 0.5) * gridSize
                            y = yMin + (j + 1) * gridSize

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
                                    "dr2dq1dv1i",
                                    "dr2dq1dv2i",
                                ]
                            ]
                            df = df[
                                np.array(df["dR"] < R[-1]) & np.array(df["dR"] >= 0)
                            ]

                            df["dT"] = df.loc[:, "T"] - t
                            df = df[
                                np.array(df["dT"] < T[-1]) & np.array(df["dT"] >= 0)
                            ]
                            if len(df) != 0:
                                theta = np.arctan2(
                                    df.loc[:, "Y"] - y, df.loc[:, "X"] - x
                                )
                                df["dtheta"] = np.where(
                                    theta < 0, 2 * np.pi + theta, theta
                                )
                                df["dr2dq1dv1i"] = list(
                                    dr2dq1
                                    * np.stack(np.array(df.loc[:, "dv"]), axis=0)[:, 0]
                                )
                                df["dr2dq1dv2i"] = list(
                                    dr2dq1
                                    * np.stack(np.array(df.loc[:, "dv"]), axis=0)[:, 1]
                                )

                                for p in range(len(df)):
                                    dr2dq1dv1ij[int(df["dT"].iloc[p] / gridSizeT)][
                                        int(df["dR"].iloc[p] / 2)
                                    ][int(8 * df["dtheta"].iloc[p] / np.pi)].append(
                                        df["dr2dq1dv1i"].iloc[p]
                                    )
                                    dr2dq1dv2ij[int(df["dT"].iloc[p] / gridSizeT)][
                                        int(df["dR"].iloc[p] / 2)
                                    ][int(8 * df["dtheta"].iloc[p] / np.pi)].append(
                                        df["dr2dq1dv2i"].iloc[p]
                                    )

            T = np.linspace(0, gridSizeT * (timeGrid - 1), timeGrid)
            R = np.linspace(0, 2 * (grid_df - 1), grid_df)
            theta = np.linspace(0, 2 * np.pi, 17)
            for i in range(len(T)):
                for j in range(len(R)):
                    for th in range(len(theta)):
                        dr1dQ1dV1Correlation[i][j][th] = np.mean(dr1dq1dv1ij[i][j][th])
                        dr1dQ1dV2Correlation[i][j][th] = np.mean(dr1dq1dv2ij[i][j][th])
                        dr2dQ1dV1Correlation[i][j][th] = np.mean(dr2dq1dv1ij[i][j][th])
                        dr2dQ1dV2Correlation[i][j][th] = np.mean(dr2dq1dv2ij[i][j][th])
                        dr1dQ1dV1Correlation_std[i][j][th] = np.std(
                            dr1dq1dv1ij[i][j][th]
                        )
                        dr1dQ1dV2Correlation_std[i][j][th] = np.std(
                            dr1dq1dv2ij[i][j][th]
                        )
                        dr2dQ1dV1Correlation_std[i][j][th] = np.std(
                            dr2dq1dv1ij[i][j][th]
                        )
                        dr2dQ1dV2Correlation_std[i][j][th] = np.std(
                            dr2dq1dv2ij[i][j][th]
                        )
                        total_dr1[i][j][th] = len(dr1dq1dv1ij[i][j][th])
                        total_dr2[i][j][th] = len(dr2dq1dv1ij[i][j][th])

            _df = []
            _df.append(
                {
                    "Filename": filename,
                    "dr1dQ1dV1Correlation": dr1dQ1dV1Correlation,
                    "dr1dQ1dV2Correlation": dr1dQ1dV2Correlation,
                    "dr2dQ1dV1Correlation": dr2dQ1dV1Correlation,
                    "dr2dQ1dV2Correlation": dr2dQ1dV2Correlation,
                    "dr1dQ1dV1Correlation_std": dr1dQ1dV1Correlation_std,
                    "dr1dQ1dV2Correlation_std": dr1dQ1dV2Correlation_std,
                    "dr2dQ1dV1Correlation_std": dr2dQ1dV1Correlation_std,
                    "dr2dQ1dV2Correlation_std": dr2dQ1dV2Correlation_std,
                    "dr1Count": total_dr1,
                    "dr2Count": total_dr2,
                }
            )

            df = pd.DataFrame(_df)
            df.to_pickle(f"databases/correlations/dfCordrdQ1V{filename}_1-4.pkl")

# space time shape-velocity correlation
if False:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    dfVelocity = pd.read_pickle(f"databases/dfVelocity{fileType}.pkl")
    grid = 27
    timeGrid = 51

    T = np.linspace(0, (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)
    theta = np.linspace(0, 2 * np.pi, 17)

    dfVelocity["dR"] = list(np.zeros([len(dfVelocity)]))
    dfVelocity["dT"] = list(np.zeros([len(dfVelocity)]))
    dfVelocity["dtheta"] = list(np.zeros([len(dfVelocity)]))

    dfVelocity["dp1dv1i"] = list(np.zeros([len(dfVelocity)]))
    dfVelocity["dp2dv2i"] = list(np.zeros([len(dfVelocity)]))

    for k in range(len(filenames)):
        filename = filenames[k]
        path_to_file = f"databases/correlations/dfCorShapeVelMidway{filename}_1-2.pkl"
        if False == exists(path_to_file):
            _df = []
            dP1dV1Correlation = np.zeros([len(T), len(R), len(theta)])
            dP2dV2Correlation = np.zeros([len(T), len(R), len(theta)])
            dP1dV1Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dP2dV2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            dP1dV1total = np.zeros([len(T), len(R), len(theta)])
            dP2dV2total = np.zeros([len(T), len(R), len(theta)])

            dp1dv1ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]  # t, r, theta
            dp2dv2ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]

            print(datetime.now().strftime("%H:%M:%S ") + filename)
            dfShapeF = dfShape[dfShape["Filename"] == filename].copy()
            dfVelocityF = dfVelocity[dfVelocity["Filename"] == filename].copy()
            n = int(len(dfShapeF) / 20)
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

                x = dfShapeF["X"].iloc[i]
                y = dfShapeF["Y"].iloc[i]
                t = dfShapeF["T"].iloc[i]
                dp1 = dfShapeF["dp"].iloc[i][0]
                dp2 = dfShapeF["dp"].iloc[i][1]
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
                        "dp1dv1i",
                        "dp2dv2i",
                    ]
                ]
                df = df[np.array(df["dR"] < R[-1]) & np.array(df["dR"] >= 0)]

                df["dT"] = df.loc[:, "T"] - t
                df = df[np.array(df["dT"] < timeGrid) & np.array(df["dT"] >= 0)]
                if len(df) != 0:
                    theta = np.arctan2(df.loc[:, "Y"] - y, df.loc[:, "X"] - x)
                    df["dtheta"] = np.where(theta < 0, 2 * np.pi + theta, theta)
                    df["dp1dv1i"] = list(
                        dp1 * np.stack(np.array(df.loc[:, "dv"]), axis=0)[:, 0]
                    )
                    df["dp2dv2i"] = list(
                        dp2 * np.stack(np.array(df.loc[:, "dv"]), axis=0)[:, 1]
                    )

                    for j in range(len(df)):
                        dp1dv1ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dp1dv1i"].iloc[j])
                        dp2dv2ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dp2dv2i"].iloc[j])

            T = np.linspace(0, (timeGrid - 1), timeGrid)
            R = np.linspace(0, 2 * (grid - 1), grid)
            theta = np.linspace(0, 2 * np.pi, 17)
            for i in range(len(T)):
                for j in range(len(R)):
                    for th in range(len(theta)):
                        dP1dV1Correlation[i][j][th] = np.mean(dp1dv1ij[i][j][th])
                        dP2dV2Correlation[i][j][th] = np.mean(dp2dv2ij[i][j][th])

                        dP1dV1Correlation_std[i][j][th] = np.std(dp1dv1ij[i][j][th])
                        dP2dV2Correlation_std[i][j][th] = np.std(dp2dv2ij[i][j][th])

                        dP1dV1total[i][j][th] = len(dp1dv1ij[i][j][th])
                        dP2dV2total[i][j][th] = len(dp2dv2ij[i][j][th])

            _df.append(
                {
                    "Filename": filename,
                    "dP1dV1Correlation": dP1dV1Correlation,
                    "dP2dV2Correlation": dP2dV2Correlation,
                    "dP1dV1Correlation_std": dP1dV1Correlation_std,
                    "dP2dV2Correlation_std": dP2dV2Correlation_std,
                    "dP1dV1Count": dP1dV1total,
                    "dP2dV2Count": dP2dV2total,
                }
            )
            dfCorrelation = pd.DataFrame(_df)
            dfCorrelation.to_pickle(
                f"databases/correlations/dfCorShapeVelMidway{filename}_1-2.pkl"
            )

# space time velocity-velocity correlation
if False:
    dfVelocity = pd.read_pickle(f"databases/dfVelocity{fileType}.pkl")
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
        path_to_file = f"databases/correlations/dfCorVelMidway{filename}_1-2.pkl"
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
            n = int(len(dfVelocityF) / 20)
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

                x = dfVelocityF["X"].iloc[i]
                y = dfVelocityF["Y"].iloc[i]
                t = dfVelocityF["T"].iloc[i]
                dv1 = dfVelocityF["dv"].iloc[i][0]
                dv2 = dfVelocityF["dv"].iloc[i][1]
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

                df["dT"] = df.loc[:, "T"] - t
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
                f"databases/correlations/dfCorVelMidway{filename}_1-2.pkl"
            )

# --------- collect all ----------

# collect all correlations
if True:
    _df = []
    for filename in filenames:

        dfCorMid_1 = pd.read_pickle(
            f"databases/correlations/dfCorMidway{filename}_1.pkl"
        )
        dfCorMid_2 = pd.read_pickle(
            f"databases/correlations/dfCorMidway{filename}_2.pkl"
        )
        dfCorMid_3 = pd.read_pickle(
            f"databases/correlations/dfCorMidway{filename}_3.pkl"
        )
        dfCorMid_4 = pd.read_pickle(
            f"databases/correlations/dfCorMidway{filename}_4.pkl"
        )
        dfCorMid_5_8 = pd.read_pickle(
            f"databases/correlations/dfCorMidway{filename}_5-8.pkl"
        )
        dfCorRho = pd.read_pickle(f"databases/correlations/dfCorRho{filename}.pkl")
        dfCorRhoQ = pd.read_pickle(f"databases/correlations/dfCorRhoQ{filename}.pkl")

        dfCorRhoV = pd.read_pickle(
            f"databases/correlations/dfCorRhoV{filename}_1-4.pkl"
        )
        dfCordrdQ1V = pd.read_pickle(
            f"databases/correlations/dfCordrdQ1V{filename}_1-4.pkl"
        )
        dfCorShapeVel = pd.read_pickle(
            f"databases/correlations/dfCorShapeVelMidway{filename}_1-2.pkl"
        )
        dfCorVel = pd.read_pickle(
            f"databases/correlations/dfCorVelMidway{filename}_1-2.pkl"
        )

        dP1dP1 = np.nan_to_num(dfCorMid_1["dP1dP1Correlation"].iloc[0])[:, :27]
        dP1dP1_std = np.nan_to_num(dfCorMid_1["dP1dP1Correlation_std"].iloc[0])[:, :27]
        dP1dP1total = np.nan_to_num(dfCorMid_1["dP1dP1Count"].iloc[0])[:, :27]
        if np.sum(dP1dP1) == 0:
            print("dP1dP1")

        dP2dP2 = np.nan_to_num(dfCorMid_2["dP2dP2Correlation"].iloc[0])[:, :27]
        dP2dP2_std = np.nan_to_num(dfCorMid_2["dP2dP2Correlation_std"].iloc[0])[:, :27]
        dP2dP2total = np.nan_to_num(dfCorMid_2["dP2dP2Count"].iloc[0])[:, :27]
        if np.sum(dP2dP2) == 0:
            print("dP2dP2")

        dQ1dQ1 = np.nan_to_num(dfCorMid_3["dQ1dQ1Correlation"].iloc[0])[:, :27]
        dQ1dQ1_std = np.nan_to_num(dfCorMid_3["dQ1dQ1Correlation_std"].iloc[0])[:, :27]
        dQ1dQ1total = np.nan_to_num(dfCorMid_3["dQ1dQ1Count"].iloc[0])[:, :27]
        if np.sum(dQ1dQ1) == 0:
            print("dQ1dQ1")

        dQ2dQ2 = np.nan_to_num(dfCorMid_4["dQ2dQ2Correlation"].iloc[0])[:, :27]
        dQ2dQ2_std = np.nan_to_num(dfCorMid_4["dQ2dQ2Correlation_std"].iloc[0])[:, :27]
        dQ2dQ2total = np.nan_to_num(dfCorMid_4["dQ2dQ2Count"].iloc[0])[:, :27]
        if np.sum(dQ2dQ2) == 0:
            print("dQ2dQ2")

        dQ1dQ2 = np.nan_to_num(dfCorMid_5_8["dQ1dQ2Correlation"].iloc[0])
        dQ1dQ2_std = np.nan_to_num(dfCorMid_5_8["dQ1dQ2Correlation_std"].iloc[0])
        dQ1dQ2total = np.nan_to_num(dfCorMid_5_8["dQ1dQ2Count"].iloc[0])
        if np.sum(dQ1dQ2) == 0:
            print("dQ1dQ2")

        dP1dQ1 = np.nan_to_num(dfCorMid_5_8["dP1dQ1Correlation"].iloc[0])
        dP1dQ1_std = np.nan_to_num(dfCorMid_5_8["dP1dQ1Correlation_std"].iloc[0])
        dP1dQ1total = np.nan_to_num(dfCorMid_5_8["dP1dQ1Count"].iloc[0])
        if np.sum(dP1dQ1) == 0:
            print("dP1dQ1")

        dP1dQ2 = np.nan_to_num(dfCorMid_5_8["dP1dQ2Correlation"].iloc[0])
        dP1dQ2_std = np.nan_to_num(dfCorMid_5_8["dP1dQ2Correlation_std"].iloc[0])
        dP1dQ2total = np.nan_to_num(dfCorMid_5_8["dP1dQ2Count"].iloc[0])
        if np.sum(dP1dQ2) == 0:
            print("dP1dQ2")

        dP2dQ2 = np.nan_to_num(dfCorMid_5_8["dP2dQ2Correlation"].iloc[0])
        dP2dQ2_std = np.nan_to_num(dfCorMid_5_8["dP2dQ2Correlation_std"].iloc[0])
        dP2dQ2total = np.nan_to_num(dfCorMid_5_8["dP2dQ2Count"].iloc[0])
        if np.sum(dP2dQ2) == 0:
            print("dP2dQ2")

        dRhodRho = np.nan_to_num(dfCorRho["dRhodRhoCorrelation"].iloc[0])
        dRhodRho_std = np.nan_to_num(dfCorRho["dRhodRhoCorrelation_std"].iloc[0])
        count_Rho = np.nan_to_num(dfCorRho["Count"].iloc[0])

        dQ1dRho = np.nan_to_num(dfCorRhoQ["dRhodQ1Correlation"].iloc[0])
        dQ1dRho_std = np.nan_to_num(dfCorRhoQ["dRhodQ1Correlation_std"].iloc[0])
        dQ2dRho = np.nan_to_num(dfCorRhoQ["dRhodQ2Correlation"].iloc[0])
        dQ2dRho_std = np.nan_to_num(dfCorRhoQ["dRhodQ2Correlation_std"].iloc[0])
        count_RhoQ = np.nan_to_num(dfCorRhoQ["Count"].iloc[0])

        dr1dRhodV1 = np.nan_to_num(dfCorRhoV["dr1dRhodV1Correlation"].iloc[0])
        dr1dRhodV1_std = np.nan_to_num(dfCorRhoV["dr1dRhodV1Correlation_std"].iloc[0])
        dr1dRhodV1total = np.nan_to_num(dfCorRhoV["dr1Count"].iloc[0])
        if np.sum(dr1dRhodV1) == 0:
            print("dr1dRhodV1")

        dr1dRhodV2 = np.nan_to_num(dfCorRhoV["dr1dRhodV2Correlation"].iloc[0])
        dr1dRhodV2_std = np.nan_to_num(dfCorRhoV["dr1dRhodV2Correlation_std"].iloc[0])
        dr1dRhodV2total = np.nan_to_num(dfCorRhoV["dr1Count"].iloc[0])
        if np.sum(dr1dRhodV2) == 0:
            print("dr1dRhodV2")

        dr2dRhodV1 = np.nan_to_num(dfCorRhoV["dr2dRhodV1Correlation"].iloc[0])
        dr2dRhodV1_std = np.nan_to_num(dfCorRhoV["dr2dRhodV1Correlation_std"].iloc[0])
        dr2dRhodV1total = np.nan_to_num(dfCorRhoV["dr2Count"].iloc[0])
        if np.sum(dr2dRhodV1) == 0:
            print("dr2dRhodV1")

        dr2dRhodV2 = np.nan_to_num(dfCorRhoV["dr2dRhodV2Correlation"].iloc[0])
        dr2dRhodV2_std = np.nan_to_num(dfCorRhoV["dr2dRhodV2Correlation_std"].iloc[0])
        dr2dRhodV2total = np.nan_to_num(dfCorRhoV["dr2Count"].iloc[0])
        if np.sum(dr2dRhodV2) == 0:
            print("dr2dRhodV2")

        dr1dQ1dV1 = np.nan_to_num(dfCordrdQ1V["dr1dQ1dV1Correlation"].iloc[0])
        dr1dQ1dV1_std = np.nan_to_num(dfCordrdQ1V["dr1dQ1dV1Correlation_std"].iloc[0])
        dr1dQ1dV1total = np.nan_to_num(dfCordrdQ1V["dr1Count"].iloc[0])
        if np.sum(dr1dQ1dV1) == 0:
            print("dr1dQ1dV1")

        dr1dQ1dV2 = np.nan_to_num(dfCordrdQ1V["dr1dQ1dV2Correlation"].iloc[0])
        dr1dQ1dV2_std = np.nan_to_num(dfCordrdQ1V["dr1dQ1dV2Correlation_std"].iloc[0])
        dr1dQ1dV2total = np.nan_to_num(dfCordrdQ1V["dr1Count"].iloc[0])
        if np.sum(dr1dQ1dV2) == 0:
            print("dr1dQ1dV2")

        dr2dQ1dV1 = np.nan_to_num(dfCordrdQ1V["dr2dQ1dV1Correlation"].iloc[0])
        dr2dQ1dV1_std = np.nan_to_num(dfCordrdQ1V["dr2dQ1dV1Correlation_std"].iloc[0])
        dr2dQ1dV1total = np.nan_to_num(dfCordrdQ1V["dr2Count"].iloc[0])
        if np.sum(dr2dQ1dV1) == 0:
            print("dr2dQ1dV1")

        dr2dQ1dV2 = np.nan_to_num(dfCordrdQ1V["dr2dQ1dV2Correlation"].iloc[0])
        dr2dQ1dV2_std = np.nan_to_num(dfCordrdQ1V["dr2dQ1dV2Correlation_std"].iloc[0])
        dr2dQ1dV2total = np.nan_to_num(dfCordrdQ1V["dr2Count"].iloc[0])
        if np.sum(dr2dQ1dV2) == 0:
            print("dr2dQ1dV2")

        dP1dV1 = np.nan_to_num(dfCorShapeVel["dP1dV1Correlation"].iloc[0])
        dP1dV1_std = np.nan_to_num(dfCorShapeVel["dP1dV1Correlation_std"].iloc[0])
        dP1dV1total = np.nan_to_num(dfCorShapeVel["dP1dV1Count"].iloc[0])
        if np.sum(dP1dV1) == 0:
            print("dP1dV1")

        dP2dV2 = np.nan_to_num(dfCorShapeVel["dP2dV2Correlation"].iloc[0])
        dP2dV2_std = np.nan_to_num(dfCorShapeVel["dP2dV2Correlation_std"].iloc[0])
        dP2dV2total = np.nan_to_num(dfCorShapeVel["dP2dV2Count"].iloc[0])
        if np.sum(dP2dV2) == 0:
            print("dP2dV2")

        dV1dV1 = np.nan_to_num(dfCorVel["dV1dV1Correlation"].iloc[0])
        dV1dV1_std = np.nan_to_num(dfCorVel["dV1dV1Correlation_std"].iloc[0])
        dV1dV1total = np.nan_to_num(dfCorVel["dV1dV1Count"].iloc[0])
        if np.sum(dV1dV1) == 0:
            print("dV1dV1")

        dV2dV2 = np.nan_to_num(dfCorVel["dV2dV2Correlation"].iloc[0])
        dV2dV2_std = np.nan_to_num(dfCorVel["dV2dV2Correlation_std"].iloc[0])
        dV2dV2total = np.nan_to_num(dfCorVel["dV2dV2Count"].iloc[0])
        if np.sum(dV2dV2) == 0:
            print("dV2dV2")

        _df.append(
            {
                "Filename": filename,
                "dP1dP1Correlation": dP1dP1,
                "dP1dP1Correlation_std": dP1dP1_std,
                "dP1dP1Count": dP1dP1total,
                "dP2dP2Correlation": dP2dP2,
                "dP2dP2Correlation_std": dP2dP2_std,
                "dP2dP2Count": dP2dP2total,
                "dQ1dQ1Correlation": dQ1dQ1,
                "dQ1dQ1Correlation_std": dQ1dQ1_std,
                "dQ1dQ1Count": dQ1dQ1total,
                "dQ2dQ2Correlation": dQ2dQ2,
                "dQ2dQ2Correlation_std": dQ2dQ2_std,
                "dQ2dQ2Count": dQ2dQ2total,
                "dQ1dQ2Correlation": dQ1dQ2,
                "dQ1dQ2Correlation_std": dQ1dQ2_std,
                "dQ1dQ2Count": dQ1dQ2total,
                "dP1dQ1Correlation": dP1dQ1,
                "dP1dQ1Correlation_std": dP1dQ1_std,
                "dP1dQ1Count": dP1dQ1total,
                "dP1dQ2Correlation": dP1dQ2,
                "dP1dQ2Correlation_std": dP1dQ2_std,
                "dP1dQ2Count": dP1dQ2total,
                "dP2dQ2Correlation": dP2dQ2,
                "dP2dQ2Correlation_std": dP2dQ2_std,
                "dP2dQ2Count": dP2dQ2total,
                "dRho_SdRho_S": dRhodRho,
                "dRho_SdRho_S_std": dRhodRho_std,
                "Count Rho_S": count_Rho,
                "dQ1dRho_S": dQ1dRho,
                "dQ1dRho_S_std": dQ1dRho_std,
                "dQ2dRho_S": dQ2dRho,
                "dQ2dRho_S_std": dQ2dRho_std,
                "Count Rho_S Q": count_RhoQ,
                "dr1dRho_SdV1Correlation": dr1dRhodV1,
                "dr1dRho_SdV1Correlation_std": dr1dRhodV1_std,
                "dr1dRho_SdV1Count": dr1dRhodV1total,
                "dr1dRho_SdV2Correlation": dr1dRhodV2,
                "dr1dRho_SdV2Correlation_std": dr1dRhodV2_std,
                "dr1dRho_SdV2Count": dr1dRhodV2total,
                "dr2dRho_SdV1Correlation": dr2dRhodV1,
                "dr2dRho_SdV1Correlation_std": dr2dRhodV1_std,
                "dr2dRho_SdV1Count": dr2dRhodV1total,
                "dr2dRho_SdV2Correlation": dr2dRhodV2,
                "dr2dRho_SdV2Correlation_std": dr2dRhodV2_std,
                "dr2dRho_SdV2Count": dr2dRhodV2total,
                "dr1dQ1dV1Correlation": dr1dQ1dV1,
                "dr1dQ1dV1Correlation_std": dr1dQ1dV1_std,
                "dr1dQ1dV1Count": dr1dQ1dV1total,
                "dr1dQ1dV2Correlation": dr1dQ1dV2,
                "dr1dQ1dV2Correlation_std": dr1dQ1dV2_std,
                "dr1dQ1dV2Count": dr1dQ1dV2total,
                "dr2dQ1dV1Correlation": dr2dQ1dV1,
                "dr2dQ1dV1Correlation_std": dr2dQ1dV1_std,
                "dr2dQ1dV1Count": dr2dQ1dV1total,
                "dr2dQ1dV2Correlation": dr2dQ1dV2,
                "dr2dQ1dV2Correlation_std": dr2dQ1dV2_std,
                "dr2dQ1dV2Count": dr2dQ1dV2total,
                "dP1dV1Correlation": dP1dV1,
                "dP1dV1Correlation_std": dP1dV1_std,
                "dP1dV1Count": dP1dV1total,
                "dP2dV2Correlation": dP2dV2,
                "dP2dV2Correlation_std": dP2dV2_std,
                "dP2dV2Count": dP2dV2total,
                "dV1dV1Correlation": dV1dV1,
                "dV1dV1Correlation_std": dV1dV1_std,
                "dV1dV1Count": dV1dV1total,
                "dV2dV2Correlation": dV2dV2,
                "dV2dV2Correlation_std": dV2dV2_std,
                "dV2dV2Count": dV2dV2total,
            }
        )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/dfCorrelations{fileType}.pkl")
