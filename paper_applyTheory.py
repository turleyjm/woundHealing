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


# --------- Unwounded at different timepoints ----------

filenames, fileType = util.getFilesType("Unwound18h")
timepoints = [[0,14],[15,29],[30,44],[45,59],[60,74],[75,89]]

# space time cell-cell shape correlation
if False:
    for times in timepoints:
        dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
        grid = 20
        timeGrid = 15

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
            path_to_file = f"databases/postWoundPaperCorrelations/dfCorMidway{filename}_T=[{times}].pkl"
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
                dfTimepoint = dfShapef[
                    np.array(dfShapef["T"] >= times[0]) & np.array(dfShapef["T"] < times[1])
                ]
                n = int(len(dfTimepoint) / 2)
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

                    x = dfTimepoint["X"].iloc[i]
                    y = dfTimepoint["Y"].iloc[i]
                    t = dfTimepoint["T"].iloc[i]
                    dq1 = dfTimepoint["dq"].iloc[i][0, 0]
                    dq2 = dfTimepoint["dq"].iloc[i][0, 1]
                    dfTimepoint.loc[:, "dR"] = (
                        (
                            (dfTimepoint.loc[:, "X"] - x) ** 2
                            + (dfTimepoint.loc[:, "Y"] - y) ** 2
                        )
                        ** 0.5
                    ).copy()
                    df = dfTimepoint[
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
                    f"databases/postWoundPaperCorrelations/dfCorMidway{filename}_T=[{times}].pkl"
                )

# collect all correlations
if False:
    _df = []
    for filename in filenames:
        for times in timepoints:
            dfCorMid = pd.read_pickle(
                f"databases/postWoundPaperCorrelations/dfCorMidway{filename}_T=[{times}].pkl"
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

            _df.append(
                {
                    "Filename": filename,
                    "Timepoints": times,
                    "dQ1dQ1Correlation": dQ1dQ1,
                    "dQ1dQ1Correlation_std": dQ1dQ1_std,
                    "dQ1dQ1Count": dQ1dQ1total,
                    "dQ2dQ2Correlation": dQ2dQ2,
                    "dQ2dQ2Correlation_std": dQ2dQ2_std,
                    "dQ2dQ2Count": dQ2dQ2total,
                }
            )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/postWoundPaperCorrelations/dfCorrelations_timepoints{fileType}.pkl")

timepoints = [[0,29],[30,59],[60,89]]
# space time cell density correlation
if True:
    for times in timepoints:
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
                dft = df[df["T"] == t + times[0]]
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
            df.to_pickle(f"databases/postWoundPaperCorrelations/dfCorRho{filename}_T=[{times}].pkl")

# collect all correlations
if True:
    _df = []
    for filename in filenames:
        for times in timepoints:
            dfCorRho = pd.read_pickle(
                f"databases/postWoundPaperCorrelations/dfCorRho{filename}.pkl"
            )

            dRhodRho = np.nan_to_num(dfCorRho["dRhodRhoCorrelation"].iloc[0])
            dRhodRho_std = np.nan_to_num(dfCorRho["dRhodRhoCorrelation_std"].iloc[0])
            count_Rho = np.nan_to_num(dfCorRho["Count"].iloc[0])

            _df.append(
                {
                    "Filename": filename,
                    "Timepoints": times,
                    "dRho_SdRho_S": dRhodRho,
                    "dRho_SdRho_S_std": dRhodRho_std,
                    "Count Rho_S": count_Rho,
                }
            )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/postWoundPaperCorrelations/dfCorrelations_timepoints{fileType}.pkl")


# --------- Visualise wounded ----------

# Far and close to wound
if False:
    filename = "WoundL18h10"
    focus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)
    (T, X, Y, rgb) = focus.shape
    dist = sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)
    close = np.zeros([X, Y])
    far = np.zeros([X, Y])

    close[(dist[45] < 40 / scale) & (dist[45] > 0)] = 128
    far[dist[45] > 40 / scale] = 128

    close = np.asarray(close, "uint8")
    tifffile.imwrite(f"results/displayProperties/close{filename}.tif", close)
    far = np.asarray(far, "uint8")
    tifffile.imwrite(f"results/displayProperties/far{filename}.tif", far)

    filename = "WoundLJNK01"
    focus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)
    (T, X, Y, rgb) = focus.shape
    dist = sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)
    close = np.zeros([X, Y])
    far = np.zeros([X, Y])

    close[(dist[45] < 40 / scale) & (dist[45] > 0)] = 128
    far[dist[45] > 40 / scale] = 128

    close = np.asarray(close, "uint8")
    tifffile.imwrite(f"results/displayProperties/close{filename}.tif", close)
    far = np.asarray(far, "uint8")
    tifffile.imwrite(f"results/displayProperties/far{filename}.tif", far)


# --------- Correlation graphs around woundsites ----------

filenames, fileType = util.getFilesType("Unwound18h")

fileTypes = ["Unwound18h", "UnwoundJNK"]
# display all main correlations shape
if False:
    for fileType in fileTypes:
        filenames, fileType = util.getFilesType(fileType)
        dfCor = pd.read_pickle(
            f"databases/postWoundPaperCorrelations/dfCorrelations{fileType}.pkl"
        )

        fig, ax = plt.subplots(1, 4, figsize=(16, 3))

        T, R, Theta = dfCor["dRho_SdRho_S"].iloc[0].shape

        dRhodRho = np.zeros([len(filenames), T, R])
        for i in range(len(filenames)):
            RhoCount = dfCor["Count Rho_S"].iloc[i][:, :, :-1]
            dRhodRho[i] = np.sum(
                dfCor["dRho_SdRho_S"].iloc[i][:, :, :-1] * RhoCount, axis=2
            ) / np.sum(RhoCount, axis=2)

        dRhodRho = np.mean(dRhodRho, axis=0)

        maxCorr = 0.003
        t, r = np.mgrid[0:60:10, 0:50:10]
        c = ax[0].pcolor(
            t,
            r,
            dRhodRho,
            cmap="RdBu_r",
            vmin=-maxCorr,
            vmax=maxCorr,
            shading="auto",
        )
        cbar = fig.colorbar(c, ax=ax[0])
        cbar.formatter.set_powerlimits((0, 0))
        ax[0].set_xlabel("Time apart $T$ (min)")
        ax[0].set_ylabel(r"Distance apart $R$ $(\mu m)$")
        ax[0].set_title(r"$C_{\rho\rho}$ healthy", y=1.1)

        dRhodRho_n = dRhodRho - np.mean(dRhodRho[4:], axis=0)
        dRhodRho_n[0, 0] = np.nan
        c = ax[1].pcolor(
            t,
            r,
            dRhodRho_n,
            cmap="RdBu_r",
            vmin=-maxCorr / 10,
            vmax=maxCorr / 10,
            shading="auto",
        )
        cbar = fig.colorbar(c, ax=ax[1])
        cbar.formatter.set_powerlimits((0, 0))
        ax[1].set_xlabel("Time apart $T$ (min)")
        ax[1].set_ylabel(r"Distance apart $R$ $(\mu m)$")
        ax[1].set_title(r"$C^{nn}_{\rho\rho}$ healthy", y=1.1)

        T, R, Theta = dfCor["dQ1dQ1Correlation"].iloc[0].shape

        dQ1dQ1 = np.zeros([len(filenames), T, R - 1])
        dQ2dQ2 = np.zeros([len(filenames), T, R - 1])
        for i in range(len(filenames)):
            dQ1dQ1total = dfCor["dQ1dQ1Count"].iloc[i][:, :-1, :-1]
            dQ1dQ1[i] = np.sum(
                dfCor["dQ1dQ1Correlation"].iloc[i][:, :-1, :-1] * dQ1dQ1total, axis=2
            ) / np.sum(dQ1dQ1total, axis=2)

            dQ2dQ2total = dfCor["dQ2dQ2Count"].iloc[i][:, :-1, :-1]
            dQ2dQ2[i] = np.sum(
                dfCor["dQ2dQ2Correlation"].iloc[i][:, :-1, :-1] * dQ2dQ2total, axis=2
            ) / np.sum(dQ2dQ2total, axis=2)

        dQ1dQ1 = np.mean(dQ1dQ1, axis=0)
        dQ2dQ2 = np.mean(dQ2dQ2, axis=0)

        maxCorr = 0.0007
        t, r = np.mgrid[0:60:2, 0:38:2]
        c = ax[2].pcolor(
            t,
            r,
            dQ1dQ1,
            cmap="RdBu_r",
            vmin=-maxCorr,
            vmax=maxCorr,
            shading="auto",
        )
        cbar = fig.colorbar(c, ax=ax[2])
        cbar.formatter.set_powerlimits((0, 0))
        ax[2].set_xlabel("Time apart $T$ (min)")
        ax[2].set_ylabel(r"Distance apart $R$ $(\mu m)$")
        ax[2].set_title(r"$C^{11}_{qq}$ healthy", y=1.1)

        maxCorr = 0.0007
        c = ax[3].pcolor(
            t,
            r,
            dQ2dQ2,
            cmap="RdBu_r",
            vmin=-maxCorr,
            vmax=maxCorr,
            shading="auto",
        )
        cbar = fig.colorbar(c, ax=ax[3])
        cbar.formatter.set_powerlimits((0, 0))
        ax[3].set_xlabel("Time apart $T$ (min)")
        ax[3].set_ylabel(r"Distance apart $R$ $(\mu m)$")
        ax[3].set_title(r"$C^{22}_{qq}$ healthy", y=1.1)

        # plt.subplot_tool()
        plt.subplots_adjust(
            left=0.075, bottom=0.1, right=0.95, top=0.9, wspace=0.35, hspace=0.55
        )

        fig.savefig(
            f"results/mathPostWoundPaper/Main correlations {fileType}",
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )
        plt.close("all")

fileTypes = ["WoundL18h", "WoundLJNK"]
# display all main correlations shape near and far
if False:
    for fileType in fileTypes:
        filenames, fileType = util.getFilesType(fileType)
        dfCor = pd.read_pickle(
            f"databases/postWoundPaperCorrelations/dfCorrelations{fileType}.pkl"
        )

        fig, ax = plt.subplots(2, 4, figsize=(16, 8))

        T, R, Theta = dfCor["dRhodRhoClose"].iloc[0].shape

        dRhodRhoClose = np.zeros([len(filenames), T, R])
        dRhodRhoFar = np.zeros([len(filenames), T, R])
        for i in range(len(filenames)):
            dRhodRhoClosetotal = dfCor["dRhodRhoClosetotal"].iloc[i][:, :, :-1]
            dRhodRhoClose[i] = np.sum(
                dfCor["dRhodRhoClose"].iloc[i][:, :, :-1] * dRhodRhoClosetotal, axis=2
            ) / np.sum(dRhodRhoClosetotal, axis=2)

            dRhodRhoFartotal = dfCor["dRhodRhoFartotal"].iloc[i][:, :, :-1]
            dRhodRhoFar[i] = np.sum(
                dfCor["dRhodRhoFar"].iloc[i][:, :, :-1] * dRhodRhoFartotal, axis=2
            ) / np.sum(dRhodRhoFartotal, axis=2)

        dRhodRhoClose = np.mean(dRhodRhoClose, axis=0)
        dRhodRhoFar = np.mean(dRhodRhoFar, axis=0)

        maxCorr = 0.003
        t, r = np.mgrid[0:60:10, 0:50:10]
        c = ax[0, 0].pcolor(
            t,
            r,
            dRhodRhoClose,
            cmap="RdBu_r",
            vmin=-maxCorr,
            vmax=maxCorr,
            shading="auto",
        )
        cbar = fig.colorbar(c, ax=ax[0, 0])
        cbar.formatter.set_powerlimits((0, 0))
        ax[0, 0].set_xlabel("Time apart $T$ (min)")
        ax[0, 0].set_ylabel(r"Distance apart $R$ $(\mu m)$")
        ax[0, 0].set_title(r"$C_{\rho\rho}$ close to wounds", y=1.1)

        dRhodRhoClose_n = dRhodRhoClose - np.mean(dRhodRhoClose[4:], axis=0)
        dRhodRhoClose_n[0, 0] = np.nan
        c = ax[0, 1].pcolor(
            t,
            r,
            dRhodRhoClose_n,
            cmap="RdBu_r",
            vmin=-maxCorr / 10,
            vmax=maxCorr / 10,
            shading="auto",
        )
        cbar = fig.colorbar(c, ax=ax[0, 1])
        cbar.formatter.set_powerlimits((0, 0))
        ax[0, 1].set_xlabel("Time apart $T$ (min)")
        ax[0, 1].set_ylabel(r"Distance apart $R$ $(\mu m)$")
        ax[0, 1].set_title(r"$C^{nn}_{\rho\rho}$ close to wounds", y=1.1)

        t, r = np.mgrid[0:60:10, 0:50:10]
        c = ax[1, 0].pcolor(
            t,
            r,
            dRhodRhoFar,
            cmap="RdBu_r",
            vmin=-maxCorr,
            vmax=maxCorr,
            shading="auto",
        )
        cbar = fig.colorbar(c, ax=ax[1, 0])
        cbar.formatter.set_powerlimits((0, 0))
        ax[1, 0].set_xlabel("Time apart $T$ (min)")
        ax[1, 0].set_ylabel(r"Distance apart $R$ $(\mu m)$")
        ax[1, 0].set_title(r"$C_{\rho\rho}$ far from wounds", y=1.1)

        dRhodRhoFar_n = dRhodRhoFar - np.mean(dRhodRhoFar[4:], axis=0)
        dRhodRhoFar_n[0, 0] = np.nan
        c = ax[1, 1].pcolor(
            t,
            r,
            dRhodRhoFar_n,
            cmap="RdBu_r",
            vmin=-maxCorr / 10,
            vmax=maxCorr / 10,
            shading="auto",
        )
        cbar = fig.colorbar(c, ax=ax[1, 1])
        cbar.formatter.set_powerlimits((0, 0))
        ax[1, 1].set_xlabel("Time apart $T$ (min)")
        ax[1, 1].set_ylabel(r"Distance apart $R$ $(\mu m)$")
        ax[1, 1].set_title(r"$C^{nn}_{\rho\rho}$ far from wounds", y=1.1)

        T, R, Theta = dfCor["dQ1dQ1Close"].iloc[0].shape

        dQ1dQ1Close = np.zeros([len(filenames), T, R - 1])
        dQ1dQ1Far = np.zeros([len(filenames), T, R - 1])
        dQ2dQ2Close = np.zeros([len(filenames), T, R - 1])
        dQ2dQ2Far = np.zeros([len(filenames), T, R - 1])
        for i in range(len(filenames)):
            dQ1dQ1Closetotal = dfCor["dQ1dQ1Closetotal"].iloc[i][:, :-1, :-1]
            dQ1dQ1Close[i] = np.sum(
                dfCor["dQ1dQ1Close"].iloc[i][:, :-1, :-1] * dQ1dQ1Closetotal, axis=2
            ) / np.sum(dQ1dQ1Closetotal, axis=2)

            dQ1dQ1Fartotal = dfCor["dQ1dQ1Fartotal"].iloc[i][:, :-1, :-1]
            dQ1dQ1Far[i] = np.sum(
                dfCor["dQ1dQ1Far"].iloc[i][:, :-1, :-1] * dQ1dQ1Fartotal, axis=2
            ) / np.sum(dQ1dQ1Fartotal, axis=2)

            dQ2dQ2Closetotal = dfCor["dQ2dQ2Closetotal"].iloc[i][:, :-1, :-1]
            dQ2dQ2Close[i] = np.sum(
                dfCor["dQ2dQ2Close"].iloc[i][:, :-1, :-1] * dQ2dQ2Closetotal, axis=2
            ) / np.sum(dQ2dQ2Closetotal, axis=2)

            dQ2dQ2Fartotal = dfCor["dQ2dQ2Fartotal"].iloc[i][:, :-1, :-1]
            dQ2dQ2Far[i] = np.sum(
                dfCor["dQ2dQ2Far"].iloc[i][:, :-1, :-1] * dQ2dQ2Fartotal, axis=2
            ) / np.sum(dQ2dQ2Fartotal, axis=2)

        dQ1dQ1Close = np.mean(dQ1dQ1Close, axis=0)
        dQ1dQ1Far = np.mean(dQ1dQ1Far, axis=0)
        dQ2dQ2Close = np.mean(dQ2dQ2Close, axis=0)
        dQ2dQ2Far = np.mean(dQ2dQ2Far, axis=0)

        maxCorr = 0.0007
        t, r = np.mgrid[0:60:2, 0:38:2]
        c = ax[0, 2].pcolor(
            t,
            r,
            dQ1dQ1Close,
            cmap="RdBu_r",
            vmin=-maxCorr,
            vmax=maxCorr,
            shading="auto",
        )
        cbar = fig.colorbar(c, ax=ax[0, 2])
        cbar.formatter.set_powerlimits((0, 0))
        ax[0, 2].set_xlabel("Time apart $T$ (min)")
        ax[0, 2].set_ylabel(r"Distance apart $R$ $(\mu m)$")
        ax[0, 2].set_title(r"$C^{11}_{qq}$ close to wounds", y=1.1)

        c = ax[1, 2].pcolor(
            t,
            r,
            dQ1dQ1Far,
            cmap="RdBu_r",
            vmin=-maxCorr,
            vmax=maxCorr,
            shading="auto",
        )
        cbar = fig.colorbar(c, ax=ax[1, 2])
        cbar.formatter.set_powerlimits((0, 0))
        ax[1, 2].set_xlabel("Time apart $T$ (min)")
        ax[1, 2].set_ylabel(r"Distance apart $R$ $(\mu m)$")
        ax[1, 2].set_title(r"$C^{11}_{qq}$ far from wounds", y=1.1)

        c = ax[0, 3].pcolor(
            t,
            r,
            dQ2dQ2Close,
            cmap="RdBu_r",
            vmin=-maxCorr,
            vmax=maxCorr,
            shading="auto",
        )
        cbar = fig.colorbar(c, ax=ax[0, 3])
        cbar.formatter.set_powerlimits((0, 0))
        ax[0, 3].set_xlabel("Time apart $T$ (min)")
        ax[0, 3].set_ylabel(r"Distance apart $R$ $(\mu m)$")
        ax[0, 3].set_title(r"$C^{22}_{qq}$ close to wounds", y=1.1)

        c = ax[1, 3].pcolor(
            t,
            r,
            dQ2dQ2Far,
            cmap="RdBu_r",
            vmin=-maxCorr,
            vmax=maxCorr,
            shading="auto",
        )
        cbar = fig.colorbar(c, ax=ax[1, 3])
        cbar.formatter.set_powerlimits((0, 0))
        ax[1, 3].set_xlabel("Time apart $T$ (min)")
        ax[1, 3].set_ylabel(r"Distance apart $R$ $(\mu m)$")
        ax[1, 3].set_title(r"$C^{22}_{qq}$ far from wounds", y=1.1)


        # plt.subplot_tool()
        plt.subplots_adjust(
            left=0.075, bottom=0.1, right=0.95, top=0.9, wspace=0.35, hspace=0.55
        )

        fig.savefig(
            f"results/mathPostWoundPaper/Main correlations {fileType}",
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )
        plt.close("all")

# --------- Fit data post-wound ----------

grid = 19
timeGrid = 30
mlist = []
mlist_JNK = []
fileTypes = ["WoundL18h", "WoundLJNK"]

# Close to wounds
if False:

    def Corr_dQ_Integral_T(R, B, L):
        C = 0.00055
        T = 0
        k = np.linspace(0, 4, 200000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, C, L) * h, axis=0)[:, 0]

    def Corr_dQ_Integral_R(T, B, L):
        C = 0.00055
        R = 0
        k = np.linspace(0, 4, 200000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, C, L) * h, axis=0)[0]


    dfCor = pd.read_pickle(f"databases/postWoundPaperCorrelations/dfCorrelations{fileTypes[0]}.pkl")
    filenames, fileType = util.getFilesType(fileTypes[0])

    T, R, Theta = dfCor["dQ1dQ1Close"].iloc[0].shape

    dQ1dQ1Close = np.zeros([len(filenames), T-1, R - 1])
    dQ2dQ2Close = np.zeros([len(filenames), T-1, R - 1])
    for i in range(len(filenames)):
        dQ1dQ1Closetotal = dfCor["dQ1dQ1Closetotal"].iloc[i][:-1, :-1, :-1]
        dQ1dQ1Close[i] = np.sum(
            dfCor["dQ1dQ1Close"].iloc[i][:-1, :-1, :-1] * dQ1dQ1Closetotal, axis=2
        ) / np.sum(dQ1dQ1Closetotal, axis=2)
        dQ2dQ2Closetotal = dfCor["dQ2dQ2Closetotal"].iloc[i][:-1, :-1, :-1]
        dQ2dQ2Close[i] = np.sum(
            dfCor["dQ2dQ2Close"].iloc[i][:-1, :-1, :-1] * dQ2dQ2Closetotal, axis=2
        ) / np.sum(dQ2dQ2Closetotal, axis=2)

    dfCor = 0

    dQ1dQ1Close = np.mean(dQ1dQ1Close, axis=0)
    dQ2dQ2Close = np.mean(dQ2dQ2Close, axis=0)

    dfCor = pd.read_pickle(f"databases/postWoundPaperCorrelations/dfCorrelations{fileTypes[1]}.pkl")
    filenames, fileType = util.getFilesType(fileTypes[1])

    dQ1dQ1Close_JNK = np.zeros([len(filenames), T-1, R - 1])
    dQ2dQ2Close_JNK = np.zeros([len(filenames), T-1, R - 1])
    for i in range(len(filenames)):
        dQ1dQ1Closetotal = dfCor["dQ1dQ1Closetotal"].iloc[i][:-1, :-1, :-1]
        dQ1dQ1Close_JNK[i] = np.sum(
            dfCor["dQ1dQ1Close"].iloc[i][:-1, :-1, :-1] * dQ1dQ1Closetotal, axis=2
        ) / np.sum(dQ1dQ1Closetotal, axis=2)
        dQ2dQ2Closetotal = dfCor["dQ2dQ2Closetotal"].iloc[i][:-1, :-1, :-1]
        dQ2dQ2Close_JNK[i] = np.sum(
            dfCor["dQ2dQ2Close"].iloc[i][:-1, :-1, :-1] * dQ2dQ2Closetotal, axis=2
        ) / np.sum(dQ2dQ2Closetotal, axis=2)

    dfCor = 0

    dQ1dQ1Close_JNK = np.mean(dQ1dQ1Close_JNK, axis=0)
    dQ2dQ2Close_JNK = np.mean(dQ2dQ2Close_JNK, axis=0)

    T = np.linspace(0, 2 * (timeGrid - 2), timeGrid-1)
    R = np.linspace(0, 2 * (grid - 1), grid)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    # print("dQ1")

    m = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_R,
        xdata=T[1:],
        ydata=dQ1dQ1Close[:, 0][1:],
        p0=(0.006, 4),
    )[0]
    # print(float(m[0]), float(m[1]))
    mlist.append(m)

    m_JNK = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_R,
        xdata=T[1:],
        ydata=dQ1dQ1Close_JNK[:, 0][1:],
        p0=(0.006, 4),
    )[0]
    # print(float(m_JNK[0]), float(m_JNK[1]))
    mlist_JNK.append(m_JNK)

    ax[0].plot(T[1:], dQ1dQ1Close[:, 0][1:], label="control", color="m", marker="o")
    ax[0].plot(T[1:], Corr_dQ_Integral_R(T[1:], m[0], m[1]), label="Model", color="m")
    ax[0].plot(T[1:], dQ1dQ1Close_JNK[:, 0][1:], label="bsk DN", color="g", marker="o")
    ax[0].plot(T[1:], Corr_dQ_Integral_R(T[1:], m_JNK[0], m_JNK[1]), label="Model", color="g")
    ax[0].set_xlabel("Time apart $T$ (min)")
    ax[0].set_ylabel(r"$\delta q^{(1)}$ Correlation")
    ax[0].set_ylim([8e-5, 5.6e-04])
    ax[0].set_title(r"$C^{11}_{qq}(0,T)$")
    ax[0].legend(fontsize=10)
    ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    m = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_T,
        xdata=R[1:],
        ydata=dQ1dQ1Close[0][1:],
        p0=(0.006, 4),
        method="lm",
    )[0]
    # print(float(m[0]), float(m[1]))
    mlist.append(m)

    m_JNK = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_T,
        xdata=R[1:],
        ydata=dQ1dQ1Close_JNK[0][1:],
        p0=(0.006, 4),
        method="lm",
    )[0]
    # print(float(m_JNK[0]), float(m_JNK[1]))
    mlist_JNK.append(m_JNK)

    ax[1].plot(R[1:], dQ1dQ1Close[0][1:], label="control", color="m", marker="o")
    ax[1].plot(R[1:], Corr_dQ_Integral_T(R[1:], m[0], m[1]), label="Model", color="m")
    ax[1].plot(R[1:], dQ1dQ1Close_JNK[0][1:], label="bsk DN", color="g", marker="o")
    ax[1].plot(R[1:], Corr_dQ_Integral_T(R[1:], m_JNK[0], m_JNK[1]), label="Model", color="g")
    ax[1].set_xlabel(r"Distance apart $R$ $(\mu m)$")
    ax[1].set_ylabel(r"$\delta q^{(1)}$ Correlation")
    ax[1].set_ylim([-4e-5, 2.8e-04])
    ax[1].set_title(r"$C^{11}_{qq}(R,0)$")
    ax[1].legend(fontsize=10)
    ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.subplots_adjust(
        left=0.08, bottom=0.1, right=0.92, top=0.9, wspace=0.35, hspace=0.50
    )
    fig.savefig(
        f"results/mathPostWoundPaper/Correlation dQ1 control and JNK close to wounds",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    # print("dQ2")

    m = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_R,
        xdata=T[1:],
        ydata=dQ2dQ2Close[:, 0][1:],
        p0=(0.006, 4),
    )[0]
    # print(float(m[0]), float(m[1]))
    mlist.append(m)

    m_JNK = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_R,
        xdata=T[1:],
        ydata=dQ2dQ2Close_JNK[:, 0][1:],
        p0=(0.006, 4),
    )[0]
    # print(float(m_JNK[0]), float(m_JNK[1]))
    mlist_JNK.append(m_JNK)

    ax[0].plot(T[1:], dQ2dQ2Close[:, 0][1:], label="control", color="m", marker="o")
    ax[0].plot(T[1:], Corr_dQ_Integral_R(T[1:], m[0], m[1]), label="Model", color="m")
    ax[0].plot(T[1:], dQ2dQ2Close_JNK[:, 0][1:], label="bsk DN", color="g", marker="o")
    ax[0].plot(T[1:], Corr_dQ_Integral_R(T[1:], m_JNK[0], m_JNK[1]), label="Model", color="g")
    ax[0].set_xlabel("Time apart $T$ (min)")
    ax[0].set_ylabel(r"$\delta q^{(2)}$ Correlation")
    ax[0].set_ylim([8e-5, 5.6e-04])
    ax[0].set_title(r"$C^{22}_{qq}(0,T)$")
    ax[0].legend(fontsize=10)
    ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    m = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_T,
        xdata=R[1:],
        ydata=dQ2dQ2Close[0][1:],
        p0=(0.006, 4),
        method="lm",
    )[0]
    # print(float(m[0]), float(m[1]))
    mlist.append(m)

    m_JNK = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_T,
        xdata=R[1:],
        ydata=dQ2dQ2Close_JNK[0][1:],
        p0=(0.006, 4),
        method="lm",
    )[0]
    # print(float(m_JNK[0]), float(m_JNK[1]))
    mlist_JNK.append(m_JNK)

    ax[1].plot(R[1:], dQ2dQ2Close[0][1:], label="control", color="m", marker="o")
    ax[1].plot(R[1:], Corr_dQ_Integral_T(R[1:], m[0], m[1]), label="Model", color="m")
    ax[1].plot(R[1:], dQ2dQ2Close_JNK[0][1:], label="bsk DN", color="g", marker="o")
    ax[1].plot(R[1:], Corr_dQ_Integral_T(R[1:], m_JNK[0], m_JNK[1]), label="Model", color="g")
    ax[1].set_xlabel(r"Distance apart $R$ $(\mu m)$")
    ax[1].set_ylabel(r"$\delta q^{(2)}$ Correlation")
    ax[1].set_ylim([-4e-5, 2.8e-04])
    ax[1].set_title(r"$C^{22}_{qq}(R,0)$")
    ax[1].legend(fontsize=10)
    ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.subplots_adjust(
        left=0.08, bottom=0.1, right=0.92, top=0.9, wspace=0.35, hspace=0.50
    )
    fig.savefig(
        f"results/mathPostWoundPaper/Correlation dQ2 control and JNK close to wounds",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

    print("B    L")
    print("control - close")
    print(np.mean(mlist, axis=0)[0], np.mean(mlist, axis=0)[1])
    print("JNK - close")
    print(np.mean(mlist_JNK, axis=0)[0], np.mean(mlist_JNK, axis=0)[1])

mlist = []
mlist_JNK = []

# Far to wounds
if False:

    def Corr_dQ_Integral_T(R, B, L):
        C = 0.00055
        T = 0
        k = np.linspace(0, 4, 200000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, C, L) * h, axis=0)[:, 0]

    def Corr_dQ_Integral_R(T, B, L):
        C = 0.00055
        R = 0
        k = np.linspace(0, 4, 200000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, C, L) * h, axis=0)[0]


    dfCor = pd.read_pickle(f"databases/postWoundPaperCorrelations/dfCorrelations{fileTypes[0]}.pkl")
    filenames, fileType = util.getFilesType(fileTypes[0])

    T, R, Theta = dfCor["dQ1dQ1Far"].iloc[0].shape

    dQ1dQ1Far = np.zeros([len(filenames), T-1, R - 1])
    dQ2dQ2Far = np.zeros([len(filenames), T-1, R - 1])
    for i in range(len(filenames)):
        dQ1dQ1Fartotal = dfCor["dQ1dQ1Fartotal"].iloc[i][:-1, :-1, :-1]
        dQ1dQ1Far[i] = np.sum(
            dfCor["dQ1dQ1Far"].iloc[i][:-1, :-1, :-1] * dQ1dQ1Fartotal, axis=2
        ) / np.sum(dQ1dQ1Fartotal, axis=2)
        dQ2dQ2Fartotal = dfCor["dQ2dQ2Fartotal"].iloc[i][:-1, :-1, :-1]
        dQ2dQ2Far[i] = np.sum(
            dfCor["dQ2dQ2Far"].iloc[i][:-1, :-1, :-1] * dQ2dQ2Fartotal, axis=2
        ) / np.sum(dQ2dQ2Fartotal, axis=2)

    dfCor = 0

    dQ1dQ1Far = np.mean(dQ1dQ1Far, axis=0)
    dQ2dQ2Far = np.mean(dQ2dQ2Far, axis=0)

    dfCor = pd.read_pickle(f"databases/postWoundPaperCorrelations/dfCorrelations{fileTypes[1]}.pkl")
    filenames, fileType = util.getFilesType(fileTypes[1])

    dQ1dQ1Far_JNK = np.zeros([len(filenames), T-1, R - 1])
    dQ2dQ2Far_JNK = np.zeros([len(filenames), T-1, R - 1])
    for i in range(len(filenames)):
        dQ1dQ1Fartotal = dfCor["dQ1dQ1Fartotal"].iloc[i][:-1, :-1, :-1]
        dQ1dQ1Far_JNK[i] = np.sum(
            dfCor["dQ1dQ1Far"].iloc[i][:-1, :-1, :-1] * dQ1dQ1Fartotal, axis=2
        ) / np.sum(dQ1dQ1Fartotal, axis=2)
        dQ2dQ2Fartotal = dfCor["dQ2dQ2Fartotal"].iloc[i][:-1, :-1, :-1]
        dQ2dQ2Far_JNK[i] = np.sum(
            dfCor["dQ2dQ2Far"].iloc[i][:-1, :-1, :-1] * dQ2dQ2Fartotal, axis=2
        ) / np.sum(dQ2dQ2Fartotal, axis=2)

    dfCor = 0

    dQ1dQ1Far_JNK = np.mean(dQ1dQ1Far_JNK, axis=0)
    dQ2dQ2Far_JNK = np.mean(dQ2dQ2Far_JNK, axis=0)

    T = np.linspace(0, 2 * (timeGrid - 2), timeGrid-1)
    R = np.linspace(0, 2 * (grid - 1), grid)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    m = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_R,
        xdata=T[1:],
        ydata=dQ1dQ1Far[:, 0][1:],
        p0=(0.006, 4),
    )[0]
    # print(float(m[0]), float(m[1]))
    mlist.append(m)

    m_JNK = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_R,
        xdata=T[1:],
        ydata=dQ1dQ1Far_JNK[:, 0][1:],
        p0=(0.006, 4),
    )[0]
    # print(float(m_JNK[0]), float(m_JNK[1]))
    mlist_JNK.append(m_JNK)

    ax[0].plot(T[1:], dQ1dQ1Far[:, 0][1:], label="control", color="m", marker="o")
    ax[0].plot(T[1:], Corr_dQ_Integral_R(T[1:], m[0], m[1]), label="Model", color="m")
    ax[0].plot(T[1:], dQ1dQ1Far_JNK[:, 0][1:], label="bsk DN", color="g", marker="o")
    ax[0].plot(T[1:], Corr_dQ_Integral_R(T[1:], m_JNK[0], m_JNK[1]), label="Model", color="g")
    ax[0].set_xlabel("Time apart $T$ (min)")
    ax[0].set_ylabel(r"$\delta q^{(1)}$ Correlation")
    ax[0].set_ylim([8e-5, 5.6e-04])
    ax[0].set_title(r"$C^{11}_{qq}(0,T)$")
    ax[0].legend(fontsize=10)
    ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    m = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_T,
        xdata=R[1:],
        ydata=dQ1dQ1Far[0][1:],
        p0=(0.006, 4),
        method="lm",
    )[0]
    # print(float(m[0]), float(m[1]))
    mlist.append(m)

    m_JNK = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_T,
        xdata=R[1:],
        ydata=dQ1dQ1Far_JNK[0][1:],
        p0=(0.006, 4),
        method="lm",
    )[0]
    # print(float(m_JNK[0]), float(m_JNK[1]))
    mlist_JNK.append(m_JNK)

    ax[1].plot(R[1:], dQ1dQ1Far[0][1:], label="control", color="m", marker="o")
    ax[1].plot(R[1:], Corr_dQ_Integral_T(R[1:], m[0], m[1]), label="Model", color="m")
    ax[1].plot(R[1:], dQ1dQ1Far_JNK[0][1:], label="bsk DN", color="g", marker="o")
    ax[1].plot(R[1:], Corr_dQ_Integral_T(R[1:], m_JNK[0], m_JNK[1]), label="Model", color="g")
    ax[1].set_xlabel(r"Distance apart $R$ $(\mu m)$")
    ax[1].set_ylabel(r"$\delta q^{(1)}$ Correlation")
    ax[1].set_ylim([-4e-5, 2.8e-04])
    ax[1].set_title(r"$C^{11}_{qq}(R,0)$")
    ax[1].legend(fontsize=10)
    ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.subplots_adjust(
        left=0.08, bottom=0.1, right=0.92, top=0.9, wspace=0.35, hspace=0.50
    )
    fig.savefig(
        f"results/mathPostWoundPaper/Correlation dQ1 control and JNK far from wounds",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    # print("dQ2")

    m = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_R,
        xdata=T[1:],
        ydata=dQ2dQ2Far[:, 0][1:],
        p0=(0.006, 4),
    )[0]
    # print(float(m[0]), float(m[1]))
    mlist.append(m)

    m_JNK = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_R,
        xdata=T[1:],
        ydata=dQ2dQ2Far_JNK[:, 0][1:],
        p0=(0.006, 4),
    )[0]
    # print(float(m_JNK[0]), float(m_JNK[1]))
    mlist_JNK.append(m_JNK)

    ax[0].plot(T[1:], dQ2dQ2Far[:, 0][1:], label="control", color="m", marker="o")
    ax[0].plot(T[1:], Corr_dQ_Integral_R(T[1:], m[0], m[1]), label="Model", color="m")
    ax[0].plot(T[1:], dQ2dQ2Far_JNK[:, 0][1:], label="bsk DN", color="g", marker="o")
    ax[0].plot(T[1:], Corr_dQ_Integral_R(T[1:], m_JNK[0], m_JNK[1]), label="Model", color="g")
    ax[0].set_xlabel("Time apart $T$ (min)")
    ax[0].set_ylabel(r"$\delta q^{(2)}$ Correlation")
    ax[0].set_ylim([8e-5, 5.6e-04])
    ax[0].set_title(r"$C^{22}_{qq}(0,T)$")
    ax[0].legend(fontsize=10)
    ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    m = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_T,
        xdata=R[1:],
        ydata=dQ2dQ2Far[0][1:],
        p0=(0.006, 4),
        method="lm",
    )[0]
    # print(float(m[0]), float(m[1]))
    mlist.append(m)

    m_JNK = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_T,
        xdata=R[1:],
        ydata=dQ2dQ2Far_JNK[0][1:],
        p0=(0.006, 4),
        method="lm",
    )[0]
    # print(float(m_JNK[0]), float(m_JNK[1]))
    mlist_JNK.append(m_JNK)

    ax[1].plot(R[1:], dQ2dQ2Far[0][1:], label="control", color="m", marker="o")
    ax[1].plot(R[1:], Corr_dQ_Integral_T(R[1:], m[0], m[1]), label="Model", color="m")
    ax[1].plot(R[1:], dQ2dQ2Far_JNK[0][1:], label="bsk DN", color="g", marker="o")
    ax[1].plot(R[1:], Corr_dQ_Integral_T(R[1:], m_JNK[0], m_JNK[1]), label="Model", color="g")
    ax[1].set_xlabel(r"Distance apart $R$ $(\mu m)$")
    ax[1].set_ylabel(r"$\delta q^{(2)}$ Correlation")
    ax[1].set_ylim([-4e-5, 2.8e-04])
    ax[1].set_title(r"$C^{22}_{qq}(R,0)$")
    ax[1].legend(fontsize=10)
    ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.subplots_adjust(
        left=0.08, bottom=0.1, right=0.92, top=0.9, wspace=0.35, hspace=0.50
    )
    fig.savefig(
        f"results/mathPostWoundPaper/Correlation dQ2 control and JNK far from wounds",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

    print("control - Far")
    print(np.mean(mlist, axis=0)[0], np.mean(mlist, axis=0)[1])
    print("JNK - Far")
    print(np.mean(mlist_JNK, axis=0)[0], np.mean(mlist_JNK, axis=0)[1])

mlist = []
mlist_JNK = []

fileTypes = ["Unwound18h", "UnwoundJNK"]

# Healthy wounds
if False:

    def Corr_dQ_Integral_T(R, B, L):
        C = 0.00055
        T = 0
        k = np.linspace(0, 4, 200000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, C, L) * h, axis=0)[:, 0]

    def Corr_dQ_Integral_R(T, B, L):
        C = 0.00055
        R = 0
        k = np.linspace(0, 4, 200000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, C, L) * h, axis=0)[0]
    dfCor = pd.read_pickle(f"databases/postWoundPaperCorrelations/dfCorrelationsWoundLJNK.pkl")
    T, R, Theta = dfCor["dQ1dQ1Far"].iloc[0].shape

    dfCor = pd.read_pickle(f"databases/postWoundPaperCorrelations/dfCorrelations{fileTypes[0]}.pkl")
    filenames, fileType = util.getFilesType(fileTypes[0])

    dQ1dQ1 = np.zeros([len(filenames), T-1, R - 1])
    dQ2dQ2 = np.zeros([len(filenames), T-1, R - 1])
    for i in range(len(filenames)):
        dQ1dQ1total = dfCor["dQ1dQ1Count"].iloc[i][:-1, :-1, :-1]
        dQ1dQ1[i] = np.sum(
            dfCor["dQ1dQ1Correlation"].iloc[i][:-1, :-1, :-1] * dQ1dQ1total, axis=2
        ) / np.sum(dQ1dQ1total, axis=2)
        dQ2dQ2total = dfCor["dQ2dQ2Count"].iloc[i][:-1, :-1, :-1]
        dQ2dQ2[i] = np.sum(
            dfCor["dQ2dQ2Correlation"].iloc[i][:-1, :-1, :-1] * dQ2dQ2total, axis=2
        ) / np.sum(dQ2dQ2total, axis=2)

    dfCor = 0

    dQ1dQ1 = np.mean(dQ1dQ1, axis=0)
    dQ2dQ2 = np.mean(dQ2dQ2, axis=0)

    dfCor = pd.read_pickle(f"databases/postWoundPaperCorrelations/dfCorrelations{fileTypes[1]}.pkl")
    filenames, fileType = util.getFilesType(fileTypes[1])

    dQ1dQ1_JNK = np.zeros([len(filenames), T-1, R - 1])
    dQ2dQ2_JNK = np.zeros([len(filenames), T-1, R - 1])
    for i in range(len(filenames)):
        dQ1dQ1total = dfCor["dQ1dQ1Count"].iloc[i][:-1, :-1, :-1]
        dQ1dQ1_JNK[i] = np.sum(
            dfCor["dQ1dQ1Correlation"].iloc[i][:-1, :-1, :-1] * dQ1dQ1total, axis=2
        ) / np.sum(dQ1dQ1total, axis=2)
        dQ2dQ2total = dfCor["dQ2dQ2Count"].iloc[i][:-1, :-1, :-1]
        dQ2dQ2_JNK[i] = np.sum(
            dfCor["dQ2dQ2Correlation"].iloc[i][:-1, :-1, :-1] * dQ2dQ2total, axis=2
        ) / np.sum(dQ2dQ2total, axis=2)

    dfCor = 0

    dQ1dQ1_JNK = np.mean(dQ1dQ1_JNK, axis=0)
    dQ2dQ2_JNK = np.mean(dQ2dQ2_JNK, axis=0)

    T = np.linspace(0, 2 * (timeGrid - 2), timeGrid-1)
    R = np.linspace(0, 2 * (grid - 1), grid)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    m = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_R,
        xdata=T[1:],
        ydata=dQ1dQ1[:, 0][1:],
        p0=(0.006, 4),
    )[0]
    # print(float(m[0]), float(m[1]))
    mlist.append(m)

    m_JNK = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_R,
        xdata=T[1:],
        ydata=dQ1dQ1_JNK[:, 0][1:],
        p0=(0.006, 4),
    )[0]
    # print(float(m_JNK[0]), float(m_JNK[1]))
    mlist_JNK.append(m_JNK)

    ax[0].plot(T[1:], dQ1dQ1[:, 0][1:], label="control", color="m", marker="o")
    ax[0].plot(T[1:], Corr_dQ_Integral_R(T[1:], m[0], m[1]), label="Model", color="m")
    ax[0].plot(T[1:], dQ1dQ1_JNK[:, 0][1:], label="bsk DN", color="g", marker="o")
    ax[0].plot(T[1:], Corr_dQ_Integral_R(T[1:], m_JNK[0], m_JNK[1]), label="Model", color="g")
    ax[0].set_xlabel("Time apart $T$ (min)")
    ax[0].set_ylabel(r"$\delta q^{(1)}$ Correlation")
    ax[0].set_ylim([8e-5, 5.6e-04])
    ax[0].set_title(r"$C^{11}_{qq}(0,T)$")
    ax[0].legend(fontsize=10)
    ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    m = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_T,
        xdata=R[1:],
        ydata=dQ1dQ1[0][1:],
        p0=(0.006, 4),
        method="lm",
    )[0]
    # print(float(m[0]), float(m[1]))
    mlist.append(m)

    m_JNK = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_T,
        xdata=R[1:],
        ydata=dQ1dQ1_JNK[0][1:],
        p0=(0.006, 4),
        method="lm",
    )[0]
    # print(float(m_JNK[0]), float(m_JNK[1]))
    mlist_JNK.append(m_JNK)

    ax[1].plot(R[1:], dQ1dQ1[0][1:], label="control", color="m", marker="o")
    ax[1].plot(R[1:], Corr_dQ_Integral_T(R[1:], m[0], m[1]), label="Model", color="m")
    ax[1].plot(R[1:], dQ1dQ1_JNK[0][1:], label="bsk DN", color="g", marker="o")
    ax[1].plot(R[1:], Corr_dQ_Integral_T(R[1:], m_JNK[0], m_JNK[1]), label="Model", color="g")
    ax[1].set_xlabel(r"Distance apart $R$ $(\mu m)$")
    ax[1].set_ylabel(r"$\delta q^{(1)}$ Correlation")
    ax[1].set_ylim([-4e-5, 2.8e-04])
    ax[1].set_title(r"$C^{11}_{qq}(R,0)$")
    ax[1].legend(fontsize=10)
    ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.subplots_adjust(
        left=0.08, bottom=0.1, right=0.92, top=0.9, wspace=0.35, hspace=0.50
    )
    fig.savefig(
        f"results/mathPostWoundPaper/Correlation dQ1 control and JNK healthy",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    # print("dQ2")

    m = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_R,
        xdata=T[1:],
        ydata=dQ2dQ2[:, 0][1:],
        p0=(0.006, 4),
    )[0]
    # print(float(m[0]), float(m[1]))
    mlist.append(m)

    m_JNK = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_R,
        xdata=T[1:],
        ydata=dQ2dQ2_JNK[:, 0][1:],
        p0=(0.006, 4),
    )[0]
    # print(float(m_JNK[0]), float(m_JNK[1]))
    mlist_JNK.append(m_JNK)

    ax[0].plot(T[1:], dQ2dQ2[:, 0][1:], label="control", color="m", marker="o")
    ax[0].plot(T[1:], Corr_dQ_Integral_R(T[1:], m[0], m[1]), label="Model", color="m")
    ax[0].plot(T[1:], dQ2dQ2_JNK[:, 0][1:], label="bsk DN", color="g", marker="o")
    ax[0].plot(T[1:], Corr_dQ_Integral_R(T[1:], m_JNK[0], m_JNK[1]), label="Model", color="g")
    ax[0].set_xlabel("Time apart $T$ (min)")
    ax[0].set_ylabel(r"$\delta q^{(2)}$ Correlation")
    ax[0].set_ylim([8e-5, 5.6e-04])
    ax[0].set_title(r"$C^{22}_{qq}(0,T)$")
    ax[0].legend(fontsize=10)
    ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    m = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_T,
        xdata=R[1:],
        ydata=dQ2dQ2[0][1:],
        p0=(0.006, 4),
        method="lm",
    )[0]
    # print(float(m[0]), float(m[1]))
    mlist.append(m)

    m_JNK = sp.optimize.curve_fit(
        f=Corr_dQ_Integral_T,
        xdata=R[1:],
        ydata=dQ2dQ2_JNK[0][1:],
        p0=(0.006, 4),
        method="lm",
    )[0]
    # print(float(m_JNK[0]), float(m_JNK[1]))
    mlist_JNK.append(m_JNK)

    ax[1].plot(R[1:], dQ2dQ2[0][1:], label="control", color="m", marker="o")
    ax[1].plot(R[1:], Corr_dQ_Integral_T(R[1:], m[0], m[1]), label="Model", color="m")
    ax[1].plot(R[1:], dQ2dQ2_JNK[0][1:], label="bsk DN", color="g", marker="o")
    ax[1].plot(R[1:], Corr_dQ_Integral_T(R[1:], m_JNK[0], m_JNK[1]), label="Model", color="g")
    ax[1].set_xlabel(r"Distance apart $R$ $(\mu m)$")
    ax[1].set_ylabel(r"$\delta q^{(2)}$ Correlation")
    ax[1].set_ylim([-4e-5, 2.8e-04])
    ax[1].set_title(r"$C^{22}_{qq}(R,0)$")
    ax[1].legend(fontsize=10)
    ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.subplots_adjust(
        left=0.08, bottom=0.1, right=0.92, top=0.9, wspace=0.35, hspace=0.50
    )
    fig.savefig(
        f"results/mathPostWoundPaper/Correlation dQ2 control and JNK healthy",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

    print("control - Healthy")
    print(np.mean(mlist, axis=0)[0], np.mean(mlist, axis=0)[1])
    print("JNK - Healthy")
    print(np.mean(mlist_JNK, axis=0)[0], np.mean(mlist_JNK, axis=0)[1])
