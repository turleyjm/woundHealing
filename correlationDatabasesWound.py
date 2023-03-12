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

print(datetime.now().strftime("%H:%M:%S"))
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


# --------- density ----------


# --------- shape ----------

# space time cell-cell shape correlation close to wound
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

    dfShape["dq1dq1i"] = list(np.zeros([len(dfShape)]))
    dfShape["dq2dq2i"] = list(np.zeros([len(dfShape)]))

    for k in range(len(filenames)):
        filename = filenames[k]
        dist = sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)
        path_to_file = f"databases/correlations/dfCorCloseWound{filename}.pkl"
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
                np.array(dfShapeF["T"] < 20) & np.array(dfShapeF["T"] >= 0)
            ]
            n = int(len(dfClose))
            count = 0
            for i in range(n):
                if i % int((n) / 10) == 0:
                    print(datetime.now().strftime("%H:%M:%S ") + f"{10*count}%")
                    count += 1

                x = dfClose["X"].iloc[i]
                y = dfClose["Y"].iloc[i]
                t = dfClose["T"].iloc[i]
                r = dist[t, int(512 - y), int(x)]
                if r * scale < 30:
                    dp1 = dfClose["dp"].iloc[i][0]
                    dp2 = dfClose["dp"].iloc[i][1]
                    dq1 = dfClose["dq"].iloc[i][0, 0]
                    dq2 = dfClose["dq"].iloc[i][0, 1]
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
                f"databases/correlations/dfCorCloseWound{filename}.pkl"
            )

# space time cell-cell shape correlation far from wound
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

    dfShape["dq1dq1i"] = list(np.zeros([len(dfShape)]))
    dfShape["dq2dq2i"] = list(np.zeros([len(dfShape)]))

    for k in range(len(filenames)):
        filename = filenames[k]
        path_to_file = f"databases/correlations/dfCorFarWound{filename}.pkl"
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
                np.array(dfShapeF["T"] < 20) & np.array(dfShapeF["T"] >= 0)
            ]
            n = int(len(dfFar) / 5)
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

                x = dfClose["X"].iloc[i]
                y = dfClose["Y"].iloc[i]
                t = dfClose["T"].iloc[i]
                r = dist[t, int(512 - y), int(x)]
                if r * scale < 30:
                    dp1 = dfClose["dp"].iloc[i][0]
                    dp2 = dfClose["dp"].iloc[i][1]
                    dq1 = dfClose["dq"].iloc[i][0, 0]
                    dq2 = dfClose["dq"].iloc[i][0, 1]
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
                f"databases/correlations/dfCorFarWound{filename}.pkl"
            )

# --------- velocity ----------


# --------- collect all ----------

# collect all correlations
if False:
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
        dfCorMid_5 = pd.read_pickle(
            f"databases/correlations/dfCorMidway{filename}_5.pkl"
        )
        dfCorMid_6 = pd.read_pickle(
            f"databases/correlations/dfCorMidway{filename}_6.pkl"
        )
        dfCorMid_7 = pd.read_pickle(
            f"databases/correlations/dfCorMidway{filename}_7.pkl"
        )
        dfCorMid_8 = pd.read_pickle(
            f"databases/correlations/dfCorMidway{filename}_8.pkl"
        )
        dfCorRho = pd.read_pickle(f"databases/correlations/dfCorRho{filename}.pkl")
        dfCorRhoQ = pd.read_pickle(f"databases/correlations/dfCorRhoQ{filename}.pkl")

        dP1dP1 = np.nan_to_num(dfCorMid_1["dP1dP1Correlation"].iloc[0])
        dP1dP1_std = np.nan_to_num(dfCorMid_1["dP1dP1Correlation_std"].iloc[0])
        dP1dP1total = np.nan_to_num(dfCorMid_1["dP1dP1Count"].iloc[0])
        if np.sum(dP1dP1) == 0:
            print("dP1dP1")

        dP2dP2 = np.nan_to_num(dfCorMid_2["dP2dP2Correlation"].iloc[0])
        dP2dP2_std = np.nan_to_num(dfCorMid_2["dP2dP2Correlation_std"].iloc[0])
        dP2dP2total = np.nan_to_num(dfCorMid_2["dP2dP2Count"].iloc[0])
        if np.sum(dP2dP2) == 0:
            print("dP2dP2")

        dQ1dQ1 = np.nan_to_num(dfCorMid_3["dQ1dQ1Correlation"].iloc[0])
        dQ1dQ1_std = np.nan_to_num(dfCorMid_3["dQ1dQ1Correlation_std"].iloc[0])
        dQ1dQ1total = np.nan_to_num(dfCorMid_3["dQ1dQ1Count"].iloc[0])
        if np.sum(dQ1dQ1) == 0:
            print("dQ1dQ1")

        dQ2dQ2 = np.nan_to_num(dfCorMid_4["dQ2dQ2Correlation"].iloc[0])
        dQ2dQ2_std = np.nan_to_num(dfCorMid_4["dQ2dQ2Correlation_std"].iloc[0])
        dQ2dQ2total = np.nan_to_num(dfCorMid_4["dQ2dQ2Count"].iloc[0])
        if np.sum(dQ2dQ2) == 0:
            print("dQ2dQ2")

        dQ1dQ2 = np.nan_to_num(dfCorMid_5["dQ1dQ2Correlation"].iloc[0])
        dQ1dQ2_std = np.nan_to_num(dfCorMid_5["dQ1dQ2Correlation_std"].iloc[0])
        dQ1dQ2total = np.nan_to_num(dfCorMid_5["dQ1dQ2Count"].iloc[0])
        if np.sum(dQ1dQ2) == 0:
            print("dQ1dQ2")

        dP1dQ1 = np.nan_to_num(dfCorMid_6["dP1dQ1Correlation"].iloc[0])
        dP1dQ1_std = np.nan_to_num(dfCorMid_6["dP1dQ1Correlation_std"].iloc[0])
        dP1dQ1total = np.nan_to_num(dfCorMid_6["dP1dQ1Count"].iloc[0])
        if np.sum(dP1dQ1) == 0:
            print("dP1dQ1")

        dP1dQ2 = np.nan_to_num(dfCorMid_7["dP1dQ2Correlation"].iloc[0])
        dP1dQ2_std = np.nan_to_num(dfCorMid_7["dP1dQ2Correlation_std"].iloc[0])
        dP1dQ2total = np.nan_to_num(dfCorMid_7["dP1dQ2Count"].iloc[0])
        if np.sum(dP1dQ2) == 0:
            print("dP1dQ2")

        dP2dQ2 = np.nan_to_num(dfCorMid_8["dP2dQ2Correlation"].iloc[0])
        dP2dQ2_std = np.nan_to_num(dfCorMid_8["dP2dQ2Correlation_std"].iloc[0])
        dP2dQ2total = np.nan_to_num(dfCorMid_8["dP2dQ2Count"].iloc[0])
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
                "dRhodRho": dRhodRho,
                "dRhodRho_std": dRhodRho_std,
                "Count Rho": count_Rho,
                "dQ1dRho": dQ1dRho,
                "dQ1dRho_std": dQ1dRho_std,
                "dQ2dRho": dQ2dRho,
                "dQ2dRho_std": dQ2dRho_std,
                "Count Rho Q": count_RhoQ,
            }
        )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/dfCorrelations{fileType}.pkl")
