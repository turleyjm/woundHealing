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

# -------------------


# space time correlation
if True:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    grid = 42
    timeGrid = 51

    T = np.linspace(0, (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)
    theta = np.linspace(0, 2 * np.pi, 17)

    dfShape["dR"] = list(np.zeros([len(dfShape)]))
    dfShape["dT"] = list(np.zeros([len(dfShape)]))
    dfShape["dtheta"] = list(np.zeros([len(dfShape)]))

    dfShape["dp1dp1i"] = list(np.zeros([len(dfShape)]))
    # dfShape["dp2dp2i"] = list(np.zeros([len(dfShape)]))
    # dfShape["dq1dq1i"] = list(np.zeros([len(dfShape)]))
    # dfShape["dq2dq2i"] = list(np.zeros([len(dfShape)]))
    # dfShape["dq1dq2i"] = list(np.zeros([len(dfShape)]))
    # dfShape["dp1dq1i"] = list(np.zeros([len(dfShape)]))
    # dfShape["dp1dq2i"] = list(np.zeros([len(dfShape)]))
    # dfShape["dp2dq2i"] = list(np.zeros([len(dfShape)]))

    for k in range(len(filenames)):
        filename = filenames[k]
        path_to_file = f"databases/dfCorMidway{filename}_1.pkl"
        if False == exists(path_to_file):
            _df = []
            dP1dP1Correlation = np.zeros([len(T), len(R), len(theta)])
            # dP2dP2Correlation = np.zeros([len(T), len(R), len(theta)])
            # dQ1dQ1Correlation = np.zeros([len(T), len(R), len(theta)])
            # dQ2dQ2Correlation = np.zeros([len(T), len(R), len(theta)])
            # dQ1dQ2Correlation = np.zeros([len(T), len(R), len(theta)])
            # dP1dQ1Correlation = np.zeros([len(T), len(R), len(theta)])
            # dP1dQ2Correlation = np.zeros([len(T), len(R), len(theta)])
            # dP2dQ2Correlation = np.zeros([len(T), len(R), len(theta)])
            dP1dP1Correlation_std = np.zeros([len(T), len(R), len(theta)])
            # dP2dP2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            # dQ1dQ1Correlation_std = np.zeros([len(T), len(R), len(theta)])
            # dQ2dQ2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            # dQ1dQ2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            # dP1dQ1Correlation_std = np.zeros([len(T), len(R), len(theta)])
            # dP1dQ2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            # dP2dQ2Correlation_std = np.zeros([len(T), len(R), len(theta)])
            total = np.zeros([len(T), len(R), len(theta)])

            dp1dp1ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]  # t, r, theta
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
            # dq1dq2ij = [
            #     [[[] for col in range(17)] for col in range(grid)]
            #     for col in range(timeGrid)
            # ]  # t, r, theta
            # dp1dq1ij = [
            #     [[[] for col in range(17)] for col in range(grid)]
            #     for col in range(timeGrid)
            # ]
            # dp1dq2ij = [
            #     [[[] for col in range(17)] for col in range(grid)]
            #     for col in range(timeGrid)
            # ]
            # dp2dq2ij = [
            #     [[[] for col in range(17)] for col in range(grid)]
            #     for col in range(timeGrid)
            # ]

            print(filename + datetime.now().strftime("%H:%M:%S"))
            dfShapeF = dfShape[dfShape["Filename"] == filename].copy()
            n = int(len(dfShapeF)/10)
            random.seed(10)
            count = 0
            Is = []
            for i0 in range(n):
                i = int(random.random() * n)
                while i in Is:
                    i = int(random.random() * n)
                Is.append(i)
                if i0 % int((n) / 10) == 0:
                    # print(datetime.now().strftime("%H:%M:%S") + f" {10*count}%")
                    count += 1

                x = dfShapeF["X"].iloc[i]
                y = dfShapeF["Y"].iloc[i]
                t = dfShapeF["T"].iloc[i]
                dp1 = dfShapeF["dp"].iloc[i][0]
                dp2 = dfShapeF["dp"].iloc[i][1]
                dq1 = dfShapeF["dq"].iloc[i][0, 0]
                dq2 = dfShapeF["dq"].iloc[i][0, 1]
                dfShapeF.loc[:,"dR"] = ((
                    (dfShapeF.loc[:, "X"] - x) ** 2 + (dfShapeF.loc[:, "Y"] - y) ** 2
                ) ** 0.5).copy()
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
                        "dp1dp1i",
                        # "dp2dp2i",
                        # "dq1dq1i",
                        # "dq2dq2i",
                        # "dq1dq2i",
                        # "dp1dq1i",
                        # "dp1dq2i",
                        # "dp2dq2i",
                    ]
                ]
                df = df[np.array(df["dR"] < R[-1]) & np.array(df["dR"] >= 0)]

                df["dT"] = df.loc[:, "T"] - t
                df = df[np.array(df["dT"] < timeGrid) & np.array(df["dT"] >= 0)]
                if len(df) != 0:
                    theta = np.arctan2(df.loc[:, "Y"] - y, df.loc[:, "X"] - x)
                    df["dtheta"] = np.where(theta < 0, 2 * np.pi + theta, theta)
                    df["dp1dp1i"] = list(
                        np.stack(np.array(df.loc[:, "dp"]), axis=0)[:, 0] * dp1
                    )
                    # df["dp2dp2i"] = list(
                    #     np.stack(np.array(df.loc[:, "dp"]), axis=0)[:, 1] * dp2
                    # )
                    # df["dq1dq1i"] = list(
                    #     np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 0] * dq1
                    # )
                    # df["dq2dq2i"] = list(
                    #     np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 1] * dq2
                    # )
                    # df["dq1dq2i"] = list(
                    #     np.stack(np.array(df.loc[:, "dq"]), axis=0)[:, 0, 0] * dq2
                    # )
                    # df["dp1dq1i"] = list(
                    #     np.stack(np.array(df.loc[:, "dp"]), axis=0)[:, 0] * dq1
                    # )
                    # df["dp1dq2i"] = list(
                    #     np.stack(np.array(df.loc[:, "dp"]), axis=0)[:, 0] * dq2
                    # )
                    # df["dp2dq2i"] = list(
                    #     np.stack(np.array(df.loc[:, "dp"]), axis=0)[:, 1] * dq2
                    # )

                    for j in range(len(df)):
                        dp1dp1ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dp1dp1i"].iloc[j])
                        # dp2dp2ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                        #     int(8 * df["dtheta"].iloc[j] / np.pi)
                        # ].append(df["dp2dp2i"].iloc[j])
                        # dq1dq1ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                        #     int(8 * df["dtheta"].iloc[j] / np.pi)
                        # ].append(df["dq1dq1i"].iloc[j])
                        # dq2dq2ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                        #     int(8 * df["dtheta"].iloc[j] / np.pi)
                        # ].append(df["dq2dq2i"].iloc[j])
                        # dq1dq2ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                        #     int(8 * df["dtheta"].iloc[j] / np.pi)
                        # ].append(df["dq1dq2i"].iloc[j])
                        # dp1dq1ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                        #     int(8 * df["dtheta"].iloc[j] / np.pi)
                        # ].append(df["dp1dq1i"].iloc[j])
                        # dp1dq2ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                        #     int(8 * df["dtheta"].iloc[j] / np.pi)
                        # ].append(df["dp1dq2i"].iloc[j])
                        # dp2dq2ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                        #     int(8 * df["dtheta"].iloc[j] / np.pi)
                        # ].append(df["dp2dq2i"].iloc[j])

            T = np.linspace(0, (timeGrid - 1), timeGrid)
            R = np.linspace(0, 2 * (grid - 1), grid)
            theta = np.linspace(0, 2 * np.pi, 17)
            for i in range(len(T)):
                for j in range(len(R)):
                    for th in range(len(theta)):
                        dP1dP1Correlation[i][j][th] = np.mean(dp1dp1ij[i][j][th])
                        # dP2dP2Correlation[i][j][th] = np.mean(dp2dp2ij[i][j][th])
                        # dQ1dQ1Correlation[i][j][th] = np.mean(dq1dq1ij[i][j][th])
                        # dQ2dQ2Correlation[i][j][th] = np.mean(dq2dq2ij[i][j][th])
                        # dQ1dQ2Correlation[i][j][th] = np.mean(dq1dq2ij[i][j][th])
                        # dP1dQ1Correlation[i][j][th] = np.mean(dp1dq1ij[i][j][th])
                        # dP1dQ2Correlation[i][j][th] = np.mean(dp1dq2ij[i][j][th])
                        # dP2dQ2Correlation[i][j][th] = np.mean(dp2dq2ij[i][j][th])

                        dP1dP1Correlation_std[i][j][th] = np.std(dp1dp1ij[i][j][th])
                        # dP2dP2Correlation_std[i][j][th] = np.std(dp2dp2ij[i][j][th])
                        # dQ1dQ1Correlation_std[i][j][th] = np.std(dq1dq1ij[i][j][th])
                        # dQ2dQ2Correlation_std[i][j][th] = np.std(dq2dq2ij[i][j][th])
                        # dQ1dQ2Correlation_std[i][j][th] = np.std(dq1dq2ij[i][j][th])
                        # dP1dQ1Correlation_std[i][j][th] = np.std(dp1dq1ij[i][j][th])
                        # dP1dQ2Correlation_std[i][j][th] = np.std(dp1dq2ij[i][j][th])
                        # dP2dQ2Correlation_std[i][j][th] = np.std(dp2dq2ij[i][j][th])
                        # total[i][j][th] = len(dq1dq2ij[i][j][th])

            _df.append(
                {
                    "Filename": filename,
                    "dP1dP1Correlation": dP1dP1Correlation,
                    # "dP2dP2Correlation": dP2dP2Correlation,
                    # "dQ1dQ1Correlation": dQ1dQ1Correlation,
                    # "dQ2dQ2Correlation": dQ2dQ2Correlation,
                    # "dQ1dQ2Correlation": dQ1dQ2Correlation,
                    # "dP1dQ1Correlation": dP1dQ1Correlation,
                    # "dP1dQ2Correlation": dP1dQ2Correlation,
                    # "dP2dQ2Correlation": dP2dQ2Correlation,
                    "dP1dP1Correlation_std": dP1dP1Correlation_std,
                    # "dP2dP2Correlation_std": dP2dP2Correlation_std,
                    # "dQ1dQ1Correlation_std": dQ1dQ1Correlation_std,
                    # "dQ2dQ2Correlation_std": dQ2dQ2Correlation_std,
                    # "dQ1dQ2Correlation_std": dQ1dQ2Correlation_std,
                    # "dP1dQ1Correlation_std": dP1dQ1Correlation_std,
                    # "dP1dQ2Correlation_std": dP1dQ2Correlation_std,
                    # "dP2dQ2Correlation_std": dP2dQ2Correlation_std,
                    # "Count": total,
                }
            )
            dfCorrelation = pd.DataFrame(_df)
            dfCorrelation.to_pickle(f"databases/correlations/dfCorMidway{filename}_1.pkl")
