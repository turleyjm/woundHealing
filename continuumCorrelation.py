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
plt.rcParams.update({"font.size": 8})

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


# short range space time correlation
if False:
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
    dfShape["dp2dp2i"] = list(np.zeros([len(dfShape)]))
    # dfShape["dq1dq1i"] = list(np.zeros([len(dfShape)]))
    # dfShape["dq2dq2i"] = list(np.zeros([len(dfShape)]))
    # dfShape["dq1dq2i"] = list(np.zeros([len(dfShape)]))
    # dfShape["dp1dq1i"] = list(np.zeros([len(dfShape)]))
    # dfShape["dp1dq2i"] = list(np.zeros([len(dfShape)]))
    # dfShape["dp2dq2i"] = list(np.zeros([len(dfShape)]))

    for k in range(len(filenames)):
        filename = filenames[k]
        path_to_file = f"databases/dfCorMidway{filename}_1-2.pkl"
        if False == exists(path_to_file):
            _df = []
            dP1dP1Correlation = np.zeros([len(T), len(R), len(theta)])
            dP2dP2Correlation = np.zeros([len(T), len(R), len(theta)])
            # dQ1dQ1Correlation = np.zeros([len(T), len(R), len(theta)])
            # dQ2dQ2Correlation = np.zeros([len(T), len(R), len(theta)])
            # dQ1dQ2Correlation = np.zeros([len(T), len(R), len(theta)])
            # dP1dQ1Correlation = np.zeros([len(T), len(R), len(theta)])
            # dP1dQ2Correlation = np.zeros([len(T), len(R), len(theta)])
            # dP2dQ2Correlation = np.zeros([len(T), len(R), len(theta)])

            dp1dp1ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]  # t, r, theta
            dp2dp2ij = [
                [[[] for col in range(17)] for col in range(grid)]
                for col in range(timeGrid)
            ]
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

            print(filename)
            dfShapeF = dfShape[dfShape["Filename"] == filename]
            n = int(len(dfShapeF) / 10)
            random.seed(10)
            count = 0
            Is = []
            for i0 in range(n):
                i = int(random.random() * len(dfShapeF) / 10)
                while i in Is:
                    i = int(random.random() * len(dfShapeF) / 10)
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
                dfShapeF["dR"] = (
                    (dfShapeF.loc[:, "X"] - x) ** 2 + (dfShapeF.loc[:, "Y"] - y) ** 2
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
                        "dp1dp1i",
                        "dp2dp2i",
                        # "dq1dq1i",
                        # "dq2dq2i",
                        # "dq1dq2i",
                        # "dp1dq1i",
                        # "dp1dq2i",
                        # "dp2dq2i",
                    ]
                ]
                df = df[np.array(df["dR"] < R[-1]) & np.array(df["dR"] > 0)]

                df["dT"] = df.loc[:, "T"] - t
                df = df[np.array(df["dT"] < timeGrid) & np.array(df["dT"] >= 0)]
                if len(df) != 0:
                    theta = np.arctan2(df.loc[:, "Y"] - y, df.loc[:, "X"] - x)
                    df["dtheta"] = np.where(theta < 0, 2 * np.pi + theta, theta)
                    df["dp1dp1i"] = list(
                        np.stack(np.array(df.loc[:, "dp"]), axis=0)[:, 0] * dp1
                    )
                    df["dp2dp2i"] = list(
                        np.stack(np.array(df.loc[:, "dp"]), axis=0)[:, 1] * dp2
                    )
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
                        dp2dp2ij[int(df["dT"].iloc[j])][int(df["dR"].iloc[j] / 2)][
                            int(8 * df["dtheta"].iloc[j] / np.pi)
                        ].append(df["dp2dp2i"].iloc[j])
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
                        dP2dP2Correlation[i][j][th] = np.mean(dp2dp2ij[i][j][th])
                        # dQ1dQ1Correlation[i][j][th] = np.mean(dq1dq1ij[i][j][th])
                        # dQ2dQ2Correlation[i][j][th] = np.mean(dq2dq2ij[i][j][th])
                        # dQ1dQ2Correlation[i][j][th] = np.mean(dq1dq2ij[i][j][th])
                        # dP1dQ1Correlation[i][j][th] = np.mean(dp1dq1ij[i][j][th])
                        # dP1dQ2Correlation[i][j][th] = np.mean(dp1dq2ij[i][j][th])
                        # dP2dQ2Correlation[i][j][th] = np.mean(dp2dq2ij[i][j][th])

            _df.append(
                {
                    "Filename": filename,
                    "dP1dP1Correlation": dP1dP1Correlation,
                    "dP2dP2Correlation": dP2dP2Correlation,
                    # "dQ1dQ1Correlation": dQ1dQ1Correlation,
                    # "dQ2dQ2Correlation": dQ2dQ2Correlation,
                    # "dQ1dQ2Correlation": dQ1dQ2Correlation,
                    # "dP1dQ1Correlation": dP1dQ1Correlation,
                    # "dP1dQ2Correlation": dP1dQ2Correlation,
                    # "dP2dQ2Correlation": dP2dQ2Correlation,
                }
            )
            dfCorrelation = pd.DataFrame(_df)
            dfCorrelation.to_pickle(f"databases/dfCorMidway{filename}_1-2.pkl")


if False:
    dP1dP1Correlation = np.nan_to_num(dP1dP1Correlation)
    dP2dP2Correlation = np.nan_to_num(dP2dP2Correlation)
    dQ1dQ1Correlation = np.nan_to_num(dQ1dQ1Correlation)
    dQ2dQ2Correlation = np.nan_to_num(dQ2dQ2Correlation)
    dQ1dQ2Correlation = np.nan_to_num(dQ1dQ2Correlation)
    dP1dQ1Correlation = np.nan_to_num(dP1dQ1Correlation)
    dP1dQ2Correlation = np.nan_to_num(dP1dQ2Correlation)
    dP2dQ2Correlation = np.nan_to_num(dP2dQ2Correlation)

    dP1dP1CorrelationFilename = dP1dP1Correlation
    dP2dP2CorrelationFilename = dP2dP2Correlation
    dQ1dQ1CorrelationFilename = dQ1dQ1Correlation
    dQ2dQ2CorrelationFilename = dQ2dQ2Correlation
    dQ1dQ2CorrelationFilename = dQ1dQ2Correlation
    dP1dQ1CorrelationFilename = dP1dQ1Correlation
    dP1dQ2CorrelationFilename = dP1dQ2Correlation
    dP2dQ2CorrelationFilename = dP2dQ2Correlation

    dP1dP1Correlation = np.mean(dP1dP1Correlation, axis=3)
    dP2dP2Correlation = np.mean(dP2dP2Correlation, axis=3)
    dQ1dQ1Correlation = np.mean(dQ1dQ1Correlation, axis=3)
    dQ2dQ2Correlation = np.mean(dQ2dQ2Correlation, axis=3)
    dQ1dQ2Correlation = np.mean(dQ1dQ2Correlation, axis=3)
    dP1dQ1Correlation = np.mean(dP1dQ1Correlation, axis=3)
    dP1dQ2Correlation = np.mean(dP1dQ2Correlation, axis=3)
    dP2dQ2Correlation = np.mean(dP2dQ2Correlation, axis=3)

    _df.append(
        {
            "dP1dP1Correlation": dP1dP1Correlation,
            "dP2dP2Correlation": dP2dP2Correlation,
            "dQ1dQ1Correlation": dQ1dQ1Correlation,
            "dQ2dQ2Correlation": dQ2dQ2Correlation,
            "dQ1dQ2Correlation": dQ1dQ2Correlation,
            "dP1dQ1Correlation": dP1dQ1Correlation,
            "dP1dQ2Correlation": dP1dQ2Correlation,
            "dP2dQ2Correlation": dP2dQ2Correlation,
            "dP1dP1CorrelationFilename": dP1dP1CorrelationFilename,
            "dP2dP2CorrelationFilename": dP2dP2CorrelationFilename,
            "dQ1dQ1CorrelationFilename": dQ1dQ1CorrelationFilename,
            "dQ2dQ2CorrelationFilename": dQ2dQ2CorrelationFilename,
            "dQ1dQ2CorrelationFilename": dQ1dQ2CorrelationFilename,
            "dP1dQ1CorrelationFilename": dP1dQ1CorrelationFilename,
            "dP1dQ2CorrelationFilename": dP1dQ2CorrelationFilename,
            "dP2dQ2CorrelationFilename": dP2dQ2CorrelationFilename,
        }
    )

    dfCorrelation = pd.DataFrame(_df)
    dfCorrelation.to_pickle(f"databases/dfCorrelation{fileType}.pkl")


# display short range correlation
if False:
    dfCorrelation = pd.read_pickle(f"databases/dfCorrelation{fileType}.pkl")
    deltaP1Correlation = dfCorrelation["deltaP1Correlation"].iloc[0]
    deltaP2Correlation = dfCorrelation["deltaP2Correlation"].iloc[0]

    deltaP1Correlation = np.mean(deltaP1Correlation[:, :, :-1], axis=2)
    deltaP2Correlation = np.mean(deltaP2Correlation[:, :, :-1], axis=2)

    t, r = np.mgrid[0:102:2, 0:9:1]
    fig, ax = plt.subplots(2, 1, figsize=(16, 14))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    maxCorr = np.max([deltaP1Correlation, -deltaP1Correlation])

    c = ax[0].pcolor(
        t,
        r,
        deltaP1Correlation,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0])
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel(r"$R (\mu m)$ ")
    ax[0].title.set_text(r"Correlation of $\delta P_1$" + f" {fileType}")

    maxCorr = np.max([deltaP2Correlation, -deltaP2Correlation])

    c = ax[1].pcolor(
        t,
        r,
        deltaP2Correlation,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1])
    ax[1].set_xlabel("Time (min)")
    ax[1].set_ylabel(r"$R (\mu m)$")
    ax[1].title.set_text(r"Correlation of $\delta P_2$" + f" {fileType}")

    fig.savefig(
        f"results/Correlation P {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


def CorR0(t, C, a):
    return C * upperGamma(0, a * t)


def upperGamma(a, x):
    if a == 0:
        return -sc.expi(-x)
    else:
        return sc.gamma(a) * sc.gammaincc(a, x)


def binomialGamma(j, a, t):
    s = 0
    for l in range(j):
        s += (-a) ** (j - l) * (t) ** (-l) * upperGamma(l, a * t)
    s += (t) ** (-j) * upperGamma(j, a * t)
    return s


def forIntegral(y, b, R, a=0.014231800277153952, T=2, C=8.06377854e-06):
    y, R, T = np.meshgrid(y, R, T, indexing="ij")
    return C * np.exp(-y * T) * sc.jv(0, R * ((y - a) / b) ** 0.5) / y


def Integral(R, b):
    a = 0.014231800277153952
    T = 2
    C = 8.06377854e-06
    y = np.linspace(a, a * 100, 100000)
    h = y[1] - y[0]
    return np.sum(forIntegral(y, b, R, a, T, C) * h, axis=0)[:, 0]


def Integral_P2(R, b):
    a = 0.014231800277153952
    T = 2
    C = 2.89978933e-06
    y = np.linspace(a, a * 100, 100000)
    h = y[1] - y[0]
    return np.sum(forIntegral(y, b, R, a, T, C) * h, axis=0)[:, 0]


# deltaP1
if False:
    grid = 9
    timeGrid = 51

    dfCorrelation = pd.read_pickle(f"databases/dfCorrelation{fileType}.pkl")
    deltaP1Correlation = dfCorrelation["deltaP1Correlation"].iloc[0]
    deltaP1Correlation = np.mean(deltaP1Correlation[:, :, :-1], axis=2)

    T = np.linspace(0, 2 * (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)

    m = sp.optimize.curve_fit(
        f=CorR0,
        xdata=T[1:],
        ydata=deltaP1Correlation[:, 0][1:],
        p0=(0.000006, 0.01),
    )[0]

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    ax[0].plot(T[1:], deltaP1Correlation[:, 0][1:])
    ax[0].plot(T[1:], CorR0(T[1:], m[0], m[1]))
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel(r"$P_1$ Correlation")
    ax[0].set_ylim([-0.000005, 0.000025])
    ax[0].set_xlim([0, 2 * timeGrid])
    ax[0].title.set_text(r"Correlation of $\delta P_1$" + f" {fileType}")

    m = sp.optimize.curve_fit(
        f=Integral,
        xdata=R,
        ydata=deltaP1Correlation[1],
        p0=0.025,
        method="lm",
    )[0]

    ax[1].plot(R, deltaP1Correlation[1])
    ax[1].plot(R, Integral(R, m))
    ax[1].set_xlabel(r"$R (\mu m)$")
    ax[1].set_ylabel(r"$P_1$ Correlation")
    ax[1].set_ylim([-0.000005, 0.000025])
    ax[1].title.set_text(r"Correlation of $\delta P_1$" + f" {fileType}")
    fig.savefig(
        f"results/Correlation P1 in T and R {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


# deltaP2
if False:
    grid = 9
    timeGrid = 51

    dfCorrelation = pd.read_pickle(f"databases/dfCorrelation{fileType}.pkl")

    deltaP1Correlation = dfCorrelation["deltaP1Correlation"].iloc[0]
    deltaP1Correlation = np.mean(deltaP1Correlation[:, :, :-1], axis=2)
    deltaP2Correlation = dfCorrelation["deltaP2Correlation"].iloc[0]
    deltaP2Correlation = np.mean(deltaP2Correlation[:, :, :-1], axis=2)

    T = np.linspace(0, 2 * (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)

    m_P1 = sp.optimize.curve_fit(
        f=CorR0,
        xdata=T[1:],
        ydata=deltaP1Correlation[:, 0][1:],
        p0=(0.000006, 0.01),
    )[0]
    m = sp.optimize.curve_fit(
        f=CorR0,
        xdata=T[1:],
        ydata=deltaP2Correlation[:, 0][1:],
        p0=(0.000006, 0.01),
    )[0]

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    ax[0].plot(T[1:], deltaP2Correlation[:, 0][1:])
    ax[0].plot(T[1:], CorR0(T[1:], m[0], m_P1[1]))
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel(r"$P_2$ Correlation")
    ax[0].set_ylim([-0.000002, 0.00001])
    ax[0].set_xlim([0, 2 * timeGrid])
    ax[0].title.set_text(r"Correlation of $\delta P_2$" + f" {fileType}")

    m = sp.optimize.curve_fit(
        f=Integral_P2,
        xdata=R,
        ydata=deltaP2Correlation[1],
        p0=0.025,
        method="lm",
    )[0]

    ax[1].plot(R, deltaP2Correlation[1])
    ax[1].plot(R, Integral_P2(R, m))
    ax[1].set_xlabel(r"$R (\mu m)$")
    ax[1].set_ylabel(r"$P_2$ Correlation")
    ax[1].set_ylim([-0.000002, 0.00001])
    ax[1].title.set_text(r"Correlation of $\delta P_2$" + f" {fileType}")
    fig.savefig(
        f"results/Correlation P2 in T and R {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


if False:
    a = 0.014231800277153952
    y = np.linspace(a, a * 100, 100000)
    R = np.linspace(0, 10, 50)

    for r in R:
        b = 0.1
        fig, ax = plt.subplots(1, 1, figsize=(4, 8))
        ax.plot(y, forIntegral(y, b, r, a=0.014231800277153952, T=2, C=8.06377854e-06))
        ax.title.set_text(f"R={int(r)}")
        ax.set_xlabel("y")
        fig.savefig(
            f"results/Integral {fileType} R={r}",
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )
        plt.close("all")

if False:
    a = 0.014231800277153952
    b = 0.005
    y = np.linspace(a, a * 100, 100000)
    R = np.linspace(0, 10, 50)

    h = y[1] - y[0]
    fun = []
    for r in R:
        fun.append(
            sum(forIntegral(y, b, r, a=0.014231800277153952, T=2, C=8.06377854e-06) * h)
        )

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(R, fun)
    ax.set_xlabel("y")
    fig.savefig(
        f"results/P r correlation {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


def CorrdP1(R, T):
    a = 0.014231800277153952
    b = 0.02502418
    C = 8.06377854e-06
    y = np.linspace(a, a * 100, 100000)
    h = y[1] - y[0]
    return np.sum(forIntegral(y, b, R, a, T, C) * h, axis=0)[:, 0]


def CorrdP2(R, T):
    a = 0.014231800277153952
    b = 0.02502418
    C = 2.89978933e-06
    y = np.linspace(a, a * 100, 100000)
    h = y[1] - y[0]
    return np.sum(forIntegral(y, b, R, a, T, C) * h, axis=0)[:, 0]


# fit cavre for dP1
if True:
    dfCorrelation = pd.read_pickle(f"databases/dfCorrelation{fileType}.pkl")
    deltaP1Correlation = dfCorrelation["deltaP1Correlation"].iloc[0]

    deltaP1Correlation = np.mean(deltaP1Correlation[:, :, :-1], axis=2)

    t, r = np.mgrid[0:102:2, 0:18:2]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))

    maxCorr = np.max([deltaP1Correlation, -deltaP1Correlation])

    c = ax[0].pcolor(
        t,
        r,
        deltaP1Correlation,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0])
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel(r"$R (\mu m)$ ")
    ax[0].title.set_text(r"Correlation of $\delta P_1$" + f" {fileType}")

    fit_dP1 = np.zeros([t.shape[0], t.shape[1]])
    for _t in range(t.shape[0]):
        for _r in range(t.shape[1]):
            fit_dP1[_t, _r] = CorrdP1(r[_t, _r], t[_t, _r])[0]

    c = ax[1].pcolor(
        t,
        r,
        fit_dP1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1])
    ax[1].set_xlabel("Time (min)")
    ax[1].set_ylabel(r"$R (\mu m)$")
    ax[1].title.set_text(r"Model Correlation of $\delta P_1$")

    c = ax[2].pcolor(
        t,
        r,
        deltaP1Correlation - fit_dP1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2])
    ax[2].set_xlabel("Time (min)")
    ax[2].set_ylabel(r"$R (\mu m)$")
    ax[2].title.set_text(r"Differnce between curves")

    fig.savefig(
        f"results/Correlation P1 {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


# fit cavre for dP1
if True:
    dfCorrelation = pd.read_pickle(f"databases/dfCorrelation{fileType}.pkl")
    deltaP2Correlation = dfCorrelation["deltaP2Correlation"].iloc[0]

    deltaP2Correlation = np.mean(deltaP2Correlation[:, :, :-1], axis=2)

    t, r = np.mgrid[0:102:2, 0:18:2]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))

    maxCorr = np.max([deltaP2Correlation, -deltaP2Correlation])

    c = ax[0].pcolor(
        t,
        r,
        deltaP2Correlation,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0])
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel(r"$R (\mu m)$ ")
    ax[0].title.set_text(r"Correlation of $\delta P_2$" + f" {fileType}")

    fit_dP2 = np.zeros([t.shape[0], t.shape[1]])
    for _t in range(t.shape[0]):
        for _r in range(t.shape[1]):
            fit_dP2[_t, _r] = CorrdP2(r[_t, _r], t[_t, _r])[0]

    c = ax[1].pcolor(
        t,
        r,
        fit_dP2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1])
    ax[1].set_xlabel("Time (min)")
    ax[1].set_ylabel(r"$R (\mu m)$")
    ax[1].title.set_text(r"Model Correlation of $\delta P_2$")

    c = ax[2].pcolor(
        t,
        r,
        deltaP2Correlation - fit_dP2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2])
    ax[2].set_xlabel("Time (min)")
    ax[2].set_ylabel(r"$R (\mu m)$")
    ax[2].title.set_text(r"Differnce between curves")

    fig.savefig(
        f"results/Correlation P2 {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")