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


def corRho_T(T, C):
    return C / T


def corRho_R(R, C, D):
    T = 2.5
    return C / T * np.exp(-(R**2) / (4 * D * T))


def expCos(R, C, D, w):
    return C * np.exp(-D * R) * np.cos(w * R)


def exp(R, C, D):
    return C * np.exp(-D * R)


def expStretched(R, C, D, alpha):
    return C * np.exp(-D * R**alpha)


def explinear(R, C, D, m, c):
    return C * np.exp(-D * R) + m * R + c


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


# ------------------- divisons

# Distributions of divisons in x and y
if False:
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    x = np.array(dfDivisions["X"])
    y = np.array(dfDivisions["Y"])

    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    ax[0].hist(x, bins=10)
    ax[0].set(xlabel="x")

    ax[1].hist(y, bins=10)
    ax[1].set(xlabel="y")

    fig.savefig(
        f"results/xy distributions {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

# Orientation distributions
if False:
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    ori = np.array(dfDivisions["Orientation"])

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    ax.hist(ori, bins=10)
    ax.set(xlabel="Orientation")

    fig.savefig(
        f"results/ori distributions {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

# Orientation distributions by filename
if False:
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    for filename in filenames:
        df = dfDivisions[dfDivisions["Filename"] == filename]
        ori = np.array(df["Orientation"])

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.gcf().subplots_adjust(bottom=0.15)

        ax.hist(ori, bins=10)
        ax.set(xlabel="Orientation")

        fig.savefig(
            f"results/ori distributions {filename}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

# Division correlations figure
if False:
    df = pd.read_pickle(f"databases/divCorr{fileType}.pkl")
    expectedXY = df["expectedXY"].iloc[0]
    ExXExY = df["ExXExY"].iloc[0]
    divCorr = df["divCorr"].iloc[0]
    oriCorr = df["oriCorr"].iloc[0]
    df = 0
    maxCorr = np.max(expectedXY)

    if False:
        t, r = np.mgrid[10:110:10, 10:120:10]
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        plt.subplots_adjust(wspace=0.3)
        plt.gcf().subplots_adjust(bottom=0.15)

        c = ax[0, 0].pcolor(t, r, expectedXY, cmap="Reds", vmin=0, vmax=maxCorr)
        fig.colorbar(c, ax=ax[0, 0])
        ax[0, 0].set_xlabel("Time (min)")
        ax[0, 0].set_ylabel(r"$R (\mu m)$ ")
        ax[0, 0].title.set_text(f"expectedXY")

        c = ax[0, 1].pcolor(t, r, ExXExY, cmap="Reds", vmin=0, vmax=maxCorr)
        fig.colorbar(c, ax=ax[0, 1])
        ax[0, 1].set_xlabel("Time (min)")
        ax[0, 1].set_ylabel(r"$R (\mu m)$")
        ax[0, 1].title.set_text(f"ExXExY")

        c = ax[1, 0].pcolor(
            t,
            r,
            divCorr,
            cmap="RdBu_r",
            vmin=-maxCorr,
            vmax=maxCorr,
        )
        fig.colorbar(c, ax=ax[1, 0])
        ax[1, 0].set_xlabel("Time (min)")
        ax[1, 0].set_ylabel(r"$R (\mu m)$")
        ax[1, 0].title.set_text(f"Correlation")

        c = ax[1, 1].pcolor(
            t,
            r,
            oriCorr,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        fig.colorbar(c, ax=ax[1, 1])
        ax[1, 1].set_xlabel("Time (min)")
        ax[1, 1].set_ylabel(r"$R (\mu m)$")
        ax[1, 1].title.set_text(f"Correlation Orientation")

        fig.savefig(
            f"results/Division Correlation {fileType}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

    t, r = np.mgrid[10:110:10, 10:120:10]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    maxCorr = np.max(divCorr[:10])
    c = ax.pcolor(
        t,
        r,
        divCorr[:10] * 10000**2,
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
    )
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("Time apart $t$ (min)")
    ax.set_ylabel(r"Distance apart $r (\mu m)$")
    fileTitle = util.getFileTitle(fileType)

    if "Wound" in fileType:
        ax.title.set_text(
            f"Division density \n correlation "
            + r"$\bf{"
            + str(str(fileTitle).split(" ")[0])
            + "}$"
            + " "
            + r"$\bf{"
            + str(str(fileTitle).split(" ")[1])
            + "}$"
        )
    else:
        ax.title.set_text(
            f"Division density \n correlation " + r"$\bf{" + str(fileTitle) + "}$"
        )

    fig.savefig(
        f"results/Division Correlation figure {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

    t, r = np.mgrid[10:110:10, 10:120:10]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    maxCorr = np.max(divCorr[:10])
    c = ax.pcolor(
        t,
        r,
        (divCorr[:10] - np.mean(divCorr[:10, 7:10], axis=1).reshape((10, 1)))
        * 10000**2,
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
    )
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("Time apart $t$ (min)")
    ax.set_ylabel(r"Distance apart $r (\mu m)$")
    fileTitle = util.getFileTitle(fileType)

    if "Wound" in fileType:
        ax.title.set_text(
            f"Division density \n correlation "
            + r"$\bf{"
            + str(str(fileTitle).split(" ")[0])
            + "}$"
            + " "
            + r"$\bf{"
            + str(str(fileTitle).split(" ")[1])
            + "}$"
        )
    else:
        ax.title.set_text(
            f"Division density \n correlation " + r"$\bf{" + str(fileTitle) + "}$"
        )

    fig.savefig(
        f"results/Division Correlation figure remove long times {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

    t, r = np.mgrid[10:110:10, 10:120:10]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    c = ax.pcolor(
        t,
        r,
        oriCorr[:10],
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("Time apart $t$ (min)")
    ax.set_ylabel(r"Distance apart $r (\mu m)$")
    fileTitle = util.getFileTitle(fileType)

    if "Wound" in fileType:
        ax.title.set_text(
            f"Division orientation \n correlation "
            + r"$\bf{"
            + str(str(fileTitle).split(" ")[0])
            + "}$"
            + " "
            + r"$\bf{"
            + str(str(fileTitle).split(" ")[1])
            + "}$"
        )
    else:
        ax.title.set_text(
            f"Division orientation \n correlation " + r"$\bf{" + str(fileTitle) + "}$"
        )

    fig.savefig(
        f"results/Division orientation figure {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

# Division orientation correlations by filename
if False:
    T = 160
    timeStep = 10
    R = 110
    rStep = 10

    df = pd.read_pickle(f"databases/divCorr{fileType}.pkl")
    thetaCorr = df["thetaCorr"].iloc[0]
    df = 0
    time = np.array(range(int(T / timeStep))) * timeStep
    rad = np.array(range(int(R / rStep))) * rStep

    t, r = np.mgrid[0:160:10, 0:110:10]

    for m in range(len(filenames)):
        oriCorr = np.zeros([int(T / timeStep), int(R / rStep)])

        for i in range(len(time)):
            for j in range(len(rad)):
                oriCorr[i][j] = np.mean(thetaCorr[i][j][m])

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.subplots_adjust(wspace=0.3)
        plt.gcf().subplots_adjust(bottom=0.15)

        c = ax.pcolor(
            t,
            r,
            oriCorr,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        fig.colorbar(c, ax=ax)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(r"$R (\mu m)$")
        ax.title.set_text(f"Correlation Orientation {filenames[m]}")

        fig.savefig(
            f"results/Correlation Orientation {filenames[m]}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

# Division-rho correlations figure
if False:
    df = pd.read_pickle(f"databases/divRhoCorr{fileType}.pkl")
    expectedXY = df["expectedXY"].iloc[0]
    ExXExY = df["ExXExY"].iloc[0]
    divRhoCorr = df["divRhoCorr"].iloc[0]
    df = 0
    maxCorr = np.max(expectedXY)

    t, r = np.mgrid[0:160:10, 0:110:10]
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    c = ax[0, 0].pcolor(t, r, expectedXY, cmap="Reds", vmin=0, vmax=maxCorr)
    fig.colorbar(c, ax=ax[0, 0])
    ax[0, 0].set_xlabel("Time (min)")
    ax[0, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 0].title.set_text(f"expectedXY")

    c = ax[0, 1].pcolor(t, r, ExXExY, cmap="Reds", vmin=0, vmax=maxCorr)
    fig.colorbar(c, ax=ax[0, 1])
    ax[0, 1].set_xlabel("Time (min)")
    ax[0, 1].set_ylabel(r"$R (\mu m)$")
    ax[0, 1].title.set_text(f"ExXExY")

    c = ax[1, 0].pcolor(
        t,
        r,
        divRhoCorr,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
    )
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set_xlabel("Time (min)")
    ax[1, 0].set_ylabel(r"$R (\mu m)$")
    ax[1, 0].title.set_text(f"Correlation")

    fig.savefig(
        f"results/Division Rho Correlation {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


# ------------------- Shape correlations

# total comparisions
if False:
    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    total = 0
    for i in range(len(dfCor)):
        total += np.sum(dfCor["Count Rho"].iloc[i])
        total += np.sum(dfCor["Count Rho Q"].iloc[i]) * 2
        total += np.sum(dfCor["dP1dP1Count"].iloc[i])
        total += np.sum(dfCor["dP2dP2Count"].iloc[i])
        total += np.sum(dfCor["dQ1dQ1Count"].iloc[i])
        total += np.sum(dfCor["dQ2dQ2Count"].iloc[i])
        total += np.sum(dfCor["dQ1dQ2Count"].iloc[i])
        total += np.sum(dfCor["dP1dQ1Count"].iloc[i])
        total += np.sum(dfCor["dP1dQ2Count"].iloc[i])
        total += np.sum(dfCor["dP2dQ2Count"].iloc[i])

    numbers = "{:,}".format(int(total))
    print(numbers)

# display all correlations
if False:
    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    fig, ax = plt.subplots(3, 4, figsize=(30, 18))

    T, R, Theta = dfCor["dRhodRho"].iloc[0].shape

    dRhodRho = np.zeros([len(filenames), T, R])
    dQ1dRho = np.zeros([len(filenames), T, R])
    dQ2dRho = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):
        RhoCount = dfCor["Count Rho"].iloc[i][:, :, :-1]
        dRhodRho[i] = np.sum(
            dfCor["dRhodRho"].iloc[i][:, :, :-1] * RhoCount, axis=2
        ) / np.sum(RhoCount, axis=2)
        RhoQCount = dfCor["Count Rho Q"].iloc[i][:, :, :-1]
        dQ1dRho[i] = np.sum(
            dfCor["dQ1dRho"].iloc[i][:, :, :-1] * RhoQCount, axis=2
        ) / np.sum(RhoQCount, axis=2)
        dQ2dRho[i] = np.sum(
            dfCor["dQ2dRho"].iloc[i][:, :, :-1] * RhoQCount, axis=2
        ) / np.sum(RhoQCount, axis=2)

    dRhodRho = np.mean(dRhodRho, axis=0)
    dQ1dRho = np.mean(dQ1dRho, axis=0)
    dQ2dRho = np.mean(dQ2dRho, axis=0)

    maxCorr = np.max([dRhodRho, -dRhodRho])
    t, r = np.mgrid[0:180:10, 0:90:10]
    c = ax[0, 0].pcolor(
        t,
        r,
        dRhodRho,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 0])
    ax[0, 0].set_xlabel("Time (mins)")
    ax[0, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 0].title.set_text(r"$\langle \delta \rho \delta \rho \rangle$")

    c = ax[0, 1].pcolor(
        t,
        r,
        dRhodRho - dRhodRho[-1],
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 1])
    ax[0, 1].set_xlabel("Time (mins)")
    ax[0, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 1].title.set_text(r"$\langle \delta \rho \delta \rho \rangle$ nosie")

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
    fig.colorbar(c, ax=ax[0, 2])
    ax[0, 2].set_xlabel("Time (mins)")
    ax[0, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 2].title.set_text(r"$\langle \delta Q^1 \delta \rho \rangle$")

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
    fig.colorbar(c, ax=ax[0, 3])
    ax[0, 3].set_xlabel("Time (mins)")
    ax[0, 3].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 3].title.set_text(r"$\langle \delta Q^2 \delta \rho \rangle$")

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

    t, r = np.mgrid[0:102:2, 0:82:2]
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
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set_xlabel("Time (mins)")
    ax[1, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 0].title.set_text(r"$\langle \delta P_1 \delta P_1 \rangle$")

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
    fig.colorbar(c, ax=ax[1, 1])
    ax[1, 1].set_xlabel("Time (mins)")
    ax[1, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 1].title.set_text(r"$\langle \delta P_2 \delta P_2 \rangle$")

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
    fig.colorbar(c, ax=ax[1, 2])
    ax[1, 2].set_xlabel("Time (mins)")
    ax[1, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 2].title.set_text(r"$\langle \delta Q^1 \delta Q^1 \rangle$")

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
    fig.colorbar(c, ax=ax[1, 3])
    ax[1, 3].set_xlabel("Time (mins)")
    ax[1, 3].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 3].title.set_text(r"$\langle \delta Q^2 \delta Q^2 \rangle$")

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
    fig.colorbar(c, ax=ax[2, 0])
    ax[2, 0].set_xlabel("Time (mins)")
    ax[2, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 0].title.set_text(r"$\langle \delta Q^1 \delta Q^2 \rangle$")

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
    fig.colorbar(c, ax=ax[2, 1])
    ax[2, 1].set_xlabel("Time (mins)")
    ax[2, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 1].title.set_text(r"$\langle \delta P_1 \delta Q^1 \rangle$")

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
    fig.colorbar(c, ax=ax[2, 2])
    ax[2, 2].set_xlabel("Time (mins)")
    ax[2, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 2].title.set_text(r"$\langle \delta P_1 \delta Q^2 \rangle$")

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
    fig.colorbar(c, ax=ax[2, 3])
    ax[2, 3].set_xlabel("Time (mins)")
    ax[2, 3].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 3].title.set_text(r"$\langle \delta P_2 \delta Q^2 \rangle$")

    fig.savefig(
        f"results/Correlations {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# display all norm correlations
if False:
    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")
    df = pd.read_pickle(f"databases/dfShape{fileType}.pkl")

    fig, ax = plt.subplots(3, 4, figsize=(30, 18))

    T, R, Theta = dfCor["dRhodRho"].iloc[0].shape

    dRhodRho = np.zeros([len(filenames), T, R])
    dQ1dRho = np.zeros([len(filenames), T, R])
    dQ2dRho = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):
        RhoCount = dfCor["Count Rho"].iloc[i][:, :, :-1]
        dRhodRho[i] = np.sum(
            dfCor["dRhodRho"].iloc[i][:, :, :-1] * RhoCount, axis=2
        ) / np.sum(RhoCount, axis=2)
        RhoQCount = dfCor["Count Rho Q"].iloc[i][:, :, :-1]
        dQ1dRho[i] = np.sum(
            dfCor["dQ1dRho"].iloc[i][:, :, :-1] * RhoQCount, axis=2
        ) / np.sum(RhoQCount, axis=2)
        dQ2dRho[i] = np.sum(
            dfCor["dQ2dRho"].iloc[i][:, :, :-1] * RhoQCount, axis=2
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
    t, r = np.mgrid[0:180:10, 0:90:10]
    c = ax[0, 0].pcolor(
        t,
        r,
        dRhodRho,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 0])
    ax[0, 0].set_xlabel("Time (mins)")
    ax[0, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 0].title.set_text(r"$\langle \delta \rho \delta \rho \rangle$")

    c = ax[0, 1].pcolor(
        t,
        r,
        dRhodRho - dRhodRho[-1],
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 1])
    ax[0, 1].set_xlabel("Time (mins)")
    ax[0, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 1].title.set_text(r"$\langle \delta \rho \delta \rho \rangle$ nosie")

    c = ax[0, 2].pcolor(
        t,
        r,
        dQ1dRho,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 2])
    ax[0, 2].set_xlabel("Time (mins)")
    ax[0, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 2].title.set_text(r"$\langle \delta Q^1 \delta \rho \rangle$")

    c = ax[0, 3].pcolor(
        t,
        r,
        dQ2dRho,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 3])
    ax[0, 3].set_xlabel("Time (mins)")
    ax[0, 3].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 3].title.set_text(r"$\langle \delta Q^2 \delta \rho \rangle$")

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

    t, r = np.mgrid[0:102:2, 0:82:2]
    c = ax[1, 0].pcolor(
        t,
        r,
        dP1dP1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set_xlabel("Time (mins)")
    ax[1, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 0].title.set_text(r"$\langle \delta P_1 \delta P_1 \rangle$")

    c = ax[1, 1].pcolor(
        t,
        r,
        dP2dP2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 1])
    ax[1, 1].set_xlabel("Time (mins)")
    ax[1, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 1].title.set_text(r"$\langle \delta P_2 \delta P_2 \rangle$")

    c = ax[1, 2].pcolor(
        t,
        r,
        dQ1dQ1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 2])
    ax[1, 2].set_xlabel("Time (mins)")
    ax[1, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 2].title.set_text(r"$\langle \delta Q^1 \delta Q^1 \rangle$")

    c = ax[1, 3].pcolor(
        t,
        r,
        dQ2dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 3])
    ax[1, 3].set_xlabel("Time (mins)")
    ax[1, 3].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 3].title.set_text(r"$\langle \delta Q^2 \delta Q^2 \rangle$")

    c = ax[2, 0].pcolor(
        t,
        r,
        dQ1dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 0])
    ax[2, 0].set_xlabel("Time (mins)")
    ax[2, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 0].title.set_text(r"$\langle \delta Q^1 \delta Q^2 \rangle$")

    c = ax[2, 1].pcolor(
        t,
        r,
        dP1dQ1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 1])
    ax[2, 1].set_xlabel("Time (mins)")
    ax[2, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 1].title.set_text(r"$\langle \delta P_1 \delta Q^1 \rangle$")

    c = ax[2, 2].pcolor(
        t,
        r,
        dP1dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 2])
    ax[2, 2].set_xlabel("Time (mins)")
    ax[2, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 2].title.set_text(r"$\langle \delta P_1 \delta Q^2 \rangle$")

    c = ax[2, 3].pcolor(
        t,
        r,
        dP2dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 3])
    ax[2, 3].set_xlabel("Time (mins)")
    ax[2, 3].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 3].title.set_text(r"$\langle \delta P_2 \delta Q^2 \rangle$")

    fig.savefig(
        f"results/Correlations Norm {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# fit carves dRhodRho based on model
if False:
    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    T, R, Theta = dfCor["dRhodRho"].iloc[0].shape

    dRhodRho = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):
        RhoCount = dfCor["Count Rho"].iloc[i][:, :, :-1]
        dRhodRho[i] = np.sum(
            dfCor["dRhodRho"].iloc[i][:, :, :-1] * RhoCount, axis=2
        ) / np.sum(RhoCount, axis=2)

    dfCor = 0

    dRhodRho = np.mean(dRhodRho, axis=0)

    dRhodRhoN = dRhodRho - dRhodRho[-1]
    limMax = np.max(dRhodRhoN[1:, 0])
    limMin = np.min(dRhodRhoN[0, 1:])

    m = sp.optimize.curve_fit(
        f=corRho_R,
        xdata=np.linspace(10, 80, 8),
        ydata=dRhodRhoN[0, 1:],
        p0=(0.003, 10),
    )[0]

    ax[0].plot(np.linspace(10, 80, 8), dRhodRhoN[0, 1:])
    ax[0].plot(np.linspace(10, 80, 8), corRho_R(np.linspace(10, 80, 8), m[0], m[1]))
    ax[0].set_xlabel(r"$R (\mu m)$ ")
    ax[0].set_ylabel(r"$\delta\rho$ Correlation")
    ax[0].set_ylim([limMin * 1.1, limMax * 1.05])
    ax[0].title.set_text(
        r"$\langle \delta \rho \delta \rho \rangle$ noise $R$ $=Ce^{-aR}\cos(\omega R)$"
    )

    m = sp.optimize.curve_fit(
        f=corRho_T,
        xdata=np.linspace(10, 170, 17),
        ydata=dRhodRhoN[1:, 0],
        p0=0.003,
    )[0]

    ax[1].plot(np.linspace(10, 170, 17), dRhodRhoN[1:, 0])
    ax[1].plot(np.linspace(10, 170, 17), corRho_T(np.linspace(10, 170, 17), m))
    ax[1].set_xlabel(r"Time (mins)")
    ax[1].set_ylabel(r"$\delta\rho$ Correlation")
    ax[1].set_ylim([limMin * 1.1, limMax * 1.05])
    ax[1].title.set_text(
        r"$\langle \delta \rho \delta \rho \rangle$ noise $T$ $=Ce^{-aT}$"
    )

    fig.savefig(
        f"results/fit dRhodRho model",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# fit carves dRhodRho
if False:
    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(1, 3, figsize=(26, 6))

    T, R, Theta = dfCor["dRhodRho"].iloc[0].shape

    dRhodRho = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):
        RhoCount = dfCor["Count Rho"].iloc[i][:, :, :-1]
        dRhodRho[i] = np.sum(
            dfCor["dRhodRho"].iloc[i][:, :, :-1] * RhoCount, axis=2
        ) / np.sum(RhoCount, axis=2)

    dfCor = 0

    dRhodRho = np.mean(dRhodRho, axis=0)

    dRhodRhoS = np.mean(dRhodRho, axis=0)

    m = sp.optimize.curve_fit(
        f=expCos,
        xdata=np.linspace(0, 80, 9),
        ydata=dRhodRhoS,
        p0=(0.0003, 0.04, 0.01),
    )[0]

    limMax = np.max(dRhodRhoS)
    limMin = np.min(dRhodRhoS)

    ax[0].plot(np.linspace(0, 80, 9), dRhodRhoS)
    ax[0].plot(
        np.linspace(0, 80, 900), expCos(np.linspace(0, 80, 900), m[0], m[1], m[2])
    )
    ax[0].set_xlabel(r"$R (\mu m)$ ")
    ax[0].set_ylabel(r"$\delta\rho$ Correlation")
    ax[0].set_ylim([limMin * 1.1, limMax * 1.05])
    ax[0].title.set_text(
        r"$\langle \delta \rho \delta \rho \rangle$ structure $=Ce^{-aR}\cos(\omega R)$"
    )

    dRhodRhoN = dRhodRho - dRhodRho[-1]
    limMax = np.max(dRhodRhoN[1:, 0])
    limMin = np.min(dRhodRhoN[0, 1:])

    m = sp.optimize.curve_fit(
        f=expCos,
        xdata=np.linspace(10, 80, 8),
        ydata=dRhodRhoN[0, 1:],
        p0=(0.00008, 0.00004, 0.03),
    )[0]

    ax[1].plot(np.linspace(10, 80, 8), dRhodRhoN[0, 1:])
    ax[1].plot(
        np.linspace(10, 80, 800), expCos(np.linspace(10, 80, 800), m[0], m[1], m[2])
    )
    ax[1].set_xlabel(r"$R (\mu m)$ ")
    ax[1].set_ylabel(r"$\delta\rho$ Correlation")
    ax[1].set_ylim([limMin * 1.1, limMax * 1.05])
    ax[1].title.set_text(
        r"$\langle \delta \rho \delta \rho \rangle$ noise $R$ $=Ce^{-aR}\cos(\omega R)$"
    )

    m = sp.optimize.curve_fit(
        f=exp,
        xdata=np.linspace(10, 170, 17),
        ydata=dRhodRhoN[1:, 0],
        p0=(0.0003, 0.04),
    )[0]

    ax[2].plot(np.linspace(10, 170, 17), dRhodRhoN[1:, 0])
    ax[2].plot(np.linspace(10, 170, 17), exp(np.linspace(10, 170, 17), m[0], m[1]))
    ax[2].set_xlabel(r"Time (mins)")
    ax[2].set_ylabel(r"$\delta\rho$ Correlation")
    ax[2].set_ylim([limMin * 1.1, limMax * 1.05])
    ax[2].title.set_text(
        r"$\langle \delta \rho \delta \rho \rangle$ noise $T$ $=Ce^{-aT}$"
    )

    fig.savefig(
        f"results/fit dRhodRho",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# fit carves Polarisation
if False:
    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))

    T, R, Theta = dfCor["dP1dP1"].iloc[0].shape

    dP1dP1 = np.zeros([len(filenames), T, R])
    dP2dP2 = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):
        Count = dfCor["Count"].iloc[i][:, :, :-1]
        dP1dP1[i] = np.sum(dfCor["dP1dP1"].iloc[i][:, :, :-1] * Count, axis=2) / np.sum(
            Count, axis=2
        )
        dP2dP2[i] = np.sum(dfCor["dP2dP2"].iloc[i][:, :, :-1] * Count, axis=2) / np.sum(
            Count, axis=2
        )

    dfCor = 0

    dP1dP1 = np.mean(dP1dP1, axis=0)
    dP2dP2 = np.mean(dP2dP2, axis=0)

    limMax = np.max(dP1dP1[:, :-1])
    limMin = np.min(dP1dP1[:, :-1])

    m = sp.optimize.curve_fit(
        f=expCos,
        xdata=np.linspace(0, 80, 41),
        ydata=dP1dP1[0, :-1],
        p0=(0.00008, 0.00004, 1),
    )[0]

    print(m[0], m[1], m[2])

    ax[0, 0].plot(np.linspace(0, 80, 41), dP1dP1[0, :-1])
    ax[0, 0].plot(
        np.linspace(0, 80, 410), expCos(np.linspace(0, 80, 410), m[0], m[1], m[2])
    )
    ax[0, 0].set_xlabel(r"$R (\mu m)$ ")
    ax[0, 0].set_ylabel(r"$\langle \delta P_1 \delta P_1 \rangle$")
    ax[0, 0].set_ylim([limMin * 2, limMax * 1.05])
    ax[0, 0].title.set_text(
        r"$\langle \delta P_1(r,t) \delta P_1(r',t) \rangle = Ce^{-aR}\cos(\omega R)$"
    )

    m = sp.optimize.curve_fit(
        f=expStretched,
        xdata=np.linspace(0, 100, 51),
        ydata=dP1dP1[:, 0],
        p0=(0.0003, 0.04, 1),
    )[0]

    print(m[0], m[1], m[2])

    ax[0, 1].plot(np.linspace(0, 100, 51), dP1dP1[:, 0])
    ax[0, 1].plot(
        np.linspace(0, 100, 510),
        expStretched(np.linspace(0, 100, 510), m[0], m[1], m[2]),
    )
    ax[0, 1].set_xlabel(r"Time (mins)")
    ax[0, 1].set_ylabel(r"$\langle \delta P_1 \delta P_1 \rangle$")
    ax[0, 1].set_ylim([limMin * 2, limMax * 1.05])
    ax[0, 1].title.set_text(
        r"$\langle \delta P_1(r,t) \delta P_1(r,t') \rangle = Ce^{-aT^\alpha}$"
    )

    limMax = np.max(dP2dP2[:, :-1])
    limMin = np.min(dP2dP2[:, :-1])

    m = sp.optimize.curve_fit(
        f=expCos,
        xdata=np.linspace(0, 80, 41),
        ydata=dP2dP2[0, :-1],
        p0=(0.00008, 0.00004, 1),
    )[0]

    print(m[0], m[1], m[2])

    ax[1, 0].plot(np.linspace(0, 80, 41), dP2dP2[0, :-1])
    ax[1, 0].plot(
        np.linspace(0, 80, 410), expCos(np.linspace(0, 80, 410), m[0], m[1], m[2])
    )
    ax[1, 0].set_xlabel(r"$R (\mu m)$ ")
    ax[1, 0].set_ylabel(r"$\langle \delta P_2 \delta P_2 \rangle$")
    ax[1, 0].set_ylim([limMin * 2, limMax * 1.05])
    ax[1, 0].title.set_text(
        r"$\langle \delta P_2(r,t) \delta P_2(r',t) \rangle = Ce^{-aR}\cos(\omega R)$"
    )

    m = sp.optimize.curve_fit(
        f=expStretched,
        xdata=np.linspace(0, 100, 51),
        ydata=dP2dP2[:, 0],
        p0=(0.0003, 0.04, 1),
    )[0]

    print(m[0], m[1], m[2])

    ax[1, 1].plot(np.linspace(0, 100, 51), dP2dP2[:, 0])
    ax[1, 1].plot(
        np.linspace(0, 100, 510),
        expStretched(np.linspace(0, 100, 510), m[0], m[1], m[2]),
    )
    ax[1, 1].set_xlabel(r"Time (mins)")
    ax[1, 1].set_ylabel(r"$\langle \delta P_2 \delta P_2 \rangle$")
    ax[1, 1].set_ylim([limMin * 2, limMax * 1.05])
    ax[1, 1].title.set_text(
        r"$\langle \delta P_2(r,t) \delta P_2(r,t') \rangle = Ce^{-aT^\alpha}$"
    )

    fig.savefig(
        f"results/fit polarisation",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# fit carves Q
if True:
    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))

    T, R, Theta = dfCor["dQ1dQ1Correlation"].iloc[0][:, :-1, :-1].shape

    dQ1dQ1 = np.zeros([len(filenames), T, R])
    dQ2dQ2 = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):
        dQ1dQ1total = dfCor["dQ1dQ1Count"].iloc[i][:, :-1, :-1]
        dQ1dQ1[i] = np.sum(
            dfCor["dQ1dQ1Correlation"].iloc[i][:, :-1, :-1] * dQ1dQ1total, axis=2
        ) / np.sum(dQ1dQ1total, axis=2)

        dQ2dQ2total = dfCor["dQ2dQ2Count"].iloc[i][:, :-1, :-1]
        dQ2dQ2[i] = np.sum(
            dfCor["dQ2dQ2Correlation"].iloc[i][:, :-1, :-1] * dQ2dQ2total, axis=2
        ) / np.sum(dQ2dQ2total, axis=2)

    dfCor = 0

    dQ1dQ1 = np.mean(dQ1dQ1, axis=0)
    dQ2dQ2 = np.mean(dQ2dQ2, axis=0)

    limMax = np.max(dQ1dQ1[:, :-1])
    limMin = np.min(dQ1dQ1[:, :-1])

    m = sp.optimize.curve_fit(
        f=explinear,
        xdata=np.linspace(0, 80, 40),
        ydata=dQ1dQ1[0, :-1],
        p0=(0.0006, 1, -3e-06, 0.00016),
    )[0]

    print(m[0], m[1], m[2], m[3])

    ax[0, 0].plot(np.linspace(0, 80, 40), dQ1dQ1[0, :-1])
    ax[0, 0].plot(
        np.linspace(0, 80, 400),
        explinear(np.linspace(0, 80, 400), m[0], m[1], m[2], m[3]),
    )
    ax[0, 0].set_xlabel(r"$R (\mu m)$ ")
    ax[0, 0].set_ylabel(r"$\langle \delta Q_1 \delta Q_1 \rangle$")
    ax[0, 0].set_ylim([limMin * 2, limMax * 1.05])
    ax[0, 0].title.set_text(
        r"$\langle \delta Q_1(r,t) \delta Q_1(r',t) \rangle = Ce^{-aR} + mR + c$"
    )

    m = sp.optimize.curve_fit(
        f=expStretched,
        xdata=np.linspace(0, 100, 51),
        ydata=dQ1dQ1[:, 0],
        p0=(0.0003, 0.04, 1),
    )[0]

    # print(m[0], m[1], m[2])

    ax[0, 1].plot(np.linspace(0, 100, 51), dQ1dQ1[:, 0])
    ax[0, 1].plot(
        np.linspace(0, 100, 510),
        expStretched(np.linspace(0, 100, 510), m[0], m[1], m[2]),
    )
    ax[0, 1].set_xlabel(r"Time (mins)")
    ax[0, 1].set_ylabel(r"$\langle \delta Q_1 \delta Q_1 \rangle$")
    ax[0, 1].set_ylim([limMin * 2, limMax * 1.05])
    ax[0, 1].title.set_text(
        r"$\langle \delta Q_1(r,t) \delta Q_1(r,t') \rangle = Ce^{-aT^\alpha}$"
    )

    limMax = np.max(dQ2dQ2[:, :-1])
    limMin = np.min(dQ2dQ2[:, :-1])

    m = sp.optimize.curve_fit(
        f=explinear,
        xdata=np.linspace(0, 80, 40),
        ydata=dQ2dQ2[0, :-1],
        p0=(0.0006, 1, -3e-06, 0.00016),
    )[0]

    print(m[0], m[1], m[2], m[3])

    ax[1, 0].plot(np.linspace(0, 80, 40), dQ2dQ2[0, :-1])
    ax[1, 0].plot(
        np.linspace(0, 80, 400),
        explinear(np.linspace(0, 80, 400), m[0], m[1], m[2], m[3]),
    )
    ax[1, 0].set_xlabel(r"$R (\mu m)$ ")
    ax[1, 0].set_ylabel(r"$\langle \delta Q_2 \delta Q_2 \rangle$")
    ax[1, 0].set_ylim([limMin * 2, limMax * 1.05])
    ax[1, 0].title.set_text(
        r"$\langle \delta Q_2(r,t) \delta Q_2(r',t) \rangle = Ce^{-aR} + mR + c$"
    )

    m = sp.optimize.curve_fit(
        f=expStretched,
        xdata=np.linspace(0, 100, 51),
        ydata=dQ2dQ2[:, 0],
        p0=(0.0003, 0.04, 1),
    )[0]

    # print(m[0], m[1], m[2])

    ax[1, 1].plot(np.linspace(0, 100, 51), dQ2dQ2[:, 0])
    ax[1, 1].plot(
        np.linspace(0, 100, 510),
        expStretched(np.linspace(0, 100, 510), m[0], m[1], m[2]),
    )
    ax[1, 1].set_xlabel(r"Time (mins)")
    ax[1, 1].set_ylabel(r"$\langle \delta Q_2 \delta Q_2 \rangle$")
    ax[1, 1].set_ylim([limMin * 2, limMax * 1.05])
    ax[1, 1].title.set_text(
        r"$\langle \delta Q_2(r,t) \delta Q_2(r,t') \rangle = Ce^{-aT^\alpha}$"
    )

    fig.savefig(
        f"results/fit Q",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

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

# deltaP1
if False:
    plt.rcParams.update({"font.size": 7})
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

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    ax[0].plot(T[1:], deltaP1Correlation[:, 0][1:], label="Data")
    ax[0].plot(T[1:], CorR0(T[1:], m[0], m[1]), label="Model")
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel(r"$P_1$ Correlation")
    ax[0].set_ylim([-0.000005, 0.000025])
    ax[0].set_xlim([0, 2 * timeGrid])
    ax[0].title.set_text(r"Correlation of $\delta P_1$, $R=0$")
    ax[0].legend()

    m = sp.optimize.curve_fit(
        f=Integral,
        xdata=R,
        ydata=deltaP1Correlation[1],
        p0=0.025,
        method="lm",
    )[0]

    ax[1].plot(R, deltaP1Correlation[1], label="Data")
    ax[1].plot(R, Integral(R, m), label="Model")
    ax[1].set_xlabel(r"$R (\mu m)$")
    ax[1].set_ylabel(r"$P_1$ Correlation")
    ax[1].set_ylim([-0.000005, 0.000025])
    ax[1].title.set_text(r"Correlation of $\delta P_1$, $T=2$")
    ax[1].legend()
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

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    ax[0].plot(T[1:], deltaP2Correlation[:, 0][1:], label="Data")
    ax[0].plot(T[1:], CorR0(T[1:], m[0], m_P1[1]), label="Model")
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel(r"$P_2$ Correlation")
    ax[0].set_ylim([-0.000002, 0.00001])
    ax[0].set_xlim([0, 2 * timeGrid])
    ax[0].title.set_text(r"Correlation of $\delta P_2$, $R=0$")
    ax[0].legend()

    m = sp.optimize.curve_fit(
        f=Integral_P2,
        xdata=R,
        ydata=deltaP2Correlation[1],
        p0=0.025,
        method="lm",
    )[0]

    ax[1].plot(R, deltaP2Correlation[1], label="Data")
    ax[1].plot(R, Integral_P2(R, m), label="Model")
    ax[1].set_xlabel(r"$R (\mu m)$")
    ax[1].set_ylabel(r"$P_2$ Correlation")
    ax[1].set_ylim([-0.000002, 0.00001])
    ax[1].title.set_text(r"Correlation of $\delta P_2$, $T=2$")
    ax[1].legend()
    fig.savefig(
        f"results/Correlation P2 in T and R {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# Integral over model function
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

# Integral function P correlation
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

# fit cavre for dP1
if False:
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

# fit cavre for dP2
if False:

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


# ------------------- Shape-rho correlations

# Correlation Rho str
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

# Correlation Rho
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

# Correlation rho in T and R
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
