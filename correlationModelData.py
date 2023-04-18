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


def OLSfit(x, y, dy=None):
    """Find the best fitting parameters of a linear fit to the data through the
    method of ordinary least squares estimation. (i.e. find m and b for
    y = m*x + b)

    Args:
        x: Numpy array of independent variable data
        y: Numpy array of dependent variable data. Must have same size as x.
        dy: Numpy array of dependent variable standard deviations. Must be same
            size as y.

    Returns: A list with four floating point values. [m, dm, b, db]
    """
    if dy is None:
        # if no error bars, weight every point the same
        dy = np.ones(x.size)
    denom = np.sum(1 / dy**2) * np.sum((x / dy) ** 2) - (np.sum(x / dy**2)) ** 2
    m = (
        np.sum(1 / dy**2) * np.sum(x * y / dy**2)
        - np.sum(x / dy**2) * np.sum(y / dy**2)
    ) / denom
    b = (
        np.sum(x**2 / dy**2) * np.sum(y / dy**2)
        - np.sum(x / dy**2) * np.sum(x * y / dy**2)
    ) / denom
    dm = np.sqrt(np.sum(1 / dy**2) / denom)
    db = np.sqrt(np.sum(x / dy**2) / denom)
    return [m, b]


def Corr_R0(t, B, D):
    return D * -sc.expi(-B * t)


def forIntegral(k, R, T, B, C, L):
    k, R, T = np.meshgrid(k, R, T, indexing="ij")
    return C * k * np.exp(-(B + L * k**2) * T) * sc.jv(0, R * k) / (B + L * k**2)


# -------------------

grid = 26
timeGrid = 51

# B^(2)
if False:

    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    q1 = np.zeros([len(filenames), T])
    for i in range(len(filenames)):
        filename = filenames[i]
        df = dfShape[dfShape["Filename"] == filename]
        for t in range(T):
            q1[i, t] = np.mean(df["q"][df["T"] == t])[0, 0]

    time = 2 * np.array(range(T))

    Q1 = np.mean(q1, axis=0)
    bestfit = OLSfit(time, Q1)
    (m, c) = (bestfit[0], bestfit[1])

# display all correlations shape
if False:
    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    fig, ax = plt.subplots(4, 3, figsize=(16, 16))

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
    fig.colorbar(c, ax=ax[0, 0])
    ax[0, 0].set_xlabel("Time (mins)")
    ax[0, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 0].title.set_text(
        r"$\langle (\delta \rho_s + \delta \rho) (\delta \rho_s + \delta \rho) \rangle$"
    )

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
    ax[0, 1].title.set_text(r"$\langle \delta \rho \delta \rho \rangle$")

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
    c = ax[1, 0].pcolor(
        t,
        r,
        dQ2dRho,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set_xlabel("Time (mins)")
    ax[1, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 0].title.set_text(r"$\langle \delta Q^2 \delta \rho \rangle$")

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
    c = ax[1, 1].pcolor(
        t,
        r,
        dP1dP1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 1])
    ax[1, 1].set_xlabel("Time (mins)")
    ax[1, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 1].title.set_text(r"$\langle \delta P_1 \delta P_1 \rangle$")

    maxCorr = np.max([dP2dP2, -dP2dP2])
    c = ax[1, 2].pcolor(
        t,
        r,
        dP2dP2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 2])
    ax[1, 2].set_xlabel("Time (mins)")
    ax[1, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 2].title.set_text(r"$\langle \delta P_2 \delta P_2 \rangle$")

    maxCorr = np.max([dQ1dQ1, -dQ1dQ1])
    c = ax[2, 0].pcolor(
        t,
        r,
        dQ1dQ1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 0])
    ax[2, 0].set_xlabel("Time (mins)")
    ax[2, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 0].title.set_text(r"$\langle \delta Q^1 \delta Q^1 \rangle$")

    maxCorr = np.max([dQ2dQ2, -dQ2dQ2])
    c = ax[2, 1].pcolor(
        t,
        r,
        dQ2dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 1])
    ax[2, 1].set_xlabel("Time (mins)")
    ax[2, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 1].title.set_text(r"$\langle \delta Q^2 \delta Q^2 \rangle$")

    maxCorr = np.max([dQ1dQ2, -dQ1dQ2])
    c = ax[2, 2].pcolor(
        t,
        r,
        dQ1dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 2])
    ax[2, 2].set_xlabel("Time (mins)")
    ax[2, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 2].title.set_text(r"$\langle \delta Q^1 \delta Q^2 \rangle$")

    maxCorr = np.max([dP1dQ1, -dP1dQ1])
    c = ax[3, 0].pcolor(
        t,
        r,
        dP1dQ1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[3, 0])
    ax[3, 0].set_xlabel("Time (mins)")
    ax[3, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[3, 0].title.set_text(r"$\langle \delta P_1 \delta Q^1 \rangle$")

    maxCorr = np.max([dP1dQ2, -dP1dQ2])
    c = ax[3, 1].pcolor(
        t,
        r,
        dP1dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[3, 1])
    ax[3, 1].set_xlabel("Time (mins)")
    ax[3, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[3, 1].title.set_text(r"$\langle \delta P_1 \delta Q^2 \rangle$")

    maxCorr = np.max([dP2dQ2, -dP2dQ2])
    c = ax[3, 2].pcolor(
        t,
        r,
        dP2dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[3, 2])
    ax[3, 2].set_xlabel("Time (mins)")
    ax[3, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[3, 2].title.set_text(r"$\langle \delta P_2 \delta Q^2 \rangle$")

    # plt.subplot_tool()
    plt.subplots_adjust(
        left=0.075, bottom=0.1, right=0.95, top=0.9, wspace=0.4, hspace=0.45
    )

    fig.savefig(
        f"results/Correlations {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# display all norm correlations shape
if False:
    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")
    df = pd.read_pickle(f"databases/dfShape{fileType}.pkl")

    fig, ax = plt.subplots(4, 3, figsize=(16, 16))

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
    fig.colorbar(c, ax=ax[0, 0])
    ax[0, 0].set_xlabel("Time (mins)")
    ax[0, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 0].title.set_text(
        r"$\langle (\delta \rho_s + \delta \rho) (\delta \rho_s + \delta \rho) \rangle$"
    )

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
    ax[0, 1].title.set_text(r"$\langle \delta \rho \delta \rho \rangle$")

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

    c = ax[1, 0].pcolor(
        t,
        r,
        dQ2dRho,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set_xlabel("Time (mins)")
    ax[1, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 0].title.set_text(r"$\langle \delta Q^2 \delta \rho \rangle$")

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
    c = ax[1, 1].pcolor(
        t,
        r,
        dP1dP1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 1])
    ax[1, 1].set_xlabel("Time (mins)")
    ax[1, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 1].title.set_text(r"$\langle \delta P_1 \delta P_1 \rangle$")

    c = ax[1, 2].pcolor(
        t,
        r,
        dP2dP2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 2])
    ax[1, 2].set_xlabel("Time (mins)")
    ax[1, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 2].title.set_text(r"$\langle \delta P_2 \delta P_2 \rangle$")

    c = ax[2, 0].pcolor(
        t,
        r,
        dQ1dQ1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 0])
    ax[2, 0].set_xlabel("Time (mins)")
    ax[2, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 0].title.set_text(r"$\langle \delta Q^1 \delta Q^1 \rangle$")

    c = ax[2, 1].pcolor(
        t,
        r,
        dQ2dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 1])
    ax[2, 1].set_xlabel("Time (mins)")
    ax[2, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 1].title.set_text(r"$\langle \delta Q^2 \delta Q^2 \rangle$")

    c = ax[2, 2].pcolor(
        t,
        r,
        dQ1dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 2])
    ax[2, 2].set_xlabel("Time (mins)")
    ax[2, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 2].title.set_text(r"$\langle \delta Q^1 \delta Q^2 \rangle$")

    c = ax[3, 0].pcolor(
        t,
        r,
        dP1dQ1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[3, 0])
    ax[3, 0].set_xlabel("Time (mins)")
    ax[3, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[3, 0].title.set_text(r"$\langle \delta P_1 \delta Q^1 \rangle$")

    c = ax[3, 1].pcolor(
        t,
        r,
        dP1dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[3, 1])
    ax[3, 1].set_xlabel("Time (mins)")
    ax[3, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[3, 1].title.set_text(r"$\langle \delta P_1 \delta Q^2 \rangle$")

    c = ax[3, 2].pcolor(
        t,
        r,
        dP2dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[3, 2])
    ax[3, 2].set_xlabel("Time (mins)")
    ax[3, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[3, 2].title.set_text(r"$\langle \delta P_2 \delta Q^2 \rangle$")

    # plt.subplot_tool()
    plt.subplots_adjust(
        left=0.075, bottom=0.1, right=0.95, top=0.9, wspace=0.4, hspace=0.45
    )

    fig.savefig(
        f"results/Correlations Norm {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# display all correlations velocity
if False:
    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    fig, ax = plt.subplots(4, 3, figsize=(16, 16))

    T, R, Theta = dfCor["dr1dRho_SdV1Correlation"].iloc[0].shape

    dr1dRhodV1 = np.zeros([len(filenames), T - 1, R - 1])
    dr1dRhodV2 = np.zeros([len(filenames), T - 1, R - 1])
    dr2dRhodV1 = np.zeros([len(filenames), T - 1, R - 1])
    dr2dRhodV2 = np.zeros([len(filenames), T - 1, R - 1])
    for i in range(len(filenames)):
        dr1Total = dfCor["dr1dRho_SdV1Count"].iloc[i][:-1, :-1, :-1]
        dr1dRhodV1[i] = np.sum(
            dfCor["dr1dRho_SdV1Correlation"].iloc[i][:-1, :-1, :-1] * dr1Total, axis=2
        ) / np.sum(dr1Total, axis=2)
        dr1dRhodV2[i] = np.sum(
            dfCor["dr1dRho_SdV2Correlation"].iloc[i][:-1, :-1, :-1] * dr1Total, axis=2
        ) / np.sum(dr1Total, axis=2)

        dr2Total = dfCor["dr1dRho_SdV1Count"].iloc[i][:-1, :-1, :-1]
        dr2dRhodV1[i] = np.sum(
            dfCor["dr2dRho_SdV1Correlation"].iloc[i][:-1, :-1, :-1] * dr2Total, axis=2
        ) / np.sum(dr2Total, axis=2)
        dr2dRhodV2[i] = np.sum(
            dfCor["dr2dRho_SdV2Correlation"].iloc[i][:-1, :-1, :-1] * dr2Total, axis=2
        ) / np.sum(dr2Total, axis=2)

    dr1dRhodV1 = np.mean(dr1dRhodV1, axis=0)
    dr1dRhodV2 = np.mean(dr1dRhodV2, axis=0)
    dr2dRhodV1 = np.mean(dr2dRhodV1, axis=0)
    dr2dRhodV2 = np.mean(dr2dRhodV2, axis=0)

    t, r = np.mgrid[0:170:10, 0:52:2]

    maxCorr = np.max([dr1dRhodV1[:50, :25], -dr1dRhodV1[:50, :25]])
    c = ax[0, 0].pcolor(
        t,
        r,
        dr1dRhodV1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 0])
    ax[0, 0].set_xlabel("Time (mins)")
    ax[0, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 0].title.set_text(
        r"$\langle \partial_{r_1} (\delta \rho_s + \delta \rho) \delta V_1 \rangle$"
    )

    maxCorr = np.max([dr1dRhodV2, -dr1dRhodV2])
    c = ax[0, 1].pcolor(
        t,
        r,
        dr1dRhodV2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 1])
    ax[0, 1].set_xlabel("Time (mins)")
    ax[0, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 1].title.set_text(
        r"$\langle \partial_{r_1} (\delta \rho_s + \delta \rho) \delta V_2 \rangle$"
    )

    maxCorr = np.max([dr2dRhodV1, -dr2dRhodV1])
    c = ax[0, 2].pcolor(
        t,
        r,
        dr2dRhodV1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 2])
    ax[0, 2].set_xlabel("Time (mins)")
    ax[0, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 2].title.set_text(
        r"$\langle \partial_{r_2} (\delta \rho_s + \delta \rho) \delta V_1 \rangle$"
    )

    maxCorr = np.max([dr2dRhodV2, -dr2dRhodV2])
    c = ax[1, 0].pcolor(
        t,
        r,
        dr2dRhodV2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set_xlabel("Time (mins)")
    ax[1, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 0].title.set_text(
        r"$\langle \partial_{r_2} (\delta \rho_s + \delta \rho) \delta V_2 \rangle$"
    )

    dr1dQ1dV1 = np.zeros([len(filenames), T - 1, R - 1])
    dr1dQ1dV2 = np.zeros([len(filenames), T - 1, R - 1])
    dr2dQ1dV1 = np.zeros([len(filenames), T - 1, R - 1])
    dr2dQ1dV2 = np.zeros([len(filenames), T - 1, R - 1])
    for i in range(len(filenames)):
        dr1Total = dfCor["dr1dQ1dV1Count"].iloc[i][:-1, :-1, :-1]
        dr1dQ1dV1[i] = np.sum(
            dfCor["dr1dQ1dV1Correlation"].iloc[i][:-1, :-1, :-1] * dr1Total, axis=2
        ) / np.sum(dr1Total, axis=2)
        dr1dQ1dV2[i] = np.sum(
            dfCor["dr1dQ1dV2Correlation"].iloc[i][:-1, :-1, :-1] * dr1Total, axis=2
        ) / np.sum(dr1Total, axis=2)

        dr2Total = dfCor["dr2dQ1dV1Count"].iloc[i][:-1, :-1, :-1]
        dr2dQ1dV1[i] = np.sum(
            dfCor["dr2dQ1dV1Correlation"].iloc[i][:-1, :-1, :-1] * dr2Total, axis=2
        ) / np.sum(dr2Total, axis=2)
        dr2dQ1dV2[i] = np.sum(
            dfCor["dr2dQ1dV2Correlation"].iloc[i][:-1, :-1, :-1] * dr2Total, axis=2
        ) / np.sum(dr2Total, axis=2)

    dr1dQ1dV1 = np.mean(dr1dQ1dV1, axis=0)
    dr1dQ1dV2 = np.mean(dr1dQ1dV2, axis=0)
    dr2dQ1dV1 = np.mean(dr2dQ1dV1, axis=0)
    dr2dQ1dV2 = np.mean(dr2dQ1dV2, axis=0)

    maxCorr = np.max([dr1dQ1dV1, -dr1dQ1dV1])
    c = ax[1, 1].pcolor(
        t,
        r,
        dr1dQ1dV1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 1])
    ax[1, 1].set_xlabel("Time (mins)")
    ax[1, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 1].title.set_text(
        r"$\langle \partial_{r_1} \delta Q^{(1)} \delta V_1 \rangle$"
    )

    maxCorr = np.max([dr1dQ1dV2, -dr1dQ1dV2])
    c = ax[1, 2].pcolor(
        t,
        r,
        dr1dQ1dV2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 2])
    ax[1, 2].set_xlabel("Time (mins)")
    ax[1, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 2].title.set_text(
        r"$\langle \partial_{r_1} \delta Q^{(1)} \delta V_2 \rangle$"
    )

    maxCorr = np.max([dr2dQ1dV1, -dr2dQ1dV1])
    c = ax[2, 0].pcolor(
        t,
        r,
        dr2dQ1dV1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 0])
    ax[2, 0].set_xlabel("Time (mins)")
    ax[2, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 0].title.set_text(
        r"$\langle \partial_{r_2} \delta Q^{(1)} \delta V_1 \rangle$"
    )

    maxCorr = np.max([dr2dQ1dV2, -dr2dQ1dV2])
    c = ax[2, 1].pcolor(
        t,
        r,
        dr2dQ1dV2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 1])
    ax[2, 1].set_xlabel("Time (mins)")
    ax[2, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 1].title.set_text(
        r"$\langle \partial_{r_2} \delta Q^{(1)} \delta V_2 \rangle$"
    )

    T, R, Theta = dfCor["dP1dV1Correlation"].iloc[0].shape

    dP1dV1 = np.zeros([len(filenames), T - 1, R - 1])
    dP2dV2 = np.zeros([len(filenames), T - 1, R - 1])
    dV1dV1 = np.zeros([len(filenames), T - 1, R - 1])
    dV2dV2 = np.zeros([len(filenames), T - 1, R - 1])
    for i in range(len(filenames)):
        dP1dV1total = dfCor["dP1dV1Count"].iloc[i][:-1, :-1, :-1]
        dP1dV1[i] = np.sum(
            dfCor["dP1dV1Correlation"].iloc[i][:-1, :-1, :-1] * dP1dV1total, axis=2
        ) / np.sum(dP1dV1total, axis=2)
        dP2dV2total = dfCor["dP2dV2Count"].iloc[i][:-1, :-1, :-1]
        dP2dV2[i] = np.sum(
            dfCor["dP2dV2Correlation"].iloc[i][:-1, :-1, :-1] * dP2dV2total, axis=2
        ) / np.sum(dP2dV2total, axis=2)

        dV1dV1total = dfCor["dV1dV1Count"].iloc[i][:-1, :-1, :-1]
        dV1dV1[i] = np.sum(
            dfCor["dV1dV1Correlation"].iloc[i][:-1, :-1, :-1] * dV1dV1total, axis=2
        ) / np.sum(dV1dV1total, axis=2)
        dV2dV2total = dfCor["dV2dV2Count"].iloc[i][:-1, :-1, :-1]
        dV2dV2[i] = np.sum(
            dfCor["dV2dV2Correlation"].iloc[i][:-1, :-1, :-1] * dV2dV2total, axis=2
        ) / np.sum(dV2dV2total, axis=2)

    dP1dV1 = np.mean(dP1dV1, axis=0)
    dP2dV2 = np.mean(dP2dV2, axis=0)
    dV1dV1 = np.mean(dV1dV1, axis=0)
    dV2dV2 = np.mean(dV2dV2, axis=0)

    t, r = np.mgrid[0:100:2, 0:52:2]
    maxCorr = np.max([dP1dV1, -dP1dV1])
    c = ax[2, 2].pcolor(
        t,
        r,
        dP1dV1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 2])
    ax[2, 2].set_xlabel("Time (mins)")
    ax[2, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 2].title.set_text(r"$\langle \delta P_1 \delta V_1 \rangle$")

    maxCorr = np.max([dP2dV2, -dP2dV2])
    c = ax[3, 0].pcolor(
        t,
        r,
        dP2dV2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[3, 0])
    ax[3, 0].set_xlabel("Time (mins)")
    ax[3, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[3, 0].title.set_text(r"$\langle \delta P_2 \delta V_2 \rangle$")

    maxCorr = np.max([dV1dV1, -dV1dV1])
    c = ax[3, 1].pcolor(
        t,
        r,
        dV1dV1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[3, 1])
    ax[3, 1].set_xlabel("Time (mins)")
    ax[3, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[3, 1].title.set_text(r"$\langle \delta V_1 \delta V_1 \rangle$")

    maxCorr = np.max([dV2dV2, -dV2dV2])
    c = ax[3, 2].pcolor(
        t,
        r,
        dV2dV2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[3, 2])
    ax[3, 2].set_xlabel("Time (mins)")
    ax[3, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[3, 2].title.set_text(r"$\langle \delta V_2 \delta V_2 \rangle$")

    # plt.subplot_tool()
    plt.subplots_adjust(
        left=0.075, bottom=0.1, right=0.95, top=0.9, wspace=0.4, hspace=0.45
    )

    fig.savefig(
        f"results/Correlations velocity {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# std d_rRho and d_rdQ1
if False:
    grid = 9
    timeGrid = 18
    gridSize = 10
    gridSizeT = 5
    grid_df = 27
    timeGrid_df = 51

    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")

    xMax = np.max(dfShape["X"])
    xMin = np.min(dfShape["X"])
    yMax = np.max(dfShape["Y"])
    yMin = np.min(dfShape["Y"])
    xGrid = int(1 + (xMax - xMin) // gridSize)
    yGrid = int(1 + (yMax - yMin) // gridSize)

    k = 0
    std_dr1drho = np.zeros([len(filenames)])
    std_dr2drho = np.zeros([len(filenames)])
    for filename in filenames:

        dfShapeF = dfShape[dfShape["Filename"] == filename].copy()
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
                        heatmapdrho[t, i, j] = len(dfg["Area"]) / np.sum(dfg["Area"])
                        inPlaneEcad[t, i, j] = 1

            heatmapdrho[t] = heatmapdrho[t] - np.mean(
                heatmapdrho[t][inPlaneEcad[t] == 1]
            )

        hm_dr1drho = (heatmapdrho[:, 1:] - heatmapdrho[:, :-1]) / gridSize
        std_dr1drho[k] = np.std(hm_dr1drho[hm_dr1drho != 0])
        hm_dr2drho = (heatmapdrho[:, :, 1:] - heatmapdrho[:, :, :-1]) / gridSize
        std_dr2drho[k] = np.std(hm_dr2drho[hm_dr2drho != 0])
        k += 1

    xMax = np.max(dfShape["X"])
    xMin = np.min(dfShape["X"])
    yMax = np.max(dfShape["Y"])
    yMin = np.min(dfShape["Y"])
    xGrid = int(1 + (xMax - xMin) // gridSize)
    yGrid = int(1 + (yMax - yMin) // gridSize)

    k = 0
    std_dr1dQ1 = np.zeros([len(filenames)])
    std_dr2dQ1 = np.zeros([len(filenames)])
    for filename in filenames:
        dfShapeF = dfShape[dfShape["Filename"] == filename].copy()
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

            heatmapdQ1[t] = heatmapdQ1[t] - np.mean(heatmapdQ1[t][inPlaneEcad[t] == 1])

        hm_dr1dQ1 = (heatmapdQ1[:, 1:] - heatmapdQ1[:, :-1]) / gridSize
        std_dr1dQ1[k] = np.std(hm_dr1dQ1[hm_dr1dQ1 != 0])
        hm_dr2dQ1 = (heatmapdQ1[:, :, 1:] - heatmapdQ1[:, :, :-1]) / gridSize
        std_dr2dQ1[k] = np.std(hm_dr2dQ1[hm_dr2dQ1 != 0])
        k += 1

    std_dr1drho = np.mean(std_dr1drho)
    std_dr2drho = np.mean(std_dr2drho)
    std_dr1dQ1 = np.mean(std_dr1dQ1)
    std_dr2dQ1 = np.mean(std_dr2dQ1)

    _df = []

    _df.append(
        {
            "std_dr1drho": std_dr1drho,
            "std_dr2drho": std_dr2drho,
            "std_dr1dQ1": std_dr1dQ1,
            "std_dr2dQ1": std_dr2dQ1,
        }
    )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/correlations/dfstd_dr{fileType}.pkl")

# display all norm correlations shape
if False:
    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")
    df = pd.read_pickle(f"databases/dfVelocity{fileType}.pkl")
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    dfstd_dr = pd.read_pickle(f"databases/correlations/dfstd_dr{fileType}.pkl")

    fig, ax = plt.subplots(4, 3, figsize=(16, 16))

    T, R, Theta = dfCor["dr1dRho_SdV1Correlation"].iloc[0].shape

    dr1dRhodV1 = np.zeros([len(filenames), T - 1, R - 1])
    dr1dRhodV2 = np.zeros([len(filenames), T - 1, R - 1])
    dr2dRhodV1 = np.zeros([len(filenames), T - 1, R - 1])
    dr2dRhodV2 = np.zeros([len(filenames), T - 1, R - 1])
    for i in range(len(filenames)):
        dr1Total = dfCor["dr1dRho_SdV1Count"].iloc[i][:-1, :-1, :-1]
        dr1dRhodV1[i] = np.sum(
            dfCor["dr1dRho_SdV1Correlation"].iloc[i][:-1, :-1, :-1] * dr1Total, axis=2
        ) / np.sum(dr1Total, axis=2)
        dr1dRhodV2[i] = np.sum(
            dfCor["dr1dRho_SdV2Correlation"].iloc[i][:-1, :-1, :-1] * dr1Total, axis=2
        ) / np.sum(dr1Total, axis=2)

        dr2Total = dfCor["dr1dRho_SdV1Count"].iloc[i][:-1, :-1, :-1]
        dr2dRhodV1[i] = np.sum(
            dfCor["dr2dRho_SdV1Correlation"].iloc[i][:-1, :-1, :-1] * dr2Total, axis=2
        ) / np.sum(dr2Total, axis=2)
        dr2dRhodV2[i] = np.sum(
            dfCor["dr2dRho_SdV2Correlation"].iloc[i][:-1, :-1, :-1] * dr2Total, axis=2
        ) / np.sum(dr2Total, axis=2)

    std_dv = np.std(np.stack(np.array(df.loc[:, "dv"]), axis=0), axis=0)
    std_dr1drho = dfstd_dr["std_dr1drho"].iloc[0]
    std_dr2drho = dfstd_dr["std_dr2drho"].iloc[0]

    dr1dRhodV1 = np.mean(dr1dRhodV1, axis=0)
    dr1dRhodV1 = dr1dRhodV1 / (std_dr1drho * std_dv[0])
    dr1dRhodV2 = np.mean(dr1dRhodV2, axis=0)
    dr1dRhodV2 = dr1dRhodV2 / (std_dr1drho * std_dv[1])
    dr2dRhodV1 = np.mean(dr2dRhodV1, axis=0)
    dr2dRhodV1 = dr2dRhodV1 / (std_dr2drho * std_dv[0])
    dr2dRhodV2 = np.mean(dr2dRhodV2, axis=0)
    dr2dRhodV2 = dr2dRhodV2 / (std_dr2drho * std_dv[1])

    maxCorr = np.max([1, -1])
    t, r = np.mgrid[0:170:10, 0:52:2]
    c = ax[0, 0].pcolor(
        t,
        r,
        dr1dRhodV1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 0])
    ax[0, 0].set_xlabel("Time (mins)")
    ax[0, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 0].title.set_text(
        r"$\langle \partial_{r_1} (\delta \rho_s + \delta \rho) \delta V_1 \rangle$"
    )

    c = ax[0, 1].pcolor(
        t,
        r,
        dr1dRhodV2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 1])
    ax[0, 1].set_xlabel("Time (mins)")
    ax[0, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 1].title.set_text(
        r"$\langle \partial_{r_1} (\delta \rho_s + \delta \rho) \delta V_2 \rangle$"
    )

    c = ax[0, 2].pcolor(
        t,
        r,
        dr2dRhodV1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 2])
    ax[0, 2].set_xlabel("Time (mins)")
    ax[0, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 2].title.set_text(
        r"$\langle \partial_{r_2} (\delta \rho_s + \delta \rho) \delta V_1 \rangle$"
    )

    c = ax[1, 0].pcolor(
        t,
        r,
        dr2dRhodV2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set_xlabel("Time (mins)")
    ax[1, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 0].title.set_text(
        r"$\langle \partial_{r_2} (\delta \rho_s + \delta \rho) \delta V_2 \rangle$"
    )

    dr1dQ1dV1 = np.zeros([len(filenames), T - 1, R - 1])
    dr1dQ1dV2 = np.zeros([len(filenames), T - 1, R - 1])
    dr2dQ1dV1 = np.zeros([len(filenames), T - 1, R - 1])
    dr2dQ1dV2 = np.zeros([len(filenames), T - 1, R - 1])
    for i in range(len(filenames)):
        dr1Total = dfCor["dr1dQ1dV1Count"].iloc[i][:-1, :-1, :-1]
        dr1dQ1dV1[i] = np.sum(
            dfCor["dr1dQ1dV1Correlation"].iloc[i][:-1, :-1, :-1] * dr1Total, axis=2
        ) / np.sum(dr1Total, axis=2)
        dr1dQ1dV2[i] = np.sum(
            dfCor["dr1dQ1dV2Correlation"].iloc[i][:-1, :-1, :-1] * dr1Total, axis=2
        ) / np.sum(dr1Total, axis=2)

        dr2Total = dfCor["dr2dQ1dV1Count"].iloc[i][:-1, :-1, :-1]
        dr2dQ1dV1[i] = np.sum(
            dfCor["dr2dQ1dV1Correlation"].iloc[i][:-1, :-1, :-1] * dr2Total, axis=2
        ) / np.sum(dr2Total, axis=2)
        dr2dQ1dV2[i] = np.sum(
            dfCor["dr2dQ1dV2Correlation"].iloc[i][:-1, :-1, :-1] * dr2Total, axis=2
        ) / np.sum(dr2Total, axis=2)

    std_dr1dQ1 = dfstd_dr["std_dr1dQ1"].iloc[0]
    std_dr2dQ1 = dfstd_dr["std_dr2dQ1"].iloc[0]

    dr1dQ1dV1 = np.mean(dr1dQ1dV1, axis=0)
    dr1dQ1dV1 = dr1dQ1dV1 / (std_dr1dQ1 * std_dv[0])
    dr1dQ1dV2 = np.mean(dr1dQ1dV2, axis=0)
    dr1dQ1dV2 = dr1dQ1dV2 / (std_dr1dQ1 * std_dv[1])
    dr2dQ1dV1 = np.mean(dr2dQ1dV1, axis=0)
    dr2dQ1dV1 = dr2dQ1dV1 / (std_dr2dQ1 * std_dv[0])
    dr2dQ1dV2 = np.mean(dr2dQ1dV2, axis=0)
    dr2dQ1dV2 = dr2dQ1dV2 / (std_dr2dQ1 * std_dv[1])

    c = ax[1, 1].pcolor(
        t,
        r,
        dr1dQ1dV1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 1])
    ax[1, 1].set_xlabel("Time (mins)")
    ax[1, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 1].title.set_text(
        r"$\langle \partial_{r_1} \delta Q^{(1)} \delta V_1 \rangle$"
    )

    c = ax[1, 2].pcolor(
        t,
        r,
        dr1dQ1dV2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 2])
    ax[1, 2].set_xlabel("Time (mins)")
    ax[1, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 2].title.set_text(
        r"$\langle \partial_{r_1} \delta Q^{(1)} \delta V_2 \rangle$"
    )

    c = ax[2, 0].pcolor(
        t,
        r,
        dr2dQ1dV1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 0])
    ax[2, 0].set_xlabel("Time (mins)")
    ax[2, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 0].title.set_text(
        r"$\langle \partial_{r_2} \delta Q^{(1)} \delta V_1 \rangle$"
    )

    c = ax[2, 1].pcolor(
        t,
        r,
        dr2dQ1dV2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 1])
    ax[2, 1].set_xlabel("Time (mins)")
    ax[2, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 1].title.set_text(
        r"$\langle \partial_{r_2} \delta Q^{(1)} \delta V_2 \rangle$"
    )

    T, R, Theta = dfCor["dP1dV1Correlation"].iloc[0].shape

    dP1dV1 = np.zeros([len(filenames), T - 1, R - 1])
    dP2dV2 = np.zeros([len(filenames), T - 1, R - 1])
    dV1dV1 = np.zeros([len(filenames), T - 1, R - 1])
    dV2dV2 = np.zeros([len(filenames), T - 1, R - 1])
    for i in range(len(filenames)):
        dP1dV1total = dfCor["dP1dV1Count"].iloc[i][:-1, :-1, :-1]
        dP1dV1[i] = np.sum(
            dfCor["dP1dV1Correlation"].iloc[i][:-1, :-1, :-1] * dP1dV1total, axis=2
        ) / np.sum(dP1dV1total, axis=2)
        dP2dV2total = dfCor["dP2dV2Count"].iloc[i][:-1, :-1, :-1]
        dP2dV2[i] = np.sum(
            dfCor["dP2dV2Correlation"].iloc[i][:-1, :-1, :-1] * dP2dV2total, axis=2
        ) / np.sum(dP2dV2total, axis=2)

        dV1dV1total = dfCor["dV1dV1Count"].iloc[i][:-1, :-1, :-1]
        dV1dV1[i] = np.sum(
            dfCor["dV1dV1Correlation"].iloc[i][:-1, :-1, :-1] * dV1dV1total, axis=2
        ) / np.sum(dV1dV1total, axis=2)
        dV2dV2total = dfCor["dV2dV2Count"].iloc[i][:-1, :-1, :-1]
        dV2dV2[i] = np.sum(
            dfCor["dV2dV2Correlation"].iloc[i][:-1, :-1, :-1] * dV2dV2total, axis=2
        ) / np.sum(dV2dV2total, axis=2)

    std_dp = np.std(np.stack(np.array(dfShape.loc[:, "dp"]), axis=0), axis=0)

    dP1dV1 = np.mean(dP1dV1, axis=0)
    dP1dV1 = dP1dV1 / (std_dp[0] * std_dv[0])
    dP2dV2 = np.mean(dP2dV2, axis=0)
    dP2dV2 = dP2dV2 / (std_dp[1] * std_dv[1])
    dV1dV1 = np.mean(dV1dV1, axis=0)
    dV1dV1 = dV1dV1 / (std_dv[0] * std_dv[0])
    dV2dV2 = np.mean(dV2dV2, axis=0)
    dV2dV2 = dV2dV2 / (std_dv[1] * std_dv[1])

    t, r = np.mgrid[0:100:2, 0:52:2]
    c = ax[2, 2].pcolor(
        t,
        r,
        dP1dV1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 2])
    ax[2, 2].set_xlabel("Time (mins)")
    ax[2, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 2].title.set_text(r"$\langle \delta P_1 \delta V_1 \rangle$")

    c = ax[3, 0].pcolor(
        t,
        r,
        dP2dV2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[3, 0])
    ax[3, 0].set_xlabel("Time (mins)")
    ax[3, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[3, 0].title.set_text(r"$\langle \delta P_2 \delta V_2 \rangle$")

    c = ax[3, 1].pcolor(
        t,
        r,
        dV1dV1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[3, 1])
    ax[3, 1].set_xlabel("Time (mins)")
    ax[3, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[3, 1].title.set_text(r"$\langle \delta V_1 \delta V_1 \rangle$")

    c = ax[3, 2].pcolor(
        t,
        r,
        dV2dV2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[3, 2])
    ax[3, 2].set_xlabel("Time (mins)")
    ax[3, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[3, 2].title.set_text(r"$\langle \delta V_2 \delta V_2 \rangle$")

    # plt.subplot_tool()
    plt.subplots_adjust(
        left=0.075, bottom=0.1, right=0.95, top=0.9, wspace=0.4, hspace=0.45
    )

    fig.savefig(
        f"results/Correlations Norm velocity {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# deltaQ1 (model)
if False:

    def Corr_dQ1_Integral_T(R, L):
        B = 0.006533824439392692
        C = 0.00055
        T = 2
        k = np.linspace(0, 4, 200000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, C, L) * h, axis=0)[:, 0]

    def Corr_dQ1_Integral_R(T, L):
        B = 0.006533824439392692
        C = 0.00055
        R = 0
        k = np.linspace(0, 4, 200000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, C, L) * h, axis=0)[0]

    def Corr_dQ1(R, T):
        B = 0.006533824439392692
        C = 0.00055
        L = 2.1

        k = np.linspace(0, 4, 200000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, C, L) * h, axis=0)

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

    fig, ax = plt.subplots(4, 1, figsize=(4, 16))

    # m = sp.optimize.curve_fit(
    #     f=Corr_dQ1_Integral_R,
    #     xdata=T[1:],
    #     ydata=dQ1dQ1[:, 0][1:],
    #     p0=(4),
    # )[0]
    # print(m)

    ax[0].plot(T[1:], dQ1dQ1[:, 0][1:], label="Data")
    ax[0].plot(T[1:], Corr_dQ1(0, T[1:])[0], label="Model")
    # ax[0].plot(T[1:], Corr_dQ1_Integral_R(T[1:], m[0]), label="Model")
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel(r"$\delta Q^{(1)}$ Correlation")
    ax[0].set_ylim([0, 5.9e-04])
    ax[0].title.set_text(r"Correlation of $\delta Q^{(1)}$, $R=0$")
    ax[0].legend()

    # m = sp.optimize.curve_fit(
    #     f=Corr_dQ1_Integral_T,
    #     xdata=R[1:],
    #     ydata=dQ1dQ1[1][1:26],
    #     p0=(4),
    #     method="lm",
    # )[0]
    # print(m)

    ax[1].plot(R, dQ1dQ1[1][:26], label="Data")
    ax[1].plot(R, Corr_dQ1(R, 2), label="Model")
    # ax[1].plot(R, Corr_dQ1_Integral_T(R, m[0]), label="Model")
    ax[1].set_xlabel(r"$R (\mu m)$")
    ax[1].set_ylabel(r"$\delta Q^{(1)}$ Correlation")
    ax[1].set_ylim([0, 5.9e-04])
    ax[1].title.set_text(r"Correlation of $\delta Q^{(1)}$, $T=2$")
    ax[1].legend()

    Corr_dQ1dQ1 = np.swapaxes(Corr_dQ1(R, T), 0, 1)
    R, T = np.meshgrid(R, T)
    maxCorr = np.max([dQ1dQ1, -dQ1dQ1])
    c = ax[2].pcolor(
        T,
        R,
        dQ1dQ1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2])
    ax[2].set_xlabel("Time (mins)")
    ax[2].set_ylabel(r"$R (\mu m)$ ")
    ax[2].title.set_text(r"Experiment $\langle \delta Q^1 \delta Q^1 \rangle$")

    c = ax[3].pcolor(
        T,
        R,
        Corr_dQ1dQ1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[3])
    ax[3].set_xlabel("Time (mins)")
    ax[3].set_ylabel(r"$R (\mu m)$ ")
    ax[3].title.set_text(r"Model $\langle \delta Q^1 \delta Q^1 \rangle$")

    plt.subplots_adjust(
        left=0.22, bottom=0.1, right=0.93, top=0.9, wspace=0.55, hspace=0.37
    )
    fig.savefig(
        f"results/Correlation dQ1 in T and R {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# deltaQ2 (model)
if False:

    def Corr_dQ2_Integral_T(R, C):
        B = 0.006533824439392692
        L = 2.1
        T = 2
        k = np.linspace(0, 4, 200000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, C, L) * h, axis=0)[:, 0]

    def Corr_dQ2(R, T):
        B = 0.006533824439392692
        C = 0.00045
        L = 2.1

        k = np.linspace(0, 4, 200000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, C, L) * h, axis=0)

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

    fig, ax = plt.subplots(4, 1, figsize=(4, 16))
    plt.subplots_adjust(wspace=0.4)
    plt.gcf().subplots_adjust(bottom=0.15)

    ax[0].plot(T[1:], dQ2dQ2[:, 0][1:], label="Data")
    ax[0].plot(
        T[1:],
        Corr_dQ2(0, T[1:])[0],
        label="Model",
    )
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel(r"$\delta Q^{(2)}$ Correlation")
    ax[0].set_ylim([0, 6e-04])
    ax[0].title.set_text(r"Correlation of $\delta Q^{(2)}$, $R=0$")
    ax[0].legend()

    # m = sp.optimize.curve_fit(
    #     f=Corr_dQ2_Integral_T,
    #     xdata=R[1:],
    #     ydata=dQ2dQ2[1][1:26],
    #     p0=(4),
    #     method="lm",
    # )[0]
    # print(m)

    ax[1].plot(R, dQ2dQ2[1][:26], label="Data")
    ax[1].plot(R, Corr_dQ2(R, 2), label="Model")
    ax[1].set_xlabel(r"$R (\mu m)$")
    ax[1].set_ylabel(r"$\delta Q^{(2)}$ Correlation")
    ax[1].set_ylim([0, 6e-04])
    ax[1].title.set_text(r"Correlation of $\delta Q^{(2)}$, $T=2$")
    ax[1].legend()

    Corr_dQ2dQ2 = np.swapaxes(Corr_dQ2(R, T), 0, 1)
    R, T = np.meshgrid(R, T)
    maxCorr = np.max([dQ2dQ2, -dQ2dQ2])
    c = ax[2].pcolor(
        T,
        R,
        dQ2dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2])
    ax[2].set_xlabel("Time (mins)")
    ax[2].set_ylabel(r"$R (\mu m)$ ")
    ax[2].title.set_text(r"Experiment $\langle \delta Q^2 \delta Q^2 \rangle$")

    c = ax[3].pcolor(
        T,
        R,
        Corr_dQ2dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[3])
    ax[3].set_xlabel("Time (mins)")
    ax[3].set_ylabel(r"$R (\mu m)$ ")
    ax[3].title.set_text(r"Model $\langle \delta Q^2 \delta Q^2 \rangle$")

    plt.subplots_adjust(
        left=0.22, bottom=0.1, right=0.93, top=0.9, wspace=0.55, hspace=0.37
    )

    fig.savefig(
        f"results/Correlation dQ2 in T and R {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# deltaP1 (model)
if False:

    def Corr_dP1_Integral_T(R, L):
        B = 0.020119498379811963
        C = 7.01482259e-06 * 2 * L
        T = 0
        k = np.linspace(0, 4, 200000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, C, L) * h, axis=0)[:, 0]

    def Corr_dP1_Integral_R(T, L, C):
        B = 0.020119498379811963
        R = 0
        k = np.linspace(0, 4, 200000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, C, L) * h, axis=0)[0]

    def Corr_dP1(R, T):
        B = 0.020119498379811963
        L = 0.1062963
        C = 1.491299372946834e-06

        k = np.linspace(0, 4, 200000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, C, L) * h, axis=0)

    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    T, R, Theta = dfCor["dP1dP1Correlation"].iloc[0][:, :-1, :-1].shape

    dP1dP1 = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):

        dP1dP1total = dfCor["dP1dP1Count"].iloc[i][:, :-1, :-1]
        dP1dP1[i] = np.sum(
            dfCor["dP1dP1Correlation"].iloc[i][:, :-1, :-1] * dP1dP1total, axis=2
        ) / np.sum(dP1dP1total, axis=2)

    dfCor = 0

    dP1dP1 = np.mean(dP1dP1, axis=0)

    T = np.linspace(0, 2 * (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)

    fig, ax = plt.subplots(4, 1, figsize=(4, 16))
    plt.subplots_adjust(wspace=0.4)
    plt.gcf().subplots_adjust(bottom=0.15)

    # m = sp.optimize.curve_fit(
    #     f=Corr_R0,
    #     xdata=T[1:],
    #     ydata=dP1dP1[:, 0][1:],
    #     p0=(4, 0.00001),
    # )[0]
    # print(m)
    # B, D = m[0], m[1]

    ax[0].plot(T, dP1dP1[:, 0], label="Data")
    ax[0].plot(T, Corr_dP1(0, T)[0], label="Model")
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel(r"$\delta P_1$ Correlation")
    ax[0].set_ylim([-2e-06, 4e-05])
    ax[0].title.set_text(r"Correlation of $\delta P_1$, $R=0$")
    ax[0].legend()

    # m = sp.optimize.curve_fit(
    #     f=Corr_dP1_Integral_T,
    #     xdata=R,
    #     ydata=dP1dP1[0][:26],
    #     p0=(4),
    #     method="lm",
    # )[0]
    # print(m)

    ax[1].plot(R, dP1dP1[0][:26], label="Data")
    ax[1].plot(R, Corr_dP1(R, 0), label="Model")
    ax[1].set_xlabel(r"$R (\mu m)$")
    ax[1].set_ylabel(r"$\delta P_1$ Correlation")
    ax[1].set_ylim([-2e-06, 4e-05])
    ax[1].title.set_text(r"Correlation of $\delta P_1$, $T=0$")
    ax[1].legend()

    Corr_dP1dP1 = np.swapaxes(Corr_dP1(R, T), 0, 1)
    R, T = np.meshgrid(R, T)
    maxCorr = np.max([dP1dP1, -dP1dP1])
    c = ax[2].pcolor(
        T,
        R,
        dP1dP1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2])
    ax[2].set_xlabel("Time (mins)")
    ax[2].set_ylabel(r"$R (\mu m)$ ")
    ax[2].title.set_text(r"Exp. $\langle \delta P_1 \delta P_1 \rangle$")

    c = ax[3].pcolor(
        T,
        R,
        Corr_dP1dP1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[3])
    ax[3].set_xlabel("Time (mins)")
    ax[3].set_ylabel(r"$R (\mu m)$ ")
    ax[3].title.set_text(r"Model $\langle \delta P_1 \delta P_1 \rangle$", y=1)

    plt.subplots_adjust(
        left=0.22, bottom=0.1, right=0.93, top=0.9, wspace=0.4, hspace=0.45
    )

    fig.savefig(
        f"results/Correlation dP1 in T and R {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# deltaP2 (model)
if False:

    def Corr_dP2_Integral_R(T, C):
        B = 0.020119498379811963
        L = 0.1062963
        R = 0
        k = np.linspace(0, 4, 200000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, C, L) * h, axis=0)[0]

    def Corr_dP2(R, T):
        B = 0.020119498379811963
        L = 0.1062963
        C = 1.0340398305600548e-06

        k = np.linspace(0, 4, 200000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, C, L) * h, axis=0)

    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    T, R, Theta = dfCor["dP2dP2Correlation"].iloc[0][:, :-1, :-1].shape

    dP2dP2 = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):

        dP2dP2total = dfCor["dP2dP2Count"].iloc[i][:, :-1, :-1]
        dP2dP2[i] = np.sum(
            dfCor["dP2dP2Correlation"].iloc[i][:, :-1, :-1] * dP2dP2total, axis=2
        ) / np.sum(dP2dP2total, axis=2)

    dfCor = 0

    dP2dP2 = np.mean(dP2dP2, axis=0)

    T = np.linspace(0, 2 * (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)

    fig, ax = plt.subplots(4, 1, figsize=(4, 16))
    plt.subplots_adjust(wspace=0.4)
    plt.gcf().subplots_adjust(bottom=0.15)

    # m = sp.optimize.curve_fit(
    #     f=Corr_dP2_Integral_R,
    #     xdata=T,
    #     ydata=dP2dP2[:, 0],
    #     p0=(0.0001),
    # )[0]

    ax[0].plot(T, dP2dP2[:, 0], label="Data")
    ax[0].plot(T, Corr_dP2(0, T)[0], label="Model")
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel(r"$\delta P_2$ Correlation")
    ax[0].set_ylim([-2e-06, 2.7e-05])
    ax[0].title.set_text(r"Correlation of $\delta P_2$, $R=0$")
    ax[0].legend()

    ax[1].plot(R, dP2dP2[0][:26], label="Data")
    ax[1].plot(R, Corr_dP2(R, 0), label="Model")
    ax[1].set_xlabel(r"$R (\mu m)$")
    ax[1].set_ylabel(r"$\delta P_2$ Correlation")
    ax[1].set_ylim([-2e-06, 2.7e-05])
    ax[1].title.set_text(r"Correlation of $\delta P_2$, $T=0$")
    ax[1].legend()

    Corr_dP2dP2 = np.swapaxes(Corr_dP2(R, T), 0, 1)
    R, T = np.meshgrid(R, T)
    maxCorr = np.max([dP2dP2, -dP2dP2])
    c = ax[2].pcolor(
        T,
        R,
        dP2dP2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2])
    ax[2].set_xlabel("Time (mins)")
    ax[2].set_ylabel(r"$R (\mu m)$ ")
    ax[2].title.set_text(r"Exp. $\langle \delta P_2 \delta P_2 \rangle$")

    c = ax[3].pcolor(
        T,
        R,
        Corr_dP2dP2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[3])
    ax[3].set_xlabel("Time (mins)")
    ax[3].set_ylabel(r"$R (\mu m)$ ")
    ax[3].title.set_text(r"Model $\langle \delta P_2 \delta P_2 \rangle$")

    plt.subplots_adjust(
        left=0.22, bottom=0.1, right=0.93, top=0.9, wspace=0.4, hspace=0.45
    )

    fig.savefig(
        f"results/Correlation dP2 in T and R {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# deltaRho_n (model)
if False:

    def Corr_Rho_T(T, C):
        return C / T

    def Corr_Rho_R(R, D):
        C = 0.008226129102387299
        T = 50
        return C / T * np.exp(-(R**2) / (4 * D * T))

    def Corr_Rho(R, T):
        C = 0.008226129102387299
        D = 2.935490027205022
        return C / T * np.exp(-(R**2) / (4 * D * T))

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

    R = np.linspace(0, 60, 7)
    T = np.linspace(10, 170, 17)

    fig, ax = plt.subplots(4, 1, figsize=(4, 16))
    plt.subplots_adjust(wspace=0.4)
    plt.gcf().subplots_adjust(bottom=0.15)

    m = sp.optimize.curve_fit(
        f=Corr_Rho_T,
        xdata=T[4:],
        ydata=dRhodRho[5:, 0],
        p0=0.003,
    )[0]
    print(m[0])

    ax[0].plot(T, dRhodRho[1:, 0])
    ax[0].plot(T, Corr_Rho_T(T, m[0]))
    ax[0].set_xlabel(r"Time (mins)")
    ax[0].set_ylabel(r"$\delta \rho_s$ Correlation")
    ax[0].set_ylim([-2e-5, 6e-4])
    ax[0].title.set_text(r"$\langle \delta \rho_s \delta \rho_s \rangle$, $R=0$")

    m = sp.optimize.curve_fit(
        f=Corr_Rho_R,
        xdata=R[1:],
        ydata=dRhodRho[4][1:],
        p0=(10),
    )[0]
    print(m[0])

    ax[1].plot(R, dRhodRho[1])
    ax[1].plot(R, Corr_Rho_R(R, m[0]))
    ax[1].set_xlabel(r"$R (\mu m)$")
    ax[1].set_ylabel(r"$\delta \rho_s$ Correlation")
    ax[1].set_ylim([-2e-5, 6e-4])
    ax[1].title.set_text(r"$\langle \delta \rho_s \delta \rho_s \rangle$, $T=50$")

    R, T = np.meshgrid(R, T)
    maxCorr = np.max([dRhodRho[1:], -dRhodRho[1:]])
    c = ax[2].pcolor(
        T,
        R,
        dRhodRho[1:],
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2])
    ax[2].set_xlabel("Time (mins)")
    ax[2].set_ylabel(r"$R (\mu m)$ ")
    ax[2].title.set_text(r"Experiment $\langle \delta \rho_s \delta \rho_s \rangle$")

    c = ax[3].pcolor(
        T,
        R,
        Corr_Rho(R, T),
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[3])
    ax[3].set_xlabel("Time (mins)")
    ax[3].set_ylabel(r"$R (\mu m)$ ")
    ax[3].title.set_text(r"Model $\langle \delta \rho_s \delta \rho_s \rangle$")

    plt.subplots_adjust(
        left=0.22, bottom=0.1, right=0.93, top=0.9, wspace=0.4, hspace=0.45
    )

    fig.savefig(
        f"results/Correlation dRho in T and R model",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# deltaRho_s (model)
if False:

    def Corr_dRho_S(r, C, lamdba):
        return C * np.exp(-lamdba * r)

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

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    m = sp.optimize.curve_fit(
        f=Corr_dRho_S,
        xdata=R,
        ydata=dRho_SdRho_S,
        p0=(0.0003, 0.04),
    )[0]
    print(m)

    ax.plot(R, dRho_SdRho_S, label="Data")
    ax.plot(R, Corr_dRho_S(R, m[0], m[1]), label="Model")
    ax.set_xlabel(r"$R (\mu m)$")
    ax.set_ylabel(r"$\delta \rho_s$ Correlation")
    ax.set_ylim([0, 4.5e-04])
    ax.title.set_text(r"Correlation of $\delta \rho_s$")
    ax.legend()

    # plt.subplots_adjust(
    #     left=0.22, bottom=0.1, right=0.93, top=0.9, wspace=0.55, hspace=0.37
    # )
    fig.savefig(
        f"results/Correlation rho_s in T and R {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# deltaV (model)
if False:

    def Corr_dV(r, C, lamdba):
        return C * np.exp(-lamdba * r)

    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    T, R, Theta = dfCor["dV1dV1Correlation"].iloc[0].shape

    dV1dV1 = np.zeros([len(filenames), T, R])
    dV2dV2 = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):
        dV1dV1Count = dfCor["dV1dV1Count"].iloc[i][:, :, :-1]
        dV1dV1[i] = np.sum(
            dfCor["dV1dV1Correlation"].iloc[i][:, :, :-1] * dV1dV1Count, axis=2
        ) / np.sum(dV1dV1Count, axis=2)
        dV2dV2[i] = np.sum(
            dfCor["dV2dV2Correlation"].iloc[i][:, :, :-1] * dV1dV1Count, axis=2
        ) / np.sum(dV1dV1Count, axis=2)

    dfCor = 0

    dV1dV1 = np.mean(dV1dV1, axis=0)
    dV2dV2 = np.mean(dV2dV2, axis=0)

    dV1dV1_r = dV1dV1[0][0:-1]
    dV2dV2_r = dV2dV2[0][0:-1]

    R = np.linspace(0, 2 * (grid - 1), grid)

    fig, ax = plt.subplots(2, 1, figsize=(4, 8))

    m = sp.optimize.curve_fit(
        f=Corr_dV,
        xdata=R[3:],
        ydata=dV1dV1_r[3:],
        p0=(0.23, 0.04),
    )[0]
    print(m)

    ax[0].plot(R, dV1dV1_r, label="Data")
    ax[0].plot(R, Corr_dV(R, m[0], m[1]), label="Model")
    ax[0].set_xlabel(r"$R (\mu m)$")
    ax[0].set_ylabel(r"$\delta V_1$ Correlation")
    # ax[0].set_ylim([0, 4.5e-04])
    ax[0].title.set_text(r"Correlation of $\delta V_1$")
    ax[0].legend()

    m = sp.optimize.curve_fit(
        f=Corr_dV,
        xdata=R[3:],
        ydata=dV2dV2_r[3:],
        p0=(0.23, 0.04),
    )[0]
    print(m)

    ax[1].plot(R, dV2dV2_r, label="Data")
    ax[1].plot(R, Corr_dV(R, m[0], m[1]), label="Model")
    ax[1].set_xlabel(r"$R (\mu m)$")
    ax[1].set_ylabel(r"$\delta V_2$ Correlation")
    # ax[1].set_ylim([0, 4.5e-04])
    ax[1].title.set_text(r"Correlation of $\delta V_2$")
    ax[1].legend()

    plt.subplots_adjust(
        left=0.22, bottom=0.1, right=0.93, top=0.9, wspace=0.55, hspace=0.37
    )
    fig.savefig(
        f"results/Correlation dV in T and R {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")
