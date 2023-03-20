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


def Corr_R0(t, c, D):
    return D * -sc.expi(-c * t)


def forIntegral(y, R, T, c, D, L):
    y, R, T = np.meshgrid(y, R, T, indexing="ij")
    return D * np.exp(-y) * sc.jv(0, R * ((y - c * T) / (L * T)) ** 0.5) / y


def Corr_dP1(R, T):
    c = 0.014231800277153952
    D = 0.02502418
    L = 8.06377854e-06
    y = np.linspace(c * T, c * T * 200, 200000)
    h = y[1] - y[0]
    return np.sum(forIntegral(y, R, T, c, D, L) * h, axis=0)[:, 0]


def Corr_dP2(R, T):
    c = 0.014231800277153952
    D = 0.02502418
    L = 8.06377854e-06
    y = np.linspace(c * T, c * T * 200, 200000)
    h = y[1] - y[0]
    return np.sum(forIntegral(y, R, T, c, D, L) * h, axis=0)[:, 0]


# -------------------

grid = 26
timeGrid = 51

# display all correlations shape
if False:
    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    fig, ax = plt.subplots(4, 3, figsize=(16, 16))

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

    t, r = np.mgrid[0:102:2, 0:82:2]
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

    t, r = np.mgrid[0:102:2, 0:82:2]
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
if True:
    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    fig, ax = plt.subplots(4, 3, figsize=(16, 16))

    T, R, Theta = dfCor["dr1dRhodV1Correlation"].iloc[0].shape

    dr1dRhodV1 = np.zeros([len(filenames), T, R])
    dr1dRhodV2 = np.zeros([len(filenames), T, R])
    dr2dRhodV1 = np.zeros([len(filenames), T, R])
    dr2dRhodV2 = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):
        dr1Total = dfCor["dr1dRhodV1Count"].iloc[i][:, :, :-1]
        dr1dRhodV1[i] = np.sum(
            dfCor["dr1dRhodV1Correlation"].iloc[i][:, :, :-1] * dr1Total, axis=2
        ) / np.sum(dr1Total, axis=2)
        dr1dRhodV2[i] = np.sum(
            dfCor["dr1dRhodV2Correlation"].iloc[i][:, :, :-1] * dr1Total, axis=2
        ) / np.sum(dr1Total, axis=2)

        dr2Total = dfCor["dr1dRhodV1Count"].iloc[i][:, :, :-1]
        dr2dRhodV1[i] = np.sum(
            dfCor["dr2dRhodV1Correlation"].iloc[i][:, :, :-1] * dr2Total, axis=2
        ) / np.sum(dr2Total, axis=2)
        dr2dRhodV2[i] = np.sum(
            dfCor["dr2dRhodV2Correlation"].iloc[i][:, :, :-1] * dr2Total, axis=2
        ) / np.sum(dr2Total, axis=2)

    dr1dRhodV1 = np.mean(dr1dRhodV1, axis=0)
    dr1dRhodV2 = np.mean(dr1dRhodV2, axis=0)
    dr2dRhodV1 = np.mean(dr2dRhodV1, axis=0)
    dr2dRhodV2 = np.mean(dr2dRhodV2, axis=0)

    t, r = np.mgrid[0:100:2, 0:50:2]

    maxCorr = np.max([dr1dRhodV1[:50, :25], -dr1dRhodV1[:50, :25]])
    print(maxCorr)
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
    ax[0, 0].title.set_text(r"$\langle \partial_{r_1} \delta \rho \delta V_1 \rangle$")

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
    ax[0, 1].title.set_text(r"$\langle \partial_{r_1} \delta \rho \delta V_2 \rangle$")

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
    ax[0, 2].title.set_text(r"$\langle \partial_{r_2} \delta \rho \delta V_1 \rangle$")

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
    ax[1, 0].title.set_text(r"$\langle \partial_{r_2} \delta \rho \delta V_2 \rangle$")

    dr1dQ1dV1 = np.zeros([len(filenames), T, R])
    dr1dQ1dV2 = np.zeros([len(filenames), T, R])
    dr2dQ1dV1 = np.zeros([len(filenames), T, R])
    dr2dQ1dV2 = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):
        dr1Total = dfCor["dr1dQ1dV1Count"].iloc[i][:, :, :-1]
        dr1dQ1dV1[i] = np.sum(
            dfCor["dr1dQ1dV1Correlation"].iloc[i][:, :, :-1] * dr1Total, axis=2
        ) / np.sum(dr1Total, axis=2)
        dr1dQ1dV2[i] = np.sum(
            dfCor["dr1dQ1dV2Correlation"].iloc[i][:, :, :-1] * dr1Total, axis=2
        ) / np.sum(dr1Total, axis=2)

        dr2Total = dfCor["dr2dQ1dV1Count"].iloc[i][:, :, :-1]
        dr2dQ1dV1[i] = np.sum(
            dfCor["dr2dQ1dV1Correlation"].iloc[i][:, :, :-1] * dr2Total, axis=2
        ) / np.sum(dr2Total, axis=2)
        dr2dQ1dV2[i] = np.sum(
            dfCor["dr2dQ1dV2Correlation"].iloc[i][:, :, :-1] * dr2Total, axis=2
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

    dP1dV1 = np.zeros([len(filenames), T, R])
    dP2dV2 = np.zeros([len(filenames), T, R])
    dV1dV1 = np.zeros([len(filenames), T, R])
    dV2dV2 = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):
        dP1dV1total = dfCor["dP1dV1Count"].iloc[i][:, :, :-1]
        dP1dV1[i] = np.sum(
            dfCor["dP1dV1Correlation"].iloc[i][:, :, :-1] * dP1dV1total, axis=2
        ) / np.sum(dP1dV1total, axis=2)
        dP2dV2total = dfCor["dP2dV2Count"].iloc[i][:, :, :-1]
        dP2dV2[i] = np.sum(
            dfCor["dP2dV2Correlation"].iloc[i][:, :, :-1] * dP2dV2total, axis=2
        ) / np.sum(dP2dV2total, axis=2)

        dV1dV1total = dfCor["dV1dV1Count"].iloc[i][:, :, :-1]
        dV1dV1[i] = np.sum(
            dfCor["dV1dV1Correlation"].iloc[i][:, :, :-1] * dV1dV1total, axis=2
        ) / np.sum(dV1dV1total, axis=2)
        dV2dV2total = dfCor["dV2dV2Count"].iloc[i][:, :, :-1]
        dV2dV2[i] = np.sum(
            dfCor["dV2dV2Correlation"].iloc[i][:, :, :-1] * dV2dV2total, axis=2
        ) / np.sum(dV2dV2total, axis=2)

    dP1dV1 = np.mean(dP1dV1, axis=0)
    dP2dV2 = np.mean(dP2dV2, axis=0)
    dV1dV1 = np.mean(dV1dV1, axis=0)
    dV2dV2 = np.mean(dV2dV2, axis=0)

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

# display all norm correlations shape
if False:
    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")
    df = pd.read_pickle(f"databases/dfShape{fileType}.pkl")

    fig, ax = plt.subplots(4, 3, figsize=(16, 16))

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

    t, r = np.mgrid[0:102:2, 0:82:2]
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
        f"results/Correlations Norm velocity {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# -------------------


def Corr_dQ2_Integral_T2(R, L):
    c = 0.005710229096088735
    D = 0.0001453973391906898

    T = 2
    y = np.linspace(c * T, c * T * 200, 200000)
    h = y[1] - y[0]
    return np.sum(forIntegral(y, R, T, c, D, L) * h, axis=0)[:, 0]


def Corr_dQ2(R, T):
    c = 0.005710229096088735
    D = 0.0001453973391906898
    L = 0.3836950539434937
    y = np.linspace(c * T, c * T * 200, 200000)
    h = y[1] - y[0]
    return np.sum(forIntegral(y, R, T, c, D, L) * h, axis=0)[:, 0]


# deltaQ2 (model)
if False:

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

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(wspace=0.4)
    plt.gcf().subplots_adjust(bottom=0.15)

    m = sp.optimize.curve_fit(
        f=Corr_R0,
        xdata=T[1:],
        ydata=dQ2dQ2[:, 0][1:],
        p0=(0.006, 0.0001),
    )[0]

    ax[0].plot(T[1:], dQ2dQ2[:, 0][1:], label="Data")
    ax[0].plot(T[1:], Corr_R0(T[1:], m[0], m[1]), label="Model")
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel(r"$\delta Q^{(2)}$ Correlation")
    ax[0].set_ylim([-0.00003, 0.0007])
    ax[0].title.set_text(r"Correlation of $\delta Q^{(2)}$, $R=0$")
    ax[0].legend()

    m = sp.optimize.curve_fit(
        f=Corr_dQ2_Integral_T0,
        xdata=R[5:],
        ydata=dQ2dQ2[0][5:26],
        p0=0.4,
        method="lm",
    )[0]

    ax[1].plot(R, dQ2dQ2[0][:26], label="Data")
    ax[1].plot(R, Corr_dQ2_Integral_T0(R, m), label="Model")
    ax[1].set_xlabel(r"$R (\mu m)$")
    ax[1].set_ylabel(r"$\delta Q^{(2)}$ Correlation")
    ax[1].set_ylim([-0.00003, 0.0007])
    ax[1].title.set_text(r"Correlation of $\delta Q^{(2)}$, $T=0$")
    ax[1].legend()
    fig.savefig(
        f"results/Correlation dQ2 in T and R {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# -------------------


def Corr_dP2_Integral_T2(R, L):
    c = 0.020993205571310774
    D = 4.3738688812044245e-06

    T = 2
    y = np.linspace(c * T, c * T * 200, 200000)
    h = y[1] - y[0]
    return np.sum(forIntegral(y, R, T, c, D, L) * h, axis=0)[:, 0]


def Corr_dP2(R, T):
    c = 0.020993205571310774
    D = 4.3738688812044245e-06
    L = 0.0003688151928231921
    y = np.linspace(c * T, c * T * 200, 200000)
    h = y[1] - y[0]
    return np.sum(forIntegral(y, R, T, c, D, L) * h, axis=0)[:, 0]


# deltaP2 (model)
if True:

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

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(wspace=0.4)
    plt.gcf().subplots_adjust(bottom=0.15)

    m = sp.optimize.curve_fit(
        f=Corr_R0,
        xdata=T[1:],
        ydata=dP2dP2[:, 0][1:],
        p0=(0.006, 0.0001),
    )[0]

    ax[0].plot(T[1:], dP2dP2[:, 0][1:], label="Data")
    ax[0].plot(T[1:], Corr_R0(T[1:], m[0], m[1]), label="Model")
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel(r"$\delta P_2$ Correlation")
    ax[0].set_ylim([-2e-06, 1.3e-05])
    ax[0].title.set_text(r"Correlation of $\delta P_2$, $R=0$")
    ax[0].legend()

    m = sp.optimize.curve_fit(
        f=Corr_dP2_Integral_T2,
        xdata=R[:10],
        ydata=dP2dP2[1][:10],
        p0=0.4,
        method="lm",
    )[0]

    ax[1].plot(R, dP2dP2[1][:26], label="Data")
    ax[1].plot(R, Corr_dP2_Integral_T2(R, m), label="Model")
    ax[1].set_xlabel(r"$R (\mu m)$")
    ax[1].set_ylabel(r"$\delta P_2$ Correlation")
    ax[1].set_ylim([-2e-06, 1.3e-05])
    ax[1].title.set_text(r"Correlation of $\delta P_2$, $T=2$")
    ax[1].legend()
    fig.savefig(
        f"results/Correlation dP2 in T and R {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")
