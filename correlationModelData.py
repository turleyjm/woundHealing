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
scale = 123.26 / 512


def cor_R0(t, c, D):
    return D * -sc.expi(-c * t)


def forIntegral(y, R, T, c, D, L):
    y, R, T = np.meshgrid(y, R, T, indexing="ij")
    return D * np.exp(-y) * sc.jv(0, R * ((y - c * T) / (L * T)) ** 0.5) / y


def cor_T0_dQ2(R, L):
    c = 0.014231800277153952
    D = 8.06377854e-06

    T = 0
    y = np.linspace(c * T, c * T * 200, 20000)
    h = y[1] - y[0]
    return np.sum(forIntegral(y, R, T, c, D, L) * h, axis=0)[:, 0]


def Integral_dQ2(R, L):
    c = 0.005710229096088735
    D = 0.0001453973391906898

    T = 2
    y = np.linspace(c * T, c * T * 200, 200000)
    h = y[1] - y[0]
    return np.sum(forIntegral(y, R, T, c, D, L) * h, axis=0)[:, 0]


def Corr_dQ2(R, T):
    a = 0.014231800277153952
    b = 0.02502418
    C = 8.06377854e-06
    y = np.linspace(a, a * 100, 100000)
    h = y[1] - y[0]
    return np.sum(forIntegral(y, b, R, a, T, C) * h, axis=0)[:, 0]


# -------------------

grid = 27
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

# -------------------

# deltaQ2 (model)
if True:

    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    plt.rcParams.update({"font.size": 12})

    T, R, Theta = dfCor["dQ1dQ1Correlation"].iloc[0][:, :-1, :-1].shape

    dQ2dQ2 = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):
        dQ2dQ2total = dfCor["dQ2dQ2Count"].iloc[i][:, :-1, :-1]
        dQ2dQ2[i] = np.sum(
            dfCor["dQ2dQ2Correlation"].iloc[i][:, :-1, :-1] * dQ2dQ2total, axis=2
        ) / np.sum(dQ2dQ2total, axis=2)

    dQ2dQ2 = np.mean(dQ2dQ2, axis=0)

    T = np.linspace(0, 2 * (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)

    m = sp.optimize.curve_fit(
        f=cor_R0,
        xdata=T[1:],
        ydata=dQ2dQ2[:, 0][1:],
        p0=(0.000006, 0.01),
    )[0]

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    ax[0].plot(T[1:], dQ2dQ2[:, 0][1:], label="Data")
    ax[0].plot(T[1:], cor_R0(T[1:], m[0], m[1]), label="Model")
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel(r"$Q^{(2)}$ Correlation")
    # ax[0].set_ylim([-0.000005, 0.000025])
    ax[0].set_xlim([0, 2 * timeGrid])
    ax[0].title.set_text(r"Correlation of $\delta Q^{(2)}$, $R=0$")
    ax[0].legend()

    m = sp.optimize.curve_fit(
        f=Integral_dQ2,
        xdata=R[:25],
        ydata=dQ2dQ2[0][5:25],
        p0=0.025,
        method="lm",
    )[0]

    ax[1].plot(R, dQ2dQ2[0], label="Data")
    ax[1].plot(R, Integral_dQ2(R, m), label="Model")
    ax[1].set_xlabel(r"$R (\mu m)$")
    ax[1].set_ylabel(r"$Q^{(2)}$ Correlation")
    # ax[1].set_ylim([-0.000005, 0.000025])
    ax[1].title.set_text(r"Correlation of $\delta Q^{(2)}$, $T=0$")
    ax[1].legend()
    fig.savefig(
        f"results/Correlation Q2 in T and R {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# deltaP2 (model)
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
