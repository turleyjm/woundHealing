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

# display all correlations shape
if False:
    dfCor = pd.read_pickle(f"databases/dfCorrelationWound{fileType}.pkl")

    fig, ax = plt.subplots(3, 4, figsize=(20, 12))

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

    dRhodRhoClose = dRhodRhoClose - np.mean(dRhodRhoClose[10:-1], axis=0)
    dRhodRhoFar = dRhodRhoFar - np.mean(dRhodRhoFar[10:-1], axis=0)

    maxCorr = np.max(
        [
            np.nan_to_num(dRhodRhoClose),
            -np.nan_to_num(dRhodRhoClose),
        ]
    )
    t, r = np.mgrid[0:180:10, 0:70:10]

    c = ax[0, 0].pcolor(
        t,
        r,
        dRhodRhoClose,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 0])
    ax[0, 0].set_xlabel("Time (mins)")
    ax[0, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 0].title.set_text(
        r"Close to wound $\langle \delta \rho_n \delta \rho_n \rangle$"
    )

    c = ax[0, 1].pcolor(
        t,
        r,
        dRhodRhoFar,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 1])
    ax[0, 1].set_xlabel("Time (mins)")
    ax[0, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 1].title.set_text(
        r"Far from wound $\langle \delta \rho_n \delta \rho_n \rangle$"
    )

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

    dQ1dQ1Close = np.mean(dQ1dQ1Close, axis=0)[:-5]
    dQ1dQ1Far = np.mean(dQ1dQ1Far, axis=0)[:-5]
    dQ2dQ2Close = np.mean(dQ2dQ2Close, axis=0)[:-5]
    dQ2dQ2Far = np.mean(dQ2dQ2Far, axis=0)[:-5]
    print(dQ2dQ2Far[:, 0])

    t, r = np.mgrid[0:92:2, 0:52:2]

    maxCorr = np.max(
        [
            np.nan_to_num(dQ1dQ1Close),
            -np.nan_to_num(dQ1dQ1Close),
            np.nan_to_num(dQ1dQ1Far),
            -np.nan_to_num(dQ1dQ1Far),
        ]
    )
    c = ax[0, 2].pcolor(
        t,
        r,
        dQ1dQ1Close,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 2])
    ax[0, 2].set_xlabel("Time (mins)")
    ax[0, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 2].title.set_text(r"Close to wound $\langle \delta Q^1 \delta Q^1 \rangle$")

    c = ax[0, 3].pcolor(
        t,
        r,
        dQ1dQ1Far,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 3])
    ax[0, 3].set_xlabel("Time (mins)")
    ax[0, 3].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 3].title.set_text(r"Far from wound $\langle \delta Q^1 \delta Q^1 \rangle$")

    maxCorr = np.max(
        [
            np.nan_to_num(dQ2dQ2Close),
            -np.nan_to_num(dQ2dQ2Close),
            np.nan_to_num(dQ2dQ2Far),
            -np.nan_to_num(dQ2dQ2Far),
        ]
    )
    c = ax[1, 0].pcolor(
        t,
        r,
        dQ2dQ2Close,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set_xlabel("Time (mins)")
    ax[1, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 0].title.set_text(r"Close to wound $\langle \delta Q^2 \delta Q^2 \rangle$")

    c = ax[1, 1].pcolor(
        t,
        r,
        dQ2dQ2Far,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 1])
    ax[1, 1].set_xlabel("Time (mins)")
    ax[1, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 1].title.set_text(r"Far from wound $\langle \delta Q^2 \delta Q^2 \rangle$")

    dV1dV1Close = np.zeros([len(filenames), T, R - 1])
    dV1dV1Far = np.zeros([len(filenames), T, R - 1])
    dV2dV2Close = np.zeros([len(filenames), T, R - 1])
    dV2dV2Far = np.zeros([len(filenames), T, R - 1])
    for i in range(len(filenames)):
        dV1dV1Closetotal = dfCor["dV1dV1Closetotal"].iloc[i][:, :-1, :-1]
        dV1dV1Close[i] = np.sum(
            dfCor["dV1dV1Close"].iloc[i][:, :-1, :-1] * dV1dV1Closetotal, axis=2
        ) / np.sum(dV1dV1Closetotal, axis=2)

        dV1dV1Fartotal = dfCor["dV1dV1Fartotal"].iloc[i][:, :-1, :-1]
        dV1dV1Far[i] = np.sum(
            dfCor["dV1dV1Far"].iloc[i][:, :-1, :-1] * dV1dV1Fartotal, axis=2
        ) / np.sum(dV1dV1Fartotal, axis=2)

        dV2dV2Closetotal = dfCor["dV2dV2Closetotal"].iloc[i][:, :-1, :-1]
        dV2dV2Close[i] = np.sum(
            dfCor["dV2dV2Close"].iloc[i][:, :-1, :-1] * dV2dV2Closetotal, axis=2
        ) / np.sum(dV2dV2Closetotal, axis=2)

        dV2dV2Fartotal = dfCor["dV2dV2Fartotal"].iloc[i][:, :-1, :-1]
        dV2dV2Far[i] = np.sum(
            dfCor["dV2dV2Far"].iloc[i][:, :-1, :-1] * dV2dV2Fartotal, axis=2
        ) / np.sum(dV2dV2Fartotal, axis=2)

    dV1dV1Close = np.mean(dV1dV1Close, axis=0)[:-5]
    dV1dV1Far = np.mean(dV1dV1Far, axis=0)[:-5]
    dV2dV2Close = np.mean(dV2dV2Close, axis=0)[:-5]
    dV2dV2Far = np.mean(dV2dV2Far, axis=0)[:-5]

    maxCorr = np.max(
        [
            np.nan_to_num(dV1dV1Close),
            -np.nan_to_num(dV1dV1Close),
            np.nan_to_num(dV1dV1Far),
            -np.nan_to_num(dV1dV1Far),
        ]
    )
    c = ax[1, 2].pcolor(
        t,
        r,
        dV1dV1Close,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 2])
    ax[1, 2].set_xlabel("Time (mins)")
    ax[1, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 2].title.set_text(r"Close to wound $\langle \delta V_1 \delta V_1 \rangle$")

    c = ax[1, 3].pcolor(
        t,
        r,
        dV1dV1Far,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 3])
    ax[1, 3].set_xlabel("Time (mins)")
    ax[1, 3].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 3].title.set_text(r"Far from wound $\langle \delta V_1 \delta V_1 \rangle$")

    maxCorr = np.max(
        [
            np.nan_to_num(dV2dV2Close),
            -np.nan_to_num(dV2dV2Close),
            np.nan_to_num(dV2dV2Far),
            -np.nan_to_num(dV2dV2Far),
        ]
    )
    c = ax[2, 1].pcolor(
        t,
        r,
        dV2dV2Close,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 1])
    ax[2, 1].set_xlabel("Time (mins)")
    ax[2, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 1].title.set_text(r"Close to wound $\langle \delta V_2 \delta V_2 \rangle$")

    c = ax[2, 2].pcolor(
        t,
        r,
        dV2dV2Far,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 2])
    ax[2, 2].set_xlabel("Time (mins)")
    ax[2, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 2].title.set_text(r"Far from wound $\langle \delta V_2 \delta V_2 \rangle$")
    # plt.subplot_tool()
    plt.subplots_adjust(
        left=0.075, bottom=0.1, right=0.95, top=0.9, wspace=0.4, hspace=0.45
    )

    fig.savefig(
        f"results/Correlations close and far from wounds {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# deltaQ1 (model)
if False:

    def Corr_dQ1_Integral_T(R, B, L):
        C = 0.00055
        T = 0
        k = np.linspace(0, 4, 200000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, C, L) * h, axis=0)[:, 0]

    def Corr_dQ1_Integral_R(T, B, L):
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

    dfCor = pd.read_pickle(f"databases/dfCorrelationsUnwound18h.pkl")

    T, R, Theta = dfCor["dQ1dQ1Correlation"].iloc[0][:, :-1, :-1].shape

    dQ1dQ1 = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):

        dQ1dQ1total = dfCor["dQ1dQ1Count"].iloc[i][:, :-1, :-1]
        dQ1dQ1[i] = np.sum(
            dfCor["dQ1dQ1Correlation"].iloc[i][:, :-1, :-1] * dQ1dQ1total, axis=2
        ) / np.sum(dQ1dQ1total, axis=2)

    dfCor = 0

    dQ1dQ1 = np.mean(dQ1dQ1, axis=0)

    dfCor = pd.read_pickle(f"databases/dfCorrelationWound{fileType}.pkl")

    T, R, Theta = dfCor["dQ1dQ1Close"].iloc[0].shape

    dQ1dQ1Close = np.zeros([len(filenames), T, R - 1])
    dQ1dQ1Far = np.zeros([len(filenames), T, R - 1])
    for i in range(len(filenames)):
        dQ1dQ1Closetotal = dfCor["dQ1dQ1Closetotal"].iloc[i][:, :-1, :-1]
        dQ1dQ1Close[i] = np.sum(
            dfCor["dQ1dQ1Close"].iloc[i][:, :-1, :-1] * dQ1dQ1Closetotal, axis=2
        ) / np.sum(dQ1dQ1Closetotal, axis=2)

        dQ1dQ1Fartotal = dfCor["dQ1dQ1Fartotal"].iloc[i][:, :-1, :-1]
        dQ1dQ1Far[i] = np.sum(
            dfCor["dQ1dQ1Far"].iloc[i][:, :-1, :-1] * dQ1dQ1Fartotal, axis=2
        ) / np.sum(dQ1dQ1Fartotal, axis=2)

    dfCor = 0

    dQ1dQ1Close = np.mean(dQ1dQ1Close, axis=0)[:-5]
    dQ1dQ1Far = np.mean(dQ1dQ1Far, axis=0)[:-5]

    T = np.linspace(0, 2 * (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    M = sp.optimize.curve_fit(
        f=Corr_dQ1_Integral_R,
        xdata=T[1:-5],
        ydata=dQ1dQ1Close[:, 0][1:],
        p0=(0.006, 4),
    )[0]
    print(M)

    m = sp.optimize.curve_fit(
        f=Corr_dQ1_Integral_R,
        xdata=T[1:-5],
        ydata=dQ1dQ1Far[:, 0][1:],
        p0=(0.006, 4),
    )[0]
    print(m)

    ax[0].plot(T[1:-5], dQ1dQ1Close[:, 0][1:], label="Close to wound", color="g")
    ax[0].plot(T[1:-5], Corr_dQ1_Integral_R(T[1:-5], M[0], M[1]), label="Model close")
    ax[0].plot(T[1:-5], dQ1dQ1Far[:, 0][1:], label="far from wound", color="m")
    ax[0].plot(T[1:-5], Corr_dQ1_Integral_R(T[1:-5], m[0], m[1]), label="Model far")
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel(r"$\delta Q^{(1)}$ Correlation")
    ax[0].set_ylim([0, 5.9e-04])
    ax[0].title.set_text(r"Correlation of $\delta Q^{(1)}$, $R=0$")
    ax[0].legend(fontsize=12)

    M = sp.optimize.curve_fit(
        f=Corr_dQ1_Integral_T,
        xdata=R[1:],
        ydata=dQ1dQ1Close[1][1:26],
        p0=(0.006, 4),
        method="lm",
    )[0]
    print(M)

    m = sp.optimize.curve_fit(
        f=Corr_dQ1_Integral_T,
        xdata=R[1:],
        ydata=dQ1dQ1Far[1][1:26],
        p0=(0.006, 4),
        method="lm",
    )[0]
    print(m)

    ax[1].plot(R[5:], dQ1dQ1Close[0][5:26], label="Close to wound", color="g")
    ax[1].plot(R[5:], Corr_dQ1_Integral_T(R[5:], M[0], M[1]), label="Model close")
    ax[1].plot(R[5:], dQ1dQ1Far[0][5:26], label="far from wound", color="m")
    ax[1].plot(R[5:], Corr_dQ1_Integral_T(R[5:], m[0], m[1]), label="Model")
    ax[1].set_xlabel(r"$R (\mu m)$")
    ax[1].set_ylabel(r"$\delta Q^{(1)}$ Correlation")
    ax[1].set_ylim([0, 5.9e-04])
    ax[1].title.set_text(r"Correlation of $\delta Q^{(1)}$, $T=0$")
    ax[1].legend(fontsize=12)

    plt.subplots_adjust(
        left=0.22, bottom=0.1, right=0.93, top=0.9, wspace=0.55, hspace=0.37
    )
    fig.savefig(
        f"results/Correlation dQ1 close and far from wounds {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# deltaQ2 (model)
if False:

    def Corr_dQ2_Integral_T(R, B, L):
        C = 0.00055
        T = 0
        k = np.linspace(0, 4, 200000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, C, L) * h, axis=0)[:, 0]

    def Corr_dQ2_Integral_R(T, B, L):
        C = 0.00055
        R = 0
        k = np.linspace(0, 4, 200000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, C, L) * h, axis=0)[0]

    def Corr_dQ2(R, T):
        B = 0.006533824439392692
        C = 0.00055
        L = 2.1

        k = np.linspace(0, 4, 200000)
        h = k[1] - k[0]
        return np.sum(forIntegral(k, R, T, B, C, L) * h, axis=0)

    dfCor = pd.read_pickle(f"databases/dfCorrelationsUnwound18h.pkl")

    T, R, Theta = dfCor["dQ2dQ2Correlation"].iloc[0][:, :-1, :-1].shape

    dQ2dQ2 = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):

        dQ2dQ2total = dfCor["dQ2dQ2Count"].iloc[i][:, :-1, :-1]
        dQ2dQ2[i] = np.sum(
            dfCor["dQ2dQ2Correlation"].iloc[i][:, :-1, :-1] * dQ2dQ2total, axis=2
        ) / np.sum(dQ2dQ2total, axis=2)

    dfCor = 0

    dQ2dQ2 = np.mean(dQ2dQ2, axis=0)

    dfCor = pd.read_pickle(f"databases/dfCorrelationWound{fileType}.pkl")

    T, R, Theta = dfCor["dQ2dQ2Close"].iloc[0].shape

    dQ2dQ2Close = np.zeros([len(filenames), T, R - 1])
    dQ2dQ2Far = np.zeros([len(filenames), T, R - 1])
    for i in range(len(filenames)):
        dQ2dQ2Closetotal = dfCor["dQ2dQ2Closetotal"].iloc[i][:, :-1, :-1]
        dQ2dQ2Close[i] = np.sum(
            dfCor["dQ2dQ2Close"].iloc[i][:, :-1, :-1] * dQ2dQ2Closetotal, axis=2
        ) / np.sum(dQ2dQ2Closetotal, axis=2)

        dQ2dQ2Fartotal = dfCor["dQ2dQ2Fartotal"].iloc[i][:, :-1, :-1]
        dQ2dQ2Far[i] = np.sum(
            dfCor["dQ2dQ2Far"].iloc[i][:, :-1, :-1] * dQ2dQ2Fartotal, axis=2
        ) / np.sum(dQ2dQ2Fartotal, axis=2)

    dfCor = 0

    dQ2dQ2Close = np.mean(dQ2dQ2Close, axis=0)[:-5]
    dQ2dQ2Far = np.mean(dQ2dQ2Far, axis=0)[:-5]

    T = np.linspace(0, 2 * (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    M = sp.optimize.curve_fit(
        f=Corr_dQ2_Integral_R,
        xdata=T[1:-5],
        ydata=dQ2dQ2Close[:, 0][1:],
        p0=(0.006, 4),
    )[0]
    print(M)

    m = sp.optimize.curve_fit(
        f=Corr_dQ2_Integral_R,
        xdata=T[1:-5],
        ydata=dQ2dQ2Far[:, 0][1:],
        p0=(0.006, 4),
    )[0]
    print(m)

    ax[0].plot(T[1:-5], dQ2dQ2Close[:, 0][1:], label="Close to wound", color="g")
    ax[0].plot(T[1:-5], Corr_dQ2_Integral_R(T[1:-5], M[0], M[1]), label="Model close")
    ax[0].plot(T[1:-5], dQ2dQ2Far[:, 0][1:], label="far from wound", color="m")
    ax[0].plot(T[1:-5], Corr_dQ2_Integral_R(T[1:-5], m[0], m[1]), label="Model far")
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel(r"$\delta Q^{(2)}$ Correlation")
    ax[0].set_ylim([0, 5.9e-04])
    ax[0].title.set_text(r"Correlation of $\delta Q^{(2)}$, $R=0$")
    ax[0].legend(fontsize=12)

    M = sp.optimize.curve_fit(
        f=Corr_dQ2_Integral_T,
        xdata=R[1:],
        ydata=dQ2dQ2Close[1][1:26],
        p0=(0.006, 4),
        method="lm",
    )[0]
    print(M)

    m = sp.optimize.curve_fit(
        f=Corr_dQ2_Integral_T,
        xdata=R[1:],
        ydata=dQ2dQ2Far[1][1:26],
        p0=(0.006, 4),
        method="lm",
    )[0]
    print(m)

    ax[1].plot(R[5:], dQ2dQ2Close[0][5:26], label="Close to wound", color="g")
    ax[1].plot(R[5:], Corr_dQ2_Integral_T(R[5:], M[0], M[1]), label="Model close")
    ax[1].plot(R[5:], dQ2dQ2Far[0][5:26], label="far from wound", color="m")
    ax[1].plot(R[5:], Corr_dQ2_Integral_T(R[5:], m[0], m[1]), label="Model far")
    ax[1].set_xlabel(r"$R (\mu m)$")
    ax[1].set_ylabel(r"$\delta Q^{(2)}$ Correlation")
    ax[1].set_ylim([0, 5.9e-04])
    ax[1].title.set_text(r"Correlation of $\delta Q^{(2)}$, $T=0$")
    ax[1].legend(fontsize=12)

    plt.subplots_adjust(
        left=0.22, bottom=0.1, right=0.93, top=0.9, wspace=0.55, hspace=0.37
    )
    fig.savefig(
        f"results/Correlation dQ2 close and far from wounds {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# deltaV (model)
if True:

    def Corr_dV(r, C, lamdba):
        return C * np.exp(-lamdba * r)

    dfCor = pd.read_pickle(f"databases/dfCorrelationsUnwound18h.pkl")

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

    dfCor = pd.read_pickle(f"databases/dfCorrelationWound{fileType}.pkl")

    T, R, Theta = dfCor["dQ1dQ1Close"].iloc[0].shape

    dV1dV1Close = np.zeros([len(filenames), T, R - 1])
    dV1dV1Far = np.zeros([len(filenames), T, R - 1])
    dV2dV2Close = np.zeros([len(filenames), T, R - 1])
    dV2dV2Far = np.zeros([len(filenames), T, R - 1])
    for i in range(len(filenames)):
        dV1dV1Closetotal = dfCor["dV1dV1Closetotal"].iloc[i][:, :-1, :-1]
        dV1dV1Close[i] = np.sum(
            dfCor["dV1dV1Close"].iloc[i][:, :-1, :-1] * dV1dV1Closetotal, axis=2
        ) / np.sum(dV1dV1Closetotal, axis=2)
        dV1dV1Fartotal = dfCor["dV1dV1Fartotal"].iloc[i][:, :-1, :-1]
        dV1dV1Far[i] = np.sum(
            dfCor["dV1dV1Far"].iloc[i][:, :-1, :-1] * dV1dV1Fartotal, axis=2
        ) / np.sum(dV1dV1Fartotal, axis=2)

        dV2dV2Closetotal = dfCor["dV2dV2Closetotal"].iloc[i][:, :-1, :-1]
        dV2dV2Close[i] = np.sum(
            dfCor["dV2dV2Close"].iloc[i][:, :-1, :-1] * dV2dV2Closetotal, axis=2
        ) / np.sum(dV2dV2Closetotal, axis=2)
        dV2dV2Fartotal = dfCor["dV2dV2Fartotal"].iloc[i][:, :-1, :-1]
        dV2dV2Far[i] = np.sum(
            dfCor["dV2dV2Far"].iloc[i][:, :-1, :-1] * dV2dV2Fartotal, axis=2
        ) / np.sum(dV2dV2Fartotal, axis=2)

    dfCor = 0

    dV1dV1Close = np.mean(dV1dV1Close, axis=0)
    dV1dV1Far = np.mean(dV1dV1Far, axis=0)
    dV2dV2Close = np.mean(dV2dV2Close, axis=0)
    dV2dV2Far = np.mean(dV2dV2Far, axis=0)

    dV1dV1Close_r = dV1dV1Close[0][0:-1]
    dV1dV1Far_r = dV1dV1Far[0][0:-1]
    dV2dV2Close_r = dV2dV2Close[0][0:-1]
    dV2dV2Far_r = dV2dV2Far[0][0:-1]

    R = np.linspace(0, 2 * (grid - 1), grid)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    M = sp.optimize.curve_fit(
        f=Corr_dV,
        xdata=R[5:25],
        ydata=dV1dV1Close_r[5:],
        p0=(0.23, 0.04),
    )[0]
    print(M)

    m = sp.optimize.curve_fit(
        f=Corr_dV,
        xdata=R[5:25],
        ydata=dV1dV1Far_r[5:],
        p0=(0.23, 0.04),
    )[0]
    print(m)

    ax[0].plot(R[5:25], dV1dV1Close_r[5:], label="Close to wound")
    ax[0].plot(R[5:], Corr_dV(R[5:], M[0], M[1]), label="Model close")
    ax[0].plot(R[5:25], dV1dV1Far_r[5:], label="Far from wound")
    ax[0].plot(R[5:], Corr_dV(R[5:], m[0], m[1]), label="Model far")
    ax[0].set_xlabel(r"$R (\mu m)$")
    ax[0].set_ylabel(r"$\delta V_1$ Correlation")
    ax[0].set_ylim([0, 0.6])
    ax[0].title.set_text(r"Correlation of $\delta V_1$")
    ax[0].legend(fontsize=12)

    M = sp.optimize.curve_fit(
        f=Corr_dV,
        xdata=R[5:25],
        ydata=dV2dV2Close_r[5:],
        p0=(0.23, 0.04),
    )[0]
    print(M)

    m = sp.optimize.curve_fit(
        f=Corr_dV,
        xdata=R[5:25],
        ydata=dV2dV2Far_r[5:],
        p0=(0.23, 0.04),
    )[0]
    print(m)

    ax[1].plot(R[5:25], dV2dV2Close_r[5:], label="Close to wound")
    ax[1].plot(R[5:], Corr_dV(R[5:], M[0], M[1]), label="Model close")
    ax[1].plot(R[5:25], dV2dV2Far_r[5:], label="Far from wound")
    ax[1].plot(R[5:], Corr_dV(R[5:], m[0], m[1]), label="Model far")
    ax[1].set_xlabel(r"$R (\mu m)$")
    ax[1].set_ylabel(r"$\delta V_2$ Correlation")
    ax[1].set_ylim([0, 0.6])
    ax[1].title.set_text(r"Correlation of $\delta V_2$")
    ax[1].legend(fontsize=12)

    plt.subplots_adjust(
        left=0.22, bottom=0.1, right=0.93, top=0.9, wspace=0.35, hspace=0.37
    )
    fig.savefig(
        f"results/Correlation dV in T and R {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")
