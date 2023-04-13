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
if True:
    dfCor = pd.read_pickle(f"databases/dfCorrelationWound{fileType}.pkl")

    fig, ax = plt.subplots(4, 3, figsize=(16, 16))

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

    dRhodRhoClose - dRhodRhoClose[-1]
    dRhodRhoFar - dRhodRhoFar[-1]

    maxCorr = np.max(
        [
            np.nan_to_num(dRhodRhoClose),
            -np.nan_to_num(dRhodRhoClose),
            np.nan_to_num(dRhodRhoFar),
            -np.nan_to_num(dRhodRhoFar),
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
    ax[0, 0].title.set_text(r"Close to wound $\langle \delta \rho \delta \rho \rangle$")

    c = ax[1, 0].pcolor(
        t,
        r,
        dRhodRhoFar,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set_xlabel("Time (mins)")
    ax[1, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 0].title.set_text(r"Far from wound $\langle \delta \rho \delta \rho \rangle$")

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

    t, r = np.mgrid[0:102:2, 0:52:2]

    maxCorr = np.max(
        [
            np.nan_to_num(dQ1dQ1Close),
            -np.nan_to_num(dQ1dQ1Close),
            np.nan_to_num(dQ1dQ1Far),
            -np.nan_to_num(dQ1dQ1Far),
        ]
    )
    c = ax[0, 1].pcolor(
        t,
        r,
        dQ1dQ1Close,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 1])
    ax[0, 1].set_xlabel("Time (mins)")
    ax[0, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 1].title.set_text(r"Close to wound $\langle \delta Q^1 \delta Q^1 \rangle$")

    c = ax[1, 1].pcolor(
        t,
        r,
        dQ1dQ1Far,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 1])
    ax[1, 1].set_xlabel("Time (mins)")
    ax[1, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 1].title.set_text(r"Far from wound $\langle \delta Q^1 \delta Q^1 \rangle$")

    maxCorr = np.max(
        [
            np.nan_to_num(dQ2dQ2Close),
            -np.nan_to_num(dQ2dQ2Close),
            np.nan_to_num(dQ2dQ2Far),
            -np.nan_to_num(dQ2dQ2Far),
        ]
    )
    c = ax[0, 2].pcolor(
        t,
        r,
        dQ2dQ2Close,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 2])
    ax[0, 2].set_xlabel("Time (mins)")
    ax[0, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 2].title.set_text(r"Close to wound $\langle \delta Q^2 \delta Q^2 \rangle$")

    c = ax[1, 2].pcolor(
        t,
        r,
        dQ2dQ2Far,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 2])
    ax[1, 2].set_xlabel("Time (mins)")
    ax[1, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 2].title.set_text(r"Far from wound $\langle \delta Q^2 \delta Q^2 \rangle$")

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

    dV1dV1Close = np.mean(dV1dV1Close, axis=0)
    dV1dV1Far = np.mean(dV1dV1Far, axis=0)
    dV2dV2Close = np.mean(dV2dV2Close, axis=0)
    dV2dV2Far = np.mean(dV2dV2Far, axis=0)

    maxCorr = np.max(
        [
            np.nan_to_num(dV1dV1Close),
            -np.nan_to_num(dV1dV1Close),
            np.nan_to_num(dV1dV1Far),
            -np.nan_to_num(dV1dV1Far),
        ]
    )
    c = ax[2, 0].pcolor(
        t,
        r,
        dV1dV1Close,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 0])
    ax[2, 0].set_xlabel("Time (mins)")
    ax[2, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 0].title.set_text(r"Close to wound $\langle \delta V_1 \delta V_1 \rangle$")

    c = ax[3, 0].pcolor(
        t,
        r,
        dV1dV1Far,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[3, 0])
    ax[3, 0].set_xlabel("Time (mins)")
    ax[3, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[3, 0].title.set_text(r"Far from wound $\langle \delta V_1 \delta V_1 \rangle$")

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

    c = ax[3, 1].pcolor(
        t,
        r,
        dV2dV2Far,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[3, 1])
    ax[3, 1].set_xlabel("Time (mins)")
    ax[3, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[3, 1].title.set_text(r"Far from wound $\langle \delta V_2 \delta V_2 \rangle$")
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
