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
plt.rcParams.update({"font.size": 16})

# -------------------

filenames, fileType = util.getFilesType()
scale = 123.26 / 512
T = 160
timeStep = 10
R = 110
rStep = 10


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


# -------------------

# 3d Scatter plot
if False:
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    for filename in filenames:
        df = dfDivisions[dfDivisions["Filename"] == filename]
        x = np.array(df["X"])
        y = np.array(df["Y"])
        t = np.array(df["T"])
        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=t, mode="markers")])
        fig.show()
        plt.close("all")

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
if True:
    df = pd.read_pickle(f"databases/divCorr{fileType}.pkl")
    expectedXY = df["expectedXY"].iloc[0]
    ExXExY = df["ExXExY"].iloc[0]
    divCorr = df["divCorr"].iloc[0]
    oriCorr = df["oriCorr"].iloc[0]
    df = 0
    maxCorr = np.max(expectedXY)

    # in
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

    t, r = np.mgrid[5:105:10, 5:115:10]
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    maxCorr = np.max(divCorr[:10])
    c = ax.pcolor(
        t,
        r,
        divCorr[:10] * 10000**2,
        cmap="RdBu_r",
        vmin=-4.1,
        vmax=4.1,
    )
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("Time apart $t$ (min)")
    ax.set_ylabel(r"Distance apart $r (\mu m)$")
    fileTitle = util.getFileTitle(fileType)
    fileTitle = util.getBoldTitle(fileTitle)
    ax.title.set_text(f"Division density \n correlation " + fileTitle)

    fig.savefig(
        f"results/Division Correlation figure {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    maxCorr = np.max(divCorr[:10])
    c = ax.pcolor(
        t,
        r,
        (divCorr[:10] - np.mean(divCorr[:10, 7:10], axis=1).reshape((10, 1)))
        * 10000**2,
        cmap="RdBu_r",
        vmin=-4.1,
        vmax=4.1,
    )
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("Time apart $t$ (min)")
    ax.set_ylabel(r"Distance apart $r (\mu m)$")
    fileTitle = util.getFileTitle(fileType)
    fileTitle = util.getBoldTitle(fileTitle)

    ax.title.set_text(
        f"Division density \n correlation minus long \n times " + fileTitle
    )

    fig.savefig(
        f"results/Division Correlation figure remove long times {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
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
    fileTitle = util.getBoldTitle(fileTitle)
    ax.title.set_text(f"Division orientation \n correlation " + fileTitle)

    fig.savefig(
        f"results/Division orientation figure {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

# Division orientation correlations by filename
if False:
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
