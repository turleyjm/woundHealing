import os
import shutil
from math import floor, log10

from collections import Counter
import cv2
import matplotlib
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy as sp
import scipy.linalg as linalg
import shapely
import skimage as sm
import skimage.io
import skimage.measure
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
import tifffile
from skimage.draw import circle_perimeter
from scipy import optimize
import xml.etree.ElementTree as et

import cellProperties as cell
import findGoodCells as fi
import commonLiberty as cl

plt.rcParams.update({"font.size": 16})

# -------------------

filenames, fileType = cl.getFilesType()

T = 93
scale = 123.26 / 512
L = 123.26
grid = 11

_df2 = []
if False:
    for filename in filenames:

        df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")

        for t in range(T):
            dft = df[df["Time"] == t]
            Q = np.mean(dft["q"])
            for i in range(len(dft)):
                [x, y] = [
                    dft["Centroid"].iloc[i][0] * scale,
                    dft["Centroid"].iloc[i][1] * scale,
                ]
                dQ = dft["q"].iloc[i] - Q
                A = dft["Area"].iloc[i] * scale ** 2
                TrdQ = np.trace(np.matmul(Q, dQ))
                Pol = dft["Polar"].iloc[i]

                _df2.append(
                    {
                        "Filename": filename,
                        "T": t,
                        "X": x,
                        "Y": y,
                        "dQ": dQ,
                        "Q": Q,
                        "TrdQ": TrdQ,
                        "Area": A,
                        "Polar": Pol,
                    }
                )

    dfShape = pd.DataFrame(_df2)
    dfShape.to_pickle(f"databases/dfContinuum{fileType}.pkl")

else:
    dfShape = pd.read_pickle(f"databases/dfContinuum{fileType}.pkl")

# density
if False:
    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        heatmapP0 = np.zeros([grid, grid])
        dft = df[df["T"] == 0]
        for i in range(grid):
            for j in range(grid):
                x = [
                    (L - 110) / 2 + i * 110 / grid,
                    (L - 110) / 2 + (i + 1) * 110 / grid,
                ]
                y = [
                    (L - 110) / 2 + j * 110 / grid,
                    (L - 110) / 2 + (j + 1) * 110 / grid,
                ]
                dfg = cl.sortGrid(dft, x, y)
                if list(dfg["Area"]) == []:
                    A = np.nan
                else:
                    A = dfg["Area"]
                    heatmapP0[i, j] = 1 / np.mean(A)

        dx, dy = 110 / grid, 110 / grid
        x, y = np.mgrid[0:110:dx, 0:110:dy]

        fig, ax = plt.subplots(1, 2, figsize=(20, 8))
        c = ax[0].pcolor(x, y, heatmapP0, cmap="Reds", vmax=0.15, shading="auto")
        fig.colorbar(c, ax=ax[0])
        ax[0].set_xlabel(r"x $(\mu m)$")
        ax[0].set_ylabel(r"y $(\mu m)$")
        ax[0].title.set_text(r"$\rho$ " + f"{filename}")

        mu = np.mean(heatmapP0)
        simga = np.std(heatmapP0)
        heatmapP0Norm = (heatmapP0 - mu) / simga

        c = ax[1].pcolor(
            x, y, heatmapP0Norm, cmap="RdBu_r", vmin=-3, vmax=3, shading="auto"
        )
        fig.colorbar(c, ax=ax[1])
        ax[1].set_xlabel(r"x $(\mu m)$")
        ax[1].set_ylabel(r"y $(\mu m)$")
        ax[1].title.set_text(r"$\sigma$ of $\rho$ from $\mu_\rho$ " + f"{filename}")

        fig.savefig(
            f"results/P0 heatmap {filename}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")


# typical cell length
if True:
    A = []
    for t in range(T):
        A.append(np.mean(dfShape["Area"][dfShape["T"] == t] ** 0.5))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(2 * np.array(range(T)), A)
    ax.set(xlabel=r"Time", ylabel=r"Typical cell length $(\mu m)$")
    ax.title.set_text("Typical Cell Length")
    fig.savefig(
        f"results/Typical cell length {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

# delta Q
if True:
    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        heatmapdQ1 = np.zeros([grid, grid])
        heatmapdQ2 = np.zeros([grid, grid])
        dft = df[df["T"] == 0]
        for i in range(grid):
            for j in range(grid):
                x = [
                    (L - 110) / 2 + i * 110 / grid,
                    (L - 110) / 2 + (i + 1) * 110 / grid,
                ]
                y = [
                    (L - 110) / 2 + j * 110 / grid,
                    (L - 110) / 2 + (j + 1) * 110 / grid,
                ]
                dfg = cl.sortGrid(dft, x, y)
                if list(dfg["dQ"]) == []:
                    A = np.nan
                else:
                    dQ = dfg["dQ"]
                    heatmapdQ1[i, j] = np.mean(dQ)[0, 0]
                    heatmapdQ2[i, j] = np.mean(dQ)[1, 0]

        dx, dy = 110 / grid, 110 / grid
        x, y = np.mgrid[0:110:dx, 0:110:dy]

        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        plt.subplots_adjust(wspace=0.3)
        plt.gcf().subplots_adjust(bottom=0.15)
        c = ax[0, 0].pcolor(
            x, y, heatmapdQ1, cmap="RdBu_r", vmin=-0.07, vmax=0.07, shading="auto"
        )
        fig.colorbar(c, ax=ax[0, 0])
        ax[0, 0].set_xlabel(r"x $(\mu m)$")
        ax[0, 0].set_ylabel(r"y $(\mu m)$")
        ax[0, 0].title.set_text(r"$\delta Q^1$ " + f"{filename}")

        c = ax[0, 1].pcolor(
            x, y, heatmapdQ2, cmap="RdBu_r", vmin=-0.07, vmax=0.07, shading="auto"
        )
        fig.colorbar(c, ax=ax[0, 1])
        ax[0, 1].set_xlabel(r"x $(\mu m)$")
        ax[0, 1].set_ylabel(r"y $(\mu m)$")
        ax[0, 1].title.set_text(r"$\delta Q^2$ " + f"{filename}")

        mu = np.mean(heatmapdQ1)
        simga = np.std(heatmapdQ1)
        heatmapdQ1Norm = (heatmapdQ1 - mu) / simga

        c = ax[1, 0].pcolor(
            x, y, heatmapdQ1Norm, cmap="RdBu_r", vmin=-3, vmax=3, shading="auto"
        )
        fig.colorbar(c, ax=ax[1, 0])
        ax[1, 0].set_xlabel(r"x $(\mu m)$")
        ax[1, 0].set_ylabel(r"y $(\mu m)$")
        ax[1, 0].title.set_text(
            r"$\sigma$ of $\delta Q^1 from \mu_{\delta Q^1}$ " + f"{filename}"
        )

        mu = np.mean(heatmapdQ2)
        simga = np.std(heatmapdQ2)
        heatmapdQ2Norm = (heatmapdQ2 - mu) / simga

        c = ax[1, 1].pcolor(
            x, y, heatmapdQ2Norm, cmap="RdBu_r", vmin=-3, vmax=3, shading="auto"
        )
        fig.colorbar(c, ax=ax[1, 1])
        ax[1, 1].set_xlabel(r"x $(\mu m)$")
        ax[1, 1].set_ylabel(r"y $(\mu m)$")
        ax[1, 1].title.set_text(
            r"$\sigma$ of $\delta Q^2 from \mu_{\delta Q^2}$ " + f"{filename}"
        )

        fig.savefig(
            f"results/deltaQ heatmap {filename}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")


# Trace of mean Q dot delta Q
if True:
    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        heatmapTr = np.zeros([grid, grid])
        dft = df[df["T"] == 0]
        for i in range(grid):
            for j in range(grid):
                x = [
                    (L - 110) / 2 + i * 110 / grid,
                    (L - 110) / 2 + (i + 1) * 110 / grid,
                ]
                y = [
                    (L - 110) / 2 + j * 110 / grid,
                    (L - 110) / 2 + (j + 1) * 110 / grid,
                ]
                dfg = cl.sortGrid(dft, x, y)
                if list(dfg["dQ"]) == []:
                    A = np.nan
                else:
                    TrdQ = dfg["TrdQ"]
                    heatmapTr[i, j] = np.mean(TrdQ)

        dx, dy = 110 / grid, 110 / grid
        x, y = np.mgrid[0:110:dx, 0:110:dy]

        fig, ax = plt.subplots(1, 2, figsize=(20, 8))
        c = ax[0].pcolor(
            x, y, heatmapTr, cmap="RdBu_r", vmin=-0.001, vmax=0.001, shading="auto"
        )
        fig.colorbar(c, ax=ax[0])
        ax[0].set_xlabel(r"x $(\mu m)$")
        ax[0].set_ylabel(r"y $(\mu m)$")
        ax[0].title.set_text(r"Tr$(\langle Q \rangle \delta Q (r))$ " + f"{filename}")

        mu = np.mean(heatmapTr)
        simga = np.std(heatmapTr)
        heatmapTrNorm = (heatmapTr - mu) / simga

        c = ax[1].pcolor(
            x, y, heatmapTrNorm, cmap="RdBu_r", vmin=-3, vmax=3, shading="auto"
        )
        fig.colorbar(c, ax=ax[1])
        ax[1].set_xlabel(r"x $(\mu m)$")
        ax[1].set_ylabel(r"y $(\mu m)$")
        ax[1].title.set_text(
            r"$\sigma$ of Tr$(\langle Q \rangle \delta Q (r))$ from $\mu$ "
            + f"{filename}"
        )

        fig.savefig(
            f"results/trdQ heatmap {filename}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")


# polarisation
if True:
    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        heatmapW0 = np.zeros([grid, grid])
        heatmapW1 = np.zeros([grid, grid])
        dft = df[df["T"] == 0]
        for i in range(grid):
            for j in range(grid):
                x = [
                    (L - 110) / 2 + i * 110 / grid,
                    (L - 110) / 2 + (i + 1) * 110 / grid,
                ]
                y = [
                    (L - 110) / 2 + j * 110 / grid,
                    (L - 110) / 2 + (j + 1) * 110 / grid,
                ]
                dfg = cl.sortGrid(dft, x, y)
                if list(dfg["Polar"]) == []:
                    A = np.nan
                else:
                    Pol = list(dfg["Polar"])
                    W = []
                    for pol in Pol:
                        W.append(pol / np.linalg.norm(pol))

                    heatmapW0[i, j] = np.mean(W, axis=0)[0]
                    heatmapW1[i, j] = np.mean(W, axis=0)[1]

        dx, dy = 110 / grid, 110 / grid
        x, y = np.mgrid[0:110:dx, 0:110:dy]

        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        plt.subplots_adjust(wspace=0.3)
        plt.gcf().subplots_adjust(bottom=0.15)
        c = ax[0, 0].pcolor(
            x, y, heatmapW0, cmap="RdBu_r", shading="auto", vmin=-1, vmax=1
        )
        fig.colorbar(c, ax=ax[0, 0])
        ax[0, 0].set_xlabel(r"x $(\mu m)$")
        ax[0, 0].set_ylabel(r"y $(\mu m)$")
        ax[0, 0].title.set_text(r"$W_1$ " + f"{filename}")

        c = ax[0, 1].pcolor(
            x, y, heatmapW1, cmap="RdBu_r", shading="auto", vmin=-1, vmax=1
        )
        fig.colorbar(c, ax=ax[0, 1])
        ax[0, 1].set_xlabel(r"x $(\mu m)$")
        ax[0, 1].set_ylabel(r"y $(\mu m)$")
        ax[0, 1].title.set_text(r"$W_2$ " + f"{filename}")

        mu = np.mean(heatmapW0)
        simga = np.std(heatmapW0)
        heatmapW0Norm = (heatmapW0 - mu) / simga

        c = ax[1, 0].pcolor(
            x, y, heatmapW0Norm, cmap="RdBu_r", shading="auto", vmin=-3, vmax=3
        )
        fig.colorbar(c, ax=ax[1, 0])
        ax[1, 0].set_xlabel(r"x $(\mu m)$")
        ax[1, 0].set_ylabel(r"y $(\mu m)$")
        ax[1, 0].title.set_text(r"$\sigma$ of $W_1$ from $\mu$ " + f"{filename}")

        mu = np.mean(heatmapW1)
        simga = np.std(heatmapW1)
        heatmapW1Norm = (heatmapW1 - mu) / simga

        c = ax[1, 1].pcolor(
            x, y, heatmapW1Norm, cmap="RdBu_r", shading="auto", vmin=-3, vmax=3
        )
        fig.colorbar(c, ax=ax[1, 1])
        ax[1, 1].set_xlabel(r"x $(\mu m)$")
        ax[1, 1].set_ylabel(r"y $(\mu m)$")
        ax[1, 1].title.set_text(r"$\sigma$ of $W_2$ from $\mu$ " + f"{filename}")

        fig.savefig(
            f"results/W heatmap {filename}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")