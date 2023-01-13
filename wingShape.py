import os
import shutil
from math import floor, log10

from collections import Counter
import cv2
import matplotlib
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
import utils as util

plt.rcParams.update({"font.size": 16})

# -------------------

filenames, fileType = util.getFilesType()

T = 90
scale = 123.26 / 512


# -------------------

# typical cell length
if False:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    A = []
    for t in range(T):
        A.append(np.mean(dfShape["Area"][dfShape["T"] == t] ** 0.5))

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(2 * np.array(range(T)), A)
    ax.set(xlabel=r"Time", ylabel=r"Typical cell length $(\mu m)$")
    ax.title.set_text("Typical Cell Length")
    fig.savefig(
        f"results/Typical cell length {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# mean sf
if False:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    sf = []
    sfstd = []
    for t in range(T):
        sf.append(np.mean(dfShape["Shape Factor"][dfShape["T"] == t]))
        sfstd.append(
            np.std(np.stack(dfShape["Shape Factor"][dfShape["T"] == t], axis=0))
        )

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].errorbar(2 * np.array(range(T)), sf, yerr=sfstd)
    ax[0].set(xlabel=r"Time", ylabel=r"$\bar{S_f}$")
    ax[0].title.set_text(r"Mean of $S_f$")
    # ax[0].set_ylim([-0.03, 0.05])

    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        sf = []
        for t in range(T):
            sf.append(np.mean(df["Shape Factor"][df["T"] == t]))

        ax[1].plot(2 * np.array(range(T)), sf)

    ax[1].set(xlabel=r"Time", ylabel=r"$\bar{S_f}$")
    ax[1].title.set_text(r"Mean of $S_f$ indivial videos")
    # ax[1].set_ylim([-0.03, 0.05])

    fig.savefig(
        f"results/mean sf {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# Q1 tensor
if False:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    Q1 = []
    Q1std = []
    for t in range(T):
        Q1.append(np.mean(dfShape["q"][dfShape["T"] == t])[0, 0])
        Q1std.append(np.std(np.stack(dfShape["q"][dfShape["T"] == t], axis=0)[:, 0, 0]))

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].errorbar(2 * np.array(range(T)), Q1, yerr=Q1std)
    ax[0].set(xlabel=r"Time", ylabel=r"$\bar{Q}^{(1)}$")
    ax[0].title.set_text(r"Mean of $Q^{(1)}$")
    ax[0].set_ylim([-0.025, 0.06])

    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        Q1 = []
        for t in range(T):
            Q1.append(np.mean(df["q"][df["T"] == t])[0, 0])

        ax[1].plot(2 * np.array(range(T)), Q1)

    ax[1].set(xlabel=r"Time", ylabel=r"$\bar{Q}^{(1)}$")
    ax[1].title.set_text(r"Mean of $Q^{(1)}$ indivial videos")
    ax[1].set_ylim([-0.025, 0.06])

    plt.subplots_adjust(wspace=0.3)
    fig.savefig(
        f"results/mean Q1 {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# Q2 tensor
if False:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    Q2 = []
    Q2std = []
    for t in range(T):
        Q2.append(np.mean(dfShape["q"][dfShape["T"] == t])[0, 1])
        Q2std.append(np.std(np.stack(dfShape["q"][dfShape["T"] == t], axis=0)[:, 0, 1]))

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].errorbar(2 * np.array(range(T)), Q2, yerr=Q2std)
    ax[0].set(xlabel=r"Time", ylabel=r"$\bar{Q}^{(2)}$")
    ax[0].title.set_text(r"Mean of $Q^{(2)}$")
    ax[0].set_ylim([-0.03, 0.05])

    ax[1].set_ylim([-0.025, 0.05])
    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        Q2 = []
        for t in range(T):
            Q2.append(np.mean(df["q"][df["T"] == t])[0, 1])

        ax[1].plot(2 * np.array(range(T)), Q2)

    ax[1].set(xlabel=r"Time", ylabel=r"$\bar{Q}^{(2)}$")
    ax[1].title.set_text(r"Mean of $Q^{(2)}$ indivial videos")
    ax[1].set_ylim([-0.03, 0.05])

    fig.savefig(
        f"results/mean Q2 {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# mean Q2 over Q1
if False:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    Q1 = []
    Q2 = []
    Q2std = []
    for t in range(T):
        Q1.append(np.mean(dfShape["q"][dfShape["T"] == t])[0, 0])
        Q2.append(np.mean(dfShape["q"][dfShape["T"] == t])[0, 1])
        Q2std.append(np.std(np.stack(dfShape["q"][dfShape["T"] == t], axis=0)[:, 0, 1]))

    Q1max = np.max(Q1)
    Q2 = np.array(Q2) / Q1max
    Q2std = np.array(Q2std) / Q1max

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].errorbar(2 * np.array(range(T)), Q2, yerr=Q2std)
    ax[0].set(xlabel=r"Time", ylabel=r"$\bar{Q}^{(2)}/\bar{Q}^{(1)}$")
    ax[0].title.set_text(r"Mean of $Q^{(2)}$")
    ax[0].set_ylim([-0.5, 1])

    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        Q2 = []
        for t in range(T):
            Q2.append(np.mean(df["q"][df["T"] == t])[0, 1])

        Q2 = np.array(Q2) / Q1max
        ax[1].plot(2 * np.array(range(T)), Q2)

    ax[1].set(xlabel=r"Time", ylabel=r"$\bar{Q}^{(2)}/\bar{Q}^{(1)}$")
    ax[1].title.set_text(r"Mean of $Q^{(2)}$ indivial videos")
    ax[1].set_ylim([-0.5, 1])

    fig.savefig(
        f"results/mean Q2 over Q1 {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# mean P1
if False:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    P1 = []
    P1std = []
    for t in range(T):
        P1.append(np.mean(dfShape["Polar"][dfShape["T"] == t])[0])
        P1std.append(
            np.std(np.stack(dfShape["Polar"][dfShape["T"] == t], axis=0)[:, 0])
        )

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].errorbar(2 * np.array(range(T)), P1, yerr=P1std)
    ax[0].set(xlabel=r"Time", ylabel=r"$\bar{P}_1$")
    ax[0].title.set_text(r"Mean of $P_1$")
    ax[0].set_ylim([-0.01, 0.01])

    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        P1 = []
        for t in range(T):
            P1.append(np.mean(df["Polar"][df["T"] == t])[0])

        ax[1].plot(2 * np.array(range(T)), P1)

    ax[1].set(xlabel=r"Time", ylabel=r"$\bar{P}_1$")
    ax[1].title.set_text(r"Mean of $P_1$ indivial videos")
    ax[1].set_ylim([-0.01, 0.01])

    fig.savefig(
        f"results/mean P1 {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# mean P2
if False:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    P2 = []
    P2std = []
    for t in range(T):
        P2.append(np.mean(dfShape["Polar"][dfShape["T"] == t])[1])
        P2std.append(
            np.std(np.stack(dfShape["Polar"][dfShape["T"] == t], axis=0)[:, 1])
        )

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].errorbar(2 * np.array(range(T)), P2, yerr=P2std)
    ax[0].set(xlabel=r"Time", ylabel=r"$\bar{P}_2$")
    ax[0].title.set_text(r"Mean of $P_2$")
    ax[0].set_ylim([-0.01, 0.01])

    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        P2 = []
        for t in range(T):
            P2.append(np.mean(df["Polar"][df["T"] == t])[1])

        ax[1].plot(2 * np.array(range(T)), P2)

    ax[1].set(xlabel=r"Time", ylabel=r"$\bar{P}_2$")
    ax[1].title.set_text(r"Mean of $P_2$ indivial videos")
    ax[1].set_ylim([-0.01, 0.01])

    fig.savefig(
        f"results/mean P2 {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# mean rho
if False:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    rho = []
    for t in range(T):
        rho.append(1 / np.mean(dfShape["Area"][dfShape["T"] == t]))

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(2 * np.array(range(T)), rho)
    ax[0].set(xlabel=r"Time", ylabel=r"$\rho$")
    ax[0].title.set_text(r"$\rho$")
    ax[0].set_ylim([0.048, 0.1])

    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        rho = []
        for t in range(T):
            rho.append(1 / np.mean(df["Area"][df["T"] == t]))

        ax[1].plot(2 * np.array(range(T)), rho)

    ax[1].set(xlabel=r"Time", ylabel=r"$\rho$")
    ax[1].title.set_text(r"$\rho$ of indivial videos")
    ax[1].set_ylim([0.048, 0.1])

    fig.savefig(
        f"results/mean rho {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# Q_0 vs Area
if False:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    area = np.array(dfShape["Area"])
    q_0 = (
        np.stack(np.array(dfShape.loc[:, "dq"]), axis=0)[:, 0, 0] ** 2
        + np.stack(np.array(dfShape.loc[:, "dq"]), axis=0)[:, 0, 1] ** 2
    ) ** 0.5

    # set grid size
    grid = 30
    heatmap = np.zeros([grid, grid])
    q_0Max = max(q_0)
    areaMax = max(area)

    # for each spot pulls out there coords
    for i in range(len(q_0)):
        heatmap[
            int((grid - 1) * (q_0[i] / q_0Max)), int((grid - 1) * (area[i] / areaMax))
        ] += 1

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    plt.subplots_adjust(wspace=0.28)
    ax[0].hist(area, bins=30)
    ax[0].set_yscale("log")
    ax[0].set(xlabel=r"Area $(\mu m^2)$", ylabel=r"log freq")
    ax[0].title.set_text(r"Distribution of Cell Area")

    ax[1].hist(q_0, bins=30)
    ax[1].set_yscale("log")
    ax[1].set(xlabel=r"$q_0$", ylabel=r"log freq")
    ax[1].title.set_text(r"Distribution of Cell $q_0$")

    dx, dy = q_0Max / grid, areaMax / grid
    x, y = np.mgrid[0:q_0Max:dx, 0:areaMax:dy]

    # make heatmap
    c = ax[2].pcolor(
        x,
        y,
        heatmap,
        norm=colors.LogNorm(),
        cmap="Reds",
    )
    fig.colorbar(c, ax=ax[2])
    ax[2].set(xlabel=r"Area $(\mu m^2)$", ylabel=r"$q_0$")
    ax[2].title.set_text(r"Correlation of $q_0$ and area")

    fig.savefig(
        f"results/Q_0 and area correlation {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# entropy
if False:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    ent = []
    entStd = []
    for t in range(T):
        x = []
        for filename in filenames:
            q = np.stack(
                dfShape["q"][(dfShape["T"] == t) & (dfShape["Filename"] == filename)]
            )
            heatmap, xedges, yedges = np.histogram2d(
                q[:, 0, 0],
                q[:, 1, 0],
                range=[[-0.3, 0.3], [-0.15, 0.15]],
                bins=(30, 15),
            )

            prob = heatmap / q.shape[0]
            prob[prob != 0]
            p = prob[prob != 0]

            entropy = p * np.log(p)
            x.append(-sum(entropy))
        ent.append(np.mean(x))
        entStd.append(np.std(x))

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.errorbar(2 * np.array(range(T)), ent, yerr=entStd)
    ax.set(xlabel=r"Time", ylabel="Shannon entropy")
    ax.title.set_text("Shannon entropy with time")
    ax.set_ylim([2.5, 3.6])

    plt.subplots_adjust(wspace=0.3)
    fig.savefig(
        f"results/Shannon entropy {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")
