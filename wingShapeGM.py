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

fileTypes, groupTitle = util.getFilesTypes()

T = 90
scale = 123.26 / 512

# -------------------

# compare: Mean sf
if False:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for fileType in fileTypes:
        filenames = util.getFilesType(fileType)[0]
        dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
        S_f = np.zeros([len(filenames), T])
        for i in range(len(filenames)):
            filename = filenames[i]
            df = dfShape[dfShape["Filename"] == filename]
            for t in range(T):
                S_f[i, t] = np.mean(df["Shape Factor"][df["T"] == t])

        time = 2 * np.array(range(T))

        std = np.std(S_f, axis=0)
        S_f = np.mean(S_f, axis=0)
        colour, mark = util.getColorLineMarker(fileType, groupTitle)
        fileTitle = util.getFileTitle(fileType)
        ax.plot(
            time,
            S_f,
            label=fileTitle,
            color=colour,
            marker=mark,
            markevery=10,
        )
        ax.fill_between(time, S_f - std, S_f + std, alpha=0.15, color=colour)

    ax.set_ylim([0.3, 0.47])
    ax.legend(loc="upper left", fontsize=12)
    ax.set(xlabel="Time after wounding (mins)", ylabel=r"$S_f$")
    boldTitle = util.getBoldTitle(groupTitle)
    ax.title.set_text(f"Mean shape factor with \n time " + boldTitle)
    fig.savefig(
        f"results/mean S_f {groupTitle}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# compare: Mean Q1 tensor
if False:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for fileType in fileTypes:
        filenames = util.getFilesType(fileType)[0]
        dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
        q1 = np.zeros([len(filenames), T])
        for i in range(len(filenames)):
            filename = filenames[i]
            df = dfShape[dfShape["Filename"] == filename]
            for t in range(T):
                q1[i, t] = np.mean(df["q"][df["T"] == t])[0, 0]

        time = 2 * np.array(range(T))

        std = np.std(q1, axis=0)
        Q1 = np.mean(q1, axis=0)
        colour, mark = util.getColorLineMarker(fileType, groupTitle)
        fileTitle = util.getFileTitle(fileType)
        ax.plot(time, Q1, label=fileTitle, color=colour, marker=mark, markevery=10)
        ax.fill_between(time, Q1 - std, Q1 + std, alpha=0.15, color=colour)

    ax.set_ylim([-0.005, 0.045])
    ax.legend(loc="upper left", fontsize=12)
    ax.set(xlabel="Time (mins)", ylabel=r"$\bar{Q}^{(1)}$")
    boldTitle = util.getBoldTitle(groupTitle)
    ax.title.set_text(r"Mean $Q^{(1)}$" + " with \n time " + boldTitle)
    fig.savefig(
        f"results/mean Q1 {groupTitle}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# compare: Mean Q2 tensor
if False:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for fileType in fileTypes:
        filenames = util.getFilesType(fileType)[0]
        dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
        q2 = np.zeros([len(filenames), T])
        for i in range(len(filenames)):
            filename = filenames[i]
            df = dfShape[dfShape["Filename"] == filename]
            for t in range(T):
                q2[i, t] = np.mean(df["q"][df["T"] == t])[0, 1]

        time = 2 * np.array(range(T))

        std = np.std(q2, axis=0)
        Q2 = np.mean(q2, axis=0)
        colour, mark = util.getColorLineMarker(fileType, groupTitle)
        fileTitle = util.getFileTitle(fileType)
        ax.plot(time, Q2, label=fileTitle, color=colour, marker=mark, markevery=10)
        ax.fill_between(time, Q2 - std, Q2 + std, alpha=0.15, color=colour)

    ax.set_ylim([-0.025, 0.025])
    ax.legend(loc="upper left", fontsize=12)
    ax.set(xlabel="Time (mins)", ylabel=r"$\bar{Q}^{(2)}$")
    boldTitle = util.getBoldTitle(groupTitle)
    ax.title.set_text(r"Mean $Q^{(2)}$" + " with \n time " + boldTitle)
    fig.savefig(
        f"results/mean Q2 {groupTitle}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# compare: Mean P1
if False:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for fileType in fileTypes:
        filenames = util.getFilesType(fileType)[0]
        dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
        p1 = np.zeros([len(filenames), T])
        for i in range(len(filenames)):
            filename = filenames[i]
            df = dfShape[dfShape["Filename"] == filename]
            for t in range(T):
                p1[i, t] = np.mean(df["Polar"][df["T"] == t])[0]

        time = 2 * np.array(range(T))

        std = np.std(p1, axis=0)
        P1 = np.mean(p1, axis=0)
        colour, mark = util.getColorLineMarker(fileType, groupTitle)
        fileTitle = util.getFileTitle(fileType)
        ax.plot(time, P1, label=fileTitle, color=colour, marker=mark, markevery=10)
        ax.fill_between(time, P1 - std, P1 + std, alpha=0.15, color=colour)

    ax.set_ylim([-0.002, 0.002])
    ax.legend(loc="upper left", fontsize=12)
    ax.set(xlabel="Time (mins)", ylabel=r"$\bar{P}_1$")
    boldTitle = util.getBoldTitle(groupTitle)
    ax.title.set_text(r"Mean $P_1$" + " with \n time " + boldTitle)
    fig.savefig(
        f"results/mean P1 {groupTitle}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# compare: Mean P2
if False:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for fileType in fileTypes:
        filenames = util.getFilesType(fileType)[0]
        dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
        p2 = np.zeros([len(filenames), T])
        for i in range(len(filenames)):
            filename = filenames[i]
            df = dfShape[dfShape["Filename"] == filename]
            for t in range(T):
                p2[i, t] = np.mean(df["Polar"][df["T"] == t])[1]

        time = 2 * np.array(range(T))

        std = np.std(p2, axis=0)
        P2 = np.mean(p2, axis=0)
        colour, mark = util.getColorLineMarker(fileType, groupTitle)
        fileTitle = util.getFileTitle(fileType)
        ax.plot(time, P2, label=fileTitle, color=colour, marker=mark, markevery=10)
        ax.fill_between(time, P2 - std, P2 + std, alpha=0.15, color=colour)

    ax.set_ylim([-0.002, 0.002])
    ax.legend(loc="upper left", fontsize=12)
    ax.set(xlabel="Time (mins)", ylabel=r"$\bar{P}_2$")
    boldTitle = util.getBoldTitle(groupTitle)
    ax.title.set_text(r"Mean $P_2$" + " with \n time " + boldTitle)
    fig.savefig(
        f"results/mean P2 {groupTitle}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# compare: Mean rho
if False:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for fileType in fileTypes:
        filenames = util.getFilesType(fileType)[0]
        dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
        rho = np.zeros([len(filenames), T])
        for i in range(len(filenames)):
            filename = filenames[i]
            df = dfShape[dfShape["Filename"] == filename]
            for t in range(T):
                rho[i, t] = 1 / np.mean(df["Area"][df["T"] == t])

        time = 2 * np.array(range(T))

        std = np.std(rho, axis=0)
        rho = np.mean(rho, axis=0)
        colour, mark = util.getColorLineMarker(fileType, groupTitle)
        fileTitle = util.getFileTitle(fileType)
        ax.plot(time, rho, label=fileTitle, color=colour, marker=mark, markevery=10)
        ax.fill_between(time, rho - std, rho + std, alpha=0.15, color=colour)

    ax.set_ylim([0.05, 0.1])
    ax.legend(loc="upper left", fontsize=12)
    ax.set(xlabel="Time (mins)", ylabel=r"$\bar{\rho}$")
    boldTitle = util.getBoldTitle(groupTitle)
    ax.title.set_text(r"Mean $\rho$" + " with \n time " + boldTitle)
    fig.savefig(
        f"results/mean rho {groupTitle}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# compare: entropy
if True:
    times = [0, 40, 80]
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for fileType in fileTypes:
        filenames = util.getFilesType(fileType)[0]
        dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
        ent = np.zeros([len(filenames), T])
        for i in range(len(filenames)):
            filename = filenames[i]
            df = dfShape[dfShape["Filename"] == filename]
            for t in range(T):
                q = np.stack(df["q"][(df["T"] == t)])
                heatmap, xedges, yedges = np.histogram2d(
                    q[:, 0, 0],
                    q[:, 1, 0],
                    range=[[-0.1, 0.11], [-0.1, 0.11]],
                    bins=(21, 21),
                )

                prob = heatmap / q.shape[0]
                p = prob[prob != 0]
                if True:
                    if i == 0:
                        if t in times:
                            x, y = np.mgrid[-0.1:0.11:0.01, -0.1:0.11:0.01]
                            fig2, ax2 = plt.subplots(1, 1, figsize=(4, 3))
                            c = ax2.pcolor(
                                x,
                                y,
                                prob,
                                vmin=0,
                                vmax=0.04,
                                cmap="Reds",
                            )
                            plt.axvline(
                                np.mean(df["q"][(df["T"] == t)], axis=0)[0, 0],
                                linewidth=1,
                            )
                            plt.axhline(
                                np.mean(df["q"][(df["T"] == t)], axis=0)[0, 1],
                                linewidth=1,
                            )
                            fig2.colorbar(c, ax=ax2)
                            ax2.set(
                                xlabel=r"$q^{(1)}$",
                                ylabel=r"$q^{(2)}$",
                            )
                            fileTitle = util.getFileTitle(fileType)
                            boldTitle = util.getBoldTitle(fileTitle)
                            ax2.title.set_text(
                                r"$\mathbf{q}$ distribution at time"
                                + f" {t*2}mins"
                                + f"\n - {boldTitle}"
                            )

                            fig2.savefig(
                                f"results/dQ heatmap time {t*2} {fileTitle}",
                                transparent=True,
                                bbox_inches="tight",
                                dpi=300,
                            )

                entropy = p * np.log(p)
                ent[i, t] = -sum(entropy)

        time = 2 * np.array(range(T))

        std = np.std(ent, axis=0)
        ent = np.mean(ent, axis=0)
        colour, mark = util.getColorLineMarker(fileType, groupTitle)
        fileTitle = util.getFileTitle(fileType)
        ax.plot(
            time[:80],
            ent[:80],
            label=fileTitle,
            color=colour,
            marker=mark,
            markevery=10,
        )
        ax.fill_between(
            time[:80],
            ent[:80] - std[:80],
            ent[:80] + std[:80],
            alpha=0.15,
            color=colour,
        )

    ax.set_ylim([4, 4.65])
    ax.legend(loc="lower left", fontsize=12)
    ax.set(xlabel="Time (mins)", ylabel="Shannon entropy")
    boldTitle = util.getBoldTitle(groupTitle)
    ax.title.set_text("Mean Shannon entropy \n with time " + boldTitle)
    fig.savefig(
        f"results/mean entropy {groupTitle}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")
