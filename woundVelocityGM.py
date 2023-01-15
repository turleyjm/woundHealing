import os
import shutil
from math import dist, floor, log10

from collections import Counter
import cv2
import matplotlib
from matplotlib import markers
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy as sp
import scipy.linalg as linalg
from scipy.stats import pearsonr
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


def weighted_avg_and_std(values, weight, axis=0):
    average = np.average(values, weights=weight, axis=axis)
    variance = np.average((values - average) ** 2, weights=weight, axis=axis)
    return average, np.sqrt(variance)


# -------------------

fileTypes, groupTitle = util.getFilesTypes()
scale = 123.26 / 512
T = 84
timeStep = 4
R = 80
rStep = 10

# -------------------

# Individual: v with distance from wound edge and time
if False:
    for fileType in fileTypes:
        filenames = util.getFilesType(fileType)[0]
        v1 = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        area = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        dfVelocity = pd.read_pickle(f"databases/dfVelocityWound{fileType}.pkl")
        for k in range(len(filenames)):
            filename = filenames[k]
            dfFile = dfVelocity[dfVelocity["Filename"] == filename]
            if "Wound" in filename:
                t0 = util.findStartTime(filename)
            else:
                t0 = 0
            t2 = int(timeStep / 2 * (int(T / timeStep) + 1) - t0 / 2)

            for r in range(v1.shape[2]):
                for t in range(v1.shape[1]):
                    df1 = dfFile[dfFile["T"] > timeStep * t]
                    df2 = df1[df1["T"] <= timeStep * (t + 1)]
                    df3 = df2[df2["R"] > rStep * r]
                    df = df3[df3["R"] <= rStep * (r + 1)]
                    if len(df) > 0:
                        v1[k, t, r] = np.mean(df["dv"], axis=0)[0]

            inPlane = 1 - (
                sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype(int)[:t2]
                / 255
            )
            dist = (
                sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)[:t2]
                * scale
            )

            for r in range(area.shape[2]):
                for t in range(area.shape[1]):
                    t1 = int(timeStep / 2 * t - t0 / 2)
                    t2 = int(timeStep / 2 * (t + 1) - t0 / 2)
                    if t1 < 0:
                        t1 = 0
                    if t2 < 0:
                        t2 = 0
                    area[k, t, r] = (
                        np.sum(
                            inPlane[t1:t2][
                                (dist[t1:t2] > rStep * r)
                                & (dist[t1:t2] <= rStep * (r + 1))
                            ]
                        )
                        * scale**2
                    )

        V1 = np.zeros([int(T / timeStep), int(R / rStep)])
        std = np.zeros([int(T / timeStep), int(R / rStep)])
        meanArea = np.zeros([int(T / timeStep), int(R / rStep)])

        for r in range(area.shape[2]):
            for t in range(area.shape[1]):
                _V1 = v1[:, t, r][v1[:, t, r] != 0]
                _area = area[:, t, r][v1[:, t, r] != 0]
                if (len(_area) > 0) & (np.sum(_area) > 0):
                    _dd, _std = weighted_avg_and_std(_V1, _area)
                    V1[t, r] = _dd
                    std[t, r] = _std
                    meanArea[t, r] = np.mean(_area)
                else:
                    V1[t, r] = np.nan
                    std[t, r] = np.nan

        V1[meanArea < 500] = np.nan

        t, r = np.mgrid[0:T:timeStep, 0:R:rStep]
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        c = ax.pcolor(
            t,
            r,
            V1,
            vmin=-0.3,
            vmax=0.3,
            cmap="RdBu_r",
        )
        fig.colorbar(c, ax=ax)
        ax.set(
            xlabel="Time after wounding (mins)",
            ylabel=r"Distance from wound edge $(\mu m)$",
        )
        fileTitle = util.getFileTitle(fileType)
        boldTitle = util.getBoldTitle(fileTitle)
        ax.title.set_text(r"$\delta v_1$ distance and" + f"\ntime {boldTitle}")

        fig.savefig(
            f"results/v heatmap {fileTitle}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")

# Compare: Rescale Speed Towards Wound
if True:
    if (
        groupTitle == "wild type"
        or groupTitle == "JNK DN"
        or groupTitle == "Ca RNAi"
        or groupTitle == "immune ablation"
    ):
        i = 0
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        for fileType in fileTypes[1:3]:
            filenames = util.getFilesType(fileType)[0]
            dfVelocity = pd.read_pickle(f"databases/dfVelocityWound{fileType}.pkl")
            dV1 = [[] for col in range(20)]
            dV1_nor = [[] for col in range(20)]
            for k in range(len(filenames)):
                filename = filenames[k]
                dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
                dfVel = dfVelocity[dfVelocity["Filename"] == filename]
                rad = (dfWound["Area"].iloc[0] / np.pi) ** 0.5

                time = []
                dv1 = []
                dv1_std = []
                for t in range(20):
                    dft = dfVel[
                        (dfVel["T"] >= timeStep * t) & (dfVel["T"] < timeStep * (t + 1))
                    ]
                    if len(dft[dft["R"] < 40]) > 0:
                        dv = np.mean(dft["dv"][dft["R"] < 40], axis=0)
                        dv1.append(dv[0])
                        dV1[t].append(dv[0])
                        dV1_nor[t].append(dv[0] / rad)
                        dv_std = np.std(np.array(dft["dv"][dft["R"] < 40]), axis=0)
                        dv1_std.append(dv_std[0] / (len(dft)) ** 0.5)
                        time.append(timeStep * t + timeStep / 2)

                if i == 0:
                    ax[0, 0].plot(time, dv1)
                    ax[1, 0].plot(time, np.array(dv1) / rad)
                else:
                    ax[0, 0].plot(time, dv1, marker="o")
                    ax[1, 0].plot(time, np.array(dv1) / rad, marker="o")

            time = []
            dV1_mu = []
            dV1_nor_mu = []
            dV1_std = []
            dV1_nor_std = []
            for t in range(20):
                if len(dV1[t]) > 0:
                    dV1_mu.append(np.mean(dV1[t]))
                    dV1_nor_mu.append(np.mean(dV1_nor[t]))
                    dV1_std.append(np.std(dV1[t]))
                    dV1_nor_std.append(np.std(dV1_nor[t]))
                    time.append(timeStep * t)
            time = np.array(time)
            dV1_mu = np.array(dV1_mu)
            dV1_nor_mu = np.array(dV1_nor_mu)
            dV1_std = np.array(dV1_std)
            dV1_nor_std = np.array(dV1_nor_std)

            color = util.getColor(fileType)
            fileTitle = util.getFileTitle(fileType)
            if i == 0:
                ax[0, 1].plot(time, dV1_mu, label=fileTitle, color=color)
                ax[1, 1].plot(time, dV1_nor_mu, label=fileTitle, color=color)
            else:
                ax[0, 1].plot(time, dV1_mu, label=fileTitle, color=color, marker="o")
                ax[1, 1].plot(
                    time, dV1_nor_mu, label=fileTitle, color=color, marker="o"
                )

            ax[0, 1].fill_between(
                time,
                dV1_mu - dV1_std,
                dV1_mu + dV1_std,
                alpha=0.15,
                color=color,
            )
            ax[1, 1].fill_between(
                time,
                dV1_nor_mu - dV1_nor_std,
                dV1_nor_mu + dV1_nor_std,
                alpha=0.15,
                color=color,
            )
            i += 1

        ax[0, 0].set(xlabel=f"Time (mins)", ylabel=r"$V_1$ rel. to wound ($\mu/min$)")
        ax[0, 0].set_ylim([-0.3, 0.3])
        ax[1, 0].set(xlabel=f"Time (mins)", ylabel=r"Norm. $V_1$ rel. to wound")
        ax[1, 0].set_ylim([-0.005, 0.005])
        ax[0, 1].set(
            xlabel=f"Time (mins)", ylabel=r"Mean $V_1$ rel. to wound ($\mu/min$)"
        )
        ax[0, 1].set_ylim([-0.3, 0.3])
        ax[0, 1].legend(loc="lower right", fontsize=12)
        ax[1, 1].set(xlabel=f"Time (mins)", ylabel=r"Mean norm. $V_1$ rel. to wound")
        ax[1, 1].set_ylim([-0.005, 0.005])
        ax[1, 1].legend(loc="lower right", fontsize=12)

        plt.subplot_tool()
        plt.subplots_adjust(
            left=0.075, bottom=0.1, right=0.95, top=0.89, wspace=0.4, hspace=0.3
        )
        boldTitle = util.getBoldTitle(groupTitle)
        st = fig.suptitle(
            "Rescale speed towards wound with time " + boldTitle,
            fontsize=24,
        )
        st.set_y(0.97)

        fig.savefig(
            f"results/Compare rescale speed towards wound {groupTitle}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")
