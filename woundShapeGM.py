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
R = 50
rStep = 10

# -------------------


# Individual: Density with distance from wound edge and time
if False:
    for fileType in fileTypes:
        filenames = util.getFilesType(fileType)[0]
        area = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        areaCell = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        dfShape = pd.read_pickle(f"databases/dfShapeWound{fileType}.pkl")
        for k in range(len(filenames)):
            filename = filenames[k]
            dfFile = dfShape[dfShape["Filename"] == filename]
            if "Wound" in filename:
                t0 = util.findStartTime(filename)
            else:
                t0 = 0
            t2 = int(timeStep / 2 * (int(T / timeStep) + 1) - t0 / 2)

            for r in range(areaCell.shape[2]):
                for t in range(areaCell.shape[1]):
                    df1 = dfFile[dfFile["T"] > timeStep * t]
                    df2 = df1[df1["T"] <= timeStep * (t + 1)]
                    df3 = df2[df2["R"] > rStep * r]
                    df = df3[df3["R"] <= rStep * (r + 1)]
                    if len(df) > 0:
                        areaCell[k, t, r] = np.mean(df["Area"])

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

        AreaCell = np.zeros([int(T / timeStep), int(R / rStep)])
        std = np.zeros([int(T / timeStep), int(R / rStep)])
        meanArea = np.zeros([int(T / timeStep), int(R / rStep)])

        for r in range(area.shape[2]):
            for t in range(area.shape[1]):
                _areaCell = areaCell[:, t, r][areaCell[:, t, r] != 0]
                _area = area[:, t, r][areaCell[:, t, r] != 0]
                if len(_area) > 0:
                    _dd, _std = weighted_avg_and_std(_areaCell, _area)
                    AreaCell[t, r] = _dd
                    std[t, r] = _std
                    meanArea[t, r] = np.mean(_area)
                else:
                    AreaCell[t, r] = np.nan
                    std[t, r] = np.nan

        AreaCell[meanArea < 500] = np.nan

        t, r = np.mgrid[
            timeStep / 2 : T + timeStep / 2 : timeStep,
            rStep / 2 : R + rStep / 2 : rStep,
        ]
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        c = ax.pcolor(
            t,
            r,
            AreaCell,
            vmin=10,
            vmax=20,
            cmap="Reds",
        )
        fig.colorbar(c, ax=ax)
        ax.set(
            xlabel="Time after wounding (mins)",
            ylabel=r"Distance from wound edge $(\mu m)$",
        )
        fileTitle = util.getFileTitle(fileType)
        boldTitle = util.getBoldTitle(fileTitle)
        ax.title.set_text(f"Density distance and" + f"\n time {boldTitle}")

        fig.savefig(
            f"results/Area heatmap {fileTitle}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")

# Individual: Q1 with distance from wound edge and time
if True:
    for fileType in fileTypes:
        filenames = util.getFilesType(fileType)[0]
        q1 = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        q1Cont = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        area = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        dfShape = pd.read_pickle(f"databases/dfShapeWound{fileType}.pkl")
        for k in range(len(filenames)):
            filename = filenames[k]
            dfFile = dfShape[dfShape["Filename"] == filename]
            dQ1Cont = np.mean(dfFile["dq"] ** 2, axis=0)[0, 0] ** 0.5

            if "Wound" in filename:
                t0 = util.findStartTime(filename)
            else:
                t0 = 0
            t2 = int(timeStep / 2 * (int(T / timeStep) + 1) - t0 / 2)

            for r in range(q1.shape[2]):
                for t in range(q1.shape[1]):
                    df1 = dfFile[dfFile["T"] > timeStep * t]
                    df2 = df1[df1["T"] <= timeStep * (t + 1)]
                    df3 = df2[df2["R"] > rStep * r]
                    df = df3[df3["R"] <= rStep * (r + 1)]
                    if len(df) > 0:
                        q1[k, t, r] = np.mean(df["dq"], axis=0)[0, 0]
                        q1Cont[k, t, r] = np.mean(df["dq"], axis=0)[0, 0] / dQ1Cont

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

        Q1 = np.zeros([int(T / timeStep), int(R / rStep)])
        Q1Cont = np.zeros([int(T / timeStep), int(R / rStep)])
        std = np.zeros([int(T / timeStep), int(R / rStep)])
        meanArea = np.zeros([int(T / timeStep), int(R / rStep)])

        for r in range(area.shape[2]):
            for t in range(area.shape[1]):
                _Q1 = q1[:, t, r][q1[:, t, r] != 0]
                _area = area[:, t, r][q1[:, t, r] != 0]
                _Q1Cont = q1Cont[:, t, r][q1Cont[:, t, r] != 0]
                if (len(_area) > 0) & (np.sum(_area) > 0):
                    _dd, _std = weighted_avg_and_std(_Q1, _area)
                    Q1[t, r] = _dd
                    std[t, r] = _std
                    meanArea[t, r] = np.mean(_area)
                    _dd, _std = weighted_avg_and_std(_Q1Cont, _area)
                    Q1Cont[t, r] = _dd
                else:
                    Q1[t, r] = np.nan
                    std[t, r] = np.nan
                    Q1Cont[t, r] = np.nan

        Q1[meanArea < 500] = np.nan
        Q1Cont[meanArea < 500] = np.nan

        t, r = np.mgrid[
            timeStep / 2 : T + timeStep / 2 : timeStep,
            rStep / 2 : R + rStep / 2 : rStep,
        ]
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        c = ax.pcolor(
            t,
            r,
            Q1,
            vmin=-0.03,
            vmax=0.03,
            cmap="RdBu_r",
        )
        fig.colorbar(c, ax=ax)
        ax.set(
            xlabel="Time after wounding (mins)",
            ylabel=f"Distance from \n wound edge" + r"$(\mu m)$",
        )
        fileTitle = util.getFileTitle(fileType)
        boldTitle = util.getBoldTitle(fileTitle)
        if "wt" in boldTitle:
            ax.title.set_text(r"$\delta Q^{(1)}$ distance and time" + f"\n{boldTitle}")
        else:
            colour, mark = util.getColorLineMarker(fileType, groupTitle)
            plt.title(
                r"$\delta Q^{(1)}$ distance and time" + f"\n{boldTitle}", color=colour
            )

        fig.savefig(
            f"results/Q1 heatmap {fileTitle}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")

        if False:
            controlType = util.controlType(fileType)
            dfCont = pd.read_pickle(f"databases/dfShapeWound{controlType}.pkl")
            dQ1Cont = np.mean(dfCont["dq"] ** 2, axis=0)[0, 0] ** 0.5

            fig, ax = plt.subplots(2, 1, figsize=(6, 6))
            c = ax[0].pcolor(
                t,
                r,
                Q1 / dQ1Cont,
                vmin=-1.2,
                vmax=1.2,
                cmap="RdBu_r",
            )
            fig.colorbar(c, ax=ax[0])
            ax[0].set(
                xlabel="Time after wounding (mins)",
                ylabel=f"Distance from \n wound edge" + r"$(\mu m)$",
            )
            fileTitle = util.getFileTitle(fileType)
            boldTitle = util.getBoldTitle(fileTitle)
            controlTitle = util.getFileTitle(controlType)
            controlTitle = util.getBoldTitle(controlTitle)
            ax[0].title.set_text(
                r"$\delta Q^{(1)}$"
                + f"{boldTitle} heterogeneity \n compared to {controlTitle}"
            )

            c = ax[1].pcolor(
                t,
                r,
                Q1Cont,
                vmin=-1.2,
                vmax=1.2,
                cmap="RdBu_r",
            )
            fig.colorbar(c, ax=ax[1])
            ax[1].set(
                xlabel="Time after wounding (mins)",
                ylabel=f"Distance from \n wound edge" + r"$(\mu m)$",
            )
            fileTitle = util.getFileTitle(fileType)
            boldTitle = util.getBoldTitle(fileTitle)
            ax[1].title.set_text(r"$\delta Q^{(1)}$" + f"{boldTitle} heterogeneity")

            plt.subplots_adjust(
                left=0.12, bottom=0.11, right=0.9, top=0.88, wspace=0.2, hspace=0.55
            )

            fig.savefig(
                f"results/Q1 heatmap heterogeneity {fileTitle}",
                transparent=True,
                bbox_inches="tight",
                dpi=300,
            )
            plt.close("all")

# Individual: P1 with distance from wound edge and time
if False:
    for fileType in fileTypes:
        filenames = util.getFilesType(fileType)[0]
        p1 = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        p1Cont = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        area = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        dfShape = pd.read_pickle(f"databases/dfShapeWound{fileType}.pkl")
        for k in range(len(filenames)):
            filename = filenames[k]
            dfFile = dfShape[dfShape["Filename"] == filename]
            dP1Cont = np.mean(dfFile["dp"] ** 2, axis=0)[0] ** 0.5

            if "Wound" in filename:
                t0 = util.findStartTime(filename)
            else:
                t0 = 0
            t2 = int(timeStep / 2 * (int(T / timeStep) + 1) - t0 / 2)

            for r in range(p1.shape[2]):
                for t in range(p1.shape[1]):
                    df1 = dfFile[dfFile["T"] > timeStep * t]
                    df2 = df1[df1["T"] <= timeStep * (t + 1)]
                    df3 = df2[df2["R"] > rStep * r]
                    df = df3[df3["R"] <= rStep * (r + 1)]
                    if len(df) > 0:
                        p1[k, t, r] = np.mean(df["dp"], axis=0)[0]
                        p1Cont[k, t, r] = np.mean(df["dp"], axis=0)[0] / dP1Cont

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

        P1 = np.zeros([int(T / timeStep), int(R / rStep)])
        P1Cont = np.zeros([int(T / timeStep), int(R / rStep)])
        std = np.zeros([int(T / timeStep), int(R / rStep)])
        meanArea = np.zeros([int(T / timeStep), int(R / rStep)])

        for r in range(area.shape[2]):
            for t in range(area.shape[1]):
                _P1 = p1[:, t, r][p1[:, t, r] != 0]
                _area = area[:, t, r][p1[:, t, r] != 0]
                _P1Cont = p1Cont[:, t, r][p1Cont[:, t, r] != 0]
                if (len(_area) > 0) & (np.sum(_area) > 0):
                    _dd, _std = weighted_avg_and_std(_P1, _area)
                    P1[t, r] = _dd
                    std[t, r] = _std
                    meanArea[t, r] = np.mean(_area)
                    _dd, _std = weighted_avg_and_std(_P1Cont, _area)
                    P1Cont[t, r] = _dd
                else:
                    P1[t, r] = np.nan
                    std[t, r] = np.nan
                    P1Cont[t, r] = np.nan

        P1[meanArea < 500] = np.nan
        P1Cont[meanArea < 500] = np.nan

        t, r = np.mgrid[
            timeStep / 2 : T + timeStep / 2 : timeStep,
            rStep / 2 : R + rStep / 2 : rStep,
        ]
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        c = ax.pcolor(
            t,
            r,
            P1,
            vmin=-0.001,
            vmax=0.001,
            cmap="RdBu_r",
        )
        fig.colorbar(c, ax=ax)
        ax.set(
            xlabel="Time after wounding (mins)",
            ylabel=f"Distance from \n wound edge" + r"$(\mu m)$",
        )
        fileTitle = util.getFileTitle(fileType)
        boldTitle = util.getBoldTitle(fileTitle)
        ax.title.set_text(r"$\delta P_1$ distance and" + f"\n time {boldTitle}")

        fig.savefig(
            f"results/P1 heatmap {fileTitle}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")

        if False:
            controlType = util.controlType(fileType)
            dfCont = pd.read_pickle(f"databases/dfShapeWound{controlType}.pkl")
            dP1Cont = np.mean(dfCont["dp"] ** 2, axis=0)[0, 0] ** 0.5

            fig, ax = plt.subplots(2, 1, figsize=(6, 6))
            c = ax[0].pcolor(
                t,
                r,
                P1 / dP1Cont,
                vmin=-1.2,
                vmax=1.2,
                cmap="RdBu_r",
            )
            fig.colorbar(c, ax=ax[0])
            ax[0].set(
                xlabel="Time after wounding (mins)",
                ylabel=f"Distance from \n wound edge" + r"$(\mu m)$",
            )
            fileTitle = util.getFileTitle(fileType)
            boldTitle = util.getBoldTitle(fileTitle)
            controlTitle = util.getFileTitle(controlType)
            controlTitle = util.getBoldTitle(controlTitle)
            ax[0].title.set_text(
                r"$\delta P_1$"
                + f"{boldTitle} heterogeneity \n compared to {controlTitle}"
            )

            c = ax[1].pcolor(
                t,
                r,
                P1Cont,
                vmin=-1.2,
                vmax=1.2,
                cmap="RdBu_r",
            )
            fig.colorbar(c, ax=ax[1])
            ax[1].set(
                xlabel="Time after wounding (mins)",
                ylabel=f"Distance from \n wound edge" + r"$(\mu m)$",
            )
            fileTitle = util.getFileTitle(fileType)
            boldTitle = util.getBoldTitle(fileTitle)
            ax[1].title.set_text(r"$\delta P_1$" + f"{boldTitle} heterogeneity")

            plt.subplots_adjust(
                left=0.12, bottom=0.11, right=0.9, top=0.88, wspace=0.2, hspace=0.55
            )

            fig.savefig(
                f"results/P1 heatmap heterogeneity {fileTitle}",
                transparent=True,
                bbox_inches="tight",
                dpi=300,
            )
            plt.close("all")

# Compare: Rescale Q1 relative to Wound
if False:
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
            dfShape = pd.read_pickle(f"databases/dfShapeWound{fileType}.pkl")
            dQ1 = [[] for col in range(10)]
            dQ1_nor = [[] for col in range(10)]
            for k in range(len(filenames)):
                filename = filenames[k]
                dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
                dfSh = dfShape[dfShape["Filename"] == filename]
                rad = (dfWound["Area"].iloc[0] / np.pi) ** 0.5

                time = []
                dq1 = []
                dq1_std = []
                for t in range(10):
                    dft = dfSh[
                        (dfSh["T"] >= timeStep * t) & (dfSh["T"] < timeStep * (t + 1))
                    ]
                    if len(dft[dft["R"] < 40]) > 0:
                        dq = np.mean(dft["dq"][dft["R"] < 40], axis=0)
                        dq1.append(dq[0, 0])
                        dQ1[t].append(dq[0, 0])
                        dQ1_nor[t].append(dq[0, 0] / rad)
                        dq_std = np.std(np.array(dft["dq"][dft["R"] < 40]), axis=0)
                        dq1_std.append(dq_std[0, 0] / (len(dft)) ** 0.5)
                        time.append(timeStep * t + timeStep / 2)

                if i == 0:
                    ax[0, 0].plot(time, dq1)
                    ax[1, 0].plot(time, np.array(dq1) / rad)
                else:
                    ax[0, 0].plot(time, dq1, marker="o")
                    ax[1, 0].plot(time, np.array(dq1) / rad, marker="o")

            time = []
            dQ1_mu = []
            dQ1_nor_mu = []
            dQ1_std = []
            dQ1_nor_std = []
            for t in range(10):
                if len(dQ1[t]) > 0:
                    dQ1_mu.append(np.mean(dQ1[t]))
                    dQ1_nor_mu.append(np.mean(dQ1_nor[t]))
                    dQ1_std.append(np.std(dQ1[t]))
                    dQ1_nor_std.append(np.std(dQ1_nor[t]))
                    time.append(timeStep * t)
            time = np.array(time)
            dQ1_mu = np.array(dQ1_mu)
            dQ1_nor_mu = np.array(dQ1_nor_mu)
            dQ1_std = np.array(dQ1_std)
            dQ1_nor_std = np.array(dQ1_nor_std)

            colour, mark = util.getColorLineMarker(fileType, groupTitle)
            fileTitle = util.getFileTitle(fileType)
            if i == 0:
                ax[0, 1].plot(time, dQ1_mu, label=fileTitle, color=colour)
                ax[1, 1].plot(time, dQ1_nor_mu, label=fileTitle, color=colour)
            else:
                ax[0, 1].plot(
                    time,
                    dQ1_mu,
                    label=fileTitle,
                    color=colour,
                    marker=mark,
                )
                ax[1, 1].plot(
                    time,
                    dQ1_nor_mu,
                    label=fileTitle,
                    color=colour,
                    marker=mark,
                )

            ax[0, 1].fill_between(
                time,
                dQ1_mu - dQ1_std,
                dQ1_mu + dQ1_std,
                alpha=0.15,
                color=colour,
            )
            ax[1, 1].fill_between(
                time,
                dQ1_nor_mu - dQ1_nor_std,
                dQ1_nor_mu + dQ1_nor_std,
                alpha=0.15,
                color=colour,
            )
            i += 1

        ax[0, 0].set(xlabel=f"Time (mins)", ylabel=r"$Q^{(1)}$ rel. to wound")
        ax[0, 0].set_ylim([-0.02, 0.0075])
        ax[1, 0].set(xlabel=f"Time (mins)", ylabel=r"Norm. $Q^{(1)}$ rel. to wound")
        ax[1, 0].set_ylim([-0.0003, 0.00015])
        ax[0, 1].set(xlabel=f"Time (mins)", ylabel=r"Mean $Q^{(1)}$ rel. to wound")
        ax[0, 1].set_ylim([-0.02, 0.0075])
        ax[0, 1].legend(loc="lower right", fontsize=12)
        ax[1, 1].set(
            xlabel=f"Time (mins)", ylabel=r"Mean norm. $Q^{(1)}$ rel. to wound"
        )
        ax[1, 1].set_ylim([-0.0003, 0.00015])
        ax[1, 1].legend(loc="lower right", fontsize=12)

        # plt.subplot_tool()
        plt.subplots_adjust(
            left=0.075, bottom=0.1, right=0.95, top=0.89, wspace=0.4, hspace=0.3
        )
        boldTitle = util.getBoldTitle(groupTitle)
        st = fig.suptitle(
            r"Rescale $Q^{(1)}$ relative to wound" + " with time " + boldTitle,
            fontsize=24,
        )
        st.set_y(0.97)

        fig.savefig(
            f"results/Compare rescale Q1 relative to wound {groupTitle}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")

# Compare with wt Large wound: Q1 with distance from wound edge and time
if True:

    fileType = "WoundL18h"
    filenames = util.getFilesType(fileType)[0]
    q1 = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
    area = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
    dfShape = pd.read_pickle(f"databases/dfShapeWound{fileType}.pkl")
    for k in range(len(filenames)):
        filename = filenames[k]
        dfFile = dfShape[dfShape["Filename"] == filename]
        dQ1Cont = np.mean(dfFile["dq"] ** 2, axis=0)[0, 0] ** 0.5

        if "Wound" in filename:
            t0 = util.findStartTime(filename)
        else:
            t0 = 0
        t2 = int(timeStep / 2 * (int(T / timeStep) + 1) - t0 / 2)

        for r in range(q1.shape[2]):
            for t in range(q1.shape[1]):
                df1 = dfFile[dfFile["T"] > timeStep * t]
                df2 = df1[df1["T"] <= timeStep * (t + 1)]
                df3 = df2[df2["R"] > rStep * r]
                df = df3[df3["R"] <= rStep * (r + 1)]
                if len(df) > 0:
                    q1[k, t, r] = np.mean(df["dq"], axis=0)[0, 0]

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
                            (dist[t1:t2] > rStep * r) & (dist[t1:t2] <= rStep * (r + 1))
                        ]
                    )
                    * scale**2
                )

    Q1Large = np.zeros([int(T / timeStep), int(R / rStep)])
    std = np.zeros([int(T / timeStep), int(R / rStep)])
    meanArea = np.zeros([int(T / timeStep), int(R / rStep)])

    for r in range(area.shape[2]):
        for t in range(area.shape[1]):
            _Q1 = q1[:, t, r][q1[:, t, r] != 0]
            _area = area[:, t, r][q1[:, t, r] != 0]
            if (len(_area) > 0) & (np.sum(_area) > 0):
                _dd, _std = weighted_avg_and_std(_Q1, _area)
                Q1Large[t, r] = _dd
                std[t, r] = _std
                meanArea[t, r] = np.mean(_area)
            else:
                Q1Large[t, r] = np.nan
                std[t, r] = np.nan

    Q1Large[meanArea < 500] = np.nan

    for fileType in fileTypes[1:]:
        filenames = util.getFilesType(fileType)[0]
        q1 = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        q1Cont = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        area = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        dfShape = pd.read_pickle(f"databases/dfShapeWound{fileType}.pkl")
        for k in range(len(filenames)):
            filename = filenames[k]
            dfFile = dfShape[dfShape["Filename"] == filename]
            dQ1Cont = np.mean(dfFile["dq"] ** 2, axis=0)[0, 0] ** 0.5

            if "Wound" in filename:
                t0 = util.findStartTime(filename)
            else:
                t0 = 0
            t2 = int(timeStep / 2 * (int(T / timeStep) + 1) - t0 / 2)

            for r in range(q1.shape[2]):
                for t in range(q1.shape[1]):
                    df1 = dfFile[dfFile["T"] > timeStep * t]
                    df2 = df1[df1["T"] <= timeStep * (t + 1)]
                    df3 = df2[df2["R"] > rStep * r]
                    df = df3[df3["R"] <= rStep * (r + 1)]
                    if len(df) > 0:
                        q1[k, t, r] = np.mean(df["dq"], axis=0)[0, 0]
                        q1Cont[k, t, r] = np.mean(df["dq"], axis=0)[0, 0] / dQ1Cont

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

        Q1 = np.zeros([int(T / timeStep), int(R / rStep)])
        Q1Cont = np.zeros([int(T / timeStep), int(R / rStep)])
        std = np.zeros([int(T / timeStep), int(R / rStep)])
        meanArea = np.zeros([int(T / timeStep), int(R / rStep)])

        for r in range(area.shape[2]):
            for t in range(area.shape[1]):
                _Q1 = q1[:, t, r][q1[:, t, r] != 0]
                _area = area[:, t, r][q1[:, t, r] != 0]
                _Q1Cont = q1Cont[:, t, r][q1Cont[:, t, r] != 0]
                if (len(_area) > 0) & (np.sum(_area) > 0):
                    _dd, _std = weighted_avg_and_std(_Q1, _area)
                    Q1[t, r] = _dd
                    std[t, r] = _std
                    meanArea[t, r] = np.mean(_area)
                    _dd, _std = weighted_avg_and_std(_Q1Cont, _area)
                    Q1Cont[t, r] = _dd
                else:
                    Q1[t, r] = np.nan
                    std[t, r] = np.nan
                    Q1Cont[t, r] = np.nan

        Q1[meanArea < 500] = np.nan
        Q1Cont[meanArea < 500] = np.nan

        t, r = np.mgrid[
            timeStep / 2 : T + timeStep / 2 : timeStep,
            rStep / 2 : R + rStep / 2 : rStep,
        ]
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        c = ax.pcolor(
            t,
            r,
            Q1 - Q1Large,
            vmin=-0.015,
            vmax=0.015,
            cmap="RdBu_r",
        )
        fig.colorbar(c, ax=ax)
        ax.set(
            xlabel="Time after wounding (mins)",
            ylabel=f"Distance from \n wound edge" + r"$(\mu m)$",
        )
        fileTitle = util.getFileTitle(fileType)
        boldTitle = util.getBoldTitle(fileTitle)
        if "large" in boldTitle:
            colour, mark = util.getColorLineMarker(fileType, groupTitle)
            plt.title(
                r"Wild type difference in $\delta Q^{(1)}$" + f"\nfrom {boldTitle}",
                color=colour,
            )
        else:
            ax.title.set_text(
                r"Wild type difference in $\delta Q^{(1)}$" + f"\nfrom {boldTitle}"
            )

        fig.savefig(
            f"results/Q1 heatmap change large wound {fileTitle}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")
