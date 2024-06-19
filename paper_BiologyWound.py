from locale import normalize
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
from sympy import true
import tifffile
from skimage.draw import circle_perimeter
from scipy import optimize
import xml.etree.ElementTree as et
import matplotlib.colors as colors
import seaborn as sns

import cellProperties as cell
import utils as util

plt.rcParams.update({"font.size": 16})

# -------------------


def weighted_avg_and_std(values, weight, axis=0):
    average = np.average(values, weights=weight, axis=axis)
    variance = np.average((values - average) ** 2, weights=weight, axis=axis)
    return average, np.sqrt(variance)


def ControlFor(fileType):

    if fileType == "WoundLCa":
        fileTypeControl = "WoundL15h"
    elif fileType == "WoundLCa_new":
        fileTypeControl = "WoundL15h"
    elif fileType == "WoundLCa_old":
        fileTypeControl = "WoundL15h"
    elif fileType == "WoundLJNK":
        fileTypeControl = "WoundL15h"
    elif fileType == "WoundLrpr":
        fileTypeControl = "WoundL26h"

    return fileTypeControl


# ------------------- Figure 2 and Fig. Sub. 2

fileTypes, groupTitle = util.getFilesTypes(fileType="18h")
scale = 123.26 / 512
T = 93
Q1Norm = 0.01217946447300043

# Compare wounds: Large and small
if False:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.vlines(10, -100, 1500, colors="r", linestyles="dashed")
    ax.set_ylim([0, 1250])

    i = 0
    for fileType in fileTypes:
        if "Unw" in fileType:
            continue
        filenames, fileType = util.getFilesType(fileType)
        _df = []
        Area0 = []

        for filename in filenames:
            t0 = util.findStartTime(filename)
            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
            T = len(dfWound)
            area = np.array(dfWound["Area"]) * (scale) ** 2

            for t in range(T):
                if area[t] > 10:
                    _df.append({"Area": area[t], "Time": int(t0 / 2) * 2 + 2 * t})
                else:
                    _df.append({"Area": 0, "Time": int(t0 / 2) * 2 + 2 * t})

        df = pd.DataFrame(_df)
        A = []
        time = []
        std = []
        T = set(df["Time"])
        N = len(filenames)
        Area0 = np.mean(df["Area"][df["Time"] == 10])
        for t in T:
            if len(df[df["Time"] == t]) > N / 3:
                if np.mean(df["Area"][df["Time"] == t]) > 0.2 * Area0:
                    time.append(t)
                    A.append(np.mean(df["Area"][df["Time"] == t]))
                    std.append(np.std(df["Area"][df["Time"] == t]))

        A = np.array(A)
        std = np.array(std)
        colour, mark = util.getColorLineMarker(fileType, "WoundL")
        fileTitle = util.getFileTitle(fileType)
        ax.plot(time, A, label=fileTitle, marker=mark, color=colour)
        ax.fill_between(time, A - std, A + std, alpha=0.15, color=colour)

    plt.xlabel("Time after wounding (mins)")
    plt.ylabel(r"Area ($\mu m ^2$)")
    boldTitle = util.getBoldTitle(groupTitle)
    plt.title(f"Mean area of\n{boldTitle} wounds")
    plt.legend(loc="upper right", fontsize=9)
    fig.savefig(
        f"results/biologyWoundPaper/Mean area of {groupTitle} wounds",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

fileTypes, groupTitle = util.getFilesTypes(fileType="control")

# Compare wounds: 15h, 18h and 26h
if False:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.vlines(10, -100, 1500, colors="r", linestyles="dashed")
    ax.set_ylim([0, 1250])

    i = 0
    for fileType in fileTypes:
        if "WoundS" in fileType:
            continue
        filenames, fileType = util.getFilesType(fileType)
        _df = []
        Area0 = []

        for filename in filenames:
            t0 = util.findStartTime(filename)
            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
            T = len(dfWound)
            area = np.array(dfWound["Area"]) * (scale) ** 2

            for t in range(T):
                if area[t] > area[0] * 0.2:
                    _df.append({"Area": area[t], "Time": int(t0 / 2) * 2 + 2 * t})
                else:
                    _df.append({"Area": 0, "Time": int(t0 / 2) * 2 + 2 * t})

        df = pd.DataFrame(_df)
        A = []
        time = []
        std = []
        T = set(df["Time"])
        N = len(filenames)
        Area0 = np.mean(df["Area"][df["Time"] == 10])
        for t in T:
            if len(df[df["Time"] == t]) > N / 3:
                if np.mean(df["Area"][df["Time"] == t]) > 0.2 * Area0:
                    time.append(t)
                    A.append(np.mean(df["Area"][df["Time"] == t]))
                    std.append(np.std(df["Area"][df["Time"] == t]))

        A = np.array(A)
        std = np.array(std)
        colour, mark = util.getColorLineMarker(fileType, "WoundL")
        fileTitle = util.getFileTitle(fileType)
        ax.plot(time, A, label=fileTitle, marker=mark, color=colour)
        ax.fill_between(time, A - std, A + std, alpha=0.15, color=colour)

    plt.xlabel("Time after wounding (mins)")
    plt.ylabel(r"Area ($\mu m ^2$)")
    boldTitle = util.getBoldTitle(groupTitle + "s")
    plt.title(f"Mean area of\n{boldTitle} wounds")
    plt.legend(loc="upper right", fontsize=9)
    fig.savefig(
        f"results/biologyWoundPaper/Mean area of {fileType} wounds",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

T = 84
timeStep = 4
R = 50
rStep = 10

# Individual: v with distance from wound edge and time
if False:
    for fileType in fileTypes:
        filenames = util.getFilesType(fileType)[0]
        v1 = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        v1Cont = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        area = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        dfVelocity = pd.read_pickle(f"databases/dfVelocityWound{fileType}.pkl")
        for k in range(len(filenames)):
            filename = filenames[k]
            dfFile = dfVelocity[dfVelocity["Filename"] == filename]
            dv1Cont = np.mean(dfFile["dv"] ** 2, axis=0)[0] ** 0.5
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
                        v1Cont[k, t, r] = np.mean(df["dv"], axis=0)[0] / dv1Cont

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
        V1Cont = np.zeros([int(T / timeStep), int(R / rStep)])
        std = np.zeros([int(T / timeStep), int(R / rStep)])
        meanArea = np.zeros([int(T / timeStep), int(R / rStep)])

        for r in range(area.shape[2]):
            for t in range(area.shape[1]):
                _V1 = v1[:, t, r][v1[:, t, r] != 0]
                _area = area[:, t, r][v1[:, t, r] != 0]
                _V1Cont = v1Cont[:, t, r][v1Cont[:, t, r] != 0]
                if (len(_area) > 0) & (np.sum(_area) > 0):
                    _dd, _std = weighted_avg_and_std(_V1, _area)
                    V1[t, r] = _dd
                    std[t, r] = _std
                    meanArea[t, r] = np.mean(_area)
                    _dd, _std = weighted_avg_and_std(_V1Cont, _area)
                    V1Cont[t, r] = _dd
                else:
                    V1[t, r] = np.nan
                    std[t, r] = np.nan
                    V1Cont[t, r] = np.nan

        V1[meanArea < 500] = np.nan
        V1Cont[meanArea < 500] = np.nan

        t, r = np.mgrid[
            timeStep / 2 : T + timeStep / 2 : timeStep,
            rStep / 2 : R + rStep / 2 : rStep,
        ]
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
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
            ylabel=f"Distance from\nwound edge " + r"$(\mu m)$",
        )
        fileTitle = util.getFileTitle(fileType)
        boldTitle = util.getBoldTitle(fileTitle)
        ax.title.set_text(r"Mean cell velocity" + f"\n{boldTitle}")
        fig.savefig(
            f"results/biologyWoundPaper/v heatmap {fileType}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")

# Individual: Q1 with distance from wound edge and time
if False:
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
            Q1 / Q1Norm,
            vmin=-3.1,
            vmax=3.1,
            cmap="RdBu_r",
        )
        fig.colorbar(c, ax=ax)
        ax.set(
            xlabel="Time after wounding (mins)",
            ylabel=f"Distance from\nwound edge " + r"$(\mu m)$",
        )
        fileTitle = util.getFileTitle(fileType)
        boldTitle = util.getBoldTitle(fileTitle)
        ax.title.set_text(r"Normalised cell shape relative to" + f"\n{boldTitle}")

        fig.savefig(
            f"results/biologyWoundPaper/Q1 heatmap {fileType}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")

T = 180
timeStep = 10
R = 110
rStep = 10

# Individual: Divison density with distance from wound edge and time
if False:
    for fileType in fileTypes:
        filenames = util.getFilesType(fileType)[0]
        count = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        area = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
        for k in range(len(filenames)):
            filename = filenames[k]
            dfFile = dfDivisions[dfDivisions["Filename"] == filename]
            if "Wound" in filename:
                t0 = util.findStartTime(filename)
            else:
                t0 = 0
            t2 = int(timeStep / 2 * (int(T / timeStep) + 1) - t0 / 2)

            for r in range(count.shape[2]):
                for t in range(count.shape[1]):
                    df1 = dfFile[dfFile["T"] > timeStep * t]
                    df2 = df1[df1["T"] <= timeStep * (t + 1)]
                    df3 = df2[df2["R"] > rStep * r]
                    df = df3[df3["R"] <= rStep * (r + 1)]
                    count[k, t, r] = len(df)

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

        dd = np.zeros([int(T / timeStep), int(R / rStep)])
        std = np.zeros([int(T / timeStep), int(R / rStep)])
        sumArea = np.zeros([int(T / timeStep), int(R / rStep)])

        for r in range(area.shape[2]):
            for t in range(area.shape[1]):
                _area = area[:, t, r][area[:, t, r] > 800]
                _count = count[:, t, r][area[:, t, r] > 800]
                if len(_area) > 0:
                    _dd, _std = weighted_avg_and_std(_count / _area, _area)
                    dd[t, r] = _dd
                    std[t, r] = _std
                    sumArea[t, r] = np.sum(_area)
                else:
                    dd[t, r] = np.nan
                    std[t, r] = np.nan

        dd[sumArea < 600 * len(filenames)] = np.nan
        dd = dd * 10000

        t, r = np.mgrid[
            timeStep / 2 : T + timeStep / 2 : timeStep,
            rStep / 2 : R + rStep / 2 : rStep,
        ]
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        c = ax.pcolor(
            t,
            r,
            dd,
            vmin=0,
            vmax=9,
            cmap="plasma",
        )
        fig.colorbar(c, ax=ax)
        if "Wound" in fileType:
            ax.set(
                xlabel="Time after wounding (mins)",
                ylabel=r"Distance from wound $(\mu m)$",
            )
        else:
            ax.set(
                xlabel="Time (mins)",
                ylabel=r"Distance from wound $(\mu m)$",
            )
        fileTitle = util.getFileTitle(fileType)
        boldTitle = util.getBoldTitle(fileTitle)
        if "AFP" in boldTitle:
            ax.title.set_text(f"Division density\n{boldTitle}")
        else:
            colour, mark = util.getColorLineMarker(fileType, groupTitle)
            plt.title(f"Division density\n{boldTitle}", color=colour)

        fig.savefig(
            f"results/biologyWoundPaper/Division density heatmap {fileType}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")


# ------------------- Figure 3, 4, 5

fileTypes, groupTitle = util.getFilesTypes(fileType="WoundL")

# Compare wounds: Control and mutune
if False:
    for fileType in fileTypes[1:]:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.vlines(10, -100, 1500, colors="r", linestyles="dashed")
        ax.set_ylim([0, 1250])
        ax.set_xlim([0, 90])

        fileTypeControl = ControlFor(fileType)
        filenames = util.getFilesType(fileTypeControl)[0]
        _df = []
        Area0 = []

        for filename in filenames:
            t0 = util.findStartTime(filename)
            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
            T = len(dfWound)
            area = np.array(dfWound["Area"]) * (scale) ** 2

            for t in range(T):
                if area[t] > 10:
                    _df.append({"Area": area[t], "Time": int(t0 / 2) * 2 + 2 * t})
                else:
                    _df.append({"Area": 0, "Time": int(t0 / 2) * 2 + 2 * t})

        df = pd.DataFrame(_df)
        A = []
        time = []
        std = []
        T = set(df["Time"])
        N = len(filenames)
        Area0 = np.mean(df["Area"][df["Time"] == 10])
        for t in T:
            if len(df[df["Time"] == t]) > N / 3:
                if np.mean(df["Area"][df["Time"] == t]) > 0.2 * Area0:
                    time.append(t)
                    A.append(np.mean(df["Area"][df["Time"] == t]))
                    std.append(np.std(df["Area"][df["Time"] == t]))

        A = np.array(A)
        std = np.array(std)
        colour, mark = util.getColorLineMarker("WoundL18h", groupTitle)
        fileTitle = util.getFileTitle(fileTypeControl)
        ax.plot(time, A, label=fileTitle, marker=mark, color=colour)
        ax.fill_between(time, A - std, A + std, alpha=0.15, color=colour)

        filenames, fileType = util.getFilesType(fileType)
        _df = []
        Area0 = []

        for filename in filenames:
            t0 = util.findStartTime(filename)
            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
            T = len(dfWound)
            area = np.array(dfWound["Area"]) * (scale) ** 2

            for t in range(T):
                if area[t] > 10:
                    _df.append({"Area": area[t], "Time": int(t0 / 2) * 2 + 2 * t})
                else:
                    _df.append({"Area": 0, "Time": int(t0 / 2) * 2 + 2 * t})

        df = pd.DataFrame(_df)
        A = []
        time = []
        std = []
        T = set(df["Time"])
        N = len(filenames)
        Area0 = np.mean(df["Area"][df["Time"] == 10])
        for t in T:
            if len(df[df["Time"] == t]) > N / 3:
                if np.mean(df["Area"][df["Time"] == t]) > 0.2 * Area0:
                    time.append(t)
                    A.append(np.mean(df["Area"][df["Time"] == t]))
                    std.append(np.std(df["Area"][df["Time"] == t]))

        A = np.array(A)
        std = np.array(std)
        colour, mark = util.getColorLineMarker(fileType, "WoundL")
        fileTitle = util.getFileTitle(fileType)
        ax.plot(time, A, label=fileTitle, marker=mark, color=colour)
        ax.fill_between(time, A - std, A + std, alpha=0.15, color=colour)

        plt.xlabel("Time after wounding (mins)")
        plt.ylabel(r"Area ($\mu m ^2$)")
        boldTitle = util.getBoldTitle(groupTitle + "s")
        plt.title(f"Mean area of\n{boldTitle}")
        plt.legend(loc="upper right", fontsize=9)
        fig.savefig(
            f"results/biologyWoundPaper/Mean area of {fileType} wounds",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")

T = 84
timeStep = 4
R = 50
rStep = 10

# Individual: v with distance from wound edge and time
if False:
    for fileType in fileTypes[1:]:
        filenames = util.getFilesType(fileType)[0]
        v1 = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        v1Cont = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        area = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        dfVelocity = pd.read_pickle(f"databases/dfVelocityWound{fileType}.pkl")
        for k in range(len(filenames)):
            filename = filenames[k]
            dfFile = dfVelocity[dfVelocity["Filename"] == filename]
            dv1Cont = np.mean(dfFile["dv"] ** 2, axis=0)[0] ** 0.5
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
                        v1Cont[k, t, r] = np.mean(df["dv"], axis=0)[0] / dv1Cont

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
        V1Cont = np.zeros([int(T / timeStep), int(R / rStep)])
        std = np.zeros([int(T / timeStep), int(R / rStep)])
        meanArea = np.zeros([int(T / timeStep), int(R / rStep)])

        for r in range(area.shape[2]):
            for t in range(area.shape[1]):
                _V1 = v1[:, t, r][v1[:, t, r] != 0]
                _area = area[:, t, r][v1[:, t, r] != 0]
                _V1Cont = v1Cont[:, t, r][v1Cont[:, t, r] != 0]
                if (len(_area) > 0) & (np.sum(_area) > 0):
                    _dd, _std = weighted_avg_and_std(_V1, _area)
                    V1[t, r] = _dd
                    std[t, r] = _std
                    meanArea[t, r] = np.mean(_area)
                    _dd, _std = weighted_avg_and_std(_V1Cont, _area)
                    V1Cont[t, r] = _dd
                else:
                    V1[t, r] = np.nan
                    std[t, r] = np.nan
                    V1Cont[t, r] = np.nan

        V1[meanArea < 500] = np.nan
        V1Cont[meanArea < 500] = np.nan

        t, r = np.mgrid[
            timeStep / 2 : T + timeStep / 2 : timeStep,
            rStep / 2 : R + rStep / 2 : rStep,
        ]
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
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
            ylabel=f"Distance from\nwound edge" + r"$(\mu m)$",
        )
        fileTitle = util.getFileTitle(fileType)
        boldTitle = util.getBoldTitle(fileTitle)
        colour, mark = util.getColorLineMarker(fileType, groupTitle)
        plt.title(r"Mean cell velocity towards" + f"\n{boldTitle}", color=colour)

        fig.savefig(
            f"results/biologyWoundPaper/v heatmap {fileTitle}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")

# Individual: Q1 with distance from wound edge and time
if False:
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
            Q1 / Q1Norm,
            vmin=-3.1,
            vmax=3.1,
            cmap="RdBu_r",
        )
        fig.colorbar(c, ax=ax)
        ax.set(
            xlabel="Time after wounding (mins)",
            ylabel=f"Distance from\nwound edge" + r"$(\mu m)$",
        )
        fileTitle = util.getFileTitle(fileType)
        boldTitle = util.getBoldTitle(fileTitle)
        colour, mark = util.getColorLineMarker(fileType, groupTitle)
        plt.title(
            r"Normalised cell shape relative to" + f"\n{boldTitle}",
            color=colour,
        )

        fig.savefig(
            f"results/biologyWoundPaper/Q1 heatmap {fileTitle}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")

# Compare with wt Large wound: V1 with distance from wound edge and time
if False:
    for fileType in fileTypes[1:]:
        fileTypeControl = ControlFor(fileType)
        filenames = util.getFilesType(fileTypeControl)[0]
        v1 = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        area = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        dfVelocity = pd.read_pickle(f"databases/dfVelocityWound{fileTypeControl}.pkl")
        for k in range(len(filenames)):
            filename = filenames[k]
            dfFile = dfVelocity[dfVelocity["Filename"] == filename]
            dv1Cont = np.mean(dfFile["dv"] ** 2, axis=0)[0] ** 0.5
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

        V1Large = np.zeros([int(T / timeStep), int(R / rStep)])
        std = np.zeros([int(T / timeStep), int(R / rStep)])
        meanArea = np.zeros([int(T / timeStep), int(R / rStep)])

        for r in range(area.shape[2]):
            for t in range(area.shape[1]):
                _V1 = v1[:, t, r][v1[:, t, r] != 0]
                _area = area[:, t, r][v1[:, t, r] != 0]
                if (len(_area) > 0) & (np.sum(_area) > 0):
                    _dd, _std = weighted_avg_and_std(_V1, _area)
                    V1Large[t, r] = _dd
                    std[t, r] = _std
                    meanArea[t, r] = np.mean(_area)
                else:
                    V1Large[t, r] = np.nan
                    std[t, r] = np.nan

        V1Large[meanArea < 500] = np.nan

        filenames = util.getFilesType(fileType)[0]
        v1 = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        v1Cont = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        area = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        dfVelocity = pd.read_pickle(f"databases/dfVelocityWound{fileType}.pkl")
        for k in range(len(filenames)):
            filename = filenames[k]
            dfFile = dfVelocity[dfVelocity["Filename"] == filename]
            dv1Cont = np.mean(dfFile["dv"] ** 2, axis=0)[0] ** 0.5
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
                        v1Cont[k, t, r] = np.mean(df["dv"], axis=0)[0] / dv1Cont

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
        V1Cont = np.zeros([int(T / timeStep), int(R / rStep)])
        std = np.zeros([int(T / timeStep), int(R / rStep)])
        meanArea = np.zeros([int(T / timeStep), int(R / rStep)])

        for r in range(area.shape[2]):
            for t in range(area.shape[1]):
                _V1 = v1[:, t, r][v1[:, t, r] != 0]
                _area = area[:, t, r][v1[:, t, r] != 0]
                _V1Cont = v1Cont[:, t, r][v1Cont[:, t, r] != 0]
                if (len(_area) > 0) & (np.sum(_area) > 0):
                    _dd, _std = weighted_avg_and_std(_V1, _area)
                    V1[t, r] = _dd
                    std[t, r] = _std
                    meanArea[t, r] = np.mean(_area)
                    _dd, _std = weighted_avg_and_std(_V1Cont, _area)
                    V1Cont[t, r] = _dd
                else:
                    V1[t, r] = np.nan
                    std[t, r] = np.nan
                    V1Cont[t, r] = np.nan

        V1[meanArea < 500] = np.nan
        V1Cont[meanArea < 500] = np.nan

        t, r = np.mgrid[
            timeStep / 2 : T + timeStep / 2 : timeStep,
            rStep / 2 : R + rStep / 2 : rStep,
        ]
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        c = ax.pcolor(
            t,
            r,
            V1 - V1Large,
            vmin=-0.2,
            vmax=0.2,
            cmap="RdBu_r",
        )
        fig.colorbar(c, ax=ax)
        ax.set(
            xlabel="Time after wounding (mins)",
            ylabel=f"Distance from \n wound edge" + r"$(\mu m)$",
        )
        fileTitle = util.getFileTitle(fileType)
        boldTitle = util.getBoldTitle(fileTitle)
        colour, mark = util.getColorLineMarker(fileType, groupTitle)
        plt.title(
            r"Difference in mean cell velocity between" + f"\n{boldTitle} and control",
            color=colour,
            y=1.03,
        )

        fig.savefig(
            f"results/biologyWoundPaper/v heatmap change large wound {fileTitle}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")

# Compare with wt Large wound: Q1 with distance from wound edge and time
if False:
    for fileType in fileTypes[1:]:
        fileTypeControl = ControlFor(fileType)
        filenames = util.getFilesType(fileTypeControl)[0]
        q1 = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        area = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        dfShape = pd.read_pickle(f"databases/dfShapeWound{fileTypeControl}.pkl")
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
                                (dist[t1:t2] > rStep * r)
                                & (dist[t1:t2] <= rStep * (r + 1))
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
            (Q1 - Q1Large) / Q1Norm,
            vmin=-1.5,
            vmax=1.5,
            cmap="RdBu_r",
        )
        fig.colorbar(c, ax=ax)
        ax.set(
            xlabel="Time after wounding (mins)",
            ylabel=f"Distance from \n wound edge" + r"$(\mu m)$",
        )
        fileTitle = util.getFileTitle(fileType)
        boldTitle = util.getBoldTitle(fileTitle)
        colour, mark = util.getColorLineMarker(fileType, groupTitle)
        plt.title(
            r"Difference in mean cell shape between" + f"\n{boldTitle} and control",
            color=colour,
            y=1.03,
        )

        fig.savefig(
            f"results/biologyWoundPaper/Q1 heatmap change large wound {fileTitle}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")


T = 180
timeStep = 10
R = 110
rStep = 10

# Individual: Divison density with distance from wound edge and time
if False:
    for fileType in fileTypes:
        filenames = util.getFilesType(fileType)[0]
        count = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        area = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
        for k in range(len(filenames)):
            filename = filenames[k]
            dfFile = dfDivisions[dfDivisions["Filename"] == filename]
            if "Wound" in filename:
                t0 = util.findStartTime(filename)
            else:
                t0 = 0
            t2 = int(timeStep / 2 * (int(T / timeStep) + 1) - t0 / 2)

            for r in range(count.shape[2]):
                for t in range(count.shape[1]):
                    df1 = dfFile[dfFile["T"] > timeStep * t]
                    df2 = df1[df1["T"] <= timeStep * (t + 1)]
                    df3 = df2[df2["R"] > rStep * r]
                    df = df3[df3["R"] <= rStep * (r + 1)]
                    count[k, t, r] = len(df)

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

        dd = np.zeros([int(T / timeStep), int(R / rStep)])
        std = np.zeros([int(T / timeStep), int(R / rStep)])
        sumArea = np.zeros([int(T / timeStep), int(R / rStep)])

        for r in range(area.shape[2]):
            for t in range(area.shape[1]):
                _area = area[:, t, r][area[:, t, r] > 800]
                _count = count[:, t, r][area[:, t, r] > 800]
                if len(_area) > 0:
                    _dd, _std = weighted_avg_and_std(_count / _area, _area)
                    dd[t, r] = _dd
                    std[t, r] = _std
                    sumArea[t, r] = np.sum(_area)
                else:
                    dd[t, r] = np.nan
                    std[t, r] = np.nan

        dd[sumArea < 600 * len(filenames)] = np.nan
        dd = dd * 10000

        t, r = np.mgrid[
            timeStep / 2 : T + timeStep / 2 : timeStep,
            rStep / 2 : R + rStep / 2 : rStep,
        ]
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        c = ax.pcolor(
            t,
            r,
            dd,
            vmin=0,
            vmax=9,
            cmap="plasma",
        )
        fig.colorbar(c, ax=ax)
        if "Wound" in fileType:
            ax.set(
                xlabel="Time after wounding (mins)",
                ylabel=r"Distance from wound $(\mu m)$",
            )
        else:
            ax.set(
                xlabel="Time (mins)",
                ylabel=r"Distance from wound $(\mu m)$",
            )
        fileTitle = util.getFileTitle(fileType)
        boldTitle = util.getBoldTitle(fileTitle)
        colour, mark = util.getColorLineMarker(fileType, groupTitle)
        plt.title(f"Division density\n{boldTitle}", color=colour)

        fig.savefig(
            f"results/biologyWoundPaper/Division density heatmap {fileType}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")

# Compare with wt Large wound: Divison density with distance from wound edge and time
if False:
    for fileType in fileTypes[1:]:
        fileTypeControl = ControlFor(fileType)
        filenames = util.getFilesType(fileTypeControl)[0]
        count = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        area = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileTypeControl}.pkl")
        for k in range(len(filenames)):
            filename = filenames[k]
            dfFile = dfDivisions[dfDivisions["Filename"] == filename]
            if "Wound" in filename:
                t0 = util.findStartTime(filename)
            else:
                t0 = 0
            t2 = int(timeStep / 2 * (int(T / timeStep) + 1) - t0 / 2)

            for r in range(count.shape[2]):
                for t in range(count.shape[1]):
                    df1 = dfFile[dfFile["T"] > timeStep * t]
                    df2 = df1[df1["T"] <= timeStep * (t + 1)]
                    df3 = df2[df2["R"] > rStep * r]
                    df = df3[df3["R"] <= rStep * (r + 1)]
                    count[k, t, r] = len(df)

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

        dd = np.zeros([int(T / timeStep), int(R / rStep)])
        std = np.zeros([int(T / timeStep), int(R / rStep)])
        sumArea = np.zeros([int(T / timeStep), int(R / rStep)])

        for r in range(area.shape[2]):
            for t in range(area.shape[1]):
                _area = area[:, t, r][area[:, t, r] > 800]
                _count = count[:, t, r][area[:, t, r] > 800]
                if len(_area) > 0:
                    _dd, _std = weighted_avg_and_std(_count / _area, _area)
                    dd[t, r] = _dd
                    std[t, r] = _std
                    sumArea[t, r] = np.sum(_area)
                else:
                    dd[t, r] = np.nan
                    std[t, r] = np.nan

        time = np.linspace(0, T, int(T / timeStep) + 1)[:-1]
        dd[sumArea < 600 * len(filenames)] = np.nan
        ddLarge = dd * 10000

        filenames = util.getFilesType(fileType)[0]
        count = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        area = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
        for k in range(len(filenames)):
            filename = filenames[k]
            dfFile = dfDivisions[dfDivisions["Filename"] == filename]
            if "Wound" in filename:
                t0 = util.findStartTime(filename)
            else:
                t0 = 0
            t2 = int(timeStep / 2 * (int(T / timeStep) + 1) - t0 / 2)

            for r in range(count.shape[2]):
                for t in range(count.shape[1]):
                    df1 = dfFile[dfFile["T"] > timeStep * t]
                    df2 = df1[df1["T"] <= timeStep * (t + 1)]
                    df3 = df2[df2["R"] > rStep * r]
                    df = df3[df3["R"] <= rStep * (r + 1)]
                    count[k, t, r] = len(df)

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

        dd = np.zeros([int(T / timeStep), int(R / rStep)])
        std = np.zeros([int(T / timeStep), int(R / rStep)])
        sumArea = np.zeros([int(T / timeStep), int(R / rStep)])

        for r in range(area.shape[2]):
            for t in range(area.shape[1]):
                _area = area[:, t, r][area[:, t, r] > 800]
                _count = count[:, t, r][area[:, t, r] > 800]
                if len(_area) > 0:
                    _dd, _std = weighted_avg_and_std(_count / _area, _area)
                    dd[t, r] = _dd
                    std[t, r] = _std
                    sumArea[t, r] = np.sum(_area)
                else:
                    dd[t, r] = np.nan
                    std[t, r] = np.nan

        time = np.linspace(0, T, int(T / timeStep) + 1)[:-1]
        dd[sumArea < 600 * len(filenames)] = np.nan
        dd = dd * 10000

        fileTitle = util.getFileTitle(fileType)
        t, r = np.mgrid[
            timeStep / 2 : T + timeStep / 2 : timeStep,
            rStep / 2 : R + rStep / 2 : rStep,
        ]
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        c = ax.pcolor(
            t,
            r,
            dd - ddLarge,
            vmin=-5,
            vmax=5,
            cmap="RdBu_r",
        )
        fig.colorbar(c, ax=ax)
        ax.set(
            xlabel="Time after wounding (mins)",
            ylabel=r"Distance from wound $(\mu m)$",
        )
        fileTitle = util.getFileTitle(fileType)
        boldTitle = util.getBoldTitle(fileTitle)
        colour, mark = util.getColorLineMarker(fileType, groupTitle)
        plt.title(
            f"Difference in division density between\n{boldTitle} and control",
            color=colour,
            y=1.03,
        )

        fig.savefig(
            f"results/biologyWoundPaper/Change in Division density heatmap with large wt {fileTitle}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")



# ------------------- Figure S2

scale = 123.26 / 512
T = 93
fileTypes, groupTitle = util.getFilesTypes(fileType="control")

# Compare wounds: 15h, 18h and 26h
if False:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.vlines(10, -100, 1500, colors="r", linestyles="dashed")
    ax.set_ylim([0, 1250])

    i = 0
    for fileType in fileTypes:
        if "WoundS" in fileType:
            continue
        filenames, fileType = util.getFilesType(fileType)
        _df = []
        Area0 = []

        for filename in filenames:
            t0 = util.findStartTime(filename)
            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
            T = len(dfWound)
            area = np.array(dfWound["Area"]) * (scale) ** 2

            for t in range(T):
                if area[t] > area[0] * 0.2:
                    _df.append({"Area": area[t], "Time": int(t0 / 2) * 2 + 2 * t})
                else:
                    _df.append({"Area": 0, "Time": int(t0 / 2) * 2 + 2 * t})

        df = pd.DataFrame(_df)
        A = []
        time = []
        std = []
        T = set(df["Time"])
        N = len(filenames)
        Area0 = np.mean(df["Area"][df["Time"] == 10])
        for t in T:
            if len(df[df["Time"] == t]) > N / 3:
                if np.mean(df["Area"][df["Time"] == t]) > 0.2 * Area0:
                    time.append(t)
                    A.append(np.mean(df["Area"][df["Time"] == t]))
                    std.append(np.std(df["Area"][df["Time"] == t]))

        A = np.array(A)
        std = np.array(std)
        colour, mark = util.getColorLineMarker(fileType, "WoundL")
        fileTitle = util.getFileTitle(fileType)
        ax.plot(time, A, label=fileTitle, marker=mark, color=colour)
        ax.fill_between(time, A - std, A + std, alpha=0.15, color=colour)

    plt.xlabel("Time after wounding (mins)")
    plt.ylabel(r"Area ($\mu m ^2$)")
    boldTitle = util.getBoldTitle(groupTitle + "s")
    plt.title(f"Mean area of\n{boldTitle}")
    plt.legend(loc="upper right", fontsize=9)
    fig.savefig(
        f"results/biologyWoundPaper/Mean area of {fileType} wounds",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")


T = 180
timeStep = 10
fileTypes = ["Unwound18h", "Unwound15h", "Unwound26h"]

# Compare: Divison density with time
if False:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    total = 0
    for fileType in fileTypes:
        filenames = util.getFilesType(fileType)[0]

        count = np.zeros([len(filenames), int(T / timeStep)])
        area = np.zeros([len(filenames), int(T / timeStep)])
        dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
        total += len(dfDivisions)
        for k in range(len(filenames)):
            filename = filenames[k]
            t0 = util.findStartTime(filename)
            dfFile = dfDivisions[dfDivisions["Filename"] == filename]

            for t in range(count.shape[1]):
                df1 = dfFile[dfFile["T"] > timeStep * t]
                df = df1[df1["T"] <= timeStep * (t + 1)]
                count[k, t] = len(df)

            inPlane = 1 - (
                sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype(int) / 255
            )
            for t in range(area.shape[1]):
                t1 = int(timeStep / 2 * t - t0 / 2)
                t2 = int(timeStep / 2 * (t + 1) - t0 / 2)
                if t1 < 0:
                    t1 = 0
                if t2 < 0:
                    t2 = 0
                area[k, t] = np.sum(inPlane[t1:t2]) * scale**2

        time = []
        dd = []
        std = []
        for t in range(area.shape[1]):
            _area = area[:, t][area[:, t] > 0]
            _count = count[:, t][area[:, t] > 0]
            if len(_area) > 0:
                _dd, _std = weighted_avg_and_std(_count / _area, _area)
                dd.append(_dd * 10000)
                std.append(_std * 10000)
                time.append(t * timeStep + timeStep / 2)

        dd = np.array(dd)
        std = np.array(std)
        colour, mark = util.getColorLineMarker(fileType, "WoundL")
        fileTitle = util.getFileTitle(fileType)
        ax.plot(time, dd, label=fileTitle, marker=mark, color=colour)
        ax.fill_between(time, dd - std, dd + std, alpha=0.15, color=colour)

    time = np.array(time)
    ax.set(
        xlabel="Time after wounding (mins)",
        ylabel=r"Divison density ($10^{-4}\mu m^{-2}$)",
    )
    boldTitle = util.getBoldTitle("unwounded " + groupTitle + "s")
    ax.title.set_text(f"Division density with \n time of " + boldTitle)
    ax.set_ylim([0, 8.5])
    ax.legend(loc="upper left", fontsize=10)

    fig.savefig(
        f"results/biologyWoundPaper/Compared division density with time {groupTitle}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")
    print(total)


# Q_0 for prewound and early unwounded
if False:
    _df = []
    fileType = "Unwound18h"
    filenames, fileType = util.getFilesType(fileType)
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        _df.append(
            {
                "Type": "Unwounded 18h APF ",
                "Filename": filename,
                "q": np.mean(df["q"][df["T"] < 8])[0, 0],
            }
        )
        # q = np.mean(df["q"][df["T"] < 8])[0, 0]
        # print(f"{filename}: {q}")

    fileType = "Unwound15h"
    filenames, fileType = util.getFilesType(fileType)
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        _df.append(
            {
                "Type": "Unwounded 14.75h APF ",
                "Filename": filename,
                "q": np.mean(df["q"][df["T"] < 8])[0, 0],
            }
        )
        # q = np.mean(df["q"][df["T"] < 8])[0, 0]
        # print(f"{filename}: {q}")

    fileType = "Unwound26h"
    filenames, fileType = util.getFilesType(fileType)
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        _df.append(
            {
                "Type": "Unwounded 26h APF ",
                "Filename": filename,
                "q": np.mean(df["q"][df["T"] < 8])[0, 0],
            }
        )
        # q = np.mean(df["q"][df["T"] < 8])[0, 0]
        # print(f"{filename}: {q}")

    sns.set(font_scale=1.5)
    df = pd.DataFrame(_df)
    ax = sns.boxplot(
        y="q",
        x="Type",
        data=df,
        boxprops={"facecolor": "None"},
    )
    plt.draw()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_ylim([-0.005, 0.03])
    sns_plot = sns.swarmplot(data=df, y="q", x="Type")
    fig = sns_plot.get_figure()
    fig.savefig(
        f"results/biologyWoundPaper/compare q prewound",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")
    # sp.stats.ttest_ind(df["q"][df["Type"]=="Unwound 26h AFP"], df["q"][df["Type"]=="Unwound 18h AFP"])
