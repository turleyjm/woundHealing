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

import cellProperties as cell
import utils as util

plt.rcParams.update({"font.size": 16})

# -------------------

fileTypes, groupTitle = util.getFilesTypes()

scale = 123.26 / 512
T = 180
timeStep = 10
R = 110
rStep = 10
Theta = 390
thetaStep = 30


def weighted_avg_and_std(values, weight, axis=0):
    average = np.average(values, weights=weight, axis=axis)
    variance = np.average((values - average) ** 2, weights=weight, axis=axis)
    return average, np.sqrt(variance)


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
    return [m, dm, b, db]


def bestFitUnwound(fileType="Unwound18h"):
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    filenames = util.getFilesType(fileType)[0]
    count = np.zeros([len(filenames), int(T / timeStep)])
    area = np.zeros([len(filenames), int(T / timeStep)])
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
            dd.append(_dd)
            std.append(_std)
            time.append(t * 10 + timeStep / 2)
    time = np.array(time)
    dd = np.array(dd)
    std = np.array(std)
    bestfit = OLSfit(time, dd)
    (m, c) = (bestfit[0], bestfit[2])

    return m, c


# -------------------

# Compare: Divison density with time
if True:
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
        colour, mark = util.getColorLineMarker(fileType, groupTitle)
        fileTitle = util.getFileTitle(fileType)
        ax.plot(time, dd, label=fileTitle, marker=mark, color=colour)
        ax.fill_between(time, dd - std, dd + std, alpha=0.15, color=colour)

    time = np.array(time)

    if (
        groupTitle == "wild type"
        or groupTitle == "JNK DN"
        or groupTitle == "Ca RNAi"
        or groupTitle == "immune ablation"
    ):
        compare = util.compareType(groupTitle)
        (m, c) = bestFitUnwound(compare)
        ax.plot(time, (m * time + c) * 10000, color="k", label=f"Linear model")

    ax.set(
        xlabel="Time after wounding (mins)",
        ylabel=r"Divison density ($10^{-4}\mu m^{-2}$)",
    )
    boldTitle = util.getBoldTitle(groupTitle)
    ax.title.set_text(f"Division density with \n time " + boldTitle)
    ax.set_ylim([0, 7])
    ax.legend(loc="upper left", fontsize=10)

    fig.savefig(
        f"results/Compared division density with time {groupTitle}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")
    print(total)

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

        t, r = np.mgrid[0:T:timeStep, 0:R:rStep]
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        c = ax.pcolor(
            t,
            r,
            dd,
            vmin=0,
            vmax=6,
        )
        fig.colorbar(c, ax=ax)
        ax.set(
            xlabel="Time (mins)",
            ylabel=r"Distance from wound $(\mu m)$",
        )
        fileTitle = util.getFileTitle(fileType)
        boldTitle = util.getBoldTitle(fileTitle)
        ax.title.set_text(f"Division density \n {boldTitle}")

        fig.savefig(
            f"results/Division density heatmap {fileTitle}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")

# Individual: Change in divison density with distance from wound edge and time
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

        if "18h" in fileType:
            compare = "Unwound18h"
        elif "JNK" in fileType:
            compare = "UnwoundJNK"
        elif "Ca" in fileType:
            compare = "UnwoundCa"
        elif "rpr" in fileType:
            compare = "Unwoundrpr"
        (m, c) = bestFitUnwound(compare)
        time = np.linspace(0, T, int(T / timeStep) + 1)[:-1]
        for r in range(dd.shape[1]):
            dd[:, r] = dd[:, r] - (m * time + c)

        dd[sumArea < 600 * len(filenames)] = np.nan
        dd = dd * 10000

        fileTitle = util.getFileTitle(fileType)
        t, r = np.mgrid[0:T:timeStep, 0:R:rStep]
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        c = ax.pcolor(
            t,
            r,
            dd,
            vmin=-5,
            vmax=5,
            cmap="RdBu_r",
        )
        fig.colorbar(c, ax=ax)
        ax.set(
            xlabel="Time (mins)",
            ylabel=r"Distance from wound $(\mu m)$",
        )
        fileTitle = util.getFileTitle(fileType)
        boldTitle = util.getBoldTitle(fileTitle)
        ax.title.set_text(
            f"Deviation in division density: \n {boldTitle} from linear model"
        )

        fig.savefig(
            f"results/Change in Division density heatmap {fileTitle}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")

        if False:
            t, r = np.mgrid[0:T:timeStep, 0:R:rStep]
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            c = ax.pcolor(
                t,
                r,
                sumArea,
                cmap="Reds",
            )
            fig.colorbar(c, ax=ax)
            ax.set(
                xlabel="Time after wounding (mins)",
                ylabel=r"Distance from wound edge $(\mu m)$",
            )
            ax.title.set_text(f"Area of division bins distance and time {boldTitle}")

            fig.savefig(
                f"results/Divison density Area bin heatmap {fileTitle}",
                transparent=True,
                bbox_inches="tight",
                dpi=300,
            )
            plt.close("all")
