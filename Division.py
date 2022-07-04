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
from pySTARMA import starma_model
from pySTARMA import stacf_stpacf
import matplotlib.colors as colors

import cellProperties as cell
import utils as util

plt.rcParams.update({"font.size": 12})

# -------------------

filenames, fileType = util.getFilesType()
scale = 123.26 / 512
T = 180
timeStep = 8
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
    denom = np.sum(1 / dy ** 2) * np.sum((x / dy) ** 2) - (np.sum(x / dy ** 2)) ** 2
    m = (
        np.sum(1 / dy ** 2) * np.sum(x * y / dy ** 2)
        - np.sum(x / dy ** 2) * np.sum(y / dy ** 2)
    ) / denom
    b = (
        np.sum(x ** 2 / dy ** 2) * np.sum(y / dy ** 2)
        - np.sum(x / dy ** 2) * np.sum(x * y / dy ** 2)
    ) / denom
    dm = np.sqrt(np.sum(1 / dy ** 2) / denom)
    db = np.sqrt(np.sum(x / dy ** 2) / denom)
    return [m, dm, b, db]


def bestFitUnwound():
    fileType = "Unwound"
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    filenames = util.getFilesOfType(fileType)
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
            area[k, t] = np.sum(inPlane[t1:t2]) * scale ** 2
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
    bestfit = OLSfit(time, dd, dy=std)
    (m, c) = (bestfit[0], bestfit[2])

    return m, c


# -------------------


# Divison density with time

if False:
    count = np.zeros([len(filenames), int(T / timeStep)])
    area = np.zeros([len(filenames), int(T / timeStep)])
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
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
            area[k, t] = np.sum(inPlane[t1:t2]) * scale ** 2

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

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.errorbar(time, dd, yerr=std, marker="o")
    ax.set(xlabel="Time", ylabel=r"Divison density ($\mu m^{-2}$)")
    ax.title.set_text(f"Divison density with time {fileType}")
    ax.set_ylim([0, 0.0007])

    fig.savefig(
        f"results/Divison density with time {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")


if False:
    fileType = "Unwound"
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    filenames = util.getFilesOfType(fileType)
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
            area[k, t] = np.sum(inPlane[t1:t2]) * scale ** 2

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
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    bestfit = OLSfit(time, dd, dy=std)
    (m, c) = (bestfit[0], bestfit[2])
    ax.errorbar(time, dd, yerr=std, marker="o")
    ax.plot(time, m * time + c)
    ax.set(xlabel="Time", ylabel=r"Divison density ($\mu m^{-2}$)")
    ax.title.set_text(f"Divison density with time {fileType}")
    ax.set_ylim([0, 0.0007])

    fig.savefig(
        f"results/Divison density with time best fit {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

# Compare divison density with time

if False:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    labels = ["WoundS", "WoundL", "Unwound"]
    legend = ["small wound", "large wound", "unwounded"]
    dat_dd = []
    total = 0
    i = 0
    for fileType in labels:
        filenames = util.getFilesOfType(fileType)
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
                area[k, t] = np.sum(inPlane[t1:t2]) * scale ** 2

        dat_dd.append(count / area)

        time = []
        dd = []
        std = []
        for t in range(area.shape[1]):
            _area = area[:, t][area[:, t] > 0]
            _count = count[:, t][area[:, t] > 0]
            if len(_area) > 0:
                _dd, _std = weighted_avg_and_std(_count / _area, _area)
                dd.append(_dd * 10000 * timeStep / 2)
                std.append(_std)
                time.append(t * timeStep + timeStep / 2)

        ax.plot(time, dd, label=f"{legend[i]}", marker="o")
        i += 1

    ax.set(xlabel="Time (mins)", ylabel=r"Divison density ($100\mu m^{-2}$)")
    ax.title.set_text(f"Division density with time")
    ax.set_ylim([0, 20])
    ax.legend()

    fig.savefig(
        f"results/Compared division density with time",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")
    print(total)


# Compare divison density with time error bar

if True:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    labels = ["WoundS", "Unwound"]
    legend = ["small wound", "unwounded"]
    dat_dd = []
    total = 0
    i = 0
    for fileType in labels:
        filenames = util.getFilesOfType(fileType)
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
                area[k, t] = np.sum(inPlane[t1:t2]) * scale ** 2

        dat_dd.append(count / area)

        time = []
        dd = []
        std = []
        for t in range(area.shape[1]):
            _area = area[:, t][area[:, t] > 0]
            _count = count[:, t][area[:, t] > 0]
            if len(_area) > 0:
                _dd, _std = weighted_avg_and_std(_count / _area, _area)
                dd.append(_dd * 10000 * timeStep / 2)
                std.append(_std * 10000 * timeStep / 2)
                time.append((-1 + i) + t * timeStep + timeStep / 2)

        ax.errorbar(time, dd, yerr=std, label=f"{legend[i]}", marker="o")
        i += 1

    ax.set(xlabel="Time (mins)", ylabel=r"Divison density ($100\mu m^{-2}$)")
    ax.title.set_text(f"Division density with time")
    ax.set_ylim([0, 25])
    ax.legend()

    fig.savefig(
        f"results/Compared division density with time errorbars",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")
    print(total)


# Divison density with distance from wound edge

if False:
    count = np.zeros([len(filenames), int(R / rStep)])
    area = np.zeros([len(filenames), int(R / rStep)])
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    for k in range(len(filenames)):
        filename = filenames[k]
        dfFile = dfDivisions[dfDivisions["Filename"] == filename]
        t0 = util.findStartTime(filename)
        t2 = int(timeStep / 2 * (int(T / timeStep) + 1) - t0 / 2)

        for r in range(count.shape[1]):
            df1 = dfFile[dfFile["R"] > rStep * r]
            df = df1[df1["R"] <= rStep * (r + 1)]
            count[k, r] = len(df)

        inPlane = 1 - (
            sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype(int)[:t2]
            / 255
        )
        dist = (
            sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)[:t2]
            * scale
        )

        for r in range(area.shape[1]):
            area[k, r] = (
                np.sum(inPlane[(dist > rStep * r) & (dist <= rStep * (r + 1))])
                * scale ** 2
            )
            # test = np.zeros([t2,512,512])
            # test[(dist > rStep * r) & (dist <= rStep * (r + 1))] = 1
            # test = np.asarray(test, "uint8")
            # tifffile.imwrite(f"results/test.tif", test)

    radius = []
    dd = []
    std = []
    for r in range(area.shape[1]):
        _area = area[:, r][area[:, r] > 0]
        _count = count[:, r][area[:, r] > 0]
        if len(_area) > 0:
            _dd, _std = weighted_avg_and_std(_count / _area, _area)
            dd.append(_dd)
            std.append(_std)
            radius.append(r * 10 + rStep / 2)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.errorbar(radius, dd, yerr=std, marker="o")
    ax.set(xlabel="Distance", ylabel=r"Divison density ($\mu m^{-2}$)")
    ax.title.set_text(f"Divison density with distance from wound {fileType}")
    ax.set_ylim([0, 0.0007])

    fig.savefig(
        f"results/Divison density with distance {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")


# Compare divison density with distance from wound edge

if False:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    labels = ["WoundS", "WoundL"]
    for fileType in labels:
        filenames = util.getFilesOfType(fileType)
        count = np.zeros([len(filenames), int(R / rStep)])
        area = np.zeros([len(filenames), int(R / rStep)])
        dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
        for k in range(len(filenames)):
            filename = filenames[k]
            dfFile = dfDivisions[dfDivisions["Filename"] == filename]
            t0 = util.findStartTime(filename)
            t2 = int(timeStep / 2 * (int(T / timeStep) + 1) - t0 / 2)

            for r in range(count.shape[1]):
                df1 = dfFile[dfFile["R"] > rStep * r]
                df = df1[df1["R"] <= rStep * (r + 1)]
                count[k, r] = len(df)

            inPlane = 1 - (
                sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype(int)[:t2]
                / 255
            )
            dist = (
                sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)[:t2]
                * scale
            )
            for r in range(area.shape[1]):
                area[k, r] = (
                    np.sum(inPlane[(dist > rStep * r) & (dist <= rStep * (r + 1))])
                    * scale ** 2
                )

        radius = []
        dd = []
        std = []
        for r in range(area.shape[1]):
            _area = area[:, r][area[:, r] > 0]
            _count = count[:, r][area[:, r] > 0]
            if len(_area) > 0:
                _dd, _std = weighted_avg_and_std(_count / _area, _area)
                dd.append(_dd)
                std.append(_std)
                radius.append(r * 10 + rStep / 2)

        ax.plot(radius, dd, label=f"{fileType}", marker="o")

    ax.set(xlabel="Distance", ylabel=r"Divison density ($\mu m^{-2}$)")
    ax.title.set_text(f"Divison density with distance from wound")
    ax.set_ylim([0, 0.00051])
    ax.legend()

    fig.savefig(
        f"results/Compared divison density with distance",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")


# Divison density with distance from wound edge and time
if False:
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
                            (dist[t1:t2] > rStep * r) & (dist[t1:t2] <= rStep * (r + 1))
                        ]
                    )
                    * scale ** 2
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

    dd[sumArea < 8000] = np.nan
    dd = dd * 10000 * timeStep / 2

    t, r = np.mgrid[0:T:timeStep, 0:R:rStep]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    c = ax.pcolor(
        t,
        r,
        dd,
        vmin=0,
        vmax=18,
    )
    fig.colorbar(c, ax=ax)
    ax.set(xlabel="Time (mins)", ylabel=r"$R (\mu m)$")
    ax.title.set_text(f"Division density {fileType}")

    fig.savefig(
        f"results/Division density heatmap {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")


# Change in divison density with distance from wound edge and time
if False:
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
                            (dist[t1:t2] > rStep * r) & (dist[t1:t2] <= rStep * (r + 1))
                        ]
                    )
                    * scale ** 2
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

    (m, c) = bestFitUnwound()
    time = np.linspace(0, T, int(T / timeStep) + 1)[:-1]
    for r in range(dd.shape[1]):
        dd[:, r] = dd[:, r] - (m * time + c)

    dd[sumArea < 600 * len(filenames)] = np.nan
    dd = dd * 10000 * timeStep / 2

    fileTitle = util.getFileTitle(fileType)
    t, r = np.mgrid[0:T:timeStep, 0:R:rStep]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    c = ax.pcolor(
        t,
        r,
        dd,
        vmin=-20,
        vmax=20,
        cmap="RdBu_r",
    )
    fig.colorbar(c, ax=ax)
    ax.set(xlabel="Time (mins)", ylabel=r"$R (\mu m)$")
    ax.title.set_text(f"Change in division density {fileTitle}")

    fig.savefig(
        f"results/Change in Division density heatmap {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

    # t, r = np.mgrid[0:T:timeStep, 0:R:rStep]
    # fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # c = ax.pcolor(
    #     t,
    #     r,
    #     sumArea,
    #     cmap="Reds",
    # )
    # fig.colorbar(c, ax=ax)
    # ax.set(xlabel="Time (min)", ylabel=r"$R (\mu m)$")
    # ax.title.set_text(f"Area of division bins distance and time {fileType}")

    # fig.savefig(
    #     f"results/Divison density Area bin heatmap {fileType}",
    #     transparent=True,
    #     bbox_inches="tight",
    #     dpi=300,
    # )
    # plt.close("all")


# compare unwounded to hot and cold spots

if False:
    labels = ["Unwound", "WoundS", "WoundL"]
    for fileType in labels:
        filenames = util.getFilesOfType(fileType)
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
                        * scale ** 2
                    )

        dd = np.zeros([int(T / timeStep), int(R / rStep)])
        std = np.zeros([int(T / timeStep), int(R / rStep)])

        for r in range(area.shape[2]):
            for t in range(area.shape[1]):
                _area = area[:, t, r][area[:, t, r] > 800]
                _count = count[:, t, r][area[:, t, r] > 800]
                if len(_area) > 0:
                    _dd, _std = weighted_avg_and_std(_count / _area, _area)
                    dd[t, r] = _dd
                    std[t, r] = _std
                else:
                    dd[t, r] = np.nan
                    std[t, r] = np.nan

        (m, c) = bestFitUnwound()
        time = np.linspace(0, T, int(T / timeStep) + 1)[:-1]
        for r in range(dd.shape[1]):
            dd[:, r] = dd[:, r] - (m * time + c)

        if fileType == "Unwound":
            Unwound = np.std(dd[:, :-2])
        elif fileType == "WoundS":
            WoundS_cold = np.std(dd[1:8, :4])
            WoundS_hot = np.std(dd[8:13, 2:8])
        elif fileType == "WoundL":
            WoundL_cold = np.std(dd[1:9, :8])
            WoundL_hot = np.std(dd[9:15, 1:10])

    y = np.array([Unwound, WoundS_cold, WoundS_hot, WoundL_cold, WoundL_hot])
    x = ["Unwound  ", "WoundS_cold  ", "WoundS_hot  ", "WoundL_cold  ", "WoundL_hot"]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(x, y, linestyle="None", marker="o")
    ax.set(xlabel="regions", ylabel="std of div density")
    ax.title.set_text(f"fluctions of division density in key regions")

    fig.savefig(
        f"results/key regions fluctions",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")


if False:
    count = np.zeros([len(filenames), int(Theta / thetaStep)])
    area = np.zeros([len(filenames), int(Theta / thetaStep)])
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    for k in range(len(filenames)):
        filename = filenames[k]
        dfShape = pd.read_pickle(f"dat/{filename}/shape{filename}.pkl")
        Q = np.mean(dfShape["q"])
        theta0 = 0.5 * np.arctan2(Q[0, 0], Q[0, 1])
        theta0 = theta0 * 180 / np.pi

        dfFile = dfDivisions[dfDivisions["Filename"] == filename]
        t0 = util.findStartTime(filename)
        t2 = int(timeStep / 2 * (int(T / timeStep) + 1) - t0 / 2)

        for theta in range(count.shape[1]):
            df1 = dfFile[dfFile["Theta"] > thetaStep * theta]
            df = df1[df1["Theta"] <= thetaStep * (theta + 1)]
            count[k, theta] = len(df)

        inPlane = 1 - (
            sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype(int)[:t2]
            / 255
        )
        angle = (
            sm.io.imread(f"dat/{filename}/angle{filename}.tif").astype(int)[:t2]
        ) - theta0
        angle = angle % 360

        for theta in range(area.shape[1]):
            area[k, theta] = (
                np.sum(
                    inPlane[
                        (angle > thetaStep * theta) & (angle <= thetaStep * (theta + 1))
                    ]
                )
                * scale ** 2
            )
            # test = np.zeros([t2,512,512])
            # test[(dist > rStep * r) & (dist <= rStep * (r + 1))] = 1
            # test = np.asarray(test, "uint8")
            # tifffile.imwrite(f"results/test.tif", test)

    radius = []
    dd = []
    std = []
    for theta in range(area.shape[1]):
        _area = area[:, theta][area[:, theta] > 0]
        _count = count[:, theta][area[:, theta] > 0]
        if len(_area) > 0:
            _dd, _std = weighted_avg_and_std(_count / _area, _area)
            dd.append(_dd)
            std.append(_std)
            radius.append(theta * thetaStep + thetaStep / 2)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.errorbar(radius, dd, yerr=std, marker="o")
    ax.set(xlabel=r"$\theta$", ylabel=r"Divison density ($\mu m^{-2}$)")
    ax.title.set_text(f"Divison density with theta from wound {fileType}")
    # ax.set_ylim([0, 0.0007])

    fig.savefig(
        f"results/Divison density with theta {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")


# Divison density STARIMA
if False:
    weights = np.zeros([7, 7])
    for r in range(7 - 1):
        weights[r, r] = 1 / 3
        weights[r, r + 1] = 1 / 3
        weights[r + 1, r] = 1 / 3

    weights[6, 6] = 1 / 3

    fileType = "Unwound"
    filenames = util.getFilesOfType(fileType)
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
                            (dist[t1:t2] > rStep * r) & (dist[t1:t2] <= rStep * (r + 1))
                        ]
                    )
                    * scale ** 2
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

    dd[sumArea < 8000] = np.nan
    dd = dd * 10000 * timeStep

    dd = np.nan_to_num(dd[:, 1:8])

    diff_dd = dd[:-1] - dd[1:]

    timeLag = 15

    stacf = stacf_stpacf.Stacf(dd, weights, timeLag)
    stacf.estimate()
    acf = stacf.get()

    stacf = stacf_stpacf.Stacf(diff_dd, weights, timeLag)
    stacf.estimate()
    acfd1 = stacf.get()
    sigLine = np.zeros(timeLag + 1) + 1 / (dd.shape[0]) ** 0.5

    acf_1 = np.ones(timeLag + 1)
    acf_1[1:] = acf[:, 0]
    acfd1_1 = np.ones(timeLag + 1)
    acfd1_1[1:] = acfd1[:, 0]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(np.linspace(0, timeLag, timeLag + 1), acf_1, marker="o")
    ax[0].plot(
        np.linspace(0, timeLag, timeLag + 1),
        sigLine,
        "--",
        color="green",
    )
    ax[0].plot(
        np.linspace(0, timeLag, timeLag + 1),
        -sigLine,
        "--",
        color="green",
    )
    ax[0].set(xlabel="time lags", ylabel="acf")
    ax[0].title.set_text(f"Space Time Autocorrelation Function")
    ax[0].set_ylim([-1, 1])

    sigLine = np.zeros(timeLag + 1) + 1 / (diff_dd.shape[0]) ** 0.5

    ax[1].plot(np.linspace(0, timeLag, timeLag + 1), acfd1_1, marker="o")
    ax[1].plot(
        np.linspace(0, timeLag, timeLag + 1),
        sigLine,
        "--",
        color="green",
    )
    ax[1].plot(
        np.linspace(0, timeLag, timeLag + 1),
        -sigLine,
        "--",
        color="green",
    )
    ax[1].set(xlabel="time lags", ylabel="acf")
    ax[1].title.set_text(f"Space Time Autocorrelation Function")
    ax[1].set_ylim([-1, 1])

    fig.savefig(
        f"results/Space Time Autocorrelation Function",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

    # print("------- Start STARIMA -------")
    # model = starma_model.STARIMA(0, 0, (1,), dd, weights)
    # model.fit()
    # print(np.sum(model.get_item("residuals")))
    # print(model.get_item("phi"))
    # print(model.get_item("theta"))
    # print(model.get_item("sigma2") ** 0.5)

    # model = starma_model.STARIMA(1, 0, (1,), dd, weights)
    # model.fit()
    # print(np.sum(model.get_item("residuals")))
    # print(model.get_item("phi"))
    # print(model.get_item("theta"))
    # print(model.get_item("sigma2") ** 0.5)

    # model = starma_model.STARIMA(0, 1, (1,), dd, weights)
    # model.fit()
    # print(np.sum(model.get_item("residuals")))
    # print(model.get_item("phi"))
    # print(model.get_item("theta"))
    # print(model.get_item("sigma2") ** 0.5)

    # model = starma_model.STARIMA(1, 1, (1,), dd, weights)
    # model.fit()
    # print(np.sum(model.get_item("residuals")))
    # print(model.get_item("phi"))
    # print(model.get_item("theta"))
    # print(model.get_item("sigma2") ** 0.5)

    model = starma_model.STARIMA(0, 1, (1,), dd, weights)
    model.fit()

    fileType = "WoundL"
    filenames = util.getFilesOfType(fileType)
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
                            (dist[t1:t2] > rStep * r) & (dist[t1:t2] <= rStep * (r + 1))
                        ]
                    )
                    * scale ** 2
                )

    ddWS = np.zeros([int(T / timeStep), int(R / rStep)])
    std = np.zeros([int(T / timeStep), int(R / rStep)])
    sumArea = np.zeros([int(T / timeStep), int(R / rStep)])

    for r in range(area.shape[2]):
        for t in range(area.shape[1]):
            _area = area[:, t, r][area[:, t, r] > 800]
            _count = count[:, t, r][area[:, t, r] > 800]
            if len(_area) > 0:
                _dd, _std = weighted_avg_and_std(_count / _area, _area)
                ddWS[t, r] = _dd
                std[t, r] = _std
                sumArea[t, r] = np.sum(_area)
            else:
                ddWS[t, r] = np.nan
                std[t, r] = np.nan

    ddWS[sumArea < 8000] = np.nan
    ddWS = ddWS * 10000 * timeStep

    std = model.get_item("sigma2") ** 0.5
    print(std)

    ddWS = np.nan_to_num(ddWS[:, 1:8])

    sifdd = ddWS[1:] - dd[:-1]
    from scipy.stats import norm

    t, r = np.mgrid[timeStep:T:timeStep, rStep : int(R - 3 * rStep) : rStep]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(3):
        pdd = norm.cdf(sifdd / std)

        pdd[pdd > 0.5] = 1 - pdd[pdd > 0.5]
        if i > 0:
            pdd[pdd > 0.1 ** i] = 1
            pdd[pdd <= 0.1 ** i] = 0
            c = ax[i].pcolor(
                t,
                r,
                pdd,
            )
            fig.colorbar(c, ax=ax[i])
            ax[i].set(xlabel="Time (mins)", ylabel=r"$R (\mu m)$")
            ax[i].title.set_text(f"Division density change sigf {round(0.1**i,2)}")

        else:
            ax[i].title.set_text(f"Division density change p value")
            c = ax[i].pcolor(
                t,
                r,
                pdd,
                norm=colors.LogNorm(vmin=pdd.min(), vmax=pdd.max()),
            )
            fig.colorbar(c, ax=ax[i])
            ax[i].set(xlabel="Time (mins)", ylabel=r"$R (\mu m)$")

    fig.savefig(
        f"results/Division density change sigf {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")