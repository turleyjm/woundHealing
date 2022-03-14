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
import findGoodCells as fi
import utils as util

plt.rcParams.update({"font.size": 16})

# -------------------

filenames, fileType = util.getFilesType()
scale = 123.26 / 512
T = 160
timeStep = 10
R = 80
rStep = 10


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

if False:
    _df = []
    for filename in filenames:
        dfDivision = pd.read_pickle(f"dat/{filename}/dfDivision{filename}.pkl")
        t0 = findStartTime(filename)
        if "Wound" in filename:
            dist = sm.io.imread(f"dat/{filename}/distanceWound{filename}.tif").astype(
                int
            )
            for i in range(len(dfDivision)):
                t = dfDivision["T"].iloc[i]
                x = dfDivision["X"].iloc[i]
                y = dfDivision["Y"].iloc[i]
                r = dist[t, x, 512 - y]
                _df.append(
                    {
                        "Filename": filename,
                        "Label": dfDivision["Label"].iloc[i],
                        "T": int(t0 + t * 2),  # frames are taken every t2 minutes
                        "X": x,
                        "Y": y,
                        "R": r * scale,
                        "Orientation": dfDivision["Orientation"].iloc[i],
                    }
                )
        else:
            for i in range(len(dfDivision)):
                _df.append(
                    {
                        "Filename": filename,
                        "Label": dfDivision["Label"].iloc[i],
                        "T": int(
                            t0 + dfDivision["T"].iloc[i] * 2
                        ),  # frames are taken every t2 minutes
                        "X": dfDivision["X"].iloc[i],
                        "Y": dfDivision["Y"].iloc[i],
                        "Orientation": dfDivision["Orientation"].iloc[i],
                    }
                )

    dfDivisions = pd.DataFrame(_df)
    dfDivisions.to_pickle(f"databases/dfDivisions{fileType}.pkl")


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
    dat_dd = []
    total = 0
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
                dd.append(_dd)
                std.append(_std)
                time.append(t * 10 + timeStep / 2)

        ax.plot(time, dd, label=f"{fileType}", marker="o")

    ax.set(xlabel="Time", ylabel=r"Divison density ($\mu m^{-2}$)")
    ax.title.set_text(f"Divison density with time")
    ax.set_ylim([0, 0.00053])
    ax.legend()

    fig.savefig(
        f"results/Compared divison density with time",
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
            sm.io.imread(f"dat/{filename}/distanceWound{filename}.tif").astype(int)[:t2]
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
                sm.io.imread(f"dat/{filename}/distanceWound{filename}.tif").astype(int)[
                    :t2
                ]
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


if True:
    count = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
    area = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    for k in range(len(filenames)):
        filename = filenames[k]
        dfFile = dfDivisions[dfDivisions["Filename"] == filename]
        t0 = util.findStartTime(filename)
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
            sm.io.imread(f"dat/{filename}/distanceWound{filename}.tif").astype(int)[:t2]
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

    t, r = np.mgrid[0:T:timeStep, 0:R:rStep]
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    c = ax.pcolor(
        t,
        r,
        dd,
        vmin=-0.0004,
        vmax=0.0004,
        cmap="RdBu_r",
    )
    fig.colorbar(c, ax=ax)
    ax.set(xlabel="Time (min)", ylabel=r"$R (\mu m)$")
    ax.title.set_text(f"Change in divison density distance and time {fileType}")

    fig.savefig(
        f"results/Divison density heatmap {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")