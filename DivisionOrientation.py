import os
import shutil
from math import dist, floor, log10

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


# -------------------

if False:
    _df = []
    for filename in filenames:
        dfDivision = pd.read_pickle(f"dat/{filename}/dfDivision{filename}.pkl")
        dfShape = pd.read_pickle(f"dat/{filename}/shape{filename}.pkl")
        Q = np.mean(dfShape["q"])
        theta0 = np.arccos(Q[0, 0] / (Q[0, 0] ** 2 + Q[0, 1] ** 2) ** 0.5) / 2

        t0 = util.findStartTime(filename)
        if "Wound" in filename:
            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
            dist = sm.io.imread(f"dat/{filename}/distanceWound{filename}.tif").astype(
                int
            )
            for i in range(len(dfDivision)):
                t = dfDivision["T"].iloc[i]
                (x_w, y_w) = dfWound["Position"].iloc[t]
                x = dfDivision["X"].iloc[i]
                y = dfDivision["Y"].iloc[i]
                ori = dfDivision["Orientation"].iloc[i]
                ori = (ori - np.arctan2(y - y_w, x - x_w) * 180 / np.pi) % 180
                if ori > 90:
                    ori = 180 - ori
                r = dist[t, x, 512 - y]
                _df.append(
                    {
                        "Filename": filename,
                        "Label": dfDivision["Label"].iloc[i],
                        "T": int(t0 + t * 2),  # frames are taken every 2 minutes
                        "X": x,
                        "Y": y,
                        "R": r * scale,
                        "Orientation": ori,
                    }
                )
        else:
            for i in range(len(dfDivision)):
                t = dfDivision["T"].iloc[i]
                ori = (dfDivision["Orientation"].iloc[i] - theta0 * 180 / np.pi) % 180
                if ori > 90:
                    ori = 180 - ori
                _df.append(
                    {
                        "Filename": filename,
                        "Label": dfDivision["Label"].iloc[i],
                        "T": int(t0 + t * 2),  # frames are taken every t2 minutes
                        "X": dfDivision["X"].iloc[i],
                        "Y": dfDivision["Y"].iloc[i],
                        "Orientation": ori,
                    }
                )

    dfDivisions = pd.DataFrame(_df)
    dfDivisions.to_pickle(f"databases/dfDivisions{fileType}.pkl")


# Divison orientation with time Unwound

if False:
    count = np.zeros([len(filenames), int(T / timeStep)])
    orientation = np.zeros([len(filenames), int(T / timeStep)])
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    for k in range(len(filenames)):
        filename = filenames[k]
        t0 = util.findStartTime(filename)
        dfFile = dfDivisions[dfDivisions["Filename"] == filename]

        for t in range(count.shape[1]):
            df1 = dfFile[dfFile["T"] > timeStep * t]
            df = df1[df1["T"] <= timeStep * (t + 1)]

            count[k, t] = len(df)
            orientation[k, t] = np.mean(df["Orientation"])

    time = []
    dd = []
    std = []
    for t in range(count.shape[1]):
        _orientation = orientation[:, t][count[:, t] > 0]
        _count = count[:, t][count[:, t] > 0]
        if len(_count) > 0:
            _dd, _std = weighted_avg_and_std(_orientation, _count)
            dd.append(_dd)
            std.append(_std)
            time.append(t * 10 + timeStep / 2)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.errorbar(time, dd, yerr=std)
    if "Wound" in filename:
        ax.set(xlabel="Time", ylabel="Divison orientation towards wound")
    else:
        ax.set(xlabel="Time", ylabel="Divison orientation")
    ax.title.set_text(f"Divison orientation with time {fileType}")
    ax.set_ylim([0, 90])

    fig.savefig(
        f"results/Divison orientation with time {fileType}",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# Compare divison density with time

if False:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    labels = ["WoundS", "WoundL"]
    for fileType in labels:
        filenames = util.getFilesOfType(fileType)
        count = np.zeros([len(filenames), int(T / timeStep)])
        orientation = np.zeros([len(filenames), int(T / timeStep)])
        dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
        for k in range(len(filenames)):
            filename = filenames[k]
            t0 = util.findStartTime(filename)
            dfFile = dfDivisions[dfDivisions["Filename"] == filename]

            for t in range(count.shape[1]):
                df1 = dfFile[dfFile["T"] > timeStep * t]
                df = df1[df1["T"] <= timeStep * (t + 1)]

                count[k, t] = len(df)
                orientation[k, t] = np.mean(df["Orientation"])

        time = []
        dd = []
        std = []
        for t in range(count.shape[1]):
            _orientation = orientation[:, t][count[:, t] > 0]
            _count = count[:, t][count[:, t] > 0]
            if len(_count) > 0:
                _dd, _std = weighted_avg_and_std(_orientation, _count)
                dd.append(_dd)
                std.append(_std)
                time.append(t * 10 + timeStep / 2)

        ax.plot(time, dd, label=f"{fileType}", marker="o")

    if "Wound" in filename:
        ax.set(xlabel="Time", ylabel="Divison orientation towards wound")
    else:
        ax.set(xlabel="Time", ylabel="Divison orientation")
    ax.title.set_text(f"Divison orientation with time")
    ax.set_ylim([0, 90])
    ax.legend()

    fig.savefig(
        f"results/Compared divison orientation with time",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


# Divison density with distance from wound edge

if False:
    count = np.zeros([len(filenames), int(R / rStep)])
    orientation = np.zeros([len(filenames), int(R / rStep)])
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
            orientation[k, r] = np.mean(df["Orientation"])

    radius = []
    dd = []
    std = []
    for r in range(count.shape[1]):
        _orientation = orientation[:, r][count[:, r] > 0]
        _count = count[:, r][count[:, r] > 0]
        if len(_count) > 0:
            _dd, _std = weighted_avg_and_std(_orientation, _count)
            dd.append(_dd)
            std.append(_std)
            radius.append(r * 10 + rStep / 2)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.errorbar(radius, dd, yerr=std)
    if "Wound" in filename:
        ax.set(xlabel="Time", ylabel="Divison orientation towards wound")
    else:
        ax.set(xlabel="Time", ylabel="Divison orientation")
    ax.title.set_text(f"Divison orientation with distance from wound {fileType}")
    ax.set_ylim([0, 90])

    fig.savefig(
        f"results/Divison orientation with distance {fileType}",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


# Compare divison density with distance from wound edge

if True:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    labels = ["WoundS", "WoundL"]
    for fileType in labels:
        filenames = util.getFilesOfType(fileType)
        count = np.zeros([len(filenames), int(R / rStep)])
        orientation = np.zeros([len(filenames), int(R / rStep)])
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
                orientation[k, r] = np.mean(df["Orientation"])

        radius = []
        dd = []
        std = []
        for r in range(count.shape[1]):
            _orientation = orientation[:, r][count[:, r] > 0]
            _count = count[:, r][count[:, r] > 0]
            if len(_count) > 0:
                _dd, _std = weighted_avg_and_std(_orientation, _count)
                dd.append(_dd)
                std.append(_std)
                radius.append(r * 10 + rStep / 2)

        ax.plot(radius, dd, label=f"{fileType}", marker="o")

    if "Wound" in filename:
        ax.set(xlabel="Time", ylabel="Divison orientation towards wound")
    else:
        ax.set(xlabel="Time", ylabel="Divison orientation")
    ax.set_ylim([0, 90])
    ax.title.set_text(f"Divison orientation with distance from wound")
    ax.legend()

    fig.savefig(
        f"results/Compared divison orientation with distance",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


# Divison density with distance from wound edge and time


if False:
    count = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
    orientation = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
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
                orientation[k, t, r] = np.mean(df["Orientation"])

    dd = np.zeros([int(T / timeStep), int(R / rStep)])
    std = np.zeros([int(T / timeStep), int(R / rStep)])

    for r in range(count.shape[2]):
        for t in range(count.shape[1]):
            _orientation = orientation[:, t, r][count[:, t, r] > 0]
            _count = count[:, t, r][count[:, t, r] > 0]
            if len(_orientation) > 0:
                _dd, _std = weighted_avg_and_std(_orientation, _count)
                dd[t, r] = _dd
                std[t, r] = _std
            else:
                dd[t, r] = np.nan
                std[t, r] = np.nan

    t, r = np.mgrid[0:T:timeStep, 0:R:rStep]
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    c = ax.pcolor(
        t,
        r,
        dd,
        cmap="RdBu_r",
        vmin=0,
        vmax=90,
        shading="auto",
    )
    fig.colorbar(c, ax=ax)
    ax.set(xlabel="Time (min)", ylabel=r"$R (\mu m)$")
    ax.title.set_text(f"Divison orientation distance and time {fileType}")

    fig.savefig(
        f"results/Divison orientation heatmap {fileType}",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")