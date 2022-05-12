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

plt.rcParams.update({"font.size": 14})

# -------------------


def weighted_avg_and_std(values, weight, axis=0):
    average = np.average(values, weights=weight, axis=axis)
    variance = np.average((values - average) ** 2, weights=weight, axis=axis)
    return average, np.sqrt(variance)


# -------------------

filenames, fileType = util.getFilesType()
scale = 123.26 / 512
T = 90

if False:
    _df2 = []
    for filename in filenames:

        dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
        dist = sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)
        t0 = util.findStartTime(filename)
        df = pd.read_pickle(f"dat/{filename}/nucleusVelocity{filename}.pkl")

        for t in range(T):
            dft = df[df["T"] == t]
            xw, yw = dfWound["Position"].iloc[t]
            V = np.mean(dft["Velocity"])

            for i in range(len(dft)):
                x = dft["X"].iloc[i]
                y = dft["Y"].iloc[i]
                r = dist[t, int(x), int(y)]
                phi = np.arctan2(y - yw, x - xw)
                R = util.rotation_matrix(-phi)

                v = np.matmul(R, dft["Velocity"].iloc[i]) / 2
                dv = np.matmul(R, dft["Velocity"].iloc[i] - V) / 2

                _df2.append(
                    {
                        "Filename": filename,
                        "T": int(2 * t + t0),  # frames are taken every 2 minutes
                        "X": x * scale,
                        "Y": y * scale,
                        "R": r * scale,
                        "Phi": phi,
                        "v": -v * scale,
                        "dv": -dv * scale,
                    }
                )

    dfVelocity = pd.DataFrame(_df2)
    dfVelocity.to_pickle(f"databases/dfVelocityWound{fileType}.pkl")


if False:
    dfVelocity = pd.read_pickle(f"databases/dfVelocityWound{fileType}.pkl")
    time = []
    dv1 = []
    dv1_std = []
    for i in range(1, 10):
        dft = dfVelocity[(dfVelocity["T"] >= 10 * i) & (dfVelocity["T"] < 10 * (i + 1))]
        dv = np.mean(dft["v"][dft["r"] < 20], axis=0)
        dv1.append(dv[0])
        dv_std = np.std(np.array(dft["v"][dft["r"] < 20]), axis=0)
        dv1_std.append(dv_std[0] / (len(dft)) ** 0.5)
        time.append(10 * i + 5)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.errorbar(time, dv1, dv1_std, marker="o")
    ax.set(xlabel="Time (min)", ylabel=r"Speed Towards Wound ($\mu/min$)")
    ax.title.set_text(f"Speed Towards Wound with Time {fileType}")
    ax.set_ylim([0, 0.11])

    fig.savefig(
        f"results/Velocity Close to the Wound Edge {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

# compare
if False:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    labels = ["WoundS", "WoundL"]
    for fileType in labels:

        dfVelocity = pd.read_pickle(f"databases/dfVelocityWound{fileType}.pkl")
        time = []
        dv1 = []
        dv1_std = []
        for i in range(1, 10):
            dft = dfVelocity[
                (dfVelocity["T"] >= 10 * i) & (dfVelocity["T"] < 10 * (i + 1))
            ]
            dv = np.mean(dft["v"][dft["r"] < 20], axis=0)
            dv1.append(dv[0])
            dv_std = np.std(np.array(dft["v"][dft["r"] < 20]), axis=0)
            dv1_std.append(dv_std[0] / (len(dft)) ** 0.5)
            time.append(10 * i + 5)

        ax.plot(time, dv1, marker="o", label=f"{fileType}")

    ax.set(xlabel=r"Time ($min$)", ylabel=r"Speed Towards Wound ($\mu/min$)")
    ax.title.set_text(f"Speed Towards Wound with Time")
    ax.set_ylim([0, 0.11])
    ax.legend()

    fig.savefig(
        f"results/Compare Velocity Close to the Wound Edge",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")


# v with distance from wound edge and time
if True:
    T = 160
    timeStep = 10
    R = 110
    rStep = 10
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
                            (dist[t1:t2] > rStep * r) & (dist[t1:t2] <= rStep * (r + 1))
                        ]
                    )
                    * scale ** 2
                )

    V1 = np.zeros([int(T / timeStep), int(R / rStep)])
    std = np.zeros([int(T / timeStep), int(R / rStep)])
    meanArea = np.zeros([int(T / timeStep), int(R / rStep)])

    for r in range(area.shape[2]):
        for t in range(area.shape[1]):
            _V1 = v1[:, t, r][v1[:, t, r] != 0]
            _area = area[:, t, r][v1[:, t, r] != 0]
            if len(_area) > 0:
                if np.sum(_area) > 0:
                    _dd, _std = weighted_avg_and_std(_V1, _area)
                    V1[t, r] = _dd
                    std[t, r] = _std
                    meanArea[t, r] = np.mean(_area)
                else:
                    V1[t, r] = np.nan
                    std[t, r] = np.nan
            else:
                V1[t, r] = np.nan
                std[t, r] = np.nan

    V1[meanArea < 600] = np.nan
    V1 = V1 * timeStep

    t, r = np.mgrid[0:T:timeStep, 0:R:rStep]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    c = ax.pcolor(
        t,
        r,
        V1,
        vmin=-2,
        vmax=2,
        cmap="RdBu_r",
    )
    fig.colorbar(c, ax=ax)
    ax.set(xlabel="Time (min)", ylabel=r"$R (\mu m)$")
    ax.title.set_text(r"$\delta v_1$" + f" distance and time {fileType}")

    fig.savefig(
        f"results/v heatmap {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")