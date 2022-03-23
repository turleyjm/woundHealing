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

if True:
    _df2 = []
    for filename in filenames:

        df = pd.read_pickle(f"dat/{filename}/shape{filename}.pkl")
        Q = np.mean(df["q"])
        theta = np.arctan2(Q[0, 1], Q[0, 0]) / 2
        R = util.rotation_matrix(-theta)
        dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
        dist = sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)
        t0 = util.findStartTime(filename)

        for t in range(T):
            dft = df[df["Time"] == t]
            Q = np.matmul(R, np.matmul(np.mean(dft["q"]), np.matrix.transpose(R)))
            P = np.matmul(R, np.mean(dft["Polar"]))
            xw, yw = dfWound["Position"].iloc[t]

            for i in range(len(dft)):
                x = dft["Centroid"].iloc[i][0]
                y = dft["Centroid"].iloc[i][1]
                r = dist[t, int(512 - y), int(x)]
                phi = np.arctan2(y - yw, x - xw)
                Rw = util.rotation_matrix(-phi)

                q = np.matmul(R, np.matmul(dft["q"].iloc[i], np.matrix.transpose(R)))
                dq = q - Q
                dq = np.matmul(Rw, np.matmul(dq, np.matrix.transpose(Rw)))

                A = dft["Area"].iloc[i] * scale ** 2
                dp = np.matmul(R, dft["Polar"].iloc[i]) - P
                dp = np.matmul(Rw, np.matmul(dp, np.matrix.transpose(Rw)))

                _df2.append(
                    {
                        "Filename": filename,
                        "T": int(2 * t + t0),
                        "X": x * scale,
                        "Y": y * scale,
                        "R": r * scale,
                        "Phi": phi,
                        "Area": A,
                        "dq": dq,
                        "dp": dp,
                    }
                )

    dfShape = pd.DataFrame(_df2)
    dfShape.to_pickle(f"databases/dfShapeWound{fileType}.pkl")


if False:
    dfShape = pd.read_pickle(f"databases/dfShapeWound{fileType}.pkl")
    time = []
    dQ1 = []
    dQ1_std = []
    for i in range(10):
        dft = dfShape[(dfShape["T"] >= 10 * i) & (dfShape["T"] < 10 * (i + 1))]
        dQ = np.mean(dft["dq"][dft["R"] < 20], axis=0)
        dQ1.append(dQ[0, 0])
        dQ_std = np.std(np.array(dft["dq"][dft["R"] < 20]), axis=0)
        dQ1_std.append(dQ_std[0, 0] / (len(dft)) ** 0.5)
        time.append(10 * i + 5)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.errorbar(time, dQ1, dQ1_std, marker="o")
    ax.set(xlabel="Time (min)", ylabel=r"$\delta Q^{(1)}$")
    ax.title.set_text(
        r"$\delta Q^{(1)}$ Close to the Wound Edge with Time" + f" {fileType}"
    )
    ax.set_ylim([-0.02, 0.004])

    fig.savefig(
        f"results/dQ1 Close to the Wound Edge {fileType}",
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

        dfShape = pd.read_pickle(f"databases/dfShapeWound{fileType}.pkl")
        time = []
        dQ1 = []
        dQ1_std = []
        for i in range(10):
            dft = dfShape[(dfShape["T"] >= 10 * i) & (dfShape["T"] < 10 * (i + 1))]
            dQ = np.mean(dft["dq"][dft["r"] < 20], axis=0)
            dQ1.append(dQ[0, 0])
            dQ_std = np.std(np.array(dft["dq"][dft["r"] < 20]), axis=0)
            dQ1_std.append(dQ_std[0, 0] / (len(dft)) ** 0.5)
            time.append(10 * i + 5)

        ax.plot(time, dQ1, marker="o", label=f"{fileType}")

    ax.set(xlabel="Time (min)", ylabel="Elongation Towards the Wound")
    ax.title.set_text("Elongation Close to the Wound Edge with Time")
    ax.set_ylim([-0.02, 0.004])
    ax.legend()

    fig.savefig(
        f"results/dQ1 Close to the Wound Edge Compare",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")


# Divison density with distance from wound edge and time
if True:
    T = 160
    timeStep = 10
    R = 110
    rStep = 10
    count = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
    area = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
    dfShape = pd.read_pickle(f"databases/dfShapeWound{fileType}.pkl")
    for k in range(len(filenames)):
        filename = filenames[k]
        dfFile = dfShape[dfShape["Filename"] == filename]
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

    t, r = np.mgrid[0:T:timeStep, 0:R:rStep]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    c = ax.pcolor(
        t,
        r,
        dd,
        vmin=0.04,
        vmax=0.08,
        cmap="Reds",
    )
    fig.colorbar(c, ax=ax)
    ax.set(xlabel="Time (min)", ylabel=r"$R (\mu m)$")
    ax.title.set_text(f"Density distance and time {fileType}")

    fig.savefig(
        f"results/Density heatmap {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")