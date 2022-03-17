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

plt.rcParams.update({"font.size": 14})

# -------------------

filenames, fileType = util.getFilesType()
scale = 123.26 / 512
T = 90

if False:
    _df2 = []
    for filename in filenames:

        df = pd.read_pickle(f"dat/{filename}/shape{filename}.pkl")
        Q = np.mean(df["q"])
        theta = np.arctan2(Q[0, 1], Q[0, 0]) / 2
        R = util.rotation_matrix(-theta)
        dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
        dist = sm.io.imread(f"dat/{filename}/distanceWound{filename}.tif").astype(int)
        t0 = util.findStartTime(filename)

        for t in range(T):
            dft = df[df["Time"] == t]
            Q = np.matmul(R, np.matmul(np.mean(dft["q"]), np.matrix.transpose(R)))
            P = np.matmul(R, np.mean(dft["Polar"]))
            xw, yw = dfWound["Position"].iloc[t]

            for i in range(len(dft)):
                x = dft["Centroid"].iloc[i][0]
                y = dft["Centroid"].iloc[i][1]
                r = dist[t, int(x), int(y)]
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
                        "r": r * scale,
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
        dQ = np.mean(dft["dq"][dft["r"] < 20], axis=0)
        dQ1.append(dQ[0, 0])
        dQ_std = np.std(np.array(dft["dq"][dft["r"] < 20]), axis=0)
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
if True:
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

    ax.set(xlabel="Time (min)", ylabel=r"$\delta Q^{(1)}$")
    ax.title.set_text(r"$\delta Q^{(1)}$ Close to the Wound Edge with Time")
    ax.set_ylim([-0.02, 0.004])
    ax.legend()

    fig.savefig(
        f"results/dQ1 Close to the Wound Edge Compare",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")