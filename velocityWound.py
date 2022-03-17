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

        dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
        dist = sm.io.imread(f"dat/{filename}/distanceWound{filename}.tif").astype(int)
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

                _df2.append(
                    {
                        "Filename": filename,
                        "T": int(2 * t + t0),
                        "X": x * scale,
                        "Y": y * scale,
                        "r": r * scale,
                        "Phi": phi,
                        "v": -v * scale,
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
        dQ = np.mean(dft["v"][dft["r"] < 20], axis=0)
        dv1.append(dQ[0])
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
if True:
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
            dQ = np.mean(dft["v"][dft["r"] < 20], axis=0)
            dv1.append(dQ[0])
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