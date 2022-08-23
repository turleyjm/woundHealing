import os
from pickle import FALSE
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
from scipy import stats
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

plt.rcParams.update({"font.size": 12})

# -------------------

filenames, fileType = util.getFilesType()
scale = 123.26 / 512
T = 180
timeStep = 8
R = 60
rStep = 10


def weighted_avg_and_std(values, weight, axis=0):
    average = np.average(values, weights=weight, axis=axis)
    variance = np.average((values - average) ** 2, weights=weight, axis=axis)
    return average, np.sqrt(variance)


def HolmBonferroni(df, alpha):

    df = df.sort_values(by=["P-value"])
    n = len(df)
    for i in range(n):
        p = df["P-value"].iloc[i]
        if p < alpha / (n - i):
            df["Sig"].iloc[i] = star(p * (n - i))
        else:
            df["Sig"].iloc[i] = False

    df = df.sort_values(by=["R"])

    return df


def star(p):

    if p >= 0.05:
        return 0
    else:
        return int(-np.log10(p))


# -------------------


# Divison orientation with respect to tissue over time
if False:
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")

    count = np.zeros([int(T / timeStep), 10])
    for t in range(int(T / timeStep)):
        df1 = dfDivisions[dfDivisions["T"] > timeStep * t]
        df = df1[df1["T"] <= timeStep * (t + 1)]
        for i in range(len(df)):
            ori = df["Orientation"].iloc[i]
            count[t, int(ori / 9)] += 1

    t, r = np.mgrid[0:T:timeStep, 0:100:10]
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    c = ax.pcolor(
        t,
        r,
        count,
        vmin=0,
        vmax=45,
    )
    fig.colorbar(c, ax=ax)
    ax.set(xlabel="Time (mins)", ylabel="Orientation")
    ax.title.set_text(
        f"Divison orientation with respect to tissue over time {fileType}"
    )

    fig.savefig(
        f"results/Divison orientation with respect to tissue over time {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")


# Divison orientation with respect to tissue
if False:
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.hist(dfDivisions["Orientation"])
    ax.set(xlabel="Orientation", ylabel="Number of Divisions")
    ax.title.set_text(f"Divison orientation with respect to tissue {fileType}")

    fig.savefig(
        f"results/Divison orientation with respect to tissue {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")


# Divison orientation with respect to a wound over time
if False:
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")

    count = np.zeros([int(T / timeStep), 10])
    for t in range(int(T / timeStep)):
        df1 = dfDivisions[dfDivisions["T"] > timeStep * t]
        df = df1[df1["T"] <= timeStep * (t + 1)]
        for i in range(len(df)):
            ori = df["Orientation Wound"].iloc[i]
            count[t, int(ori / 9)] += 1

    t, r = np.mgrid[0:T:timeStep, 0:100:10]
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    c = ax.pcolor(
        t,
        r,
        count,
        vmin=0,
        vmax=45,
    )
    fig.colorbar(c, ax=ax)
    ax.set(xlabel="Time (mins)", ylabel="Orientation")
    ax.title.set_text(
        f"Divison orientation with respect to a wound over time {fileType}"
    )

    fig.savefig(
        f"results/Divison orientation with respect to a wound over time {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")


# Divison orientation with respect to a wound
if False:
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.hist(dfDivisions["Orientation Wound"])
    ax.set(xlabel="Orientation", ylabel="Number of Divisions")
    ax.title.set_text(f"Divison orientation with respect to a wound {fileType}")

    fig.savefig(
        f"results/Divison orientation with respect to a wound {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")


# Divison orientation with respect to a wound over distance from wound
if False:
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")

    count = np.zeros([int(R / rStep), 10])
    rad = []
    mu = []
    std = []
    for r in range(int(R / rStep)):
        df1 = dfDivisions[dfDivisions["R"] > rStep * r]
        df = df1[df1["R"] <= rStep * (r + 1)]
        mu.append(np.mean(df["Orientation Wound"]))
        std.append(np.std(df["Orientation Wound"]))
        rad.append(rStep * r + rStep / 2)
        for i in range(len(df)):
            ori = df["Orientation Wound"].iloc[i]
            count[r, int(ori / 9)] += 1

    t, r = np.mgrid[0:R:rStep, 0:100:10]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    c = ax[0].pcolor(
        t,
        r,
        count,
        vmin=0,
        vmax=80,
    )
    fig.colorbar(c, ax=ax[0])
    ax[0].set(xlabel=r"Distance from wound ($\mu m^{-2}$)", ylabel="Orientation")

    ax[1].errorbar(rad, mu, yerr=std)
    ax[1].set_ylim([0, 90])
    ax[1].set(xlabel=r"Distance from wound ($\mu m^{-2}$)", ylabel="Orientation")

    fig.suptitle(
        f"Divison orientation with respect to a wound over distance from wound {fileType}"
    )

    fig.savefig(
        f"results/Divison orientation with respect to a wound over distance from wound {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")


# Divison orientation with respect to a wound over distance from wound
if False:
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")

    count = np.zeros([int(R / rStep), 10])
    rad = []
    mu = []
    std = []
    for r in range(int(R / rStep)):
        df1 = dfDivisions[dfDivisions["R"] > rStep * r]
        df2 = df1[df1["T"] < 50]
        df = df2[df2["R"] <= rStep * (r + 1)]
        mu.append(np.mean(df["Orientation Wound"]))
        std.append(np.std(df["Orientation Wound"]))
        rad.append(rStep * r + rStep / 2)
        for i in range(len(df)):
            ori = df["Orientation Wound"].iloc[i]
            count[r, int(ori / 9)] += 1

    t, r = np.mgrid[0:R:rStep, 0:100:10]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    c = ax[0].pcolor(
        t,
        r,
        count,
        vmin=0,
        vmax=80,
    )
    fig.colorbar(c, ax=ax[0])
    ax[0].set(xlabel=r"Distance from wound ($\mu m^{-2}$)", ylabel="Orientation")

    ax[1].errorbar(rad, mu, yerr=std)
    ax[1].set_ylim([0, 90])
    ax[1].set(xlabel=r"Distance from wound ($\mu m^{-2}$)", ylabel="Orientation")

    fig.suptitle(
        f"Divison orientation with respect to a wound over distance from wound {fileType}"
    )

    fig.savefig(
        f"results/Divison orientation WRTW over distance from wound {fileType} short time",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")


# Divison orientation with respect to a wound over distance from wound t-tests
if False:
    alpha = 0.05
    dfDivisions = pd.read_pickle(f"databases/dfDivisionsUnwound.pkl")
    dfDivisionsS = pd.read_pickle(f"databases/dfDivisionsWoundS.pkl")
    dfDivisionsL = pd.read_pickle(f"databases/dfDivisionsWoundL.pkl")

    count = np.zeros([int(R / rStep), 10])
    rad = []
    mu = []
    std = []
    muS = []
    stdS = []
    muL = []
    stdL = []
    _dfttestS = []
    _dfttestL = []
    fig, ax = plt.subplots(3, 6, figsize=(30, 15))
    for r in range(int(R / rStep)):
        df1 = dfDivisions[dfDivisions["R"] > rStep * r]
        df = df1[df1["R"] <= rStep * (r + 1)]
        df1 = dfDivisionsS[dfDivisionsS["R"] > rStep * r]
        dfS = df1[df1["R"] <= rStep * (r + 1)]
        df1 = dfDivisionsL[dfDivisionsL["R"] > rStep * r]
        dfL = df1[df1["R"] <= rStep * (r + 1)]

        ax[0, r].hist(df["Orientation Wound"], 10)
        ax[0, r].set(xlabel=r"Orientation Wound", ylabel=r"number")
        ax[0, r].title.set_text(f"Unwound R={rStep * r}")

        ax[1, r].hist(dfS["Orientation Wound"], 10)
        ax[1, r].set(xlabel=r"Orientation Wound", ylabel=r"number")
        ax[1, r].title.set_text(f"Wound Small R={rStep * r}")

        ax[2, r].hist(dfL["Orientation Wound"], 10)
        ax[2, r].set(xlabel=r"Orientation Wound", ylabel=r"number")
        ax[2, r].title.set_text(f"Wound Large R={rStep * r}")

        mu.append(np.mean(df["Orientation Wound"]))
        std.append(np.std(df["Orientation Wound"]))
        muS.append(np.mean(dfS["Orientation Wound"]))
        stdS.append(np.std(dfS["Orientation Wound"]))
        muL.append(np.mean(dfL["Orientation Wound"]))
        stdL.append(np.std(dfL["Orientation Wound"]))
        rad.append(rStep * r + rStep / 2)

        _dfttestS.append(
            {
                "R": rStep * r,
                "P-value": stats.ttest_ind(
                    df["Orientation Wound"], dfS["Orientation Wound"]
                ).pvalue,
                "Sig": star(
                    stats.ttest_ind(
                        df["Orientation Wound"], dfS["Orientation Wound"]
                    ).pvalue
                ),
            }
        )
        _dfttestL.append(
            {
                "R": rStep * r,
                "P-value": stats.ttest_ind(
                    df["Orientation Wound"], dfL["Orientation Wound"]
                ).pvalue,
                "Sig": star(
                    stats.ttest_ind(
                        df["Orientation Wound"], dfL["Orientation Wound"]
                    ).pvalue
                ),
            }
        )

    fig.savefig(
        f"results/Divison orientation with respect to a wound",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

    dfttestS = pd.DataFrame(_dfttestS)
    dfttestL = pd.DataFrame(_dfttestL)

    dfttestS = HolmBonferroni(dfttestS, alpha)
    dfttestL = HolmBonferroni(dfttestL, alpha)

    print("Small")
    print(dfttestS)
    print("Large")
    print(dfttestL)

    rad = np.array(rad)
    # fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # ax[0].errorbar(rad, mu, yerr=std, label=f"Unwound")
    # ax[0].errorbar(rad + 1, muS, yerr=stdS, label=f"Small wound")
    # ax[0].set_ylim([0, 90])
    # ax[0].legend()
    # ax[0].set(xlabel=r"Distance from wound ($\mu m^{-2}$)", ylabel="Orientation")
    # ax[0].title.set_text(f"Small wound")

    # ax[1].errorbar(rad, mu, yerr=std, label=f"Unwound")
    # ax[1].errorbar(rad + 1, muL, yerr=stdL, label=f"Large wound")
    # ax[1].set_ylim([0, 90])
    # ax[1].legend()
    # ax[1].set(xlabel=r"Distance from wound ($\mu m^{-2}$)", ylabel="Orientation")
    # ax[1].title.set_text(f"Large wound")

    # fig.suptitle(
    #     f"Divison orientation with respect to a wound over distance from wound"
    # )

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    ax.errorbar(rad - 1, mu, yerr=std, label=f"Unwounded", marker="o")
    ax.errorbar(rad, muS, yerr=stdS, label=f"Small wound", marker="o")
    ax.errorbar(rad + 1, muL, yerr=stdL, label=f"Large wound", marker="o")
    ax.set_ylim([0, 90])
    ax.set_xlim([0, 60])

    ax.legend(loc="lower right")
    ax.set(xlabel=r"Distance from wound ($\mu m$)", ylabel="Mean orientation")
    # ax.title.set_text(
    #     f"Divison orientation with respect to a wound over distance from wound"
    # )

    fig.savefig(
        f"results/Divison orientation with respect to a wound over distance from wound t-test",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")


timeStep = 16
R = 80
rStep = 20


# Divison orientation with distance from wound edge and time
if False:
    ori = np.zeros([int(T / timeStep), int(R / rStep)])
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    for r in range(ori.shape[1]):
        for t in range(ori.shape[0]):
            df1 = dfDivisions[dfDivisions["T"] > timeStep * t]
            df2 = df1[df1["T"] <= timeStep * (t + 1)]
            df3 = df2[df2["R"] > rStep * r]
            df = df3[df3["R"] <= rStep * (r + 1)]
            ori[t, r] = np.mean(df["Orientation Wound"])

    t, r = np.mgrid[0:T:timeStep, 0:R:rStep]
    fig, ax = plt.subplots(1, 1, figsize=(6, 2))
    c = ax.pcolor(
        t,
        r,
        ori,
        vmin=70,
        vmax=20,
        cmap="RdBu_r",
    )
    fig.colorbar(c, ax=ax)
    ax.set(xlabel="Time (mins)", ylabel=r"$R (\mu m)$")
    # ax.title.set_text(f"Division ori {fileType}")

    fig.savefig(
        f"results/Division ori heatmap {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

# Divison orientation with distance from wound edge and time
if False:
    ori = np.zeros([int(T / timeStep), int(R / rStep)])
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    for filename in filenames:
        dfF = dfDivisions[dfDivisions["Filename"] == filename]
        for r in range(ori.shape[1]):
            for t in range(ori.shape[0]):
                df1 = dfF[dfF["T"] > timeStep * t]
                df2 = df1[df1["T"] <= timeStep * (t + 1)]
                df3 = df2[df2["R"] > rStep * r]
                df = df3[df3["R"] <= rStep * (r + 1)]
                ori[t, r] = np.mean(df["Orientation Wound"])

        t, r = np.mgrid[0:T:timeStep, 0:R:rStep]
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        c = ax.pcolor(
            t,
            r,
            ori,
            vmin=70,
            vmax=20,
            cmap="RdBu_r",
        )
        fig.colorbar(c, ax=ax)
        ax.set(xlabel="Time (mins)", ylabel=r"$R (\mu m)$")
        # ax.title.set_text(f"Division ori {fileType}")

        fig.savefig(
            f"results/Division ori heatmap {filename}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")


# Divison orientation with direction from wound
if True:
    ori = np.zeros([int(T / timeStep), int(R / rStep)])
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    dfDivisions = dfDivisions[dfDivisions["R"] < 60]
    for i in range(8):
        df1 = dfDivisions[dfDivisions["Theta"] > 45 * i]
        df = df1[df1["Theta"] < 45 * (i + 1)]

        j = i % 4
        k = i // 4
        ax[k, j].hist(df["Orientation Wound"], 9)
        ax[k, j].set(xlabel=r"Orientation Wound", ylabel=r"number")
        ax[k, j].title.set_text(f"{45*(i)}<" r"$\theta$" f"<{45*(i+1)}")
        ax[k, j].set_xlim([0, 90])

    fig.savefig(
        f"results/Division Orientation Wound direction {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")