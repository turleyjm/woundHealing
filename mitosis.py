import os
from math import floor, log10

import cv2
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg as linalg
import shapely
import skimage as sm
import skimage.io
import skimage.measure
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
import tifffile
from skimage.draw import circle_perimeter
from scipy import optimize

import cellProperties as cell
import findGoodCells as fi

plt.rcParams.update({"font.size": 20})
plt.ioff()
pd.set_option("display.width", 1000)

f = open("pythonText.txt", "r")

filenames = f.read()
filenames = filenames.split(", ")

T = 181

muAll = []

for filename in filenames:

    functionTitle = "Division time"

    df = pd.read_pickle(f"dat/{filename}/mitosisTracks{filename}.pkl")

    n = int(len(df) / 3)

    mu = []

    for i in range(n):
        prop = list(df["Time"][df["Chain"] == "parent"])[i][-1]
        mu.append(prop)

    fig, ax = plt.subplots()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xlabel("Time")
    ax.hist(mu, density=False, bins=18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim([0, 180])
    fig.savefig(
        f"results/{functionTitle} of {filename}", dpi=200, transparent=True,
    )

    plt.close("all")

    functionTitle = "Division Orientation"

    if "Wound" in filename:
        wound = True
    else:
        wound = False

    if wound == True:

        wound = sm.io.imread(f"dat/{filename}/woundsite{filename}.tif").astype("uint8")

        dist = np.zeros([T, 512, 512])
        for t in range(T):
            img = 255 - wound[t]
            img = sp.ndimage.morphology.distance_transform_edt(img)
            dist[t] = fi.imgrcxy(img)

        dfSub = df[df["Division While Wounded"] == "Y"]

        m = int(len(dfSub) / 3)

        distance = []
        mu = []

        for i in range(m):
            prop = list(dfSub[functionTitle][dfSub["Chain"] == "parent"])[i]
            (Cx, Cy) = dfSub["Position"][dfSub["Chain"] == "parent"].iloc[i][-1]
            t = list(dfSub["Time"][dfSub["Chain"] == "parent"])[i][-1]

            Cx = int(Cx)
            Cy = int(Cy)

            distance.append(dist[t, Cx, Cy])

            if prop > 90:
                prop = 180 - prop
            mu.append(prop)

            if dist[t, Cx, Cy] < 100:
                muAll.append(prop)

        fig, ax = plt.subplots()
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.xlabel("Degrees")
        ax.hist(mu, density=False, bins=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim([0, 90])
        fig.savefig(
            f"results/{functionTitle} of {filename}", dpi=200, transparent=True,
        )

        plt.close("all")

        fig, ax = plt.subplots()
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.xlabel("Distance")
        ax.hist(distance, density=False, bins=20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.savefig(
            f"results/Divsion Distance of {filename}", dpi=200, transparent=True,
        )

        plt.close("all")

    else:

        for i in range(n):
            prop = list(df[functionTitle][df["Chain"] == "parent"])[i]

            if prop > 90:
                prop = 180 - prop
            mu.append(prop)

        fig, ax = plt.subplots()
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.xlabel("Degrees")
        ax.hist(mu, density=False, bins=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim([0, 90])
        fig.savefig(
            f"results/{functionTitle} of {filename}", dpi=200, transparent=True,
        )

        plt.close("all")


fig, ax = plt.subplots()
plt.gcf().subplots_adjust(bottom=0.15)
plt.xlabel("Degrees")
ax.hist(muAll, density=False, bins=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim([0, 90])
fig.savefig(
    f"results/{functionTitle} of All Videos", dpi=200, transparent=True,
)
