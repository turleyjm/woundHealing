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


for filename in filenames:

    functionTitle = "Migration Velocity"

    df = pd.read_pickle(f"dat/{filename}/nucleusTracks{filename}.pkl")

    _df2 = []

    for i in range(len(df)):
        t = df["t"][i]
        x = df["x"][i]
        y = df["y"][i]

        m = len(t)

        for j in range(m - 1):
            v = np.array([x[j + 1] - x[j], -(y[j + 1] - y[j])])  # -y as coord change

            s = ((v[0]) ** 2 + (v[1]) ** 2) ** 0.5

            _df2.append(
                {"v": v, "Speed": s, "Time": t[j],}
            )

    df2 = pd.DataFrame(_df2)

    iso = []
    V = []
    mu = []
    S = []
    errS = []
    for t in range(T - 1):
        prop = list(df2[f"v"][df2["Time"] == t])
        V.append(cell.mean(prop))
        mu.append(((cell.mean(prop)[0] ** 2 + cell.mean(prop)[1] ** 2) ** 0.5))

        Ori = []
        for i in range(len(prop)):

            v = prop[i]
            x = v[0]
            y = v[1]

            c = (x ** 2 + y ** 2) ** 0.5

            if x == 0 and y == 0:
                continue
            else:
                Ori.append(np.array([x, y]) / c)

        n = len(Ori)

        OriDash = sum(Ori) / n

        rho = ((OriDash[0]) ** 2 + (OriDash[1]) ** 2) ** 0.5

        OriSigma = sum(((Ori - OriDash) ** 2) / n) ** 0.5

        OriSigma = sum(OriSigma)

        iso.append(rho / OriSigma)

        prop = list(df2[f"Speed"][df2["Time"] == t])
        S.append(cell.mean(prop))
        errS.append(cell.sd(prop) / len(prop) ** 0.5)

    x = range(T - 1)
    fig = plt.figure(1, figsize=(9, 8))
    plt.gcf().subplots_adjust(left=0.2)
    plt.plot(x, mu)
    plt.title(cell.mean(V))
    plt.xlabel("Time")
    plt.ylabel(f"Global movement of the tissue")
    plt.gcf().subplots_adjust(bottom=0.2)
    fig.savefig(
        f"results/{functionTitle} of {filename}", dpi=300, transparent=True,
    )
    plt.close("all")

    functionTitle = "Migration Isotopy"

    x = range(T - 1)
    fig = plt.figure(1, figsize=(9, 8))
    plt.gcf().subplots_adjust(left=0.2)
    plt.plot(x, iso)

    plt.xlabel("Time")
    plt.ylabel(f"isotopy of the tissue")
    plt.gcf().subplots_adjust(bottom=0.2)
    fig.savefig(
        f"results/{functionTitle} of {filename}", dpi=300, transparent=True,
    )
    plt.close("all")

    # ------------------------

    x = range(T - 1)
    fig = plt.figure(1, figsize=(9, 8))
    plt.gcf().subplots_adjust(left=0.2)
    plt.errorbar(x, mu, yerr=errS, fmt=".")
    plt.xlabel("Time")
    plt.ylabel(f"Speed")
    plt.gcf().subplots_adjust(bottom=0.2)
    fig.savefig(
        f"results/Migration Speed of {filename}", dpi=300, transparent=True,
    )
    plt.close("all")
