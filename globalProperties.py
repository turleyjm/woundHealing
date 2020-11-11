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

    functionTitle = "Shape Factor"

    df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")

    mu = []
    err = []

    for t in range(T):
        prop = list(df[f"{functionTitle}"][df["Time"] == t])
        mu.append(cell.mean(prop))
        err.append(cell.sd(prop) / len(prop) ** 0.5)

    x = range(T)

    fig = plt.figure(1, figsize=(9, 8))
    plt.gcf().subplots_adjust(left=0.2)
    plt.errorbar(x, mu, yerr=err, fmt="o")

    plt.xlabel("Time")
    plt.ylabel(f"{functionTitle}")
    plt.gcf().subplots_adjust(bottom=0.2)
    fig.savefig(
        f"results/{functionTitle} of {filename}", dpi=300, transparent=True,
    )
    plt.close("all")

    # ----------------------------

    functionTitle = "Orientation"

    df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")

    iso = []
    for t in range(T):
        prop = list(df[f"Q"][df["Time"] == t])
        Ori = []
        for i in range(len(prop)):

            Q = prop[i]
            v1 = Q[0]
            x = v1[0]
            y = v1[1]

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

    x = range(T)
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
