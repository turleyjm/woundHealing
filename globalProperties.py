import os
import shutil
from math import floor, log10

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
import commonLiberty as cl

plt.rcParams.update({"font.size": 16})

# -------------------

filenames, fileType = cl.getFilesType()

T = 181
scale = 147.91 / 512

run = True
if run:
    fig = plt.figure(1, figsize=(9, 8))
    time = range(T)
    for filename in filenames:

        df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")
        sf0 = np.mean(list(df["Shape Factor"][df["Time"] == 0]))

        mu = []
        err = []

        for t in time:
            prop = list(df["Shape Factor"][df["Time"] == t])
            mu.append(np.mean(prop) / sf0)
            err.append(np.std(prop) / len(prop) ** 0.5)

        plt.plot(time, mu)

    plt.xlabel("Time")
    plt.ylabel(f"Shape Factor relative")
    fig.savefig(
        f"results/Shape Factor", dpi=300, transparent=True,
    )
    plt.close("all")

# ----------------------------

run = True
if run:
    fig = plt.figure(1, figsize=(9, 8))
    time = range(T)
    for filename in filenames:

        df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")
        A0 = np.mean(list(df["Area"][df["Time"] == 0] * scale ** 2))

        mu = []
        err = []

        for t in time:
            prop = list(df["Area"][df["Time"] == t] * scale ** 2)
            mu.append(np.mean(prop) / A0)
            err.append(np.std(prop) / len(prop) ** 0.5)

        plt.plot(time, mu)
        mu = np.array(mu)

    plt.xlabel("Time")
    plt.ylabel("Area relative")
    fig.savefig(
        f"results/Area", dpi=300, transparent=True,
    )
    plt.close("all")


# ----------------------------

run = False
if run:
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
        f"results/Orientation of {filename}", dpi=300, transparent=True,
    )
    plt.close("all")

