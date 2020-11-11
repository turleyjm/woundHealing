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

    mu = []

    for i in range(n):
        prop = list(df[functionTitle][df["Chain"] == "parent"])[i]
        mu.append(prop)

    fig, ax = plt.subplots()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xlabel("Degrees")
    ax.hist(mu, density=False, bins=18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim([0, 90])
    fig.savefig(
        f"results/{functionTitle} of {filename}", dpi=200, transparent=True,
    )

    plt.close("all")
