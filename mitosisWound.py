import os
import shutil
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

scale = 147.91 / 512
bandWidth = 20  # in microns
pixelWidth = bandWidth / scale

f = open("pythonText.txt", "r")

filenames = f.read()
filenames = filenames.split(", ")

_df2 = []

for filename in filenames:

    dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
    dfDivisions = pd.read_pickle(f"dat/{filename}/mitosisTracks{filename}.pkl")
    df = dfDivisions[dfDivisions["Chain"] == "parent"]
    n = int(len(df))

    for i in range(n):

        label = df["Label"].iloc[i]
        ori = df["Division Orientation"].iloc[i]
        C = df["Position"].iloc[i]
        T = df["Time"].iloc[i]
        t = T[-1]
        [Cx, Cy] = dfWound["Centroid"].iloc[t]
        x = (C[-1][0] - Cx) * scale
        y = (C[-1][1] - Cy) * scale
        _df2.append(
            {
                "Filename": filename,
                "Label": label,
                "Wound Orientation": ori,
                "X": x,
                "Y": y,
                "R": (x ** 2 + y ** 2) ** 0.5,
                "T": t,
            }
        )

dfDivisions = pd.DataFrame(_df2)

time = dfDivisions["T"]
radius = dfDivisions["R"]
orientation = dfDivisions["Wound Orientation"]

fig = plt.figure(1, figsize=(9, 8))
plt.hist(time, 18)
plt.xlabel("Time")
fig.savefig(
    f"results/Division time after wounding", dpi=300, transparent=True,
)
plt.close("all")

fig = plt.figure(1, figsize=(9, 8))
plt.hist(orientation, 9)
plt.xlabel("Orientation")
fig.savefig(
    f"results/Division Orientation", dpi=300, transparent=True,
)
plt.close("all")

fig = plt.figure(1, figsize=(9, 8))
plt.scatter(
    time, radius, s=10,
)
plt.xlabel("Time")
plt.ylabel(f"Radius")
fig.savefig(
    f"results/Time and Radius", dpi=300, transparent=True,
)
plt.close("all")

fig = plt.figure(1, figsize=(9, 8))
plt.scatter(
    time, radius, c=orientation, s=10,
)
plt.colorbar()
plt.xlabel("Time")
plt.ylabel(f"Radius")
fig.savefig(
    f"results/Time, Orientation and Radius", dpi=300, transparent=True,
)
plt.close("all")
