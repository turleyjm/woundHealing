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

plt.rcParams.update({"font.size": 20})
plt.ioff()
pd.set_option("display.width", 1000)

scale = 147.91 / 512
bandWidth = 20  # in microns
pixelWidth = bandWidth / scale

f = open("pythonText.txt", "r")

filenames = f.read()
filenames = filenames.split(", ")

TrS = []
sf = []

for filename in filenames:

    df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")

    df2 = df[df["Time"] < 5]
    for i in range(len(df2)):

        TrS.append(df2["Trace(S)"].iloc[i])
        sf.append(df2["Shape Factor"].iloc[i])

# sf_a = np.linspace(0, 0.98, 9800)

# TrS_a = (0.1 - 0.1*(1 - sf_a)**0.5) / (1 - sf_a)**0.5 + 1/(2*np.pi)

fig = plt.figure(1, figsize=(9, 8))
plt.scatter(
    sf, TrS, s=1,
)
# plt.plot(sf_a, TrS_a, 'r')
plt.xlabel("sf")
plt.ylabel(f"TrS")
fig.savefig(
    f"results/TrS and sf", dpi=300, transparent=True,
)
plt.close("all")
