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

background = 4

for filename in filenames:

    vidFile = f"dat/{filename}/3dFilter{filename}.tif"

    vid = sm.io.imread(vidFile).astype(int)

    (T, Z, X, Y) = vid.shape

    mu0 = []

    for z in range(Z):
        mu0.append(vid[0][z][vid[0][z] > background])
        vid[0][z][vid[0][z] <= background] = 0

    count = 0
    epsilon = 0
    for z in range(Z):
        epsilon += cell.mean(mu0[z]) * len(mu0[z])
        count += len(mu0[z])

    mu0 = epsilon / count

    for t in range(T - 1):
        mu = []
        for z in range(Z):
            if len(vid[t + 1][z][vid[t + 1][z] > background]) != 0:
                mu.append(vid[t + 1][z][vid[t + 1][z] > background])
                vid[t + 1][z][vid[t + 1][z] <= background] = 0

        count = 0
        epsilon = 0
        for z in range(len(mu)):
            epsilon += cell.mean(mu[z]) * len(mu[z])
            count += len(mu[z])

        mu = epsilon / count

        ratio = mu0 / mu

        vid[t + 1] = vid[t + 1] * ratio
        vid[t + 1][vid[t + 1] > 255] = 255

    vid = np.asarray(vid, "uint8")
    tifffile.imwrite(f"dat/{filename}/nuclei{filename}.tif", vid)

    # --------

    vidFile = f"dat/{filename}/focusBleachH2{filename}.tif"

    vid = sm.io.imread(vidFile).astype(int)

    (T, X, Y) = vid.shape

    mu0 = vid[0][vid[0] > background]

    mu0 = cell.mean(mu0)

    for t in range(T - 1):
        mu = vid[t + 1][vid[t + 1] > background]
        vid[t + 1][vid[t + 1] <= background] = 0

        mu = cell.mean(mu)

        ratio = mu0 / mu

        vid[t + 1] = vid[t + 1] * ratio
        vid[t + 1][vid[t + 1] > 255] = 255

    vid = np.asarray(vid, "uint8")
    tifffile.imwrite(f"dat/{filename}/focusH2{filename}.tif", vid)

    # --------

    vidFile = f"dat/{filename}/focusBleachEcad{filename}.tif"

    vid = sm.io.imread(vidFile).astype(int)

    (T, X, Y) = vid.shape

    mu0 = vid[0][vid[0] > background]

    mu0 = cell.mean(mu0)

    for t in range(T - 1):
        mu = vid[t + 1][vid[t + 1] > background]
        vid[t + 1][vid[t + 1] <= background] = 0

        mu = cell.mean(mu)

        ratio = mu0 / mu

        vid[t + 1] = vid[t + 1] * ratio
        vid[t + 1][vid[t + 1] > 255] = 255

    vid = np.asarray(vid, "uint8")
    tifffile.imwrite(f"dat/{filename}/focusEcad{filename}.tif", vid)
