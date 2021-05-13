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


dfDensityUnwound = pd.read_pickle(f"databases/dfDensityUnwound.pkl")
dfDensityWoundS = pd.read_pickle(f"databases/dfDensityWoundS.pkl")
dfDensityWoundL = pd.read_pickle(f"databases/dfDensityWoundL.pkl")

scale = 147.91 / 512

Rbin = 5
Tbin = 10
R = range(0, 80, Rbin)
T = range(0, 180, Tbin)

run = False
if run:
    fig = plt.figure(1, figsize=(9, 8))
    position = range(5, 85, Rbin)
    density = []
    for r in R:
        area = np.mean(dfDensityUnwound["Area"][dfDensityUnwound["R"] == r])
        n = np.mean(dfDensityUnwound["Number"][dfDensityUnwound["R"] == r])

        density.append(n / area)

    plt.plot(position[1:], density[1:], label="Unwound")

    density = []
    for r in R:
        area = np.mean(dfDensityWoundS["Area"][dfDensityWoundS["R"] == r])
        n = np.mean(dfDensityWoundS["Number"][dfDensityWoundS["R"] == r])

        density.append(n / area)

    plt.plot(position, density, label="WoundS")

    density = []
    for r in R:
        area = np.mean(dfDensityWoundL["Area"][dfDensityWoundL["R"] == r])
        n = np.mean(dfDensityWoundL["Number"][dfDensityWoundL["R"] == r])

        density.append(n / area)

    plt.plot(position, density, label="WoundL")

    plt.ylabel("Density of Divisons")
    plt.xlabel("Wound Distance")
    plt.ylim([0, 0.01])
    plt.legend()
    plt.title(f"Division Density")
    fig.savefig(
        f"results/Division Density All",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

    # -------------------------------

    fig = plt.figure(1, figsize=(9, 8))
    time = range(10, 190, Tbin)
    density = []
    for t in T:
        area = np.mean(dfDensityUnwound["Area"][dfDensityUnwound["T"] == t])
        n = np.mean(dfDensityUnwound["Number"][dfDensityUnwound["T"] == t])
        density.append(n / area)

    plt.plot(time, density, label="Unwound")

    density = []
    for t in T:
        area = np.mean(dfDensityWoundS["Area"][dfDensityWoundS["T"] == t])
        n = np.mean(dfDensityWoundS["Number"][dfDensityWoundS["T"] == t])
        density.append(n / area)

    plt.plot(time, density, label="WoundS")

    density = []
    for t in T:
        area = np.mean(dfDensityWoundL["Area"][dfDensityWoundL["T"] == t])
        n = np.mean(dfDensityWoundL["Number"][dfDensityWoundL["T"] == t])
        density.append(n / area)

    plt.plot(time, density, label="WoundL")

    plt.ylabel("Density of Divisons")
    plt.xlabel("Time (mins)")
    plt.ylim([0, 0.012])
    plt.legend()
    plt.title(f"Division time")
    fig.savefig(
        f"results/Division time All",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

run = True
if run:
    fig = plt.figure(1, figsize=(9, 8))
    plt.gcf().subplots_adjust(left=0.18)
    position = range(5, 85, Rbin)
    density = []
    for r in R:
        area = np.mean(dfDensityUnwound["Area"][dfDensityUnwound["R"] == r])
        n = np.mean(dfDensityUnwound["Number"][dfDensityUnwound["R"] == r])

        density.append(n / area)

    c = np.polyfit(position[2:], density[2:], 0)

    density = []
    for r in R:
        area = np.mean(dfDensityWoundS["Area"][dfDensityWoundS["R"] == r])
        n = np.mean(dfDensityWoundS["Number"][dfDensityWoundS["R"] == r])

        density.append(n / area)

    densityChange = np.array(density) - c

    plt.plot(position, densityChange, label="WoundS")

    density = []
    for r in R:
        area = np.mean(dfDensityWoundL["Area"][dfDensityWoundL["R"] == r])
        n = np.mean(dfDensityWoundL["Number"][dfDensityWoundL["R"] == r])

        density.append(n / area)

    densityChange = np.array(density) - c

    plt.plot(position, densityChange, label="WoundL")

    plt.plot(range(90), np.zeros(90))

    plt.ylabel(r"Density of Divisons $(\mu m^{-2})$")
    plt.xlabel(r"Wound Distance $(\mu m)$")
    plt.ylim([-0.008, 0.005])
    plt.xlim([0, 85])
    plt.legend()
    plt.title(f"Division Density")
    fig.savefig(
        f"results/Division Density wound change",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

    # -------------------------------

    fig = plt.figure(1, figsize=(9, 8))
    plt.gcf().subplots_adjust(left=0.18)
    time = range(10, 190, Tbin)
    density = []
    for t in T:
        area = np.mean(dfDensityUnwound["Area"][dfDensityUnwound["T"] == t])
        n = np.mean(dfDensityUnwound["Number"][dfDensityUnwound["T"] == t])
        density.append(n / area)

    m, c = np.polyfit(time, density, 1)

    density = []
    for t in T:
        area = np.mean(dfDensityWoundS["Area"][dfDensityWoundS["T"] == t])
        n = np.mean(dfDensityWoundS["Number"][dfDensityWoundS["T"] == t])
        density.append(n / area)

    densityChange = np.array(density) - (m * np.array(time) + c)

    plt.plot(time, densityChange, label="WoundS")

    density = []
    for t in T:
        area = np.mean(dfDensityWoundL["Area"][dfDensityWoundL["T"] == t])
        n = np.mean(dfDensityWoundL["Number"][dfDensityWoundL["T"] == t])
        density.append(n / area)

    densityChange = np.array(density) - (m * np.array(time) + c)

    plt.plot(time, densityChange, label="WoundL")

    plt.plot(range(190), np.zeros(190))

    plt.ylabel(r"Density of Divisons $(\mu m^{-2})$")
    plt.xlabel("Time (mins)")
    plt.ylim([-0.008, 0.005])
    plt.xlim([0, 190])
    plt.legend()
    plt.title(f"Division time")
    fig.savefig(
        f"results/Division time wound change",
        dpi=300,
        transparent=True,
    )
    plt.close("all")
