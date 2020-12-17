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


def ThreeD(a):
    lst = [[[] for col in range(a)] for col in range(a)]
    return lst


def sortGrid(dfVelocity, x, y):

    xMin = x[0]
    xMax = x[1]
    yMin = y[0]
    yMax = y[1]

    dfxmin = dfVelocity[dfVelocity["X"] > xMin]
    dfx = dfxmin[dfxmin["X"] < xMax]

    dfymin = dfx[dfx["Y"] > yMin]
    df = dfymin[dfymin["Y"] < yMax]

    return df


# -------------------

f = open("pythonText.txt", "r")

filenames = f.read()
filenames = filenames.split(", ")

T = 181
scale = 147.91 / 512
grid = 10


_df2 = []
for filename in filenames:

    functionTitle = "Migration Velocity"

    df = pd.read_pickle(f"dat/{filename}/nucleusTracks{filename}.pkl")

    for i in range(len(df)):
        t = df["t"][i]
        x = df["x"][i]
        y = df["y"][i]
        label = df["Label"][i]

        m = len(t)
        tMax = t[-1]

        if m > 1:
            for j in range(m - 1):
                t0 = t[j]
                x0 = x[j]
                y0 = y[j]

                tdelta = tMax - t0
                if tdelta > 5:
                    t5 = t[j + 5]
                    x5 = x[j + 5]
                    y5 = y[j + 5]

                    v = np.array([(x5 - x0) / t5, (y5 - y0) / t5])

                    _df2.append(
                        {"Label": label, "T": t0, "X": x0, "Y": y0, "velocity": v}
                    )
                else:
                    tEnd = t[-1]
                    xEnd = x[-1]
                    yEnd = y[-1]

                    v = np.array([(xEnd - x0) / tEnd, (yEnd - y0) / tEnd])

                    _df2.append(
                        {"Label": label, "T": t0, "X": x0, "Y": y0, "velocity": v}
                    )

dfVelocity = pd.DataFrame(_df2)

dfVelocity = dfVelocity[dfVelocity["T"] < 10]

a = ThreeD(grid)

for i in range(grid):
    for j in range(grid):
        x = [(512 / grid) * j, (512 / grid) * j + 512 / grid]
        y = [(512 / grid) * i, (512 / grid) * i + 512 / grid]
        dfxy = sortGrid(dfVelocity, x, y)
        a[i][j] = list(dfxy["velocity"])
        a[i][j] = cell.mean(a[i][j])

x, y = np.meshgrid(np.linspace(0, 512 * scale, grid), np.linspace(0, 512 * scale, grid))

u = np.zeros([grid, grid])
v = np.zeros([grid, grid])

for i in range(grid):
    for j in range(grid):
        u[i, j] = a[i][j][0]
        v[i, j] = a[i][j][1]

fig = plt.figure(1, figsize=(9, 8))
plt.quiver(x, y, u, v)
fig.savefig(
    f"results/velocity field", dpi=300, transparent=True,
)
plt.close("all")

