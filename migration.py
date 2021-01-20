import os
import shutil
from math import floor, log10

from collections import Counter
import cv2
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

plt.rcParams.update({"font.size": 20})

# -------------------

filenames, fileType = cl.getFilesType()

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

                    v = np.array([(x5 - x0) / 5, (y5 - y0) / 5])

                    _df2.append(
                        {"Label": label, "T": t0, "X": x0, "Y": y0, "velocity": v}
                    )
                else:
                    tEnd = t[-1]
                    xEnd = x[-1]
                    yEnd = y[-1]

                    v = np.array([(xEnd - x0) / (tEnd - t0), (yEnd - y0) / (tEnd - t0)])

                    _df2.append(
                        {
                            "Filename": filename,
                            "Label": label,
                            "T": int(t0),
                            "X": x0,
                            "Y": y0,
                            "velocity": v,
                        }
                    )

dfVelocity = pd.DataFrame(_df2)

createFolder("results/video/")
for t in range(T):
    dfVelocityT = dfVelocity[dfVelocity["T"] == t]

    a = ThreeD(grid)

    for i in range(grid):
        for j in range(grid):
            x = [(512 / grid) * j, (512 / grid) * j + 512 / grid]
            y = [(512 / grid) * i, (512 / grid) * i + 512 / grid]
            dfxy = sortGrid(dfVelocityT, x, y)
            a[i][j] = list(dfxy["Velocity"])
            if a[i][j] == []:
                a[i][j] = np.array([0, 0])
            else:
                a[i][j] = np.mean(a[i][j])

    x, y = np.meshgrid(
        np.linspace(0, 512 * scale, grid), np.linspace(0, 512 * scale, grid),
    )

    u = np.zeros([grid, grid])
    v = np.zeros([grid, grid])

    for i in range(grid):
        for j in range(grid):
            u[i, j] = a[i][j][0]
            v[i, j] = a[i][j][1]

    fig = plt.figure(1, figsize=(9, 8))
    plt.quiver(x, y, u, v, scale=10)
    plt.title(f"time = {t}")
    fig.savefig(
        f"results/video/Velocity field{t}", dpi=300, transparent=True,
    )
    plt.close("all")

# make video
img_array = []

for t in range(T):
    img = cv2.imread(f"results/video/Velocity field{t}.png")
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter(
    f"results/Velocity field{fileType}.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 5, size,
)
for i in range(len(img_array)):
    out.write(img_array[i])

out.release()
cv2.destroyAllWindows()

shutil.rmtree("results/video")

# Mean migration

for filename in filenames:

    dft = dfVelocity[dfVelocity["Filename"] == filename]
    x = []
    y = []
    xt = 0
    yt = 0
    for t in range(T - 1):
        df = dft[dft["T"] == t]
        v = np.mean(list(df["velocity"]))
        xt += v[0]
        yt += v[1]
        x.append(xt)
        y.append(yt)

    plt.plot(x, y)

plt.xlabel("x")
plt.ylabel(f"y")
fig.savefig(
    f"results/migration", dpi=300, transparent=True,
)
plt.close("all")
