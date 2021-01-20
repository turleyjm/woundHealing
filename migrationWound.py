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


finish = []
for filename in filenames:

    dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")

    area = np.array(dfWound["Area"]) * (scale) ** 2
    t = 0
    while pd.notnull(area[t]):
        t += 1

    finish.append(t - 1)

meanFinish = int(min(finish))


_df2 = []
for filename in filenames:
    dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
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

                    [wx, wy] = dfWound["Centroid"].iloc[int(t0)]
                    _df2.append(
                        {
                            "Filename": filename,
                            "Label": label,
                            "T": t0,
                            "X": x0 - wx,
                            "Y": y0 - wy,
                            "Velocity": v,
                        }
                    )
                else:
                    tEnd = t[-1]
                    xEnd = x[-1]
                    yEnd = y[-1]

                    v = np.array([(xEnd - x0) / (tEnd - t0), (yEnd - y0) / (tEnd - t0)])

                    [wx, wy] = dfWound["Centroid"].iloc[int(t0)]
                    _df2.append(
                        {
                            "Filename": filename,
                            "Label": label,
                            "T": t0,
                            "X": x0 - wx,
                            "Y": y0 - wy,
                            "Velocity": v,
                        }
                    )

dfvelocity = pd.DataFrame(_df2)

_dfVelocity = []
for filename in filenames:
    df = dfvelocity[dfvelocity["Filename"] == filename]
    for t in range(T - 1):
        dft = df[df["T"] == t]
        V = np.mean(list(dft["Velocity"]))
        for i in range(len(dft)):
            _dfVelocity.append(
                {
                    "Filename": filename,
                    "Label": dft["Label"].iloc[i],
                    "T": dft["T"].iloc[i],
                    "X": dft["X"].iloc[i],
                    "Y": dft["Y"].iloc[i],
                    "Velocity": dft["Velocity"].iloc[i] - V,
                }
            )
dfVelocity = pd.DataFrame(_dfVelocity)

cl.createFolder("results/video/")
for t in range(meanFinish):
    dfVelocityT = dfVelocity[dfVelocity["T"] == t]

    a = cl.ThreeD(grid)

    for i in range(grid):
        for j in range(grid):
            x = [(512 / grid) * j - 256, (512 / grid) * j + 512 / grid - 256]
            y = [(512 / grid) * i - 256, (512 / grid) * i + 512 / grid - 256]
            dfxy = cl.sortGrid(dfVelocityT, x, y)
            a[i][j] = list(dfxy["Velocity"])
            if a[i][j] == []:
                a[i][j] = np.array([0, 0])
            else:
                a[i][j] = np.mean(a[i][j])

    x, y = np.meshgrid(
        np.linspace(-256 * scale, 256 * scale, grid),
        np.linspace(-256 * scale, 256 * scale, grid),
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
        f"results/video/Velocity field wound centred {t}", dpi=300, transparent=True,
    )
    plt.close("all")

# make video
img_array = []

for t in range(meanFinish):
    img = cv2.imread(f"results/video/Velocity field wound centred {t}.png")
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter(
    f"results/Velocity field wound centred {fileType}.mp4",
    cv2.VideoWriter_fourcc(*"DIVX"),
    3,
    size,
)
for i in range(len(img_array)):
    out.write(img_array[i])

out.release()
cv2.destroyAllWindows()

shutil.rmtree("results/video")

