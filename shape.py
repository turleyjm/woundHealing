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

plt.rcParams.update({"font.size": 14})

# -------------------

filenames, fileType = cl.getFilesType()

T = 181
scale = 147.91 / 512
grid = 10


_df2 = []
for filename in filenames:

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
                        {
                            "Filename": filename,
                            "Label": label,
                            "Time": t0,
                            "X": x0,
                            "Y": y0,
                            "Velocity": v,
                        }
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
                            "Time": int(t0),
                            "X": x0,
                            "Y": y0,
                            "Velocity": v,
                        }
                    )

dfVelocity = pd.DataFrame(_df2)

_df2 = []
for filename in filenames:
    dfUnwound = dfVelocity[dfVelocity["Filename"] == filename]
    xt = 256
    yt = 256
    xList = []
    yList = []
    xList.append(xt)
    yList.append(yt)
    for t in range(T - 1):
        df = dfUnwound[dfUnwound["Time"] == t]
        v = np.mean(list(df["Velocity"]), axis=0)
        xt += v[0]
        yt += v[1]
        xList.append(xt)
        yList.append(yt)

    dist = np.ones([T, 512, 512])

    for t in range(T):
        x = int(xList[t])
        y = int(yList[t])
        dist[t, x, y] = 0
        dist[t] = sp.ndimage.morphology.distance_transform_edt(dist[t])

    df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")

    for t in range(T):
        dft = df[df["Time"] == t]
        x = int(xList[t])
        y = int(yList[t])
        Q = np.mean(dft["q"])
        for i in range(len(dft)):
            [x, y] = [dft["Centroid"].iloc[i][0] - x, dft["Centroid"].iloc[i][1] - y]
            r = dist[
                int(t), int(dft["Centroid"].iloc[i][0]), int(dft["Centroid"].iloc[i][1])
            ]
            if r == 0:
                r = -1
            q = dft["q"].iloc[i] - Q
            sf = dft["Shape Factor"].iloc[i]
            A = dft["Area"].iloc[i]

            _df2.append(
                {
                    "Filename": filename,
                    "Time": t,
                    "X": x,
                    "Y": y,
                    "R": r,
                    "Theta": np.arctan2(y, x),
                    "q": q,
                    "Shape Factor": sf,
                    "Area": A,
                }
            )

dfShape = pd.DataFrame(_df2)


#  ------------------- Area kymograph

run = True
if run:
    grid = 50
    heatmapA = np.zeros([int(T), grid])
    for i in range(T):
        for j in range(grid):
            r = [100 / grid * j / scale, (100 / grid * j + 100 / grid) / scale]
            t = [i, i + 1]
            dfr = cl.sortRadius(dfShape, t, r)
            if list(dfr["Area"]) == []:
                Ar = np.nan
            else:
                Ar = dfr["Area"]
                heatmapA[int(i), j] = np.mean(Ar)

    dt, dr = 1, 100 / grid
    t, r = np.mgrid[0:181:dt, 0:100:dr]
    # z_min, z_max = heatmapA.min(), heatmapA.max()
    # midpoint = (1 - z_min) / (z_max - z_min)
    # orig_cmap = matplotlib.cm.seismic
    # shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")

    fig, ax = plt.subplots()
    c = ax.pcolor(t, r, heatmapA, cmap="Reds", vmin=100, vmax=225)
    fig.colorbar(c, ax=ax)
    plt.xlabel("Time (mins)")
    plt.ylabel(r"Distance from wound edge $(\mu m)$")
    plt.title(f"Area {fileType}")
    fig.savefig(
        f"results/Area kymograph {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")


#  ------------------- Shape Factor kymograph

run = False
if run:
    grid = 50
    heatmapSf = np.zeros([int(T), grid])
    for i in range(T):
        for j in range(grid):
            r = [100 / grid * j / scale, (100 / grid * j + 100 / grid) / scale]
            t = [i, i + 1]
            dfr = cl.sortRadius(dfShape, t, r)
            if list(dfr["Shape Factor"]) == []:
                Sf = np.nan
            else:
                Sf = dfr["Shape Factor"]
                heatmapSf[int(i), j] = np.mean(Sf)

    dt, dr = 1, 100 / grid
    t, r = np.mgrid[0:181:dt, 0:100:dr]
    # z_min, z_max = heatmapSf.min(), heatmapSf.max()
    # midpoint = (1 - z_min) / (z_max - z_min)
    # orig_cmap = matplotlib.cm.seismic
    # shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")

    fig, ax = plt.subplots()
    c = ax.pcolor(t, r, heatmapSf, cmap="Reds", vmin=0.3, vmax=0.5)
    fig.colorbar(c, ax=ax)
    plt.xlabel("Time (mins)")
    plt.ylabel(r"Distance from wound edge $(\mu m)$")
    plt.title(f"Shape Factor {fileType}")
    fig.savefig(
        f"results/Shape Factor kymograph {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")


#  ------------------- Orientation kymograph

run = False
if run:
    grid = 50
    heatmapOri = np.zeros([int(T), grid])
    for i in range(T):
        for j in range(grid):
            r = [100 / grid * j / scale, (100 / grid * j + 100 / grid) / scale]
            t = [i, i + 1]
            dfr = cl.sortRadius(dfShape, t, r)
            if list(dfr["q"]) == []:
                ori = np.nan
            else:
                V = []
                for k in range(len(dfr)):
                    q = dfr["q"].iloc[k]
                    phi = dfr["Theta"].iloc[k] * 2
                    R = cl.rotation_matrix(-phi)
                    V.append(np.matmul(R, q[0]))

                V = np.mean(V, axis=0)
                ori = abs(np.arctan2(V[1], V[0])) % (np.pi / 2)

                heatmapOri[int(i), j] = 180 * ori / np.pi

    dt, dr = 1, 100 / grid
    t, r = np.mgrid[0:181:dt, 0:100:dr]

    fig, ax = plt.subplots()
    c = ax.pcolor(t, r, heatmapOri, cmap="Reds")  # vmin=0, vmax=90)
    fig.colorbar(c, ax=ax)
    plt.xlabel("Time (min)")
    plt.ylabel(r"Distance from wound edge $(\mu m)$")
    plt.title(f"Orientation {fileType}")
    fig.savefig(
        f"results/Orientation kymograph {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")

#  ------------------- q tensor kymograph

run = False
if run:
    grid = 50
    heatmapq1 = np.zeros([int(T), grid])
    heatmapq2 = np.zeros([int(T), grid])
    for i in range(T):
        for j in range(grid):
            r = [100 / grid * j / scale, (100 / grid * j + 100 / grid) / scale]
            t = [i, i + 1]
            dfr = cl.sortRadius(dfShape, t, r)
            if list(dfr["q"]) == []:
                ori = np.nan
            else:
                Q = []
                for k in range(len(dfr)):
                    q = dfr["q"].iloc[k]
                    phi = dfr["Theta"].iloc[k] * 2
                    R = cl.rotation_matrix(-phi)
                    Q.append(np.matmul(R, q))

                Q = np.mean(Q, axis=0)

                heatmapq1[int(i), j] = Q[0, 0]
                heatmapq2[int(i), j] = Q[0, 1]

    dt, dr = 1, 100 / grid
    t, r = np.mgrid[0:181:dt, 0:100:dr]

    fig, ax = plt.subplots()
    c = ax.pcolor(t, r, heatmapq1, cmap="RdBu_r", vmin=-0.03, vmax=0.03)
    fig.colorbar(c, ax=ax)
    plt.xlabel("Time (min)")
    plt.ylabel(r"Distance from wound edge $(\mu m)$")
    plt.title(f"Q1 {fileType}")
    fig.savefig(
        f"results/Q1 kymograph {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")

    fig, ax = plt.subplots()
    c = ax.pcolor(t, r, heatmapq2, cmap="RdBu_r", vmin=-0.03, vmax=0.03)
    fig.colorbar(c, ax=ax)
    plt.xlabel("Time (min)")
    plt.ylabel(r"Distance from wound edge $(\mu m)$")
    plt.title(f"Q2 {fileType}")
    fig.savefig(
        f"results/Q2 kymograph {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")

