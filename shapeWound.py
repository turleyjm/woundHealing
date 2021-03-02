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
R = 40

finish = []
woundEdge = []
for filename in filenames:

    dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")

    area = np.array(dfWound["Area"]) * (scale) ** 2
    t = 0
    while pd.notnull(area[t]):
        t += 1

    finish.append(t - 1)
    woundEdge.append((area[0] / np.pi) ** 0.5)

meanFinish = int(np.mean(finish))
medianFinish = int(np.median(finish))
minFinish = int(min(finish))
woundEdge = np.mean(woundEdge)


_df2 = []
for filename in filenames:
    dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
    df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")
    dist = sm.io.imread(f"dat/{filename}/distanceWound{filename}.tif").astype(float)

    for t in range(T):
        dft = df[df["Time"] == t]
        [wx, wy] = dfWound["Position"].iloc[int(t)]
        Q = np.mean(dft["q"])
        for i in range(len(dft)):
            [x, y] = [dft["Centroid"].iloc[i][0] - wx, dft["Centroid"].iloc[i][1] - wy]
            r = dist[
                int(t), int(dft["Centroid"].iloc[i][0]), int(dft["Centroid"].iloc[i][1])
            ]
            if r == 0:
                r = -1
            q = dft["q"].iloc[i] - Q
            sf = dft["Shape Factor"].iloc[i]
            A = dft["Area"].iloc[i] * scale ** 2
            P = dft["Perimeter"].iloc[i] * scale

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
                    "Shape Index": P / A ** 0.5,
                }
            )

dfShape = pd.DataFrame(_df2)

#  -------------------

run = False
if run:
    fig = plt.figure(1, figsize=(9, 8))
    time = range(T)
    for filename in filenames:

        df = dfShape[dfShape["Filename"] == filename]
        dfr = df[df["R"] > R]

        mu = []
        err = []

        for t in time:
            prop = list(dfr["Area"][dfr["Time"] == t])
            mu.append(np.mean(prop))
            err.append(np.std(prop) / len(prop) ** 0.5)

        plt.plot(time, mu)

    plt.xlabel("Time (mins)")
    plt.ylabel(r"Area ($\mu m^2$)")
    plt.title(f"Area {R}" + r"$\mu m $ back from wound")
    fig.savefig(
        f"results/Area back from Wound {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")

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
    c = ax.pcolor(t, r, heatmapA, cmap="Reds")
    fig.colorbar(c, ax=ax)
    plt.axvline(x=medianFinish)
    plt.text(medianFinish + 2, 50, "Median Finish Time", size=10, rotation=90)
    plt.xlabel("Time (mins)")
    plt.ylabel(r"Distance from wound edge $(\mu m)$")
    plt.title(f"Area {fileType}")
    fig.savefig(
        f"results/Area kymograph {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")


#  ------------------- Shape index kymograph

run = False
if run:
    grid = 50
    heatmapSI = np.zeros([int(T), grid])
    for i in range(T):
        for j in range(grid):
            r = [100 / grid * j / scale, (100 / grid * j + 100 / grid) / scale]
            t = [i, i + 1]
            dfr = cl.sortRadius(dfShape, t, r)
            if list(dfr["Shape Index"]) == []:
                Si = np.nan
            else:
                Si = dfr["Shape Index"]
                heatmapSI[int(i), j] = np.mean(Si)

    dt, dr = 1, 100 / grid
    t, r = np.mgrid[0:181:dt, 0:100:dr]

    fig, ax = plt.subplots()
    c = ax.pcolor(t, r, heatmapSI, cmap="Reds")
    fig.colorbar(c, ax=ax)
    plt.axvline(x=medianFinish)
    plt.text(medianFinish + 2, 50, "Median Finish Time", size=10, rotation=90)
    plt.xlabel("Time (mins)")
    plt.ylabel(r"Distance from wound edge $(\mu m)$")
    plt.title(f"Shape index {fileType}")
    fig.savefig(
        f"results/Shape index kymograph {fileType}", dpi=300, transparent=True,
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
    plt.axvline(x=medianFinish)
    plt.text(medianFinish + 2, 50, "Median Finish Time", size=10, rotation=90)
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
    plt.axvline(x=medianFinish)
    plt.text(medianFinish + 2, 50, "Median Finish Time", size=10, rotation=90)
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
    plt.axvline(x=medianFinish)
    plt.text(medianFinish + 2, 50, "Median Finish Time", size=10, rotation=90)
    plt.xlabel("Time (min)")
    plt.ylabel(r"Distance from wound edge $(\mu m)$")
    plt.title(f"Q1 {fileType}")
    fig.savefig(
        f"results/Q1 kymograph {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")

    # p = np.zeros([15, 3])
    # for t in range(10):

    # d = heatmapq1[t * 4, 3:30]
    # x = np.array(range(len(d))) * 2 + 6
    # p[t] = np.polyfit(x, d, 2)

    # fig, ax = plt.subplots()
    # plt.plot(x, d)
    # plt.plot(x, (p[t, 0] * x ** 2 + p[t, 1] * x + p[t, 2]))
    # plt.ylim(-0.04, 0.01)
    # plt.xlabel(r"Distance from Wound Edge $(\mu m)$")
    # plt.ylabel(r"Q1")
    # fig.savefig(
    #     f"results/Q1 t={t*4} {fileType}", dpi=300, transparent=True,
    # )
    # plt.close("all")

    fig, ax = plt.subplots()
    c = ax.pcolor(t, r, heatmapq2, cmap="RdBu_r", vmin=-0.03, vmax=0.03)
    fig.colorbar(c, ax=ax)
    plt.axvline(x=medianFinish)
    plt.text(medianFinish + 2, 50, "Median Finish Time", size=10, rotation=90)
    plt.xlabel("Time (min)")
    plt.ylabel(r"Distance from wound edge $(\mu m)$")
    plt.title(f"Q2 {fileType}")
    fig.savefig(
        f"results/Q2 kymograph {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")

