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

    for i in range(len(df)):
        t = df["Time"][i]
        [wx, wy] = dfWound["Position"].iloc[int(t)]
        [x, y] = [df["Centroid"][i][0] - wx, df["Centroid"][i][1] - wy]
        r = dist[int(t), int(df["Centroid"][i][0]), int(df["Centroid"][i][1])]
        if r == 0:
            r = -1
        q = df["q"][i]
        sf = df["Shape Factor"][i]
        A = df["Area"][i]

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

# normalisation of area and shape factor
# _dfShape = []
# for filename in filenames:

#     dfFile = df2[df2["Filename"] == filename]
#     dfTime = dfFile[dfFile["Time"] == 0]
#     dfr = dfTime[dfTime["Filename"] == filename]
#     sf0 = np.mean(dfr["Shape Factor"][dfr["R"] > R / scale])
#     A0 = np.mean(dfr["Area"][dfr["R"] > R / scale])

#     for i in range(len(dfFile)):
#         _dfShape.append(
#             {
#                 "Filename": dfFile["Filename"].iloc[i],
#                 "Time": dfFile["Time"].iloc[i],
#                 "X": dfFile["X"].iloc[i],
#                 "Y": dfFile["Y"].iloc[i],
#                 "R": dfFile["R"].iloc[i],
#                 "Theta": dfFile["Theta"].iloc[i],
#                 "q": dfFile["q"].iloc[i],
#                 "Shape Factor": dfFile["Shape Factor"].iloc[i] / sf0,
#                 "Area": dfFile["Area"].iloc[i] / A0,
#             }
#         )
# dfShape = pd.DataFrame(_dfShape)

#  -------------------

run = True
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
            mu.append(np.mean(prop) * scale ** 2)
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
    grid = 40
    heatmapA = np.zeros([int(T / 4), grid])
    for i in range(0, 180, 4):
        for j in range(grid):
            r = [80 / grid * j / scale, (80 / grid * j + 80 / grid) / scale]
            t = [i, i + 4]
            dfr = cl.sortRadius(dfShape, t, r)
            if list(dfr["Area"]) == []:
                Ar = np.nan
            else:
                Ar = dfr["Area"]
                heatmapA[int(i / 4), j] = np.mean(Ar)

    dt, dr = 4, 80 / grid
    t, r = np.mgrid[0:180:dt, 0:80:dr]
    # z_min, z_max = heatmapA.min(), heatmapA.max()
    # midpoint = (1 - z_min) / (z_max - z_min)
    # orig_cmap = matplotlib.cm.seismic
    # shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")

    fig, ax = plt.subplots()
    c = ax.pcolor(t, r, heatmapA, cmap="Reds", vmin=100, vmax=225)
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


#  ------------------- Shape Factor kymograph

run = True
if run:
    grid = 40
    heatmapSf = np.zeros([int(T / 4), grid])
    for i in range(0, 180, 4):
        for j in range(grid):
            r = [80 / grid * j / scale, (80 / grid * j + 80 / grid) / scale]
            t = [i, i + 4]
            dfr = cl.sortRadius(dfShape, t, r)
            if list(dfr["Shape Factor"]) == []:
                Sf = np.nan
            else:
                Sf = dfr["Shape Factor"]
                heatmapSf[int(i / 4), j] = np.mean(Sf)

    dt, dr = 4, 80 / grid
    t, r = np.mgrid[0:180:dt, 0:80:dr]
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

run = True
if run:
    grid = 40
    heatmapOri = np.zeros([int(T / 4), grid])
    for i in range(0, 180, 4):
        for j in range(grid):
            r = [80 / grid * j / scale, (80 / grid * j + 80 / grid) / scale]
            t = [i, i + 4]
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

                heatmapOri[int(i / 4), j] = 180 * ori / np.pi

    dt, dr = 4, 80 / grid
    t, r = np.mgrid[0:180:dt, 0:80:dr]

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
