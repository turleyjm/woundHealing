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

plt.rcParams.update({"font.size": 20})

# -------------------

filenames, fileType = cl.getFilesType()

T = 181
scale = 147.91 / 512
grid = 10


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

meanFinish = int(min(finish))
woundEdge = np.mean(woundEdge)


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
                            "Time": t0,
                            "X": x0 - wx,
                            "Y": y0 - wy,
                            "R": ((x0 - wx) ** 2 + (y0 - wy) ** 2) ** 0.5,
                            "Theta": np.arctan2(y0 - wy, x0 - wx),
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
                            "Time": t0,
                            "X": x0 - wx,
                            "Y": y0 - wy,
                            "R": ((x0 - wx) ** 2 + (y0 - wy) ** 2) ** 0.5,
                            "Theta": np.arctan2(y0 - wy, x0 - wx),
                            "Velocity": v,
                        }
                    )

dfvelocity = pd.DataFrame(_df2)

#  ------------------- Velocity - mean velocity

_dfVelocity = []
for filename in filenames:
    df = dfvelocity[dfvelocity["Filename"] == filename]
    for t in range(T - 1):
        dft = df[df["Time"] == t]
        V = np.mean(list(dft["Velocity"]))
        for i in range(len(dft)):
            _dfVelocity.append(
                {
                    "Filename": filename,
                    "Label": dft["Label"].iloc[i],
                    "Time": dft["Time"].iloc[i],
                    "X": dft["X"].iloc[i],
                    "Y": dft["Y"].iloc[i],
                    "R": dft["R"].iloc[i],
                    "Theta": dft["Theta"].iloc[i],
                    "Velocity": dft["Velocity"].iloc[i] - V,
                }
            )
dfVelocity = pd.DataFrame(_dfVelocity)

#  ------------------- Velocity feild

run = False
if run:
    cl.createFolder("results/video/")
    for t in range(meanFinish):
        dfVelocityT = dfVelocity[dfVelocity["Time"] == t]

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
            f"results/video/Velocity field wound centred {t}",
            dpi=300,
            transparent=True,
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


#  ------------------- Velocity field scaled by wound size
run = False
if run:
    dfVelocityScale = dfVelocity
    for filename in filenames:

        woundScale = dfWound["Area"].iloc[0] ** 0.5
        dfVelocityScale["Velocity"][dfVelocityScale["Filename"] == filename] = (
            dfVelocityScale["Velocity"][dfVelocityScale["Filename"] == filename]
            * woundScale
        )

    cl.createFolder("results/video/")
    for t in range(meanFinish):
        dfVelocityScaleT = dfVelocity[dfVelocityScale["Time"] == t]

        a = cl.ThreeD(grid)

        for i in range(grid):
            for j in range(grid):
                x = [(512 / grid) * j - 256, (512 / grid) * j + 512 / grid - 256]
                y = [(512 / grid) * i - 256, (512 / grid) * i + 512 / grid - 256]
                dfxy = cl.sortGrid(dfVelocityScaleT, x, y)
                a[i][j] = list(dfxy["Velocity"])
                if a[i][j] == []:
                    a[i][j] = np.array([0, 0])
                else:
                    a[i][j] = np.mean(a[i][j], axis=0)

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
        plt.quiver(x, y, u, v, scale=1000)
        plt.title(f"time = {t}")
        fig.savefig(
            f"results/video/Velocity field scaled wound centred {t}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

    # make video
    img_array = []

    for t in range(meanFinish):
        img = cv2.imread(f"results/video/Velocity field scaled wound centred {t}.png")
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(
        f"results/Velocity field scaled wound centred {fileType}.mp4",
        cv2.VideoWriter_fourcc(*"DIVX"),
        3,
        size,
    )
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
    cv2.destroyAllWindows()

    shutil.rmtree("results/video")


#  ------------------- Mean migration path

run = False

if run:
    fig = plt.figure(1, figsize=(9, 8))
    x = []
    y = []
    xt = 0
    yt = 0
    for t in range(T - 1):
        df = dfvelocity[dfvelocity["Time"] == t]
        v = np.mean(list(df["Velocity"]), axis=0)
        xt += v[0]
        yt += v[1]
        x.append(xt)
        y.append(yt)

        plt.plot(x, y)

    plt.xlabel("x")
    plt.ylabel(f"y")
    fig.savefig(
        f"results/migration path {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")

#  ------------------- Radial Velocity

run = True

if run:
    grid = 40
    heatmap = np.zeros([int(T / 4), grid])
    for i in range(0, 180, 4):
        for j in range(grid):
            r = [80 / grid * j / scale, (80 / grid * j + 80 / grid) / scale]
            t = [i, i + 4]
            dfr = cl.sortRadius(dfVelocity, t, r)
            if list(dfr["Velocity"]) == []:
                Vr = 0
            else:
                Vr = []
                for k in range(len(dfr)):
                    v = dfr["Velocity"].iloc[k]
                    theta = dfr["Theta"].iloc[k]
                    R = cl.rotation_matrix(-theta)
                    Vr.append(-np.matmul(R, v)[0])

                heatmap[int(i / 4), j] = np.mean(Vr) * scale

    dt, dr = 4, 80 / grid
    t, r = np.mgrid[0:180:dt, 0:80:dr]
    z_min, z_max = -0.5, 0.5
    midpoint = 1 - z_max / (z_max + abs(z_min))
    orig_cmap = matplotlib.cm.seismic
    shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")
    fig, ax = plt.subplots()
    c = ax.pcolor(t, r, heatmap, cmap=shifted_cmap, vmin=z_min, vmax=z_max)
    fig.colorbar(c, ax=ax)
    c.set_label(r"Velocity $(\mu m/min)$")
    plt.xlabel("Time (min)")
    plt.ylabel(r"Distance from wound center $(\mu m)$")
    fig.savefig(
        f"results/Radial Velocity kymograph {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")


#  ------------------- Radial Laganian Velocity

run = True

if run:

    woundDist = np.linspace(woundEdge, woundEdge + 40, 41)
    lagMigration = []

    grid = 40
    heatmap = np.zeros([meanFinish, grid])
    for i in range(meanFinish):
        for j in range(grid):
            r = [80 / grid * j / scale, (80 / grid * j + 80 / grid) / scale]
            t = [i, i]
            dfr = cl.sortRadius(dfVelocity, t, r)
            if list(dfr["Velocity"]) == []:
                Vr = 0
            else:
                Vr = []
                for k in range(len(dfr)):
                    v = dfr["Velocity"].iloc[k]
                    theta = dfr["Theta"].iloc[k]
                    R = cl.rotation_matrix(-theta)
                    Vr.append(-np.matmul(R, v)[0])

                heatmap[i, j] = np.mean(Vr) * scale

    for d in woundDist:
        d0 = d
        migration = 0
        for t in range(meanFinish):
            d = d - heatmap[t, int(d / 2)]

        lagMigration.append(d0 - d)

    fig, ax = plt.subplots()
    plt.plot(woundDist - woundEdge, lagMigration)
    plt.xlabel(r"Start from Wound Edge $(\mu m)$")
    plt.ylabel(r"Migration $(\mu m)$")
    fig.savefig(
        f"results/Laganian Migration {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")

