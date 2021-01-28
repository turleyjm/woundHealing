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

#  ------------------- Velocity feild
run = False
if run:
    cl.createFolder("results/video/")
    for t in range(T - 1):
        dfVelocityT = dfVelocity[dfVelocity["Time"] == t]

        a = cl.ThreeD(grid)

        for i in range(grid):
            for j in range(grid):
                x = [(512 / grid) * j, (512 / grid) * j + 512 / grid]
                y = [(512 / grid) * i, (512 / grid) * i + 512 / grid]
                dfxy = cl.sortGrid(dfVelocityT, x, y)
                a[i][j] = list(dfxy["Velocity"])
                if a[i][j] == []:
                    a[i][j] = np.array([0, 0])
                else:
                    a[i][j] = np.mean(a[i][j], axis=0)

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

    for t in range(T - 1):
        img = cv2.imread(f"results/video/Velocity field{t}.png")
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(
        f"results/Velocity field {fileType}.mp4",
        cv2.VideoWriter_fourcc(*"DIVX"),
        5,
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
        df = dfVelocity[dfVelocity["Time"] == t]
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

#  ------------------- Compare migration path of unwounded, wounded and woundsite

run = False
if run:

    fig = plt.figure(1, figsize=(9, 8))

    fileType = "Unwound"
    cwd = os.getcwd()
    Fullfilenames = os.listdir(cwd + "/dat")
    filenames = []
    for filename in Fullfilenames:
        if fileType in filename:
            filenames.append(filename)

    filenames.sort()

    xList = [[] for col in range(T - 1)]
    yList = [[] for col in range(T - 1)]
    for filename in filenames:
        dfUnwound = dfVelocity[dfVelocity["Filename"] == filename]
        xt = 0
        yt = 0
        for t in range(T - 1):
            df = dfUnwound[dfUnwound["Time"] == t]
            v = np.mean(list(df["Velocity"]), axis=0)
            xt += v[0]
            yt += v[1]
            xList[t].append(xt)
            yList[t].append(yt)

    for t in range(T - 1):
        xList[t] = np.mean(xList[t])
        yList[t] = np.mean(yList[t])

    xList = np.array(xList) * scale
    yList = np.array(yList) * scale

    plt.plot(xList, yList)

    fileType = "Wound"
    cwd = os.getcwd()
    Fullfilenames = os.listdir(cwd + "/dat")
    filenames = []
    for filename in Fullfilenames:
        if fileType in filename:
            filenames.append(filename)

    filenames.sort()

    xList = [[] for col in range(T - 1)]
    yList = [[] for col in range(T - 1)]
    for filename in filenames:
        dfUnwound = dfVelocity[dfVelocity["Filename"] == filename]
        xt = 0
        yt = 0
        for t in range(T - 1):
            df = dfUnwound[dfUnwound["Time"] == t]
            v = np.mean(list(df["Velocity"]), axis=0)
            xt += v[0]
            yt += v[1]
            xList[t].append(xt)
            yList[t].append(yt)

    for t in range(T - 1):
        xList[t] = np.mean(xList[t])
        yList[t] = np.mean(yList[t])

    xList = np.array(xList) * scale
    yList = np.array(yList) * scale

    plt.plot(xList, yList)

    xList = [[] for col in range(T)]
    yList = [[] for col in range(T)]
    for filename in filenames:
        dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
        x0, y0 = dfWound["Position"].iloc[0]
        for t in range(T):
            xList[t].append(dfWound["Position"].iloc[t][0] - x0)
            yList[t].append(dfWound["Position"].iloc[t][1] - y0)

    for t in range(T):
        xList[t] = np.mean(xList[t])
        yList[t] = np.mean(yList[t])

    xList = np.array(xList) * scale
    yList = np.array(yList) * scale

    plt.plot(xList, yList)

    plt.legend(("Unwounded", "Wounded", "Wounsite"), loc="upper right")
    plt.xlabel("x")
    plt.ylabel(f"y")

    fig.savefig(
        f"results/migration path Unwounded Wounded and Wounsite",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

#  ------------------- Radial Velocity

run = True
if run:

    position = []
    xf = 256
    yf = 256
    position.append((xf, yf))

    for t in range(T - 1):

        x = [xf - 150, xf + 150]
        y = [yf - 150, yf + 150]
        dfxy = cl.sortGrid(dfVelocity[dfVelocity["Time"] == t], x, y)

        v = np.mean(list(dfxy["Velocity"]), axis=0)

        xf = xf + v[0]
        yf = yf + v[1]

        position.append((xf, yf))

    position.append((xf, yf))

    dfVelocityCenter = dfVelocity

    for t in range(T - 1):

        dfVelocityCenter["X"][dfVelocityCenter["Time"] == t] = (
            dfVelocity["X"][dfVelocity["Time"] == t] - position[t][0]
        )
        dfVelocityCenter["Y"][dfVelocityCenter["Time"] == t] = (
            dfVelocity["Y"][dfVelocity["Time"] == t] - position[t][1]
        )

    _dfvelocity = []
    for filename in filenames:
        df = dfVelocityCenter[dfVelocityCenter["Filename"] == filename]
        for t in range(T - 1):
            dft = df[df["Time"] == t]
            V = np.mean(list(dft["Velocity"]), axis=0)
            for i in range(len(dft)):
                _dfvelocity.append(
                    {
                        "Filename": filename,
                        "Label": dft["Label"].iloc[i],
                        "Time": dft["Time"].iloc[i],
                        "X": dft["X"].iloc[i],
                        "Y": dft["Y"].iloc[i],
                        "R": (dft["X"].iloc[i] ** 2 + dft["Y"].iloc[i] ** 2) ** 0.5,
                        "Theta": np.arctan2(dft["Y"].iloc[i], dft["X"].iloc[i]),
                        "Velocity": dft["Velocity"].iloc[i] - V,
                    }
                )
    dfvelocity = pd.DataFrame(_dfvelocity)

    grid = 40
    heatmap = np.zeros([int(T / 4), grid])
    for i in range(0, 180, 4):
        for j in range(grid):
            r = [80 / grid * j / scale, (80 / grid * j + 80 / grid) / scale]
            t = [i, i + 4]
            dfr = cl.sortRadius(dfvelocity, t, r)
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
