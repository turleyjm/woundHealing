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
                        {"Label": label, "T": t0, "X": x0, "Y": y0, "Velocity": v}
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
                            "Velocity": v,
                        }
                    )

dfVelocity = pd.DataFrame(_df2)

#  ------------------- Velocity feild
run = False
if run:
    cl.createFolder("results/video/")
    for t in range(T - 1):
        dfVelocityT = dfVelocity[dfVelocity["T"] == t]

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
        df = dfVelocity[dfVelocity["T"] == t]
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

run = True
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
            df = dfUnwound[dfUnwound["T"] == t]
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
            df = dfUnwound[dfUnwound["T"] == t]
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
