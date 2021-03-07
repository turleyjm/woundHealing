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
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
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


def model(x, coeffs):
    return coeffs / x


def residuals(coeffs, y, x):
    return y - model(x, coeffs)


def model2(t, coeffs):
    return coeffs[0] * np.exp(coeffs[1] * t)


def residuals2(coeffs, y, t):
    return y - model2(t, coeffs)


def model3(M, A, c):
    return (A / M[1]) * np.exp(-c * M[0])


def densityDrift(filename):

    df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")
    A0 = np.mean(list(df["Area"][df["Time"] == 0] * scale ** 2))

    mu = []
    time = range(181)
    for t in time:
        prop = list(df["Area"][df["Time"] == t] * scale ** 2)
        mu.append(np.mean(prop) / A0)
    for i in range(7):
        mu.append(mu[-1])
    mu = np.array(mu)

    mu = movingAverage(mu, 8)

    density = 1 / mu

    return density


def movingAverage(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def driftCorrection(r, theta, D, time):

    s = r * (1 - (D[time] / D[time + 1]) ** 0.5)
    Vd = s * np.array([np.cos(theta), np.sin(theta)])
    return Vd


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

meanFinish = int(np.mean(finish))
medianFinish = int(np.median(finish))
minFinish = int(min(finish))
woundEdge = np.mean(woundEdge)


_df2 = []
for filename in filenames:
    dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
    df = pd.read_pickle(f"dat/{filename}/nucleusTracks{filename}.pkl")
    dist = sm.io.imread(f"dat/{filename}/distanceWound{filename}.tif").astype(float)

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
                r = dist[int(t0), int(x0), int(y0)]
                if r == 0:
                    r = -1

                tdelta = tMax - t0
                if tdelta > 5:
                    t5 = t[j + 5]
                    x5 = x[j + 5]
                    y5 = y[j + 5]

                    v = np.array([(x5 - x0) / 5, (y5 - y0) / 5])

                    [wx, wy] = dfWound["Position"].iloc[int(t0)]
                    _df2.append(
                        {
                            "Filename": filename,
                            "Label": label,
                            "Time": t0,
                            "X": x0 - wx,
                            "Y": y0 - wy,
                            "R": r,
                            "Theta": np.arctan2(y0 - wy, x0 - wx),
                            "Velocity": v,
                        }
                    )
                else:
                    tEnd = t[-1]
                    xEnd = x[-1]
                    yEnd = y[-1]

                    v = np.array([(xEnd - x0) / (tEnd - t0), (yEnd - y0) / (tEnd - t0)])

                    [wx, wy] = dfWound["Position"].iloc[int(t0)]
                    _df2.append(
                        {
                            "Filename": filename,
                            "Label": label,
                            "Time": t0,
                            "X": x0 - wx,
                            "Y": y0 - wy,
                            "R": r,
                            "Theta": np.arctan2(y0 - wy, x0 - wx),
                            "Velocity": v,
                        }
                    )

dfvelocity = pd.DataFrame(_df2)

radius = [[] for col in range(T)]
for filename in filenames:

    dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")

    # area = dfWound["Area"].iloc[0] * (scale) ** 2
    # print(f"{filename} {area} {2*(area/np.pi)**0.5}")

    time = np.array(dfWound["Time"])
    area = np.array(dfWound["Area"]) * (scale) ** 2
    for t in range(T):
        if pd.isnull(area[t]):
            area[t] = 0

    for t in range(T):
        radius[t].append((area[t] / np.pi) ** 0.5)

for t in range(T):
    radius[t] = np.mean(radius[t])

#  ------------------- Velocity - mean velocity + density drift correction

_dfVelocity = []
for filename in filenames:
    # change in density drift
    D = densityDrift(filename)

    df = dfvelocity[dfvelocity["Filename"] == filename]
    for t in range(T - 1):
        dft = df[df["Time"] == t]
        V = np.mean(list(dft["Velocity"]), axis=0)
        for i in range(len(dft)):
            r = dft["R"].iloc[i]
            theta = dft["Theta"].iloc[i]
            time = int(dft["Time"].iloc[i])
            Vd = driftCorrection(r, theta, D, time)
            _dfVelocity.append(
                {
                    "Filename": filename,
                    "Label": dft["Label"].iloc[i],
                    "Time": dft["Time"].iloc[i],
                    "X": dft["X"].iloc[i],
                    "Y": dft["Y"].iloc[i],
                    "R": dft["R"].iloc[i],
                    "Theta": dft["Theta"].iloc[i],
                    "Velocity": dft["Velocity"].iloc[i] - V + Vd,
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
    heatmapErr = np.zeros([int(T / 4), grid])
    for i in range(0, 180, 4):
        for j in range(grid):
            r = [80 / grid * j / scale, (80 / grid * j + 80 / grid) / scale]
            t = [i, i + 4]
            dfr = cl.sortRadius(dfVelocity, t, r)
            if list(dfr["Velocity"]) == []:
                Vr = np.nan
            else:
                Vr = []
                for k in range(len(dfr)):
                    v = dfr["Velocity"].iloc[k]
                    theta = dfr["Theta"].iloc[k]
                    R = cl.rotation_matrix(-theta)
                    Vr.append(-np.matmul(R, v)[0])

                heatmap[int(i / 4), j] = np.mean(Vr) * scale
                heatmapErr[int(i / 4), j] = np.std(Vr) * scale / len(Vr)

    dt, dr = 4, 80 / grid
    t, r = np.mgrid[0:180:dt, 0:80:dr]
    z_min, z_max = -0.35, 0.35
    midpoint = 1 - z_max / (z_max + abs(z_min))
    orig_cmap = matplotlib.cm.RdBu_r
    shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")

    fig, ax = plt.subplots()
    c = ax.pcolor(t, r, heatmap, cmap=shifted_cmap, vmin=z_min, vmax=z_max)
    fig.colorbar(c, ax=ax)
    plt.axvline(x=medianFinish)
    plt.text(medianFinish + 2, 45, "Median Finish Time", size=10, rotation=90)
    plt.xlabel("Time (min)")
    plt.ylabel(r"Distance from wound edge $(\mu m)$")
    plt.title(f"Velocity {fileType}")
    fig.savefig(
        f"results/Radial Velocity kymograph DDC {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")

    # -----------

    r = np.array(range(40)) * 2 + 1
    t = np.array(range(45)) * 4
    T, R = np.meshgrid(t, r)
    for i in range(len(R[0])):
        R[:, i] += int(radius[4 * i])

    xdata = np.vstack((T.ravel(), R.ravel()))

    coeffs = curve_fit(model3, xdata, heatmap.ravel())[0]

    T = 181
    heatmapModel = np.zeros([int(T / 4), grid])
    for i in range(0, 180, 4):
        for j in range(grid):
            heatmapModel[int(i / 4), j] = model3(
                [i, 2 * j + radius[i] + 1], coeffs[0], coeffs[1]
            )

    dt, dr = 4, 80 / grid
    t, r = np.mgrid[0:180:dt, 0:80:dr]

    fig, ax = plt.subplots()
    c = ax.pcolor(t, r, heatmapModel, cmap=shifted_cmap, vmin=z_min, vmax=z_max)
    fig.colorbar(c, ax=ax)
    plt.axvline(x=medianFinish)
    plt.text(medianFinish + 2, 45, "Median Finish Time", size=10, rotation=90)
    plt.xlabel("Time (mins)")
    plt.ylabel(r"Distance from wound edge $(\mu m)$")
    plt.title(f"Velocity model {fileType}")
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.20)
    fig.savefig(
        f"results/Radial Velocity model kymograph DDC {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

    # -----------

    D = np.zeros(40)
    Dmodel = np.zeros(40)
    Dpred = np.zeros(40)
    r0 = radius[0]
    R = np.linspace(0, 80, 200)
    for r in range(40):

        D[r] = sum(heatmap[:, r]) * 4
        Dmodel[r] = sum(heatmapModel[:, r]) * 4
        Dpred[r] = r0 + r * 2 - ((r0 + r * 2) ** 2 - r0 ** 2) ** 0.5

    fig, ax = plt.subplots()
    plt.plot(np.array(range(40)) * 2, D, label="exprement")
    plt.plot(np.array(range(40)) * 2, Dmodel, label="model")
    # plt.plot(np.array(range(40)) * 2, Dpred, label="model 2")
    plt.xlabel(r"Start from Wound Edge $(\mu m)$")
    plt.ylabel(r"Migration $(\mu m)$")
    plt.ylim([0, 7.5])
    plt.legend()
    fig.savefig(
        f"results/Migration {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")

    dt, dr = 4, 80 / grid
    t, r = np.mgrid[0:180:dt, 0:80:dr]

    fig, ax = plt.subplots()
    c = ax.pcolor(
        t, r, heatmap - heatmapModel, cmap=shifted_cmap, vmin=z_min, vmax=z_max
    )
    fig.colorbar(c, ax=ax)
    plt.axvline(x=medianFinish)
    plt.text(medianFinish + 2, 45, "Median Finish Time", size=10, rotation=90)
    plt.xlabel("Time (mins)")
    plt.ylabel(r"Distance from wound edge $(\mu m)$")
    plt.title(f"Velocity data - model {fileType}")
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.20)
    fig.savefig(
        f"results/Radial Velocity data - model kymograph DDC {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

    # make into a video
    run = False
    if run:
        cl.createFolder("results/video/")

        vidDist = np.ones([45, 511, 511])

        for t in range(45):
            r = radius[t * 4]
            if r == 0:
                r = scale
            rr, cc = sm.draw.circle(255, 255, r / scale)
            img = np.ones([511, 511])
            img[rr, cc] = 0
            vidDist[t] = sp.ndimage.morphology.distance_transform_edt(img)

        vid = np.zeros([45, 511, 511])
        for t in range(45):
            img = vidDist[t]
            vid[t][img == 0] = np.nan
            for r in range(grid):
                vid[t][img > 2 * r / scale] = heatmap[t, r]

        dx, dy = scale, scale
        x, y = np.mgrid[
            -255 * scale : 256 * scale : dx, -255 * scale : 256 * scale : dy
        ]
        z_min, z_max = -0.35, 0.35
        midpoint = 1 - z_max / (z_max + abs(z_min))
        orig_cmap = matplotlib.cm.RdBu_r
        shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")

        for t in range(45):
            fig, ax = plt.subplots()
            c = ax.pcolor(x, y, vid[t], cmap=shifted_cmap, vmin=z_min, vmax=z_max)
            fig.colorbar(c, ax=ax)
            plt.title(f"Velocity {fileType} {int(4*t)}mins")
            fig.savefig(
                f"results/video/Velocity Video {fileType} {t}",
                dpi=300,
                transparent=True,
            )
            plt.close("all")

        # make video
        img_array = []

        for t in range(45):
            img = cv2.imread(f"results/video/Velocity Video {fileType} {t}.png")
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        out = cv2.VideoWriter(
            f"results/Velocity Video {fileType}.mp4",
            cv2.VideoWriter_fourcc(*"DIVX"),
            3,
            size,
        )
        for i in range(len(img_array)):
            out.write(img_array[i])

        out.release()
        cv2.destroyAllWindows()

        shutil.rmtree("results/video")


#  ------------------- Rotational Velocity

run = False
if run:
    for filename in filenames:
        df = dfVelocity[dfVelocity["Filename"] == filename]
        grid = 50
        heatmap = np.zeros([int(T / 4), grid])
        heatmapErr = np.zeros([int(T / 4), grid])
        for i in range(0, 180, 4):
            for j in range(grid):
                r = [100 / grid * j / scale, (100 / grid * j + 100 / grid) / scale]
                t = [i, i + 4]
                dfr = cl.sortRadius(df, t, r)
                if list(dfr["Velocity"]) == []:
                    Vr = np.nan
                else:
                    Vr = []
                    for k in range(len(dfr)):
                        v = dfr["Velocity"].iloc[k]
                        theta = dfr["Theta"].iloc[k]
                        R = cl.rotation_matrix(-theta)
                        Vr.append(-np.matmul(R, v)[1])

                    heatmap[int(i / 4), j] = np.mean(Vr) * scale
                    heatmapErr[int(i / 4), j] = np.std(Vr) * scale / len(Vr)

        dt, dr = 4, 100 / grid
        t, r = np.mgrid[0:180:dt, 0:100:dr]
        z_min, z_max = -0.35, 0.35
        midpoint = 1 - z_max / (z_max + abs(z_min))
        orig_cmap = matplotlib.cm.RdBu_r
        shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")

        fig, ax = plt.subplots()
        c = ax.pcolor(t, r, heatmap, cmap=shifted_cmap, vmin=z_min, vmax=z_max)
        fig.colorbar(c, ax=ax)
        plt.axvline(x=medianFinish)
        plt.text(medianFinish + 2, 50, "Median Finish Time", size=10, rotation=90)
        plt.xlabel("Time (min)")
        plt.ylabel(r"Distance from wound edge $(\mu m)$")
        plt.title(f"Rotational Velocity {fileType}")
        fig.savefig(
            f"results/rotational Velocity kymograph DDC {filename}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")
