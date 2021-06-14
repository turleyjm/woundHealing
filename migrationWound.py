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

plt.rcParams.update({"font.size": 8})

# -------------------


def model(t, r, Acoeffs, Bcoeffs, Ccoeffs):

    A = Acoeffs[0] * r + Acoeffs[1]
    B = Bcoeffs[0] * r + Bcoeffs[1]
    C = Ccoeffs[0] * r + Ccoeffs[1]

    return A * t ** 2 + B * t + C


def Polynomial(x, coeffs):
    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]
    return a * x ** 2 + b * x + c


def residualsPolynomial(coeffs, y, x):
    return y - Polynomial(x, coeffs)


def Polynomialt0(x, coeffs):
    a = coeffs[0]
    b = coeffs[1]
    t0 = coeffs[2]
    return a * (x - t0) ** 2 + b * (x - t0)


def residualsPolynomialt0(coeffs, y, x):
    return y - Polynomialt0(x, coeffs)


def Linear(x, coeffs):
    m = coeffs[0]
    c = coeffs[1]
    return m * x + c


def residualsLinear(coeffs, y, x):
    return y - Linear(x, coeffs)


def Gaussian(x, coeffs):

    mu = coeffs[0]
    sigma = coeffs[1]
    A = coeffs[2]

    return (A / (sigma * (2 * np.pi) ** 0.5)) * np.exp(
        -0.5 * ((x - mu) ** 2) / (sigma ** 2)
    )


def residualsGaussian(coeffs, y, x):
    return y - Gaussian(x, coeffs)


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

if fileType == "WoundL":
    fast = 70
    med = 100
else:
    fast = 35
    med = 50

# get only fast/med/slow videos
# fileType = "fastHealersS"
# fileType = "medHealersS"
# fileType = "slowHealersS"
# fileType = "fastHealersL"
# fileType = "medHealersL"
# fileType = "slowHealersL"
# newFilenames = []
# for i in range(len(filenames)):
#     if "fastHealers" in fileType:
#         if finish[i] < fast:
#             newFilenames.append(filenames[i])
#     if "medHealers" in fileType:
#         if finish[i] < med:
#             if finish[i] > fast:
#                 newFilenames.append(filenames[i])
#     if "slowHealers" in fileType:
#         if finish[i] > med:
#             newFilenames.append(filenames[i])

# filenames = newFilenames

run = False
if run:
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

                        v = np.array(
                            [(xEnd - x0) / (tEnd - t0), (yEnd - y0) / (tEnd - t0)]
                        )

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
                        "Velocity": dft["Velocity"].iloc[i] - V,  # removed Vd
                    }
                )
    dfVelocity = pd.DataFrame(_dfVelocity)
    dfVelocity.to_pickle(f"databases/dfVelocity{fileType}.pkl")

else:
    dfVelocity = pd.read_pickle(f"databases/dfVelocity{fileType}.pkl")
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

        circle1 = plt.Circle((0, 0), radius[t], color="r")

        fig, ax = plt.subplots(figsize=(5, 5))
        plt.quiver(x, y, u, v, scale=10)
        ax.add_patch(circle1)
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


#  ------------------- Velocity feild single video

run = False
if run:
    for filename in filenames:

        radius = []

        dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")

        # area = dfWound["Area"].iloc[0] * (scale) ** 2
        # print(f"{filename} {area} {2*(area/np.pi)**0.5}")

        time = np.array(dfWound["Time"])
        area = np.array(dfWound["Area"]) * (scale) ** 2

        t = 0
        while pd.notnull(area[t]):
            t += 1

        finish = t - 1

        for t in range(T):
            if pd.isnull(area[t]):
                area[t] = 0

        for t in range(T):
            radius.append((area[t] / np.pi) ** 0.5)

        df = dfVelocity[dfVelocity["Filename"] == filename]
        cl.createFolder("results/video/")
        for t in range(T - 1):
            dfVelocityT = df[df["Time"] == t]

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

            circle1 = plt.Circle((0, 0), radius[t], color="r")

            fig, ax = plt.subplots(figsize=(5, 5))
            plt.quiver(x, y, u, v, scale=10)
            ax.add_patch(circle1)
            plt.title(f"time = {t}")
            fig.savefig(
                f"results/video/Velocity field wound centred {t}",
                dpi=300,
                transparent=True,
            )
            plt.close("all")

        # make video
        img_array = []

        for t in range(T - 1):
            img = cv2.imread(f"results/video/Velocity field wound centred {t}.png")
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        out = cv2.VideoWriter(
            f"results/Velocity field wound centred {filename}.mp4",
            cv2.VideoWriter_fourcc(*"DIVX"),
            3,
            size,
        )
        for i in range(len(img_array)):
            out.write(img_array[i])

        out.release()
        cv2.destroyAllWindows()

        shutil.rmtree("results/video")

#  ------------------- Winding number

run = False
if run:
    for filename in filenames:

        try:
            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
            area = np.array(dfWound["Area"]) * (scale) ** 2
            radius = (area[0] / np.pi) ** 0.5

            df = dfVelocity[dfVelocity["Filename"] == filename]
            dfVelocityT = df[df["Time"] < 10]

            r = [0, 20 / scale]
            Theta = np.linspace(-np.pi, np.pi, 11)
            x = []
            y = []
            U = []
            V = []
            for i in range(10):
                theta = [Theta[i], Theta[i + 1]]
                df = cl.sortSection(dfVelocityT, r, theta)
                x.append((radius + 20) * np.cos(np.mean(theta)))
                y.append((radius + 20) * np.sin(np.mean(theta)))
                v = np.mean(df["Velocity"])
                U.append(v[0])
                V.append(v[1])

            r = [40 / scale, 60 / scale]
            for i in range(10):
                theta = [Theta[i], Theta[i + 1]]
                df = cl.sortSection(dfVelocityT, r, theta)
                x.append((radius + 50) * np.cos(np.mean(theta)))
                y.append((radius + 50) * np.sin(np.mean(theta)))
                v = np.mean(df["Velocity"])
                np.isnan(v)
                U.append(v[0])
                V.append(v[1])

            thetaInner = []
            thetaOuter = []
            for i in range(10):
                thetaInner.append(np.arctan2(V[i], U[i]))
                thetaOuter.append(np.arctan2(V[10 + i], U[10 + i]))
            thetaInner.append(np.arctan2(V[0], U[0]))
            thetaOuter.append(np.arctan2(V[10], U[10]))

            windingInner = 0
            windingOuter = 0
            for i in range(10):
                dtheta = thetaInner[i + 1] - thetaInner[i]
                if dtheta > np.pi:
                    dtheta = -2 * np.pi + dtheta
                elif dtheta < -np.pi:
                    dtheta = 2 * np.pi + dtheta

                windingInner += dtheta

                dtheta = thetaOuter[i + 1] - thetaOuter[i]
                if dtheta > np.pi:
                    dtheta = -2 * np.pi + dtheta
                elif dtheta < -np.pi:
                    dtheta = 2 * np.pi + dtheta

                windingOuter += dtheta

            windingInner = windingInner / (2 * np.pi)
            windingOuter = windingOuter / (2 * np.pi)

            circle1 = plt.Circle((0, 0), radius, color="r")
            fig, ax = plt.subplots(figsize=(5, 5))
            plt.quiver(x, y, U, V, scale=10)
            ax.add_patch(circle1)
            plt.suptitle(f"Winding number bins {filename}")
            plt.title(
                f"Outer = {round(windingOuter,1)}, Inner = {round(windingInner,1)}"
            )
            fig.savefig(
                f"results/Winding number bins {filename}",
                dpi=300,
                transparent=True,
            )
            plt.close("all")
        except:
            continue


#  ------------------- Winding number

run = True
if run:
    for filename in filenames:

        dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
        area = np.array(dfWound["Area"]) * (scale) ** 2
        radius = (area[0] / np.pi) ** 0.5

        dfFile = dfVelocity[dfVelocity["Filename"] == filename]

        windingI = []
        windingO = []
        for t in range(T):
            try:
                dfVelocityT = dfFile[dfFile["Time"] == t]

                r = [0, 20 / scale]
                Theta = np.linspace(-np.pi, np.pi, 11)
                x = []
                y = []
                U = []
                V = []
                for i in range(10):
                    theta = [Theta[i], Theta[i + 1]]
                    df = cl.sortSection(dfVelocityT, r, theta)
                    x.append((radius + 20) * np.cos(np.mean(theta)))
                    y.append((radius + 20) * np.sin(np.mean(theta)))
                    v = np.mean(df["Velocity"])
                    U.append(v[0])
                    V.append(v[1])

                r = [40 / scale, 60 / scale]
                for i in range(10):
                    theta = [Theta[i], Theta[i + 1]]
                    df = cl.sortSection(dfVelocityT, r, theta)
                    x.append((radius + 50) * np.cos(np.mean(theta)))
                    y.append((radius + 50) * np.sin(np.mean(theta)))
                    v = np.mean(df["Velocity"])
                    np.isnan(v)
                    U.append(v[0])
                    V.append(v[1])

                thetaInner = []
                thetaOuter = []
                for i in range(10):
                    thetaInner.append(np.arctan2(V[i], U[i]))
                    thetaOuter.append(np.arctan2(V[10 + i], U[10 + i]))
                thetaInner.append(np.arctan2(V[0], U[0]))
                thetaOuter.append(np.arctan2(V[10], U[10]))

                windingInner = 0
                windingOuter = 0
                for i in range(10):
                    dtheta = thetaInner[i + 1] - thetaInner[i]
                    if dtheta > np.pi:
                        dtheta = -2 * np.pi + dtheta
                    elif dtheta < -np.pi:
                        dtheta = 2 * np.pi + dtheta

                    windingInner += dtheta

                    dtheta = thetaOuter[i + 1] - thetaOuter[i]
                    if dtheta > np.pi:
                        dtheta = -2 * np.pi + dtheta
                    elif dtheta < -np.pi:
                        dtheta = 2 * np.pi + dtheta

                    windingOuter += dtheta

                windingI.append(windingInner / (2 * np.pi))
                windingO.append(windingOuter / (2 * np.pi))
            except:
                windingI.append(np.nan)
                windingO.append(np.nan)

        windingI = np.array(windingI)
        windingO = np.array(windingO)

        t = range(T)
        # fig, ax = plt.subplots(figsize=(5, 5))
        # plt.scatter(t, windingI, marker=".", label="Inner")
        # plt.scatter(t, windingO + 0.10, marker=".", label="Outer")
        # plt.legend()
        # plt.title(f"Winding Number {filename}")
        # fig.savefig(
        #     f"results/Winding number with time {filename}",
        #     dpi=300,
        #     transparent=True,
        # )
        # plt.close("all")

        for i in range(T):
            try:
                windingI[i] = round(windingI[i])
                windingO[i] = round(windingO[i])
            except:
                windingI[i] = np.nan
                windingO[i] = np.nan

        inRowInner = [0]
        for i in range(T - 2):
            if windingI[i] == windingI[i + 1] == windingI[i + 2]:
                inRowInner.append(1)
            else:
                inRowInner.append(0)
        inRowInner.append(0)
        inRowInner = np.array(inRowInner)

        inRowOuter = [0]
        for i in range(T - 2):
            if windingO[i] == windingO[i + 1] == windingO[i + 2]:
                inRowOuter.append(1)
            else:
                inRowOuter.append(0)
        inRowOuter.append(0)
        inRowOuter = np.array(inRowOuter)

        windingInnerRow = sp.ndimage.morphology.binary_dilation(inRowInner).astype(
            windingI.dtype
        )
        windingOuterRow = sp.ndimage.morphology.binary_dilation(inRowOuter).astype(
            windingO.dtype
        )

        for i in range(T):
            if windingInnerRow[i] == 1:
                windingInnerRow[i] = windingI[i]
            else:
                windingInnerRow[i] = np.nan

        for i in range(T):
            if windingOuterRow[i] == 1:
                windingOuterRow[i] = windingO[i]
            else:
                windingOuterRow[i] = np.nan

        fig, ax = plt.subplots(figsize=(5, 5))
        plt.scatter(t, windingInnerRow, marker=".", label="Inner")
        plt.scatter(t, windingOuterRow + 0.10, marker=".", label="Outer")
        plt.xlim([0,181])
        plt.legend()
        plt.title(f"Winding Number {filename}")
        fig.savefig(
            f"results/Winding number with time 3 in row {filename}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")


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
        f"results/migration path {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

#  ------------------- Radial Velocity

run = False
if run:
    grid = 40
    heatmap = np.zeros([int(T / 4), grid])
    heatmapErr = np.zeros([int(T / 4), grid])
    for i in range(0, 180, 4):
        for j in range(grid):
            r = [80 / grid * j / scale, (80 / grid * j + 80 / grid) / scale]
            t = [i, i + 4]
            if r[0] < 0:
                r[0] = 0
            if r[1] < 0:
                r[1] = 0
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

    # -----------

    r = np.array(range(40)) * 2 + 1
    t = np.array(range(45)) * 4

    A = []
    B = []
    C = []

    for i in range(40):
        m = leastsq(
            residualsPolynomial,
            x0=(1, 10, 0.4),
            args=(heatmap[:, i][: int(medianFinish / 4)], t[: int(medianFinish / 4)]),
        )[0]
        A.append(m[0])
        B.append(m[1])
        C.append(m[2])

        # fig, ax = plt.subplots()
        # plt.plot(
        #     t[: int(medianFinish / 4)],
        #     heatmap[:, i][: int(medianFinish / 4)],
        #     label="exprement",
        # )
        # plt.plot(
        #     t[: int(medianFinish / 4)], Polynomial(t[: int(medianFinish / 4)], m), label="model"
        # )
        # plt.xlabel(r"Time")
        # plt.ylabel(r"Velocity $(\mu m mins^{-1})$")
        # plt.ylim([-0.4, 0.6])
        # plt.legend()
        # fig.savefig(
        #     f"results/Velocity Model {filename} r = {2*i}",
        #     dpi=300,
        #     transparent=True,
        # )
        # plt.close("all")

    Acoeffs = leastsq(residualsLinear, x0=(0, 0), args=(A, r))[0]

    Bcoeffs = leastsq(residualsLinear, x0=(0, 0), args=(B, r))[0]

    Ccoeffs = leastsq(residualsLinear, x0=(0, 0), args=(C, r))[0]

    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    plt.subplots_adjust(wspace=0.4)
    plt.gcf().subplots_adjust(bottom=0.15)
    ax[0].plot(r, A, label="exprement")
    ax[0].plot(r, Acoeffs[0] * r + Acoeffs[1], label="model")
    ax[0].set(xlabel="r", ylabel="A")
    ax[0].legend()

    ax[1].plot(r, B, label="exprement")
    ax[1].plot(r, Bcoeffs[0] * r + Bcoeffs[1], label="model")
    ax[1].set(xlabel="r", ylabel="B")
    ax[1].legend()

    ax[2].plot(r, C, label="exprement")
    ax[2].plot(r, Ccoeffs[0] * r + Ccoeffs[1], label="model")
    ax[2].set(xlabel="r", ylabel="C")
    ax[2].legend()

    fig.savefig(
        f"results/Model Coeff {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

    T = 181
    heatmapModel = np.zeros([int(T / 4), grid])
    for i in range(int(medianFinish / 4)):
        for j in range(grid):
            heatmapModel[i, j] = model(t[i], r[j], Acoeffs, Bcoeffs, Ccoeffs)

    # -----------

    D = np.zeros(40)
    Dmodel = np.zeros(40)
    for r in range(40):

        D[r] = sum(heatmap[: int(medianFinish / 4), r]) * 4
        Dmodel[r] = sum(heatmapModel[: int(medianFinish / 4), r]) * 4

    r = np.array(range(40)) * 2 + 1

    fig, ax = plt.subplots()
    plt.plot(r, D, label="exprement")
    plt.plot(r, Dmodel, label="model")
    plt.xlabel(r"r $(\mu m)$")
    plt.ylabel(r"Migration $(\mu m)$")
    plt.ylim([0, 12])
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.legend()
    fig.savefig(
        f"results/Migration {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

    dt, dr = 4, 80 / grid
    t, r = np.mgrid[0:180:dt, 1:81:dr]
    z_min, z_max = -0.4, 0.4
    midpoint = 1 - z_max / (z_max + abs(z_min))
    orig_cmap = matplotlib.cm.RdBu_r
    shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")

    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)
    c = ax[0].pcolor(t, r, heatmap, cmap=shifted_cmap, vmin=z_min, vmax=z_max)
    fig.colorbar(c, ax=ax[0])
    ax[0].axvline(x=medianFinish)
    ax[0].text(medianFinish + 2, 45, "Median Finish Time", size=6, rotation=90)
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel(r"Distance from wound edge $(\mu m)$")
    ax[0].title.set_text(f"Velocity {fileType}")

    c = ax[1].pcolor(t, r, heatmapModel, cmap=shifted_cmap, vmin=z_min, vmax=z_max)
    fig.colorbar(c, ax=ax[1])
    ax[1].axvline(x=medianFinish)
    ax[1].text(medianFinish + 2, 45, "Median Finish Time", size=6, rotation=90)
    ax[1].set_xlabel("Time (min)")
    ax[1].set_ylabel(r"Distance from wound edge $(\mu m)$")
    ax[1].title.set_text(f"Velocity Model {fileType}")

    c = ax[2].pcolor(
        t, r, heatmap - heatmapModel, cmap=shifted_cmap, vmin=z_min, vmax=z_max
    )
    fig.colorbar(c, ax=ax[2])
    ax[2].axvline(x=medianFinish)
    ax[2].text(medianFinish + 2, 45, "Median Finish Time", size=6, rotation=90)
    ax[2].set_xlabel("Time (min)")
    ax[2].set_ylabel(r"Distance from wound edge $(\mu m)$")
    ax[2].title.set_text(f"Velocity difference {fileType}")

    fig.savefig(
        f"results/Radial Velocity kymograph {fileType}",
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
        z_min, z_max = -0.5, 0.5
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
            f"results/rotational Velocity kymograph {filename}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")


#  ------------------- Radial Velocity indial videos
heatmaps = []
run = False
if run:

    Acoeffs = []
    Bcoeffs = []
    Ccoeffs = []
    t0 = []
    Am = [[], [], []]
    Ac = [[], [], []]
    Bm = [[], [], []]
    Bc = [[], [], []]
    Cm = [[], [], []]
    Cc = [[], [], []]

    for filename in filenames:

        dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")

        area = np.array(dfWound["Area"]) * (scale) ** 2
        t = 0
        while pd.notnull(area[t]):
            t += 1
        finish = t

        df = dfVelocity[dfVelocity["Filename"] == filename]
        grid = 40
        heatmap = np.zeros([int(T / 4), grid])
        heatmapErr = np.zeros([int(T / 4), grid])
        for i in range(0, 180, 4):
            for j in range(grid):
                r = [80 / grid * j / scale, (80 / grid * j + 80 / grid) / scale]
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
                        Vr.append(-np.matmul(R, v)[0])

                    heatmap[int(i / 4), j] = np.mean(Vr) * scale
                    heatmapErr[int(i / 4), j] = np.std(Vr) * scale / len(Vr)

        heatmaps.append(heatmap)

        r = np.array(range(40)) * 2 + 1
        t = np.array(range(45)) * 4

        A = []
        B = []
        C = []

        for i in range(40):
            m = leastsq(
                residualsPolynomial,
                x0=(1, 10, 0.4),
                args=(heatmap[:, i][: int(finish / 4)], t[: int(finish / 4)]),
            )[0]
            A.append(m[0])
            B.append(m[1])
            C.append(m[2])

            # fig, ax = plt.subplots()
            # plt.plot(
            #     t[: int(finish / 4)],
            #     heatmap[:, i][: int(finish / 4)],
            #     label="exprement",
            # )
            # plt.plot(
            #     t[: int(finish / 4)], Polynomial(t[: int(finish / 4)], m), label="model"
            # )
            # plt.xlabel(r"Time")
            # plt.ylabel(r"Velocity $(\mu m mins^{-1})$")
            # plt.ylim([-0.4, 0.6])
            # plt.legend()
            # fig.savefig(
            #     f"results/Velocity Model {filename} r = {2*i}",
            #     dpi=300,
            #     transparent=True,
            # )
            # plt.close("all")

        Acoeff = leastsq(residualsLinear, x0=(0, 0), args=(A, r))[0]

        Bcoeff = leastsq(residualsLinear, x0=(0, 0), args=(B, r))[0]

        Ccoeff = leastsq(residualsLinear, x0=(0, 0), args=(C, r))[0]

        # fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        # plt.subplots_adjust(wspace=0.4)
        # plt.gcf().subplots_adjust(bottom=0.15)
        # ax[0].plot(r, A, label="exprement")
        # ax[0].plot(r, Acoeff[0] * r + Acoeff[1], label="model")
        # ax[0].set(xlabel="r", ylabel="A")
        # ax[0].legend()

        # ax[1].plot(r, B, label="exprement")
        # ax[1].plot(r, Bcoeff[0] * r + Bcoeff[1], label="model")
        # ax[1].set(xlabel="r", ylabel="B")
        # ax[1].legend()

        # ax[2].plot(r, C, label="exprement")
        # ax[2].plot(r, Ccoeff[0] * r + Ccoeff[1], label="model")
        # ax[2].set(xlabel="r", ylabel="C")
        # ax[2].legend()

        # fig.savefig(
        #     f"results/Model Coeff {filename}", dpi=300, transparent=True,
        # )
        # plt.close("all")

        # T = 181
        # heatmapModel = np.zeros([int(T / 4), grid])
        # for i in range(int(finish / 4)):
        #     for j in range(grid):
        #         heatmapModel[i, j] = model(t[i], r[j], Acoeff, Bcoeff, Ccoeff)

        # # -----------

        # dt, dr = 4, 80 / grid
        # t, r = np.mgrid[0:180:dt, 1:81:dr]
        # z_min, z_max = -0.4, 0.4
        # midpoint = 1 - z_max / (z_max + abs(z_min))
        # orig_cmap = matplotlib.cm.RdBu_r
        # shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")

        # fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        # plt.subplots_adjust(wspace=0.3)
        # plt.gcf().subplots_adjust(bottom=0.15)
        # c = ax[0].pcolor(t, r, heatmap, cmap=shifted_cmap, vmin=z_min, vmax=z_max)
        # fig.colorbar(c, ax=ax[0])
        # ax[0].axvline(x=finish)
        # ax[0].text(finish + 2, 45, "Median Finish Time", size=6, rotation=90)
        # ax[0].set_xlabel("Time (min)")
        # ax[0].set_ylabel(r"Distance from wound edge $(\mu m)$")
        # ax[0].title.set_text(f"Velocity {filename}")

        # c = ax[1].pcolor(t, r, heatmapModel, cmap=shifted_cmap, vmin=z_min, vmax=z_max)
        # fig.colorbar(c, ax=ax[1])
        # ax[1].axvline(x=finish)
        # ax[1].text(finish + 2, 45, "Median Finish Time", size=6, rotation=90)
        # ax[1].set_xlabel("Time (min)")
        # ax[1].set_ylabel(r"Distance from wound edge $(\mu m)$")
        # ax[1].title.set_text(f"Velocity Model {filename}")

        # c = ax[2].pcolor(
        #     t, r, heatmap - heatmapModel, cmap=shifted_cmap, vmin=z_min, vmax=z_max
        # )
        # fig.colorbar(c, ax=ax[2])
        # ax[2].axvline(x=finish)
        # ax[2].text(finish + 2, 45, "Median Finish Time", size=6, rotation=90)
        # ax[2].set_xlabel("Time (min)")
        # ax[2].set_ylabel(r"Distance from wound edge $(\mu m)$")
        # ax[2].title.set_text(f"Velocity difference {filename}")

        # fig.savefig(
        #     f"results/Radial Velocity kymograph {filename}", dpi=300, transparent=True,
        # )
        # plt.close("all")

        r = np.array(range(40)) * 2 + 1
        A0 = leastsq(residualsLinear, x0=(0, 0), args=(A, r))[0][1]
        B0 = leastsq(residualsLinear, x0=(0, 0), args=(B, r))[0][1]
        C0 = leastsq(residualsLinear, x0=(0, 0), args=(C, r))[0][1]

        if B0 ** 2 - 4 * A0 * C0 >= 0:
            t0.append(
                [
                    (B0 + (B0 ** 2 - 4 * A0 * C0) ** 0.5) / (2 * A0),
                    (B0 - (B0 ** 2 - 4 * A0 * C0) ** 0.5) / (2 * A0),
                ]
            )
        else:
            t0.append(
                [
                    np.nan,
                    np.nan,
                ]
            )

        if finish < fast:
            Am[0].append(Acoeff[0])
            Ac[0].append(Acoeff[1])
            Bm[0].append(Bcoeff[0])
            Bc[0].append(Bcoeff[1])
            Cm[0].append(Ccoeff[0])
            Cc[0].append(Ccoeff[1])
        elif finish < med:
            Am[1].append(Acoeff[0])
            Ac[1].append(Acoeff[1])
            Bm[1].append(Bcoeff[0])
            Bc[1].append(Bcoeff[1])
            Cm[1].append(Ccoeff[0])
            Cc[1].append(Ccoeff[1])
        else:
            Am[2].append(Acoeff[0])
            Ac[2].append(Acoeff[1])
            Bm[2].append(Bcoeff[0])
            Bc[2].append(Bcoeff[1])
            Cm[2].append(Ccoeff[0])
            Cc[2].append(Ccoeff[1])

    T0 = []
    x = []
    for i in range(len(t0)):
        if np.nan == t0[i][0]:
            continue
        else:
            T0.append(t0[i][0])
            T0.append(t0[i][1])
            x.append(i + 1)
            x.append(i + 1)

    fig = plt.figure(1, figsize=(9, 8))
    plt.scatter(x, T0)
    plt.title(r"0=A_0 t_0^2 - B_0 t_0 + C_0")
    plt.xlabel("filename")
    plt.ylabel("t0")
    fig.savefig(
        f"results/t0 coeff",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

    AmErr = []
    AcErr = []
    BmErr = []
    BcErr = []
    CmErr = []
    CcErr = []

    for i in range(3):
        AmErr.append(np.std(Am[i]))
        AcErr.append(np.std(Ac[i]))
        BmErr.append(np.std(Bm[i]))
        BcErr.append(np.std(Bc[i]))
        CmErr.append(np.std(Cm[i]))
        CcErr.append(np.std(Cc[i]))
        Am[i] = np.mean(Am[i])
        Ac[i] = np.mean(Ac[i])
        Bm[i] = np.mean(Bm[i])
        Bc[i] = np.mean(Bc[i])
        Cm[i] = np.mean(Cm[i])
        Cc[i] = np.mean(Cc[i])

    x = ["Fast", "Med", "Slow"]

    fig, ax = plt.subplots(2, 3)
    plt.subplots_adjust(wspace=0.8)
    plt.suptitle(
        r"$v(r,t) = A(r)t^2 + B(r)t+ C(r)$ with $A(r) = M_A r + A_0, B(r) = M_B r + B_0, C(r) = M_C r + C_0$"
    )
    ax[0, 0].errorbar(x, Am, yerr=AmErr, marker="o", ls="none")
    ax[0, 0].set(ylabel=r"$M_A$")

    ax[1, 0].errorbar(x, Ac, yerr=AcErr, marker="o", ls="none")
    ax[1, 0].set(ylabel=r"$A_0$")

    ax[0, 1].errorbar(x, Bm, yerr=BmErr, marker="o", ls="none")
    ax[0, 1].set(ylabel=r"$M_A$")

    ax[1, 1].errorbar(x, Bc, yerr=BcErr, marker="o", ls="none")
    ax[1, 1].set(ylabel=r"$B_0$")

    ax[0, 2].errorbar(x, Cm, yerr=CmErr, marker="o", ls="none")
    ax[0, 2].set(ylabel=r"$M_C$")

    ax[1, 2].errorbar(x, Cc, yerr=CcErr, marker="o", ls="none")
    ax[1, 2].set(ylabel=r"$C_0$")

    fig.savefig(
        f"results/model coeff of each vid {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

    fig, ax = plt.subplots()
    t0_1 = []
    t0_2 = []
    for i in range(len(t0)):
        if t0[i][0] == np.nan:
            t0_1.append(0)
            t0_2.append(0)
        else:
            t0_1.append(t0[i][0])
            t0_2.append(t0[i][1])

    ax.scatter(range(1, int(len(filenames)) + 1), t0_1)
    ax.scatter(range(1, int(len(filenames)) + 1), t0_2)
    ax.set(ylabel=r"$t_0$")

    fig.savefig(
        f"results/velocity t0 {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


#  ------------------- Radial Velocity grouped vs indial

run = False
if run:
    if fileType == "WoundL":
        fileTypes = "fastHealersL", "medHealersL", "slowHealersL", "WoundL"
    else:
        fileTypes = "fastHealersS", "medHealersS", "slowHealersS", "WoundS"
    for fileType in fileTypes:
        dfVelocity = pd.read_pickle(f"databases/dfVelocity{fileType}.pkl")
        radius = [[] for col in range(T)]
        for filename in filenames:

            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")

            time = np.array(dfWound["Time"])
            area = np.array(dfWound["Area"]) * (scale) ** 2
            for t in range(T):
                if pd.isnull(area[t]):
                    area[t] = 0

            for t in range(T):
                radius[t].append((area[t] / np.pi) ** 0.5)

        for t in range(T):
            radius[t] = np.mean(radius[t])

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

        r = np.array(range(40)) * 2 + 1
        t = np.array(range(45)) * 4

        A = []
        B = []
        C = []

        for i in range(40):
            m = leastsq(
                residualsPolynomial,
                x0=(1, 10, 0.4),
                args=(
                    heatmap[:, i][: int(medianFinish / 4)],
                    t[: int(medianFinish / 4)],
                ),
            )[0]
            A.append(m[0])
            B.append(m[1])
            C.append(m[2])

        Acoeffs = leastsq(residualsLinear, x0=(0, 0), args=(A, r))[0]

        Bcoeffs = leastsq(residualsLinear, x0=(0, 0), args=(B, r))[0]

        Ccoeffs = leastsq(residualsLinear, x0=(0, 0), args=(C, r))[0]

        # Am = []
        # Ac = []
        # Bm = []
        # Bc = []
        # Cm = []
        # Cc = []
        # AmErr = []
        # AcErr = []
        # BmErr = []
        # BcErr = []
        # CmErr = []
        # CcErr = []

        Am.append(Acoeffs[0])
        Ac.append(Acoeffs[1])
        Bm.append(Bcoeffs[0])
        Bc.append(Bcoeffs[1])
        Cm.append(Ccoeffs[0])
        Cc.append(Ccoeffs[1])
        AmErr.append(0)
        AcErr.append(0)
        BmErr.append(0)
        BcErr.append(0)
        CmErr.append(0)
        CcErr.append(0)

    x = [r"$F$", r"$M$", r"$S$", r"$\bar F$", r"$\bar M$", r"$\bar S$", "All"]

    plt.rcParams.update({"font.size": 10})

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    plt.subplots_adjust(wspace=0.6)
    plt.suptitle(
        r"$v(r,t) = A(r)t^2 + B(r)t+ C(r)$ with $A(r) = M_A r + A_0, B(r) = M_B r + B_0, C(r) = M_C r + C_0$"
    )
    ax[0, 0].errorbar(x, Am, yerr=AmErr, marker="o", ls="none")
    ax[0, 0].set(ylabel=r"$M_A$")

    ax[1, 0].errorbar(x, Ac, yerr=AcErr, marker="o", ls="none")
    ax[1, 0].set(ylabel=r"$A_0$")

    ax[0, 1].errorbar(x, Bm, yerr=BmErr, marker="o", ls="none")
    ax[0, 1].set(ylabel=r"$M_B$")

    ax[1, 1].errorbar(x, Bc, yerr=BcErr, marker="o", ls="none")
    ax[1, 1].set(ylabel=r"$B_0$")

    ax[0, 2].errorbar(x, Cm, yerr=CmErr, marker="o", ls="none")
    ax[0, 2].set(ylabel=r"$M_C$")

    ax[1, 2].errorbar(x, Cc, yerr=CcErr, marker="o", ls="none")
    ax[1, 2].set(ylabel=r"$C_0$")

    fig.savefig(
        f"results/model coeff compare slow fast mean {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")
