import os
import shutil
from math import floor, log10, tau

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
from scipy.stats import pearsonr
import shapely
import skimage as sm
import skimage.io
import skimage.measure
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
import tifffile
from skimage.draw import circle_perimeter
from scipy import optimize
from scipy.interpolate import UnivariateSpline
import xml.etree.ElementTree as et

import cellProperties as cell
import findGoodCells as fi
import commonLiberty as cl


# -------------------


def derivative(f, a, h=0.01):

    return (f(a + h) - f(a - h)) / (2 * h)


# -------------------

filenames, fileType = cl.getFilesType()
scale = 147.91 / 512

dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
dfVelocity = pd.read_pickle(f"databases/dfVelocity{fileType}.pkl")

All_dA = []
All_dq1 = []
All_dw = []
All_v = []
halfLifes = []
quarterLifes = []
healTimes = []

if False:
    for filename in filenames:
        T = 181
        plt.rcParams.update({"font.size": 16})
        fig, ax = plt.subplots(2, 2, figsize=(9, 8))
        plt.subplots_adjust(wspace=0.4, bottom=0.1)
        dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
        area = np.array(dfWound["Area"]) * (scale) ** 2

        area0 = area[0]
        i = 0
        while area[i] > 0:
            i += 1
        finish = i

        i = 0
        while area[i] > area0 / 2:
            i += 1
        halfLife = i
        halfLifes.append(halfLife)

        area0 = area[0]
        i = 0
        while area[i] > area0 / 4:
            i += 1
        quarterLife = i
        quarterLifes.append(quarterLife)

        for t in range(T):
            if pd.isnull(area[t]):
                area[t] = 0

        healTime = sum(area) / area[0]
        healTimes.append(int(healTime))

        df = dfShape[dfShape["Filename"] == filename]
        grid = 50
        heatmapA = np.zeros([int(T / 4), grid])
        for i in range(45):
            for j in range(grid):
                r = [100 / grid * j / scale, (100 / grid * j + 100 / grid) / scale]
                t = [4 * i, 4 * i + 4]
                dfr = cl.sortRadius(df, t, r)
                if list(dfr["Area"]) == []:
                    Ar = np.nan
                else:
                    Ar = dfr["Area"]
                    heatmapA[int(i), j] = np.mean(Ar)

        Area = np.mean(heatmapA[:, 0:10], axis=1)
        t = np.array(range(45)) * 4
        spl_A = UnivariateSpline(t, Area / np.max(Area), k=5)
        xs = np.linspace(0, 180, 1000)

        ax[0, 0].plot(t, Area)
        ax[0, 0].plot(xs, spl_A(xs) * np.max(Area))
        ax[0, 0].set_xlabel("Time (mins)")
        ax[0, 0].set_ylabel(r"Cell Area $(\mu m^2)$")
        ax[0, 0].set_ylim([5, 35])

        heatmapq1 = np.zeros([int(T / 4), grid])
        grid = 40
        for i in range(45):
            for j in range(grid):
                r = [80 / grid * j / scale, (80 / grid * j + 80 / grid) / scale]
                t = [4 * i, 4 * i + 4]
                dfr = cl.sortRadius(df, t, r)
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

        q1 = np.mean(heatmapq1[:, 0:10], axis=1)
        t = np.array(range(45)) * 4
        spl_q1 = UnivariateSpline(t, q1, k=5)

        ax[0, 1].plot(t, q1)
        ax[0, 1].plot(xs, spl_q1(xs))
        ax[0, 1].set_xlabel("Time (mins)")
        ax[0, 1].set_ylabel("q1")
        ax[0, 1].set_ylim([-0.035, 0.035])

        df = dfVelocity[dfVelocity["Filename"] == filename]
        grid = 40
        heatmap = np.zeros([int(T / 4), grid])
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

        v = np.mean(heatmap[:, 0:10], axis=1)
        t = np.array(range(45)) * 4
        spl_v = UnivariateSpline(t, v, k=5)

        ax[1, 0].plot(t, v)
        ax[1, 0].plot(xs, spl_v(xs))
        ax[1, 0].set_xlabel("Time (mins)")
        ax[1, 0].set_ylabel(r"v $(\mu m)$")
        ax[1, 0].set_ylim([-0.5, 0.5])

        area = np.array(dfWound["Area"]) * (scale) ** 2

        for t in range(T):
            if pd.isnull(area[t]):
                area[t] = 0

        T = np.array(range(181))
        spl_w = UnivariateSpline(T, area / 10, k=5)

        ax[1, 1].plot(T, area)
        ax[1, 1].plot(xs, spl_w(xs) * 10)
        ax[1, 1].axvline(halfLife, color="r")
        ax[1, 1].set_xlabel("Time (mins)")
        ax[1, 1].set_ylabel(r"Wound Area $(\mu m^2)$")
        ax[1, 1].set_ylim([0, 1300])
        plt.suptitle(f"Wound healing properties {filename}")
        fig.savefig(
            f"results/Wound healing properties {filename}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

        dA = []
        dq1 = []
        dw = []
        v = []

        for t in T:
            dA.append(derivative(spl_A, t))
            dq1.append(derivative(spl_q1, t))
            dw.append(derivative(spl_w, t) * 10)
            v.append(spl_v(t))
            if t < finish - 5:
                All_dA.append(derivative(spl_A, t))
                All_dq1.append(derivative(spl_q1, t))
                All_dw.append(derivative(spl_w, t) * 10)
                All_v.append(spl_v(t))

        if fileType == "WoundL":
            limWound = [-120, 20]
        else:
            limWound = [-40, 10]

        plt.rcParams.update({"font.size": 18})
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        plt.subplots_adjust(wspace=0.5)
        plt.gcf().subplots_adjust(bottom=0.15)
        ax[0, 0].scatter(
            dw[0 : finish - 5], dA[0 : finish - 5], c=range(finish - 5), cmap="RdBu"
        )
        ax[0, 0].set(xlabel=r"$\frac{dw_A}{dt}$", ylabel=r"$\frac{dc_A}{dt}$")
        ax[0, 0].set_xlim(limWound)
        ax[0, 0].set_ylim([-0.03, 0.02])

        ax[0, 1].scatter(
            dw[0 : finish - 5], dq1[0 : finish - 5], c=range(finish - 5), cmap="RdBu"
        )
        ax[0, 1].set(xlabel=r"$\frac{dw_A}{dt}$", ylabel=r"$\frac{dq_1}{dt}$")
        ax[0, 1].set_xlim(limWound)
        ax[0, 1].set_ylim([-0.0005, 0.002])

        ax[1, 0].scatter(
            dw[0 : finish - 5], v[0 : finish - 5], c=range(finish - 5), cmap="RdBu"
        )
        ax[1, 0].set(xlabel=r"$\frac{dw_A}{dt}$", ylabel=r"$v$")
        ax[1, 0].set_xlim(limWound)
        ax[1, 0].set_ylim([-0.2, 0.4])

        ax[1, 1].scatter(
            dq1[0 : finish - 5], v[0 : finish - 5], c=range(finish - 5), cmap="RdBu"
        )
        ax[1, 1].set(xlabel=r"$\frac{dq_1}{dt}$", ylabel=r"$v$")
        ax[1, 1].set_xlim([-0.0005, 0.002])
        ax[1, 1].set_ylim([-0.2, 0.4])

        plt.suptitle(f"Corriation change in properties {filename}")

        fig.savefig(
            f"results/Corriation change in properties {filename}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

        # print(f"dw {min(dw)}, {max(dw)}")
        # print(f"dq1 {min(dq1)}, {max(dq1)}")
        # print(f"v {min(v)}, {max(v)}")

    limWound = [-120, 20]
    xx = np.linspace(-120, 20, num=500)

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.4, bottom=0.1)
    result = sp.stats.linregress(All_dw, All_dA)
    ax[0, 0].scatter(All_dw, All_dA)
    ax[0, 0].plot(xx, result.slope * xx + result.intercept, "r")
    ax[0, 0].set(xlabel=r"$\frac{dw_A}{dt}$", ylabel=r"$\frac{dc_A}{dt}$")
    ax[0, 0].set_xlim(limWound)
    ax[0, 0].title.set_text(f"Correlation = {round(pearsonr(All_dw, All_dA)[0], 3)}")

    result = sp.stats.linregress(All_dw, All_dq1)
    ax[0, 1].scatter(All_dw, All_dq1)
    ax[0, 1].plot(xx, result.slope * xx + result.intercept, "r")
    ax[0, 1].set(xlabel=r"$\frac{dw_A}{dt}$", ylabel=r"$\frac{dq_1}{dt}$")
    ax[0, 1].set_xlim(limWound)
    ax[0, 1].set_ylim([-0.0005, 0.002])
    ax[0, 1].title.set_text(f"Correlation = {round(pearsonr(All_dw, All_dq1)[0], 3)}")

    result = sp.stats.linregress(All_dw, All_v)
    ax[1, 0].scatter(All_dw, All_v)
    ax[1, 0].plot(xx, result.slope * xx + result.intercept, "r")
    ax[1, 0].set(xlabel=r"$\frac{dw_A}{dt}$", ylabel=r"$v$")
    ax[1, 0].set_xlim(limWound)
    ax[1, 0].set_ylim([-0.2, 0.4])
    ax[1, 0].title.set_text(f"Correlation = {round(pearsonr(All_dw, All_v)[0], 3)}")

    xx = np.linspace(-0.2, 0.4, num=500)
    result = sp.stats.linregress(All_dq1, All_v)

    ax[1, 1].scatter(All_dq1, All_v)
    ax[1, 1].plot(xx, result.slope * xx + result.intercept, "r")
    ax[1, 1].set(xlabel=r"$\frac{dq_1}{dt}$", ylabel=r"$v$")
    ax[1, 1].set_xlim([-0.0005, 0.002])
    ax[1, 1].set_ylim([-0.2, 0.4])
    ax[1, 1].title.set_text(f"Correlation = {round(pearsonr(All_dq1, All_v)[0], 3)}")

    plt.suptitle(f"Corriation change in properties all {fileType}")

    fig.savefig(
        f"results/Corriation change in properties all {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


if False:
    for filename in filenames:
        T = 181
        plt.rcParams.update({"font.size": 14})
        fig, ax = plt.subplots(2, 2, figsize=(9, 8))
        plt.subplots_adjust(wspace=0.5)
        dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
        area = np.array(dfWound["Area"]) * (scale) ** 2

        area0 = area[0]
        i = 0
        while area[i] > 0:
            i += 1
        finish = i

        df = dfShape[dfShape["Filename"] == filename]
        grid = 50
        heatmapA = np.zeros([int(T / 4), grid])
        for i in range(45):
            for j in range(grid):
                r = [100 / grid * j / scale, (100 / grid * j + 100 / grid) / scale]
                t = [4 * i, 4 * i + 4]
                dfr = cl.sortRadius(df, t, r)
                if list(dfr["Area"]) == []:
                    Ar = np.nan
                else:
                    Ar = dfr["Area"]
                    heatmapA[int(i), j] = np.mean(Ar)

        Area = np.mean(heatmapA[:, 0:10], axis=1)
        t = np.array(range(45)) * 4
        spl_A = UnivariateSpline(t, Area / np.max(Area), k=5)
        xs = np.linspace(0, 180, 1000)

        # ax[0, 0].plot(t, Area)
        ax[0, 0].plot(xs, spl_A(xs) * np.max(Area), label=r"$0-20 \mu m$")

        Area = np.mean(heatmapA[:, 10:20], axis=1)
        t = np.array(range(45)) * 4
        spl_A = UnivariateSpline(t, Area / np.max(Area), k=5)
        xs = np.linspace(0, 180, 1000)

        # ax[0, 0].plot(t, Area)
        ax[0, 0].plot(xs, spl_A(xs) * np.max(Area), label=r"$20-40 \mu m$")

        Area = np.mean(heatmapA[:, 20:30], axis=1)
        t = np.array(range(45)) * 4
        spl_A = UnivariateSpline(t, Area / np.max(Area), k=5)
        xs = np.linspace(0, 180, 1000)

        # ax[0, 0].plot(t, Area)
        ax[0, 0].plot(xs, spl_A(xs) * np.max(Area), label=r"$40-60 \mu m$")

        Area = np.mean(heatmapA[:, 30:40], axis=1)
        t = np.array(range(45)) * 4
        spl_A = UnivariateSpline(t, Area / np.max(Area), k=5)
        xs = np.linspace(0, 180, 1000)

        # ax[0, 0].plot(t, Area)
        ax[0, 0].plot(xs, spl_A(xs) * np.max(Area), label=r"$60-80 \mu m$")
        ax[0, 0].set_xlabel("Time (mins)")
        ax[0, 0].set_ylabel(r"Cell Area $(\mu m^2)$")
        ax[0, 0].set_ylim([5, 35])
        ax[0, 0].legend()

        heatmapq1 = np.zeros([int(T / 4), grid])
        grid = 40
        for i in range(45):
            for j in range(grid):
                r = [80 / grid * j / scale, (80 / grid * j + 80 / grid) / scale]
                t = [4 * i, 4 * i + 4]
                dfr = cl.sortRadius(df, t, r)
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

        q1 = np.mean(heatmapq1[:, 0:10], axis=1)
        t = np.array(range(45)) * 4
        spl_q1 = UnivariateSpline(t, q1, k=5)

        # ax[0, 1].plot(t, q1)
        ax[0, 1].plot(xs, spl_q1(xs))

        q1 = np.mean(heatmapq1[:, 10:20], axis=1)
        t = np.array(range(45)) * 4
        spl_q1 = UnivariateSpline(t, q1, k=5)

        # ax[0, 1].plot(t, q1)
        ax[0, 1].plot(xs, spl_q1(xs))

        q1 = np.mean(heatmapq1[:, 20:30], axis=1)
        t = np.array(range(45)) * 4
        spl_q1 = UnivariateSpline(t, q1, k=5)

        # ax[0, 1].plot(t, q1)
        ax[0, 1].plot(xs, spl_q1(xs))

        q1 = np.mean(heatmapq1[:, 30:40], axis=1)
        t = np.array(range(45)) * 4
        spl_q1 = UnivariateSpline(t, q1, k=5)

        # ax[0, 1].plot(t, q1)
        ax[0, 1].plot(xs, spl_q1(xs))
        ax[0, 1].set_xlabel("Time (mins)")
        ax[0, 1].set_ylabel("q1")
        ax[0, 1].set_ylim([-0.035, 0.035])

        df = dfVelocity[dfVelocity["Filename"] == filename]
        grid = 40
        heatmap = np.zeros([int(T / 4), grid])
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

        v = np.mean(heatmap[:, 0:10], axis=1)
        t = np.array(range(45)) * 4
        spl_v = UnivariateSpline(t, v, k=5)

        # ax[1, 0].plot(t, v)
        ax[1, 0].plot(xs, spl_v(xs))

        v = np.mean(heatmap[:, 10:20], axis=1)
        t = np.array(range(45)) * 4
        spl_v = UnivariateSpline(t, v, k=5)

        # ax[1, 0].plot(t, v)
        ax[1, 0].plot(xs, spl_v(xs))

        v = np.mean(heatmap[:, 20:30], axis=1)
        t = np.array(range(45)) * 4
        spl_v = UnivariateSpline(t, v, k=5)

        # ax[1, 0].plot(t, v)
        ax[1, 0].plot(xs, spl_v(xs))

        v = np.mean(heatmap[:, 30:40], axis=1)
        t = np.array(range(45)) * 4
        spl_v = UnivariateSpline(t, v, k=5)

        # ax[1, 0].plot(t, v)
        ax[1, 0].plot(xs, spl_v(xs))
        ax[1, 0].set_xlabel("Time (mins)")
        ax[1, 0].set_ylabel(r"v $(\mu m)$")
        ax[1, 0].set_ylim([-0.5, 0.5])

        area = np.array(dfWound["Area"]) * (scale) ** 2

        for t in range(T):
            if pd.isnull(area[t]):
                area[t] = 0

        T = np.array(range(181))
        spl_w = UnivariateSpline(T, area / 10, k=5)

        ax[1, 1].plot(T, area)
        ax[1, 1].plot(xs, spl_w(xs) * 10)
        ax[1, 1].set_xlabel("Time (mins)")
        ax[1, 1].set_ylabel(r"Wound Area $(\mu m^2)$")
        ax[1, 1].set_ylim([0, 1300])
        plt.suptitle(f"Wound healing properties {filename}")
        fig.savefig(
            f"results/Wound healing properties 4 layers {filename}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

# q0_dq1_dq2 close and far from wounds


def rotation_matrix(theta):

    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    return R


if True:

    T = 181
    scale = 147.91 / 512

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

    _df = []
    for filename in filenames:
        df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")
        distance = sm.io.imread(f"dat/{filename}/distanceWound{filename}.tif").astype(
            int
        )

        T = np.linspace(0, 170, 18)
        for t in T:

            dft = cl.sortTime(df, [t, t + 10])
            m = len(dft)

            q_c = []
            q_f = []
            for i in range(m):
                x, y = dft["Centroid"].iloc[i]
                tau = dft["Time"].iloc[i]
                dist = distance[int(tau), int(x), int(y)]
                if dist * scale < 40:
                    q_c.append(dft["q"].iloc[i])
                else:
                    q_f.append(dft["q"].iloc[i])

            Q_c = np.mean(q_c, axis=0)
            thetastar = np.arctan2(Q_c[0, 1], Q_c[0, 0])
            q0_c = (2 * Q_c[0, 0] ** 2 + 2 * Q_c[0, 1] ** 2) ** 0.5

            R = rotation_matrix(thetastar)

            qr_c = np.matmul(R.transpose(), q_c)
            Qr_c = np.matmul(R.transpose(), Q_c)

            dQr_c = qr_c - Qr_c

            dQr1_c = []
            dQr2_c = []
            for i in range(len(dQr_c)):
                dQr1_c.append(dQr_c[i][0, 0])
                dQr2_c.append(dQr_c[i][0, 1])
            dQr1_c = np.array(dQr1_c)
            dQr2_c = np.array(dQr2_c)

            # ----

            Q_f = np.mean(q_f, axis=0)
            thetastar = np.arctan2(Q_f[0, 1], Q_f[0, 0])
            q0_f = (2 * Q_f[0, 0] ** 2 + 2 * Q_f[0, 1] ** 2) ** 0.5

            R = rotation_matrix(thetastar)

            qr_f = np.matmul(R.transpose(), q_f)
            Qr_f = np.matmul(R.transpose(), Q_f)

            dQr_f = qr_f - Qr_f

            dQr1_f = []
            dQr2_f = []
            for i in range(len(dQr_f)):
                dQr1_f.append(dQr_f[i][0, 0])
                dQr2_f.append(dQr_f[i][0, 1])
            dQr1_f = np.array(dQr1_f)
            dQr2_f = np.array(dQr2_f)

            _df.append(
                {
                    "Filename": filename,
                    "q0_c": q0_c,
                    "dQr1_c": np.mean(dQr1_c ** 2),
                    "dQr2_c": np.mean(dQr2_c ** 2),
                    "q0_f": q0_f,
                    "dQr1_f": np.mean(dQr1_f ** 2),
                    "dQr2_f": np.mean(dQr2_f ** 2),
                    "T": t,
                }
            )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/dfq0_dq1_dq2CloseFar{fileType}.pkl")
else:
    df = pd.read_pickle(f"databases/dfq0_dq1_dq2CloseFar{fileType}.pkl")


if True:

    T = np.linspace(0, 170, 18)
    q0_c = []
    dQr1_c = []
    dQr2_c = []
    mseq0_c = []
    msedQr1_c = []
    msedQr2_c = []
    q0_f = []
    dQr1_f = []
    dQr2_f = []
    mseq0_f = []
    msedQr1_f = []
    msedQr2_f = []
    n = len(filenames)
    for t in T:
        q0_c.append(np.mean(df["q0_c"][df["T"] == t]))
        dQr1_c.append(np.mean(df["dQr1_c"][df["T"] == t]))
        dQr2_c.append(np.mean(df["dQr2_c"][df["T"] == t]))
        mseq0_c.append(np.std(df["q0_c"][df["T"] == t]) / n ** 0.5)
        msedQr1_c.append(np.std(df["dQr1_c"][df["T"] == t]) / n ** 0.5)
        msedQr2_c.append(np.std(df["dQr2_c"][df["T"] == t]) / n ** 0.5)
        q0_f.append(np.mean(df["q0_f"][df["T"] == t]))
        dQr1_f.append(np.mean(df["dQr1_f"][df["T"] == t]))
        dQr2_f.append(np.mean(df["dQr2_f"][df["T"] == t]))
        mseq0_f.append(np.std(df["q0_f"][df["T"] == t]) / n ** 0.5)
        msedQr1_f.append(np.std(df["dQr1_f"][df["T"] == t]) / n ** 0.5)
        msedQr2_f.append(np.std(df["dQr2_f"][df["T"] == t]) / n ** 0.5)

    fig, ax = plt.subplots(2, 3, figsize=(30, 18))

    ax[0, 0].errorbar(T, q0_c, yerr=mseq0_c)
    ax[0, 0].set_xlabel("Time (mins)")
    ax[0, 0].set_ylabel(r"$q_0$")
    ax[0, 0].set_ylim([0, 0.06])

    ax[0, 1].errorbar(T, dQr1_c, yerr=msedQr1_c)
    ax[0, 1].set_xlabel("Time (mins)")
    ax[0, 1].set_ylabel(r"$(\delta q_1)^2$")
    ax[0, 1].set_ylim([0.0004, 0.001])
    ax[0, 1].title.set_text("Close to Wound")

    ax[0, 2].errorbar(T, dQr2_c, yerr=msedQr2_c)
    ax[0, 2].set_xlabel("Time (mins)")
    ax[0, 2].set_ylabel(r"$(\delta q_2)^2$")
    ax[0, 2].set_ylim([0.0004, 0.001])

    ax[1, 0].errorbar(T, q0_f, yerr=mseq0_f)
    ax[1, 0].set_xlabel("Time (mins)")
    ax[1, 0].set_ylabel(r"$q_0$")
    ax[1, 0].set_ylim([0, 0.06])

    ax[1, 1].errorbar(T, dQr1_f, yerr=msedQr1_f)
    ax[1, 1].set_xlabel("Time (mins)")
    ax[1, 1].set_ylabel(r"$(\delta q_1)^2$")
    ax[1, 1].set_ylim([0.0004, 0.001])
    ax[1, 1].title.set_text("Far from Wound")

    ax[1, 2].errorbar(T, dQr2_f, yerr=msedQr2_f)
    ax[1, 2].set_xlabel("Time (mins)")
    ax[1, 2].set_ylabel(r"$(\delta q_2)^2$")
    ax[1, 2].set_ylim([0.0004, 0.001])

    plt.suptitle(f"Shape with time {fileType}")

    fig.savefig(
        f"results/q0_dq1_dq2 close far {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")