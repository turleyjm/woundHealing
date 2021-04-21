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

for filename in filenames:
    T = 181
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(2, 2, figsize=(9, 8))
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
    ax[0, 0].set_ylabel(r"Area $(\mu m^2)$")
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
    spl_w = UnivariateSpline(T, area, k=1)

    ax[1, 1].plot(T, area)
    ax[1, 1].plot(xs, spl_w(xs))
    ax[1, 1].axvline(halfLife, color="r")
    ax[1, 1].set_xlabel("Time (mins)")
    ax[1, 1].set_ylabel(r"Wound Size $(\mu m^2)$")
    ax[1, 1].set_ylim([0, 1300])
    plt.suptitle(f"Wound healing properties {filename}")
    fig.savefig(
        f"results/Wound healing properties {filename}", dpi=300, transparent=True,
    )
    plt.close("all")

    dA = []
    dq1 = []
    dw = []
    v = []

    for t in T:
        dA.append(derivative(spl_A, t))
        dq1.append(derivative(spl_q1, t))
        dw.append(derivative(spl_w, t))
        v.append(spl_v(t))
        if t < finish - 5:
            All_dA.append(derivative(spl_A, t))
            All_dq1.append(derivative(spl_q1, t))
            All_dw.append(derivative(spl_w, t))
            All_v.append(spl_v(t))

    if fileType == "WoundL":
        limWound = [-120, 20]
    else:
        limWound = [-40, 10]

    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.4)
    plt.gcf().subplots_adjust(bottom=0.15)
    ax[0, 0].scatter(
        dw[0 : finish - 5], dA[0 : finish - 5], c=range(finish - 5), cmap="RdBu"
    )
    ax[0, 0].set(xlabel=r"$\frac{dw_A}{dt}$", ylabel=r"$\frac{dA}{dt}$")
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

plt.rcParams.update({"font.size": 14})
fig, ax = plt.subplots(2, 2, figsize=(12, 12))
plt.subplots_adjust(wspace=0.4, bottom=0.1)
result = sp.stats.linregress(All_dw, All_dA)
ax[0, 0].scatter(All_dw, All_dA)
ax[0, 0].plot(xx, result.slope * xx + result.intercept, "r")
ax[0, 0].set(xlabel=r"$\frac{dw_A}{dt}$", ylabel=r"$\frac{dA}{dt}$")
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

print(halfLifes)
print(quarterLifes)
print(healTimes)
