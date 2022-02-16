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
import xml.etree.ElementTree as et

import cellProperties as cell
import findGoodCells as fi
import utils as util

plt.rcParams.update({"font.size": 10})

# -------------------

filenames, fileType = util.getFilesType()
scale = 123 / 512

fig = plt.figure(1, figsize=(9, 8))

T = 93

sf = []
endTime = []
divisions = []
R = [[] for col in range(T)]
for filename in filenames:

    dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
    # dfMitosis = pd.read_pickle(f"dat/{filename}/nucleusVelocity{filename}.pkl")

    # divisions.append(int(len(dfMitosis) / 3))
    sf.append(dfWound["Shape Factor"].iloc[0])
    time = np.array(dfWound["Time"])
    area = np.array(dfWound["Area"]) * (scale) ** 2

    tf = sum(area > 0)
    endTime.append(tf)
    for t in range(T):
        if pd.isnull(area[t]):
            area[t] = 0

    # df = dfMitosis[dfMitosis["Chain"] == "parent"]
    # count = 0
    # for i in range(len(df)):
    #     if df["Time"].iloc[i][-1] < tf:
    #         count += 1
    # divisions.append(count)

    for t in range(T):
        R[t].append(area[t])

    plt.plot(time, area)

plt.gcf().subplots_adjust(left=0.2)
plt.xlabel("Time")
plt.ylabel(r"Area ($\mu m ^2$)")
fig.savefig(
    f"results/Wound Area {fileType}",
    dpi=300,
    transparent=True,
)
plt.close("all")

err = []
for t in range(T):
    err.append(np.std(R[t]) / (len(R[t]) ** 0.5))
    R[t] = np.mean(R[t])

meanFinish = []
for filename in filenames:

    dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")

    area = np.array(dfWound["Area"]) * (scale) ** 2
    t = 0
    while pd.notnull(area[t]):
        t += 1

    meanFinish.append(t - 1)

meanFinish = np.mean(meanFinish)

if False:
    fig = plt.figure(1, figsize=(9, 8))
    plt.errorbar(time, R, yerr=err)
    plt.gcf().subplots_adjust(left=0.2)
    plt.title(f"Mean finish time = {meanFinish}")
    plt.suptitle("Wound Area")
    plt.xlabel("Time (mins)")
    plt.ylabel(r"Area ($\mu m ^2$)")
    fig.savefig(
        f"results/Wound Area Mean {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

    R = np.array(R)
    err = np.array(err)

    fig = plt.figure(1, figsize=(9, 8))
    plt.errorbar(time, (R / np.pi) ** 0.5, yerr=(err / np.pi) ** 0.5)
    plt.gcf().subplots_adjust(left=0.2)
    plt.title(f"Mean finish time = {meanFinish}")
    plt.suptitle("Wound Radius")
    plt.xlabel("Time (mins)")
    plt.ylabel(r"Radius ($\mu m$)")
    fig.savefig(
        f"results/Wound Radius Mean {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

#  ------------------- Radius around wound thats fully in frame

if False:
    rList = [[] for col in range(T)]
    for filename in filenames:
        dist = sm.io.imread(f"dat/{filename}/distanceWound{filename}.tif").astype(
            "uint16"
        )
        for t in range(T):
            Max = dist[t].max()
            dist[t][1:511] = Max
            rList[t].append(dist[t].min())

    for t in range(T):
        rList[t] = np.mean(rList[t])

    t = range(T)
    rList = np.array(rList) * scale

    fig = plt.figure(1, figsize=(9, 8))
    plt.plot(t, rList)
    plt.ylim([0, 80])

    plt.xlabel(r"Time (mins)")
    plt.ylabel(r"distance from wound edge to frame edge ($\mu m$)")

    fig.savefig(
        f"results/max distance from wound edge {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


if False:
    fig = plt.figure(1, figsize=(9, 8))
    plt.scatter(sf, endTime)
    plt.xlabel(r"sf")
    plt.ylabel(r"end time")

    fig.savefig(
        f"results/correlation {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


if False:
    cor, a = pearsonr(endTime, divisions)
    fig = plt.figure(1, figsize=(9, 8))
    plt.scatter(endTime, divisions)
    plt.xlabel(r"end time")
    plt.ylabel(r"divisions")
    plt.title(f"Pearsons Correlation = {cor}")

    fig.savefig(
        f"results/correlation healing time and mitosis {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


if False:
    x = np.array(range(len(endTime))) + 1
    fig = plt.figure(1, figsize=(9, 8))
    plt.scatter(x, endTime)
    plt.xlabel(r"video")
    plt.ylabel(r"end time")

    fig.savefig(
        f"results/end time {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


if False:
    for filename in filenames:
        dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
        vid = sm.io.imread(f"dat/{filename}/ecadFocus{filename}.tif").astype(int)
        woundsite = sm.io.imread(f"dat/{filename}/woundsite{filename}.tif").astype(int)

        wound = np.zeros([181, 20])
        intWound = np.zeros([181, 20])
        for t in range(T):
            x, y = dfWound["Position"].iloc[t]
            x = int(x)
            y = 512 - int(y)
            img = np.zeros([512, 512])
            img[y, x] = 1
            img = sp.ndimage.morphology.distance_transform_edt(1 - img)
            intensity = []
            intensityWound = []
            for r in range(20):
                intensity.append(
                    np.mean(vid[t][(5 * r / scale < img) & (img < 5 * (r + 1) / scale)])
                )
                intensityWound.append(
                    np.mean(
                        woundsite[t][
                            (5 * r / scale < img) & (img < 5 * (r + 1) / scale)
                        ]
                    )
                )

            wound[t] = intensity
            intWound[t] = intensityWound

        t, r = np.mgrid[0:181:1, 1:100:5]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        c = ax[0].pcolor(t, r, wound)
        fig.colorbar(c, ax=ax[0])
        ax[0].set_xlabel("Time (mins)")
        ax[0].set_ylabel(r"Distance from wound center $(\mu m)$")
        ax[0].title.set_text(f"Intensity {filename}")

        c = ax[1].pcolor(t, r, 255 - intWound)
        fig.colorbar(c, ax=ax[1])
        ax[1].set_xlabel("Time (mins)")
        ax[1].set_ylabel(r"Distance from wound center $(\mu m)$")
        ax[1].title.set_text(f"Wound {filename}")
        fig.savefig(
            f"results/Radial Wound {filename}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")