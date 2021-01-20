import os
import shutil
from math import floor, log10

from collections import Counter
import cv2
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

scale = 147.91 / 512
bandWidth = 20  # in microns
pixelWidth = bandWidth / scale


def qVideoHeatmap(df, band):

    cl.createFolder("results/video/")
    uniqueTimes = list(set(df["Time"]))
    uniqueTimes = sorted(uniqueTimes)

    q1 = df["q1"]
    q1Max = max(max(q1), -min(q1))
    q2 = df["q2"]
    q2Max = max(max(q2), -min(q2))

    heatmaps = np.zeros([len(uniqueTimes), 30, 30])

    for t in uniqueTimes:
        df2 = df[df["Time"] == t]

        q1 = list(df2["q1"])
        q2 = list(df2["q2"])

        heatmap, xedges, yedges = np.histogram2d(
            q1, q2, range=[[-q1Max, q1Max], [-q2Max, q2Max]], bins=30
        )
        m = len(df2)
        heatmaps[t] = heatmap / m

    zMax = heatmaps.max()
    zMin = 0
    dx, dy = q1Max / 15, q2Max / 15
    x, y = np.mgrid[-q1Max:q1Max:dx, -q2Max:q2Max:dy]

    for t in uniqueTimes:

        heatmap = heatmaps[t]

        fig, ax = plt.subplots()
        c = ax.pcolor(x, y, heatmap, cmap="Reds", vmin=zMin, vmax=zMax)
        fig.colorbar(c, ax=ax)
        fig.savefig(
            f"results/video/heatmap{t}", dpi=300, transparent=True,
        )
        plt.close("all")

    img_array = []
    for t in uniqueTimes:
        img = cv2.imread(f"results/video/heatmap{t}.png")
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(
        f"results/qHeatmapBand{band}.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 5, size
    )
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
    cv2.destroyAllWindows()

    shutil.rmtree("results/video")


# -------------------

filenames, fileType = cl.getFilesType()

_dfRadial = []
for filename in filenames:

    df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")
    dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
    T = len(dfWound)

    wound = sm.io.imread(f"dat/{filename}/woundsite{filename}.tif").astype("uint8")

    dist = []
    for t in range(T):
        img = 255 - fi.imgrcxy(wound[t])
        dist.append(sp.ndimage.morphology.distance_transform_edt(img))

    mu = []
    err = []

    for t in range(T):
        df2 = df[df["Time"] == t]
        (Cx, Cy) = dfWound["Centroid"][t]
        for i in range(len(df2)):
            area = df2["Area"].iloc[i]
            (x, y) = df2["Centroid"].iloc[i]
            sf = df2["Shape Factor"].iloc[i]
            TrS = df2["Trace(S)"].iloc[i]

            x = int(x)
            y = int(y)
            distance = dist[t][x, y]
            q = df2["q"].iloc[i]
            phi = np.arctan2(y - Cy, x - Cx)

            R = cl.rotation_matrix(-phi)

            qr = np.matmul(q, R.transpose())
            qw = np.matmul(R, qr)

            _dfRadial.append(
                {
                    "Wound Oriented q": qw,
                    "q1": qw[0, 0],
                    "q2": qw[0, 1],
                    "Centroid": (x, y),
                    "Time": t,
                    "Area": area,
                    "Wound Edge Distance": distance,
                    "filename": filename,
                    "Shape Factor": sf,
                    "TrS": TrS,
                }
            )

dfRadial = pd.DataFrame(_dfRadial)

finished = False
band = 1
while finished != True:
    df = cl.sortBand(dfRadial, band, pixelWidth)

    if len(df) == 0:
        finished = True
    else:
        uniqueTimes = list(set(df["Time"]))
        uniqueTimes = sorted(uniqueTimes)

        # qVideoHeatmap(df, band)

        theta = []
        sf = []
        errsf = []
        area = []
        errA = []
        TrS = []
        errTrS = []
        for t in uniqueTimes:

            prop = df["Wound Oriented q"][df["Time"] == t]
            q = np.mean(list(prop))
            phi = np.arctan2(q[1, 0], q[0, 0]) / 2
            if phi > np.pi / 2:
                phi = np.pi / 2 - phi
            elif phi < 0:
                phi = -phi
            theta.append(phi)

            prop = df["Shape Factor"][df["Time"] == t]
            sf.append(np.mean(list(prop)))
            errsf.append(np.std(list(prop)) / (len(prop) ** 0.5))

            prop = df["Area"][df["Time"] == t] * (scale ** 2)
            area.append(np.mean(list(prop)))
            errA.append(np.std(list(prop)) / (len(prop) ** 0.5))

            prop = df["TrS"][df["Time"] == t] * (scale ** 2)
            TrS.append(np.mean(list(prop)))
            errTrS.append(np.std(list(prop)) / (len(prop) ** 0.5))

        x = range(len(uniqueTimes))

        # -------------------

        fig = plt.figure(1, figsize=(9, 8))
        plt.gcf().subplots_adjust(left=0.2)
        plt.plot(x, theta)

        plt.xlabel("Time (mins)")
        plt.ylabel(f"theta")
        plt.gcf().subplots_adjust(bottom=0.2)
        fig.savefig(
            f"results/theta close to wound band{band} {fileType}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

        # -------------------

        fig = plt.figure(1, figsize=(9, 8))
        plt.gcf().subplots_adjust(left=0.2)
        plt.errorbar(x, sf, yerr=errsf, fmt="o")

        plt.xlabel("Time (mins)")
        plt.ylabel(f"sf")
        plt.gcf().subplots_adjust(bottom=0.2)
        fig.savefig(
            f"results/sf close to wound band{band} {fileType}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

        # -------------------

        fig = plt.figure(1, figsize=(9, 8))
        plt.gcf().subplots_adjust(left=0.2)
        plt.errorbar(x, area, yerr=errA, fmt="o")

        plt.xlabel("Time (mins)")
        plt.ylabel(r"Area ($\mu m^2$)")
        plt.gcf().subplots_adjust(bottom=0.2)
        fig.savefig(
            f"results/Area close to wound band{band} {fileType}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

        # -------------------

        fig = plt.figure(1, figsize=(9, 8))
        plt.gcf().subplots_adjust(left=0.2)
        plt.errorbar(x, TrS, yerr=errTrS, fmt="o")

        plt.xlabel("Time (mins)")
        plt.ylabel(f"TrS")
        plt.gcf().subplots_adjust(bottom=0.2)
        fig.savefig(
            f"results/TrS close to wound band{band} {fileType}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

        band += 1

