import os
from math import floor, log10

import cv2
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

import cellProperties as cell
import findGoodCells as fi

plt.rcParams.update({"font.size": 20})
plt.ioff()
pd.set_option("display.width", 1000)

scale = 147.91 / 512


def rotation_matrix(theta):

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)],])

    return R


f = open("pythonText.txt", "r")

filenames = f.read()
filenames = filenames.split(", ")

_dfRadial = []
for filename in filenames:

    df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")
    dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
    T = len(dfWound)

    wound = sm.io.imread(f"dat/{filename}/woundsite{filename}.tif").astype("uint8")

    dist = np.zeros([T, 512, 512])
    for t in range(T):
        img = fi.imgrcxy(wound[t])
        dist[t] = sp.ndimage.morphology.distance_transform_edt(img)

    mu = []
    err = []

    for t in range(T):
        df2 = df[df["Time"] == t]
        (Cx, Cy) = dfWound["Centroid"][t]
        for i in range(len(df2)):
            area = df2["Area"].iloc[i]
            (x, y) = df2["Centroid"].iloc[i]
            sf = df2["Shape Factor"].iloc[i]
            x = int(x)
            y = int(y)
            distance = dist[t][x, y]
            Q = -df2["Q"].iloc[i]
            phi = np.arctan2(y - Cy, x - Cx)

            R = rotation_matrix(-phi)

            Qr = np.matmul(Q, R.transpose())
            Qw = np.matmul(R, Qr)

            _dfRadial.append(
                {
                    "Wound Oriented Q": Qw,
                    "Centroid": (x, y),
                    "Time": t,
                    "Area": area,
                    "Wound Edge Distance": distance,
                    "filename": filename,
                    "Shape Factor": sf,
                }
            )

dfRadial = pd.DataFrame(_dfRadial)

df = dfRadial[dfRadial["Wound Edge Distance"] < 50]

uniqueTimes = list(set(df["Time"]))

uniqueTimes = sorted(uniqueTimes)
q1 = []
errq1 = []
q2 = []
errq2 = []
sf = []
errsf = []
theta = []
area = []
errA = []
for t in uniqueTimes:

    prop = df["Wound Oriented Q"][df["Time"] == t]

    Q = cell.mean(list(prop))
    sd = cell.sd(list(prop))
    q1.append(Q[0, 0])
    # errq1.append(sd[0, 0] / (len(prop) ** 0.5))
    q2.append(Q[1, 0])
    # errq2.append(sd[1, 0] / (len(prop) ** 0.5))
    phi = np.arctan2(Q[1, 0], Q[0, 0]) / 2
    if phi > np.pi / 2:
        phi = np.pi / 2 - phi
    theta.append(phi)

    prop = df["Area"][df["Time"] == t] * (scale ** 2)
    area.append(cell.mean(list(prop)))
    errA.append(cell.sd(list(prop)) / (len(prop) ** 0.5))

    prop = df["Shape Factor"][df["Time"] == t]
    sf.append(cell.mean(list(prop)))
    errsf.append(cell.sd(list(prop)) / (len(prop) ** 0.5))

x = range(len(uniqueTimes))

# -------------------

fig = plt.figure(1, figsize=(9, 8))
plt.gcf().subplots_adjust(left=0.2)
plt.plot(x, theta)

plt.xlabel("Time")
plt.ylabel(f"theta")
plt.gcf().subplots_adjust(bottom=0.2)
fig.savefig(
    f"results/theta close to wound", dpi=300, transparent=True,
)
plt.close("all")

# -------------------

fig = plt.figure(1, figsize=(9, 8))
plt.gcf().subplots_adjust(left=0.2)
plt.errorbar(x, area, yerr=errA, fmt="o")

plt.xlabel("Time")
plt.ylabel(f"Area")
plt.gcf().subplots_adjust(bottom=0.2)
fig.savefig(
    f"results/Area close to wound", dpi=300, transparent=True,
)
plt.close("all")

# -------------------

fig = plt.figure(1, figsize=(9, 8))
plt.gcf().subplots_adjust(left=0.2)
plt.errorbar(x, q1, yerr=errq1, fmt="o")

plt.xlabel("Time")
plt.ylabel(f"q1")
plt.gcf().subplots_adjust(bottom=0.2)
fig.savefig(
    f"results/q1 close to wound", dpi=300, transparent=True,
)
plt.close("all")

fig = plt.figure(1, figsize=(9, 8))
plt.gcf().subplots_adjust(left=0.2)
plt.errorbar(x, q2, yerr=errq2, fmt="o")

plt.xlabel("Time")
plt.ylabel(f"q2")
plt.gcf().subplots_adjust(bottom=0.2)
fig.savefig(
    f"results/q2 close to wound", dpi=300, transparent=True,
)
plt.close("all")

fig = plt.figure(1, figsize=(9, 8))
plt.gcf().subplots_adjust(left=0.2)
plt.errorbar(x, sf, yerr=errsf, fmt="o")

plt.xlabel("Time")
plt.ylabel(f"sf")
plt.gcf().subplots_adjust(bottom=0.2)
fig.savefig(
    f"results/sf close to wound", dpi=300, transparent=True,
)
plt.close("all")

# ----------------------------

df2 = dfRadial[dfRadial["Wound Edge Distance"] < 100]
df = df2[df2["Wound Edge Distance"] > 50]

uniqueTimes = list(set(df["Time"]))

uniqueTimes = sorted(uniqueTimes)
q1 = []
errq1 = []
q2 = []
errq2 = []
sf = []
errsf = []
for t in uniqueTimes:

    prop = df["Wound Oriented Q"][df["Time"] == t]
    Q = cell.mean(list(prop))
    sd = cell.sd(list(prop))
    q1.append(Q[0, 0])
    errq1.append(sd[0, 0] / (len(prop) ** 0.5))
    q2.append(Q[1, 0])
    errq2.append(sd[1, 0] / (len(prop) ** 0.5))

    prop = df["Shape Factor"][df["Time"] == t]
    sf.append(cell.mean(list(prop)))
    errsf.append(cell.sd(list(prop)) / (len(prop) ** 0.5))

x = range(len(uniqueTimes))

fig = plt.figure(1, figsize=(9, 8))
plt.gcf().subplots_adjust(left=0.2)
plt.errorbar(x, q1, yerr=errq1, fmt="o")

plt.xlabel("Time")
plt.ylabel(f"q1")
plt.gcf().subplots_adjust(bottom=0.2)
fig.savefig(
    f"results/q1 close to wound band 2", dpi=300, transparent=True,
)
plt.close("all")

fig = plt.figure(1, figsize=(9, 8))
plt.gcf().subplots_adjust(left=0.2)
plt.errorbar(x, q2, yerr=errq2, fmt="o")

plt.xlabel("Time")
plt.ylabel(f"q2")
plt.gcf().subplots_adjust(bottom=0.2)
fig.savefig(
    f"results/q2 close to wound band 2", dpi=300, transparent=True,
)
plt.close("all")

fig = plt.figure(1, figsize=(9, 8))
plt.gcf().subplots_adjust(left=0.2)
plt.errorbar(x, sf, yerr=errsf, fmt="o")

plt.xlabel("Time")
plt.ylabel(f"sf")
plt.gcf().subplots_adjust(bottom=0.2)
fig.savefig(
    f"results/sf close to wound band 2", dpi=300, transparent=True,
)
plt.close("all")

# ----------------------------

df2 = dfRadial[dfRadial["Wound Edge Distance"] < 150]
df = df2[df2["Wound Edge Distance"] > 100]

uniqueTimes = list(set(df["Time"]))

uniqueTimes = sorted(uniqueTimes)
q1 = []
errq1 = []
q2 = []
errq2 = []
sf = []
errsf = []
for t in uniqueTimes:

    prop = df["Wound Oriented Q"][df["Time"] == t]
    Q = cell.mean(list(prop))
    sd = cell.sd(list(prop))
    q1.append(Q[0, 0])
    errq1.append(sd[0, 0] / (len(prop) ** 0.5))
    q2.append(Q[1, 0])
    errq2.append(sd[1, 0] / (len(prop) ** 0.5))

    prop = df["Shape Factor"][df["Time"] == t]
    sf.append(cell.mean(list(prop)))
    errsf.append(cell.sd(list(prop)) / (len(prop) ** 0.5))

x = range(len(uniqueTimes))

fig = plt.figure(1, figsize=(9, 8))
plt.gcf().subplots_adjust(left=0.2)
plt.errorbar(x, q1, yerr=errq1, fmt="o")

plt.xlabel("Time")
plt.ylabel(f"q1")
plt.gcf().subplots_adjust(bottom=0.2)
fig.savefig(
    f"results/q1 close to wound band 3", dpi=300, transparent=True,
)
plt.close("all")

fig = plt.figure(1, figsize=(9, 8))
plt.gcf().subplots_adjust(left=0.2)
plt.errorbar(x, q2, yerr=errq2, fmt="o")

plt.xlabel("Time")
plt.ylabel(f"q2")
plt.gcf().subplots_adjust(bottom=0.2)
fig.savefig(
    f"results/q2 close to wound band 3", dpi=300, transparent=True,
)
plt.close("all")

fig = plt.figure(1, figsize=(9, 8))
plt.gcf().subplots_adjust(left=0.2)
plt.errorbar(x, sf, yerr=errsf, fmt="o")

plt.xlabel("Time")
plt.ylabel(f"sf")
plt.gcf().subplots_adjust(bottom=0.2)
fig.savefig(
    f"results/sf close to wound band 3", dpi=300, transparent=True,
)
plt.close("all")
