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
scale = 147.91 / 512

_dfRadial = []
for filename in filenames:

    df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")
    dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
    T = 90

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
        areaMean = np.mean(df2["Area"])
        sfMean = np.mean(df2["Shape Factor"])
        for i in range(len(df2)):
            area = df2["Area"].iloc[i] - areaMean
            (x, y) = df2["Centroid"].iloc[i]
            sf = df2["Shape Factor"].iloc[i] - sfMean
            TrS = df2["Trace(S)"].iloc[i]

            x = int(x)
            y = int(y)
            distance = dist[t][x, y] * scale
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

R = np.linspace(0, 80, num=17)
T = np.linspace(0, 90, num=19)

thetaHeatmap = np.zeros([18, 16])
sfHeatmap = np.zeros([18, 16])
areaHeatmap = np.zeros([18, 16])


for i in range(18):
    t = T[0 + i : 2 + i]
    for j in range(16):
        r = R[0 + j : 2 + j]
        df = cl.sortBand(dfRadial, r, t)

        prop = df["Wound Oriented q"]
        q = np.mean(list(prop))
        phi = np.arctan2(q[1, 0], q[0, 0]) / 2
        if phi > np.pi / 2:
            phi = np.pi / 2 - phi
        elif phi < 0:
            phi = -phi
        thetaHeatmap[int(t[0] / 5), int(r[0] / 5)] = 180 * phi / np.pi

        prop = df["Shape Factor"]
        sfHeatmap[int(t[0] / 5), int(r[0] / 5)] = np.mean(list(prop))

        prop = df["Area"] * (scale ** 2)
        areaHeatmap[int(t[0] / 5), int(r[0] / 5)] = np.mean(list(prop))


dt, dr = 5, 5
t, r = np.mgrid[0:90:dt, 0:80:dr]
z_min, z_max = 0, 90
fig, ax = plt.subplots()
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.20)
c = ax.pcolor(t, r, thetaHeatmap, cmap="coolwarm", vmin=z_min, vmax=z_max)
plt.xlabel(r"Time (mins)")
plt.ylabel(r"distance from wound edge ($\mu m$)")
fig.colorbar(c, ax=ax)
fig.savefig(
    f"results/theta kymograph {fileType}", dpi=300, transparent=True,
)
plt.close("all")

# -----------------

dt, dr = 5, 5
t, r = np.mgrid[0:90:dt, 0:80:dr]
z_min, z_max = -0.08, 0.08
midpoint = 1 - z_max / (z_max + abs(z_min))
orig_cmap = matplotlib.cm.seismic
shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")
fig, ax = plt.subplots()
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.20)
c = ax.pcolor(t, r, sfHeatmap, cmap=shifted_cmap, vmin=z_min, vmax=z_max)
plt.xlabel(r"Time (mins)")
plt.ylabel(r"distance from wound edge ($\mu m$)")
fig.colorbar(c, ax=ax)
fig.savefig(
    f"results/delta shape factor kymograph {fileType}", dpi=300, transparent=True,
)
plt.close("all")

# -----------------

dt, dr = 5, 5
t, r = np.mgrid[0:90:dt, 0:80:dr]
z_min, z_max = -4, 4
midpoint = 1 - z_max / (z_max + abs(z_min))
orig_cmap = matplotlib.cm.seismic
shifted_cmap = cl.shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")
fig, ax = plt.subplots()
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.20)
c = ax.pcolor(t, r, areaHeatmap, cmap=shifted_cmap, vmin=z_min, vmax=z_max)
plt.xlabel(r"Time (mins)")
plt.ylabel(r"distance from wound edge ($\mu m$)")
fig.colorbar(c, ax=ax)
fig.savefig(
    f"results/delta area kymograph {fileType}", dpi=300, transparent=True,
)
plt.close("all")
