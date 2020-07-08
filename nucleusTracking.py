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

import cell_properties as cell
import find_good_cells as fi

plt.rcParams.update({"font.size": 20})
plt.ioff()
pd.set_option("display.width", 1000)

cwd = os.getcwd()

width = 184.88  # width in um
timewidth = 180.008

wound = sm.io.imread("dat_nucleus/binary_wound_wound16h01.tif").astype(float)
C = []
for t in range(len(wound)):
    img_wound = wound[t]
    img_label = sm.measure.label(img_wound, background=0, connectivity=1)
    contour = sm.measure.find_contours(img_label == 1, level=0)[0]
    poly = sm.measure.approximate_polygon(contour, tolerance=1)
    wound_polygon = Polygon(poly)
    c = cell.centroid(wound_polygon)
    c = np.array(c)
    c = c * (width / 512)
    C.append(c)

filename = "wound16h01"

df = pd.read_pickle(f"databases/nucleusVertice{filename}.pkl")

N = 7
time = 31
timeframes = 10

H = []
for t in range(timeframes):
    H.append([])
    for x in range(N):
        H[t].append([])
        for y in range(N):
            H[t][x].append([])


for i in range(len(df)):

    X = df.iloc[i, 1]
    Y = df.iloc[i, 2]
    T = df.iloc[i, 4]
    A = len(T)

    for i in range(A - 1):

        V = np.array([X[i + 1] - X[i], Y[i + 1] - Y[i]])
        x = int(X[i] * N / width)
        y = int(Y[i] * N / width)
        t = int(T[i] * timeframes / 5400)  # change

        H[t][x][y].append(V)

for t in range(timeframes):
    for x in range(N):
        for y in range(N):
            m = len(H[t][x][y])
            if m > 0:
                H[t][x][y] = cell.mean(H[t][x][y])
            else:
                H[t][x][y] = np.array([0, 0])

for t in range(timeframes):
    u, v = np.meshgrid(np.linspace(1, N, N), np.linspace(1, N, N))
    u0, v0 = np.meshgrid(np.linspace(1, N, N), np.linspace(1, N, N))

    prop = []  # correct with wieghted average
    for i in range(N):
        for j in range(N):
            prop.append(H[t][i][j])
    mu = cell.mean(prop)

    for i in range(N):  # change of coord
        for j in range(N):
            V = H[t][i][j]
            V0 = H[t][i][j] - mu
            x = V[0]
            y = -V[1]
            u[(N - 1) - j, i] = x
            v[(N - 1) - j, i] = y
            x = V0[0]
            y = -V0[1]
            u0[(N - 1) - j, i] = x
            v0[(N - 1) - j, i] = y

    x, y = np.meshgrid(np.linspace(1, N, N), np.linspace(1, N, N))

    fig = plt.figure()
    plt.quiver(x, y, u, v)
    fig.savefig(
        "results/migration/" + f"import/store/nucleusMovement{filename}_{t*6}",
        dpi=300,
        transparent=True,
    )
    fig = plt.figure()
    plt.quiver(x, y, u0, v0)
    fig.savefig(
        "results/migration/"
        + f"import/store/nucleusMovementRemoveDrift{filename}_{t*6}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

V = []
for t in range(12):
    V.append([])

l = np.array(range(12)) * 10

for i in range(len(df)):

    T = df.iloc[i, 4]

    if len(T) > 5:
        X = df.iloc[i, 1]
        Y = df.iloc[i, 2]
        x0 = X[0]
        y0 = Y[0]
        x1 = X[-1]
        y1 = Y[-1]
        t0 = int(T[0] / timewidth)
        t1 = int(T[-1] / timewidth)
        (Cx0, Cy0) = C[t0]  # change
        (Cx1, Cy1) = C[t1]

        r0 = ((Cx0 - x0) ** 2 + (Cy0 - y0) ** 2) ** 0.5
        r1 = ((Cx1 - x1) ** 2 + (Cy1 - y1) ** 2) ** 0.5

        u = (r0 - r1) / len(T)
        R = int(r0 / 15)
        V[R].append(u)

err = []
for i in range(12):
    if len(V[i]) == 0:
        V[i] = [0]
    err.append(cell.sd(V[i]) / (len(V[i]) ** 0.5))
    V[i] = cell.mean(V[i])


fig = plt.figure()
plt.errorbar(l, V, err)
fig.savefig(
    "results/migration/" + f"migration{filename}", dpi=300, transparent=True,
)

# kymograph

M = 25
timeframes = 30

H = []
for r in range(M):
    H.append([])
    for t in range(timeframes):
        H[r].append([])


for i in range(len(df)):

    X = df.iloc[i, 1]
    Y = df.iloc[i, 2]
    T = df.iloc[i, 4]
    A = len(T)

    for i in range(A - 1):

        t = T[i]
        t = int(t / timewidth)  # change
        (cx0, cy0) = C[t]
        (cx1, cy1) = C[t + 1]
        x0 = X[i]
        x1 = X[i + 1]
        y0 = Y[i]
        y1 = Y[i + 1]
        r0 = ((cx0 - x0) ** 2 + (cy0 - y0) ** 2) ** 0.5
        r1 = ((cx1 - x1) ** 2 + (cy1 - y1) ** 2) ** 0.5
        r = int(r0 * M / 188)

        H[r][t].append(r1 - r0)

for r in range(M):
    for t in range(timeframes):
        m = len(H[r][t])
        if m > 0:
            H[r][t] = cell.mean(H[r][t])
        else:
            H[r][t] = 0

fig, ax = plt.subplots()

pos = ax.imshow(H, origin=["lower"])

fig.colorbar(pos)
fig.savefig(
    f"results/migration/kymograph{filename}.png", dpi=300, transparent=True,
)
plt.close()
