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
from collections import Counter

import cell_properties as cell
import find_good_cells as fi

plt.rcParams.update({"font.size": 20})
plt.ioff()
pd.set_option("display.width", 1000)
filename = "HelenH2"

df_mitosis = pd.read_pickle(f"databases/mitosisPoly{filename}.pkl")

binary = sm.io.imread(f"dat_binary/binary_prob_HelenEcad.tif").astype(float)

vid_labels = []
T = len(binary)

for t in range(T):
    img = binary[t]
    mu = cell.mean(cell.mean(img))
    if mu < 130:
        img = img
        img = 255 - img

    vid_labels.append(sm.measure.label(img, background=0, connectivity=1))

vid_labels = np.asarray(vid_labels, "uint16")
tifffile.imwrite(f"dat_binary/vid_labels_{filename}.tif", vid_labels)

unique = set(list(df_mitosis.iloc[:, 0]))

_df4 = []

for label in unique:

    df3 = df_mitosis.loc[lambda df_mitosis: df_mitosis["Label"] == label, :]

    polygonsParent = []
    polygonsDaughter1 = []
    polygonsDaughter2 = []

    (Cx, Cy) = df3.iloc[0, 2][-1]
    t0 = df3.iloc[0, 1][-1]

    if t0 < len(vid_labels) - 5:

        Cx = int(Cx)
        Cy = int(Cy)

        img_label = vid_labels[t0][Cy, Cx]  # change coord

        cell_pos = np.zeros([512, 512])

        cell_pos[vid_labels[t0] == img_label] = 1

        labels = vid_labels[t0 + 4][cell_pos == 1]

        uniqueLabels = set(list(labels))

        count = Counter(labels)
        c = []
        for l in uniqueLabels:
            c.append(count[l])

        uniqueLabels = list(uniqueLabels)
        daughterLabel1 = uniqueLabels[c.index(max(c))]

        c.remove(max(c))
        uniqueLabels.remove(daughterLabel1)

        daughterLabel2 = uniqueLabels[c.index(max(c))]

        iter = 0

        while (daughterLabel1 != daughterLabel2) and (iter <= 4):

            savedLabel1 = daughterLabel1

            labels = vid_labels[t0 + 3 - iter][
                vid_labels[t0 + 4 - iter] == daughterLabel1
            ]
            uniqueLabels = set(list(labels))
            count = Counter(labels)
            c = []
            for l in uniqueLabels:
                c.append(count[l])

            uniqueLabels = list(uniqueLabels)
            daughterLabel1 = uniqueLabels[c.index(max(c))]

            # ----

            savedLabel2 = daughterLabel2

            labels = vid_labels[t0 + 3 - iter][
                vid_labels[t0 + 4 - iter] == daughterLabel2
            ]
            uniqueLabels = set(list(labels))
            count = Counter(labels)
            c = []
            for l in uniqueLabels:
                c.append(count[l])

            uniqueLabels = list(uniqueLabels)
            daughterLabel2 = uniqueLabels[c.index(max(c))]

            iter = iter + 1

        parentLabel = daughterLabel1

        t0 = t0 + 5 - iter  # time of cytokinesis

        if len(vid_labels) > t0 + 10:
            t_fin = 9
        else:
            t_fin = len(vid_labels) - t0 - 1

        daughterLabel1 = savedLabel1

        contour = sm.measure.find_contours(vid_labels[t0] == daughterLabel1, level=0)[0]
        poly = sm.measure.approximate_polygon(contour, tolerance=1)
        polygonsDaughter1.append(poly)

        # --

        daughterLabel2 = savedLabel2

        contour = sm.measure.find_contours(vid_labels[t0] == daughterLabel2, level=0)[0]
        poly = sm.measure.approximate_polygon(contour, tolerance=1)
        polygonsDaughter2.append(poly)

        for i in range(t_fin):

            labels = vid_labels[t0 + 1 + i][vid_labels[t0 + i] == daughterLabel1]

            uniqueLabels = set(list(labels))
            count = Counter(labels)
            c = []
            for l in uniqueLabels:
                c.append(count[l])

            uniqueLabels = list(uniqueLabels)
            daughterLabel1 = uniqueLabels[c.index(max(c))]

            contour = sm.measure.find_contours(
                vid_labels[t0 + 1 + i] == daughterLabel1, level=0
            )[0]
            poly = sm.measure.approximate_polygon(contour, tolerance=1)
            polygonsDaughter1.append(poly)

            # ----

            labels = vid_labels[t0 + 1 + i][vid_labels[t0 + i] == daughterLabel2]

            uniqueLabels = set(list(labels))
            count = Counter(labels)
            c = []
            for l in uniqueLabels:
                c.append(count[l])

            uniqueLabels = list(uniqueLabels)
            daughterLabel2 = uniqueLabels[c.index(max(c))]

            contour = sm.measure.find_contours(
                vid_labels[t0 + 1 + i] == daughterLabel2, level=0
            )[0]
            poly = sm.measure.approximate_polygon(contour, tolerance=1)
            polygonsDaughter2.append(poly)

        if 0 < t0 - 10:
            t_mitosis = 9
        else:
            t_mitosis = t0 - 1

        contour = sm.measure.find_contours(vid_labels[t0 - 1] == parentLabel, level=0)[
            0
        ]
        poly = sm.measure.approximate_polygon(contour, tolerance=1)
        polygonsParent.append(poly)

        for i in range(t_mitosis):
            labels = vid_labels[t0 - i - 2][vid_labels[t0 - i - 1] == parentLabel]

            uniqueLabels = set(list(labels))
            count = Counter(labels)
            c = []
            for l in uniqueLabels:
                c.append(count[l])

            uniqueLabels = list(uniqueLabels)
            parentLabel = uniqueLabels[c.index(max(c))]

            contour = sm.measure.find_contours(
                vid_labels[t0 - i - 2] == parentLabel, level=0
            )[0]
            poly = sm.measure.approximate_polygon(contour, tolerance=1)
            polygonsParent.append(poly)

        _df4.append(
            {
                "Label": label,
                "Parent": polygonsParent,
                "Daughter1": polygonsDaughter1,
                "Daughter2": polygonsDaughter2,
            }
        )

df4 = pd.DataFrame(_df4)

A = []
for t in range(20):
    A.append([])

Sf = []
for t in range(20):
    Sf.append([])

for i in range(len(df4)):
    polygonsParent = df4.iloc[i, 1]
    polygonsDaughter1 = df4.iloc[i, 2]
    polygonsDaughter2 = df4.iloc[i, 3]

    for j in range(len(polygonsParent)):
        poly = polygonsParent[j]
        polygon = Polygon(poly)
        area = cell.area(polygon)
        shape = cell.shape_factor(polygon)

        A[9 - j].append(area)
        Sf[9 - j].append(shape)

    for j in range(len(polygonsDaughter1)):
        poly = polygonsDaughter1[j]
        polygon = Polygon(poly)
        area = cell.area(polygon)
        shape = cell.shape_factor(polygon)

        A[10 + j].append(area)
        Sf[10 + j].append(shape)

        # ----

        poly = polygonsDaughter2[j]
        polygon = Polygon(poly)
        area = cell.area(polygon)
        shape = cell.shape_factor(polygon)

        A[10 + j].append(area)
        Sf[10 + j].append(shape)

errA = []
errSf = []

for t in range(20):
    errA.append(cell.sd(A[t]) / len(A[t]) ** 0.5)
    errSf.append(cell.sd(Sf[t]) / len(Sf[t]) ** 0.5)
    A[t] = cell.mean(A[t])
    Sf[t] = cell.mean(Sf[t])

x = np.array(range(20))

fig = plt.figure(1, figsize=(8, 8))
plt.errorbar(x, A, errA)
fig.savefig(
    "results/mitosis/" + f"cytokinesis_area", dpi=300, transparent=True,
)
plt.close("all")

fig = plt.figure(1, figsize=(8, 8))
plt.errorbar(x, Sf, errSf)
fig.savefig(
    "results/mitosis/" + f"cytokinesis_sf", dpi=300, transparent=True,
)
plt.close("all")
