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

# -----


def round_sig(x, sig=2):

    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def plot_dist(prop, function_name, function_title, filename, bins=40, xlim="None"):
    """produces a bar plot with mean line from the colume col of table df"""

    # mu = cell.mean(prop)
    # sigma = cell.sd(prop)
    # sigma = float(sigma)
    # sigma = round_sig(sigma, 3)
    fig, ax = plt.subplots()
    plt.gcf().subplots_adjust(bottom=0.15)
    ax.hist(prop, density=False, bins=bins)
    ax.set_xlabel(function_name, y=0.13)
    # ax.axvline(mu, c="k", label="mean")
    # ax.axvline(mu + sigma, c="k", label=r"$\sigma$", ls="--")
    # ax.axvline(mu - sigma, c="k", ls="--")
    # ax.axvline(med, c='r', label='median')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlim != "None":
        ax.set_xlim(xlim)
    # plt.suptitle(f"Distribution of {function_name}", y=1)
    # plt.suptitle(r"$\sigma$" + f" = {sigma}", y=0.95)
    # fig.legend(loc="upper right", fontsize=18, bbox_to_anchor=(0.9, 0.85))
    fig.savefig(
        "results/bar_graphs/" + f"_dist_{function_title}_{filename}.png",
        dpi=200,
        transparent=True,
    )


# ------

plt.rcParams.update({"font.size": 20})
plt.ioff()
pd.set_option("display.width", 1000)

wound = sm.io.imread("dat_nucleus/Helen_wound.tif").astype(float)
Centroids = []
for t in range(len(wound)):
    img_wound = wound[t]
    img_wound[img_wound == 0] = 1
    img_wound[img_wound == 255] = 0
    img_wound[img_wound == 1] = 255
    img_label = sm.measure.label(img_wound, background=0, connectivity=1)
    contour = sm.measure.find_contours(img_label == 1, level=0)[0]
    poly = sm.measure.approximate_polygon(contour, tolerance=1)
    wound_polygon = Polygon(poly)
    c = cell.centroid(wound_polygon)
    c = np.array(c)
    Centroids.append(c)

filename = "HelenH2"

df_mitosis = pd.read_pickle(f"databases/mitosisPoly{filename}.pkl")

binary = sm.io.imread(f"dat_binary/binary_prob_HelenEcad.tif").astype(float)
trackBinary = binary

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
    tm = df3.iloc[0, 1][-1]

    Cx = int(Cx)
    Cy = int(Cy)
    t = tm

    parentLabel = vid_labels[tm][Cy, Cx]  # change coord

    divided = False

    while divided == False and (t + 1 < T):

        labels = vid_labels[t + 1][vid_labels[t] == parentLabel]

        uniqueLabels = set(list(labels))
        if 0 in uniqueLabels:
            uniqueLabels.remove(0)

        count = Counter(labels)
        c = []
        for l in uniqueLabels:
            c.append(count[l])

        uniqueLabels = list(uniqueLabels)
        mostLabel = uniqueLabels[c.index(max(c))]
        C = max(c)

        c.remove(max(c))
        uniqueLabels.remove(mostLabel)

        if c == []:
            Cdash = 0
        else:
            mostLabel2nd = uniqueLabels[c.index(max(c))]
            Cdash = max(c)

        if Cdash / C > 0.5:
            divided = True
            daughterLabel1 = mostLabel
            daughterLabel2 = mostLabel2nd
        else:
            t += 1
            parentLabel = mostLabel

    if divided == True:

        tc = t  # time of cytokinesis

        if len(vid_labels) > tc + 11:
            t_fin = 9
        else:
            t_fin = len(vid_labels) - tc - 2

        trackBinary[tc + 1][vid_labels[tc + 1] == daughterLabel1] = 200

        contour = sm.measure.find_contours(
            vid_labels[tc + 1] == daughterLabel1, level=0
        )[0]
        poly = sm.measure.approximate_polygon(contour, tolerance=1)
        polygonsDaughter1.append(poly)

        # --

        trackBinary[tc + 1][vid_labels[tc + 1] == daughterLabel2] = 150

        contour = sm.measure.find_contours(
            vid_labels[tc + 1] == daughterLabel2, level=0
        )[0]
        poly = sm.measure.approximate_polygon(contour, tolerance=1)
        polygonsDaughter2.append(poly)

        for i in range(t_fin):

            labels = vid_labels[tc + 2 + i][vid_labels[tc + 1 + i] == daughterLabel1]

            uniqueLabels = set(list(labels))
            if 0 in uniqueLabels:
                uniqueLabels.remove(0)

            count = Counter(labels)
            c = []
            for l in uniqueLabels:
                c.append(count[l])

            uniqueLabels = list(uniqueLabels)
            daughterLabel1 = uniqueLabels[c.index(max(c))]

            trackBinary[tc + 2 + i][vid_labels[tc + 2 + i] == daughterLabel1] = 200

            contour = sm.measure.find_contours(
                vid_labels[tc + 2 + i] == daughterLabel1, level=0
            )[0]
            poly = sm.measure.approximate_polygon(contour, tolerance=1)
            polygonsDaughter1.append(poly)

            # ----

            labels = vid_labels[tc + 2 + i][vid_labels[tc + 1 + i] == daughterLabel2]

            uniqueLabels = set(list(labels))
            if 0 in uniqueLabels:
                uniqueLabels.remove(0)

            count = Counter(labels)
            c = []
            for l in uniqueLabels:
                c.append(count[l])

            uniqueLabels = list(uniqueLabels)
            daughterLabel2 = uniqueLabels[c.index(max(c))]

            trackBinary[tc + 2 + i][vid_labels[tc + 2 + i] == daughterLabel2] = 150

            contour = sm.measure.find_contours(
                vid_labels[tc + 2 + i] == daughterLabel2, level=0
            )[0]
            poly = sm.measure.approximate_polygon(contour, tolerance=1)
            polygonsDaughter2.append(poly)

        if 0 < tc - 10:
            t_mitosis = 9
        else:
            t_mitosis = tc

        trackBinary[tc][vid_labels[tc] == parentLabel] = 100
        contour = sm.measure.find_contours(vid_labels[tc] == parentLabel, level=0)[0]
        poly = sm.measure.approximate_polygon(contour, tolerance=1)
        polygonsParent.append(poly)

        for i in range(t_mitosis):
            labels = vid_labels[tc - i - 1][vid_labels[tc - i] == parentLabel]

            uniqueLabels = set(list(labels))
            if 0 in uniqueLabels:
                uniqueLabels.remove(0)

            count = Counter(labels)
            c = []
            for l in uniqueLabels:
                c.append(count[l])

            uniqueLabels = list(uniqueLabels)
            parentLabel = uniqueLabels[c.index(max(c))]

            trackBinary[tc - i - 1][vid_labels[tc - i - 1] == parentLabel] = 100

            contour = sm.measure.find_contours(
                vid_labels[tc - i - 1] == parentLabel, level=0
            )[0]
            poly = sm.measure.approximate_polygon(contour, tolerance=1)
            polygonsParent.append(poly)

        _df4.append(
            {
                "Label": label,
                "Parent": polygonsParent,
                "Daughter1": polygonsDaughter1,
                "Daughter2": polygonsDaughter2,
                "Time difference": tc - tm,
                "Cytokineses time": tc,
            }
        )

df4 = pd.DataFrame(_df4)


trackBinary = np.asarray(trackBinary, "uint8")
tifffile.imwrite(f"results/mitosis/tracks_{filename}.tif", trackBinary)

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

x = np.array(range(20)) - 10

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

time_delay = []
for i in range(len(df4)):
    time = df4.iloc[i][4]
    time_delay.append(time)

plot_dist(time_delay, "time-lag", "time-lag", filename, bins=20)

div_ori = []

for i in range(len(df4)):
    polygonDaughter1 = df4.iloc[i, 2][0]
    polygonDaughter2 = df4.iloc[i, 3][0]
    tc = df4.iloc[i, 5]

    polygonDaughter1 = Polygon(polygonDaughter1)
    polygonDaughter2 = Polygon(polygonDaughter2)

    (Cx, Cy) = Centroids[tc + 1]
    [x0, y0] = cell.centroid(polygonDaughter1)
    [x1, y1] = cell.centroid(polygonDaughter2)

    xm = (x0 + x1) / 2
    ym = (y0 + y1) / 2
    v = np.array([x0 - x1, y0 - y1])
    w = np.array([xm - Cx, ym - Cy])

    phi = np.arccos(np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w)))

    if phi > np.pi / 2:
        theta = np.pi - phi
    else:
        theta = phi
    div_ori.append(theta * (180 / np.pi))

plot_dist(div_ori, "div_ori", "div_ori_cells", filename, bins=15, xlim=[0, 90])
