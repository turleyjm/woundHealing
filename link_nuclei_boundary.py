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

for label in unique:

    df3 = df_mitosis.loc[lambda df_mitosis: df_mitosis["Label"] == label, :]

    polygons_parent = []
    polygons_daughter1 = []
    polygons_daughter2 = []

    (Cx, Cy) = df3.iloc[0, 2][-1]
    t0 = df3.iloc[0, 1][-1]

    if t0 < len(vid_labels) - 5:

        Cx = int(Cx)
        Cy = int(Cy)

        img_label = vid_labels[t0][Cy, Cx]

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

        contour = sm.measure.find_contours(
            vid_labels[t0 + 4] == daughterLabel1, level=0
        )[0]
        poly = sm.measure.approximate_polygon(contour, tolerance=1)
        polygons_daughter1.append(poly)

        contour = sm.measure.find_contours(
            vid_labels[t0 + 4] == daughterLabel2, level=0
        )[0]
        poly = sm.measure.approximate_polygon(contour, tolerance=1)
        polygons_daughter2.append(poly)

        iter = 0

        while (daughterLabel1 != daughterLabel2) and (iter <= 4):

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

            contour = sm.measure.find_contours(
                vid_labels[t0 + 3 - iter] == daughterLabel1, level=0
            )[0]
            poly = sm.measure.approximate_polygon(contour, tolerance=1)
            polygons_daughter1.append(poly)

            # ----

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

            contour = sm.measure.find_contours(
                vid_labels[t0 + 3 - iter] == daughterLabel2, level=0
            )[0]
            poly = sm.measure.approximate_polygon(contour, tolerance=1)
            polygons_daughter2.append(poly)

            iter = iter + 1

        polygons_daughter1.pop(-1)
        polygons_daughter2.pop(-1)

        polygons_parent
