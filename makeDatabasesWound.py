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


def sortGrid(dfVelocity, x, y):

    xMin = x[0]
    xMax = x[1]
    yMin = y[0]
    yMax = y[1]

    dfxmin = dfVelocity[dfVelocity["X"] > xMin]
    dfx = dfxmin[dfxmin["X"] < xMax]

    dfymin = dfx[dfx["Y"] > yMin]
    df = dfymin[dfymin["Y"] < yMax]

    return df


f = open("pythonText.txt", "r")

filename = f.read()

start = (256, 306)  # if not all wounds are centred

vidFile = f"dat/{filename}/outPlane{filename}.tif"  # change

vidWound = sm.io.imread(vidFile).astype(int)

(T, X, Y) = vidWound.shape

vidLabels = []  # labels all wound and out of plane areas
vidLabelsrc = []
for t in range(T):
    img = sm.measure.label(vidWound[t], background=0, connectivity=1)
    imgxy = fi.imgrcxy(img)
    vidLabelsrc.append(img)
    vidLabels.append(imgxy)

_dfWound = []

label = vidLabels[0][start]
contour = sm.measure.find_contours(vidLabels[0] == label, level=0)[0]
poly = sm.measure.approximate_polygon(contour, tolerance=1)
polygon = Polygon(poly)
(Cx, Cy) = cell.centroid(polygon)
vidWound[0][vidLabelsrc[0] != label] = 0

t = 0
_dfWound.append(
    {
        "Time": t,
        "Polygon": polygon,
        "Position": (Cx, Cy),
        "Centroid": (Cx, Cy),
        "Contour": contour,
        "Area": polygon.area,
        "Shape Factor": cell.shapeFactor(polygon),
        "Circularity": cell.circularity(polygon),
    }
)

mostLabel = label

finished = False


# find the woundsite using migration of cells

xf = Cx
yf = Cy
position = []

dfNucleus = pd.read_pickle(f"dat/{filename}/nucleusTracks{filename}.pkl")
_df2 = []

for i in range(len(dfNucleus)):
    t = dfNucleus["t"][i]
    x = dfNucleus["x"][i]
    y = dfNucleus["y"][i]
    label = dfNucleus["Label"][i]

    m = len(t)
    tMax = t[-1]

    if m > 1:
        for j in range(m - 1):
            t0 = t[j]
            x0 = x[j]
            y0 = y[j]

            tdelta = tMax - t0
            if tdelta > 5:
                t5 = t[j + 5]
                x5 = x[j + 5]
                y5 = y[j + 5]

                v = np.array([(x5 - x0) / 5, (y5 - y0) / 5])

                _df2.append(
                    {"Label": label, "T": t0, "X": x0, "Y": y0, "velocity": v,}
                )
            else:
                tEnd = t[-1]
                xEnd = x[-1]
                yEnd = y[-1]

                v = np.array([(xEnd - x0) / (tEnd - t0), (yEnd - y0) / (tEnd - t0)])

                _df2.append(
                    {"Label": label, "T": t0, "X": x0, "Y": y0, "velocity": v,}
                )

dfVelocity = pd.DataFrame(_df2)

for t in range(T - 1):

    x = [xf - 150, xf + 150]
    y = [yf - 150, yf + 150]
    dfxy = sortGrid(dfVelocity[dfVelocity["T"] == t], x, y)

    v = np.mean(list(dfxy["velocity"]), axis=0)

    xf = xf + v[0]
    yf = yf + v[1]

    position.append((xf, yf))

    x = 512 - int(yf)  # change coord
    y = int(xf)

position.append((xf, yf))

# track wound with time
t = 0
while t < 180 and finished != True:

    labels = vidLabels[t + 1][vidLabels[t] == mostLabel]

    uniqueLabels = set(list(labels))
    if 0 in uniqueLabels:
        uniqueLabels.remove(0)

    if len(uniqueLabels) == 0:
        finished = True
    else:
        count = Counter(labels)
        c = []
        for l in uniqueLabels:
            c.append(count[l])

        uniqueLabels = list(uniqueLabels)
        mostLabel = uniqueLabels[c.index(max(c))]
        C = max(c)

        if C < 50:
            finished = True
        else:
            contour = sm.measure.find_contours(vidLabels[t + 1] == mostLabel, level=0)[
                0
            ]
            poly = sm.measure.approximate_polygon(contour, tolerance=1)
            polygon = Polygon(poly)
            (Cx, Cy) = cell.centroid(polygon)
            vidWound[t + 1][vidLabelsrc[t + 1] != mostLabel] = 0

            t += 1

            _dfWound.append(
                {
                    "Time": t,
                    "Polygon": polygon,
                    "Position": position[t - 1],
                    "Centroid": cell.centroid(polygon),
                    "Contour": contour,
                    "Area": polygon.area,
                    "Shape Factor": cell.shapeFactor(polygon),
                    "Circularity": cell.circularity(polygon),
                }
            )

vidEcad = sm.io.imread(f"dat/{filename}/focusEcad{filename}.tif").astype(int)
vidH2 = sm.io.imread(f"dat/{filename}/focusH2{filename}.tif").astype(int)

vidEcad[vidWound == 255] = 0

vidEcad = np.asarray(vidEcad, "uint8")
tifffile.imwrite(f"dat/{filename}/woundMaskEcad{filename}.tif", vidEcad)

vidH2[vidWound == 255] = 0

vidH2 = np.asarray(vidH2, "uint8")
tifffile.imwrite(f"dat/{filename}/woundMaskH2{filename}.tif", vidH2)


tf = t
for t in range(tf, T - 1):
    vidWound[t + 1][vidLabels[t + 1] != 256] = 0
    [x, y] = [int(position[t - 1][0]), int(512 - position[t - 1][1])]
    vidWound[t + 1][y - 2 : y + 2, x - 2 : x + 2] = 255
    _dfWound.append(
        {"Time": t, "Position": position[t - 1],}
    )

dfWound = pd.DataFrame(_dfWound)
dfWound.to_pickle(f"dat/{filename}/woundsite{filename}.pkl")

vidWound = np.asarray(vidWound, "uint8")
tifffile.imwrite(f"dat/{filename}/woundsite{filename}.tif", vidWound)

# display Wound

vid = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(float)

vid[:, :, :, 2][vidWound == 255] = 100

vid = np.asarray(vid, "uint8")
tifffile.imwrite(f"dat/{filename}/highlightWound{filename}.tif", vid)

dist = []
for t in range(T):
    img = 255 - fi.imgrcxy(vidWound[t])
    dist.append(sp.ndimage.morphology.distance_transform_edt(img))

dist = np.asarray(dist, "uint16")
tifffile.imwrite(f"dat/{filename}/distanceWound{filename}.tif", dist)

