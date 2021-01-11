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
from collections import Counter

import cellProperties as cell
import findGoodCells as fi

plt.rcParams.update({"font.size": 20})
plt.ioff()
pd.set_option("display.width", 1000)


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


# Apply code for an example
# x = np.r_[36, 36, 19, 18, 33, 26]
# y = np.r_[14, 10, 28, 31, 18, 26]
# comp_curv = ComputeCurvature()
# curvature = comp_curv.fit(x, y)

# theta_fit = np.linspace(-np.pi, np.pi, 180)
# x_fit = comp_curv.xc + comp_curv.r*np.cos(theta_fit)
# y_fit = comp_curv.yc + comp_curv.r*np.sin(theta_fit)
# plt.plot(x_fit, y_fit, 'k--', label='fit', lw=2)
# plt.plot(x, y, 'ro', label='data', ms=8, mec='b', mew=1)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('curvature = {:.3e}'.format(curvature))
# plt.show()

f = open("pythonText.txt", "r")

filename = f.read()

start = (256, 256)  # if not all wounds are centred

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
        "Centroid": (Cx, Cy),
        "Contour": contour,
        "Area": polygon.area,
    }
)

mostLabel = label

finished = False

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
                    "Centroid": cell.centroid(polygon),
                    "Contour": contour,
                    "Area": polygon.area,
                }
            )

tf = t
for t in range(tf, T - 1):
    vidWound[t + 1][vidLabels[t + 1] != 256] = 0

dfWound = pd.DataFrame(_dfWound)

vidEcad = sm.io.imread(f"dat/{filename}/focusEcad{filename}.tif").astype(int)
vidH2 = sm.io.imread(f"dat/{filename}/focusH2{filename}.tif").astype(int)

vidEcad[vidWound == 255] = 0

vidEcad = np.asarray(vidEcad, "uint8")
tifffile.imwrite(f"dat/{filename}/focusEcad{filename}.tif", vidEcad)

vidH2[vidWound == 255] = 0

vidH2 = np.asarray(vidH2, "uint8")
tifffile.imwrite(f"dat/{filename}/focusH2{filename}.tif", vidH2)

# find the woundsite after epithelialisation

tf = dfWound["Time"].iloc[-1] + 1
xf = dfWound["Centroid"].iloc[-1][0]
yf = dfWound["Centroid"].iloc[-1][1]

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

for t in range(tf, T - 1):

    x = [xf - 100, xf + 100]
    y = [yf - 100, yf + 100]
    dfxy = sortGrid(dfVelocity[dfVelocity["T"] == t], x, y)

    v = cell.mean(list(dfxy["velocity"]))

    xf = xf + v[0]
    yf = yf + v[1]

    _dfWound.append(
        {"Time": t, "Centroid": [xf, yf],}
    )

    x = 512 - int(yf)  # change coord
    y = int(xf)
    vidWound[t][x - 6 : x + 6, y - 6 : y + 6] = 255

vidWound[T - 1][x - 6 : x + 6, y - 6 : y + 6] = 255
_dfWound.append(
    {"Time": T - 1, "Centroid": [xf, yf],}
)

dfWound = pd.DataFrame(_dfWound)

vidWound = np.asarray(vidWound, "uint8")
tifffile.imwrite(f"dat/{filename}/woundsite{filename}.tif", vidWound)

dfWound.to_pickle(f"dat/{filename}/woundsite{filename}.pkl")

# display Wound

vid = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(float)

vid[:, :, :, 2][vidWound == 255] = 255

vid = np.asarray(vid, "uint8")
tifffile.imwrite(f"dat/{filename}/highlightWound{filename}.tif", vid)

