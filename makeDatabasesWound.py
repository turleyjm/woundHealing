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

filenames = f.read()
filenames = filenames.split(", ")

i = 0
starts = [
    (230, 250),
]  # if not all wounds are centred

for filename in filenames:

    if "Unwound" in filename:
        wound = False
    else:
        wound = True

    vidFile = f"dat/{filename}/outPlane{filename}.tif"  # change

    vidWound = sm.io.imread(vidFile).astype(int)

    (T, X, Y) = vidWound.shape

    vidLabels = []  # labels all wound and out of plane areas

    for t in range(T):
        img = sm.measure.label(vidWound[t], background=0, connectivity=1)
        imgxy = fi.imgrcxy(img)
        vidLabels.append(imgxy)
    vidLabels = np.asarray(vidLabels, "uint8")

    vidOutPlane = np.zeros([181, 514, 514])
    vidOutPlane[:, 1:513, 1:513] = vidLabels

    for t in range(len(vidWound)):  # removes small areas

        imgLabels = np.unique(vidOutPlane[t])[1:]

        for label in imgLabels:
            contour = sm.measure.find_contours(vidOutPlane[t] == label, level=0)[0]
            poly = sm.measure.approximate_polygon(contour, tolerance=1)
            try:
                polygon = Polygon(poly)
                a = cell.area(polygon)

                if a < 250:
                    vidOutPlane[t][vidOutPlane[t] == label] = 0
            except:
                continue

    binary = vidOutPlane[:, 1:513, 1:513]

    binary[binary > 0] = 255

    for t in range(T):
        binary[t] = fi.imgxyrc(binary[t])

    vidOutPlane = np.asarray(binary, "uint8")
    tifffile.imwrite(f"dat/{filename}/outPlane{filename}.tif", vidOutPlane)

    vidLabels = []  # labels all wound and out of plane areas
    vidLabelsrc = []
    for t in range(T):
        img = sm.measure.label(vidOutPlane[t], background=0, connectivity=1)
        imgxy = fi.imgrcxy(img)
        vidLabelsrc.append(img)
        vidLabels.append(imgxy)

    if wound == True:  # If there is a wound the boundary is found quantified

        # start = (int(X / 2), int(Y / 2))  # change coords if not all wounds are centred
        start = starts[i]
        i += 1

        _dfWound = []

        label = vidLabels[0][start]
        contour = sm.measure.find_contours(vidLabels[0] == label, level=0)[0]
        poly = sm.measure.approximate_polygon(contour, tolerance=1)
        polygon = Polygon(poly)
        (Cx, Cy) = cell.centroid(polygon)
        vidWound[0][vidLabelsrc[0] != label] = 0

        m = 41

        curvature = np.array(cell.findContourCurvature(contour, m)) * len(contour)
        t = 0
        _dfWound.append(
            {
                "Time": t,
                "Polygon": polygon,
                "Centroid": (Cx, Cy),
                "Curvature": curvature,
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

                if C < 250:
                    finished = True
                else:
                    contour = sm.measure.find_contours(
                        vidLabels[t + 1] == mostLabel, level=0
                    )[0]
                    poly = sm.measure.approximate_polygon(contour, tolerance=1)
                    polygon = Polygon(poly)
                    (Cx, Cy) = cell.centroid(polygon)
                    vidWound[t + 1][vidLabelsrc[t + 1] != mostLabel] = 0

                    curvature = np.array(cell.findContourCurvature(contour, m)) * len(
                        contour
                    )

                    t += 1

                    _dfWound.append(
                        {
                            "Time": t,
                            "Polygon": polygon,
                            "Centroid": cell.centroid(polygon),
                            "Curvature": curvature,
                            "Contour": contour,
                            "Area": polygon.area,
                        }
                    )

        tf = t
        for t in range(tf, T - 1):
            vidWound[t + 1][vidLabels[t + 1] != 256] = 0

        vidWound = np.asarray(vidWound, "uint8")
        tifffile.imwrite(f"dat/{filename}/woundsite{filename}.tif", vidWound)

        dfWound = pd.DataFrame(_dfWound)
        dfWound.to_pickle(f"dat/{filename}/woundsite{filename}.pkl")

        vidEcad = sm.io.imread(f"dat/{filename}/focusEcad{filename}.tif").astype(int)
        vidH2 = sm.io.imread(f"dat/{filename}/focusH2{filename}.tif").astype(int)

        vidEcad[vidWound == 255] = 0

        vidEcad = np.asarray(vidEcad, "uint8")
        tifffile.imwrite(f"dat/{filename}/focusEcad{filename}.tif", vidEcad)

        vidH2[vidWound == 255] = 0

        vidH2 = np.asarray(vidH2, "uint8")
        tifffile.imwrite(f"dat/{filename}/focusH2{filename}.tif", vidH2)

