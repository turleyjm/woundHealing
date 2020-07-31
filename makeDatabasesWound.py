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

folder = "dat/datProbOutPlane"


cwd = os.getcwd()

files = os.listdir(cwd + f"/{folder}")

try:
    files.remove(".DS_Store")
except:
    pass

for vidFile in files:

    filename = vidFile

    filename = filename.replace("probOutPlane", "")
    filename = filename.replace(".tif", "")

    if "Unwound" in filename:
        wound = False
    else:
        wound = True

    vidFile = f"{folder}/" + vidFile

    vid = sm.io.imread(vidFile).astype(int)
    vidOutPlane = vid

    for t in range(len(vid)):

        img = vid[t]

        binary = np.zeros([514, 514])

        mu = cell.mean(cell.mean(img))

        for x in range(512):
            for y in range(512):
                if img[x, y] == 255:
                    binary[x + 1, y + 1] = 255

        imgLabel = sm.measure.label(binary, background=0, connectivity=1)
        imgLabels = np.unique(imgLabel)[1:]

        for label in imgLabels:
            contour = sm.measure.find_contours(imgLabel == label, level=0)[0]
            poly = sm.measure.approximate_polygon(contour, tolerance=1)
            try:
                polygon = Polygon(poly)
                a = cell.area(polygon)

                if a < 600:
                    binary[imgLabel == label] = 0
            except:
                continue

        binary = binary[1:513, 1:513]

        vidOutPlane[t] = binary

    vidWound = vidOutPlane
    vidOutPlane = np.asarray(vidOutPlane, "uint8")
    tifffile.imwrite(f"dat/datOutPlane/outPlane{filename}.tif", vidOutPlane)

    (T, X, Y) = vidOutPlane.shape

    if wound == True:

        start = (int(Y / 2), int(X / 2))  # change coords

        _dfWound = []

        imgWound = vidWound[0]
        imgLabel = sm.measure.label(imgWound, background=0, connectivity=1)
        label = imgLabel[start]
        contour = sm.measure.find_contours(imgLabel == label, level=0)[0]
        poly = sm.measure.approximate_polygon(contour, tolerance=1)
        polygon = Polygon(poly)
        (Cx, Cy) = cell.centroid(polygon)
        vidWound[0][imgLabel != label] = 0

        m = 41

        curvature = cell.findContourCurvature(contour, m)

        _dfWound.append(
            {
                "name": filename,
                "polygons": polygon,
                "centriods": cell.centroid(polygon),
                "curvature": curvature,
            }
        )

        # for i in range(n): # see contour
        #     vidWound[0][int(contour[i][0]), int(contour[i][1])] = curvature[i]

        for t in range(T - 1):
            imgWound = vidWound[t + 1]

            imgLabel = sm.measure.label(imgWound, background=0, connectivity=1)

            label = imgLabel[int(Cx), int(Cy)]
            contour = sm.measure.find_contours(imgLabel == label, level=0)[0]
            poly = sm.measure.approximate_polygon(contour, tolerance=1)
            polygon = Polygon(poly)
            (Cx, Cy) = cell.centroid(polygon)
            vidWound[t + 1][imgLabel != label] = 0

            curvature = cell.findContourCurvature(contour, m)

            _dfWound.append(
                {
                    "name": filename,
                    "polygon": polygon,
                    "centriod": cell.centroid(polygon),
                    "curvature": curvature,
                }
            )

            # for i in range(n): # see contour
            #     vidWound[t + 1][int(contour[i][0]), int(contour[i][1])] = curvature[i]

    dfWound = pd.DataFrame(_dfWound)
    dfWound.to_pickle(f"dat/databases/woundsite{filename}.pkl")

    vidWound = np.asarray(vidWound, "uint8")
    tifffile.imwrite(f"dat/datWound/maskWound{filename}.tif", vidWound)
