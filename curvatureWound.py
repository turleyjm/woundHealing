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

from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour


def findCurvatureAtPoints(coord, contour, curvature):

    coordCurvature = []
    n = len(contour)

    for co in coord:
        for i in range(n - 1):
            pts = contour[i]
            if pts[0] == co[0] and pts[1] == co[1]:
                coordCurvature.append(curvature[i])

    return coordCurvature


# def contourHeatmap(coord, prop, contour):

#     index = []
#     propContours = []
#     n = len(contour)
#     m = len(coord)

#     for j in range(m):
#         co = coord[j]
#         for i in range(n - 1):
#             pts = contour[i]
#             if pts[0] == co[0] and pts[1] == co[1]:
#                 index.append(i)

#     c0 = 0
#     for i in range(m):
#         c = index[i] - index[i - 1]

#         if c < c0:
#             c0 = c
#             dsicon = i

#     return propContours


f = open("pythonText.txt", "r")

filenames = f.read()
filenames = filenames.split(", ")

T = 10
numPoints = 20

trackCoord = []
trackContour = []
trackCurvature = []
centroid = []

for filename in filenames:

    vidFile = f"dat/{filename}/maskWound{filename}.tif"

    vid = sm.io.imread(vidFile).astype(int)

    avgVid = []

    for t in range(T):

        img = cell.mean(vid[t * 6 : t * 6 + 6])

        img = sp.ndimage.median_filter(img, 5)

        img[img >= 0.5] = 1
        img[img < 0.5] = 0

        avgVid.append(img)

    img = avgVid[0]

    imgLabel = sm.measure.label(img, background=0, connectivity=1)

    label = imgLabel[256, 256]
    contour = sm.measure.find_contours(imgLabel == label, level=0)[0]
    trackContour.append(contour)
    curvature = np.array(cell.findContourCurvature(contour, 21)) * len(contour)
    poly = sm.measure.approximate_polygon(contour, tolerance=1)
    polygon = Polygon(poly)
    centroid.append(cell.centroid(polygon))

    m = len(contour)
    n = m / numPoints
    start = []

    for i in range(numPoints):
        start.append(contour[int(i * n)])

    previous = start
    trackCoord.append(previous)
    coordCurvature = findCurvatureAtPoints(previous, contour, curvature)
    trackCurvature.append(coordCurvature)

    for t in range(T - 1):

        img = avgVid[t + 1]

        imgLabel = sm.measure.label(img, background=0, connectivity=1)

        label = imgLabel[256, 256]
        contour = sm.measure.find_contours(imgLabel == label, level=0)[0]
        trackContour.append(contour)
        curvature = np.array(cell.findContourCurvature(contour, 21)) * len(contour)
        poly = sm.measure.approximate_polygon(contour, tolerance=1)
        polygon = Polygon(poly)
        centroid.append(cell.centroid(polygon))

        leastSquares = []

        for i in range(numPoints):
            [xs, ys] = previous[i]

            r0 = 100

            for j in range(len(contour)):

                [x, y] = contour[j]

                r = ((x - xs) ** 2 + (y - ys) ** 2) ** 0.5

                if r < r0:
                    pts = j
                    r0 = r

            leastSquares.append(pts)

        deltaPoints = []

        c0 = 0
        for i in range(numPoints):
            c = leastSquares[i] - leastSquares[i - 1]

            if c < c0:
                c0 = c
                index = i

            if c < 0:
                c += len(contour)

            deltaPoints.append(c)

        deltaPoints.append(deltaPoints[0])
        deltaPoints.append(deltaPoints[1])
        deltaPoints.reverse()
        deltaPoints.append(deltaPoints[2])
        deltaPoints.append(deltaPoints[3])
        deltaPoints.reverse()

        rollingAverages = []

        for i in range(numPoints):
            rollingAverages.append(int(cell.mean(deltaPoints[i : i + 5])))

        points = []

        firstPoint = leastSquares[-2:] + leastSquares[:3]
        c0 = 0
        for i in range(4):
            c = firstPoint[i + 1] - firstPoint[i]

            if c < c0:
                c0 = c
                index = i

        if c0 != 0:
            for i in range(index + 1):
                firstPoint[i] = firstPoint[i] - len(contour)

        mu = cell.mean(firstPoint)

        if mu < 0:
            mu += len(contour)

        mu = int(mu)

        points.append(contour[mu])
        count = mu

        for i in range(numPoints - 1):
            count += rollingAverages[i]

            if count > len(contour):
                count = count - len(contour)
            points.append(contour[count])

        trackCoord.append(points)
        previous = points
        coordCurvature = findCurvatureAtPoints(previous, contour, curvature)
        trackCurvature.append(coordCurvature)

    seeTracks = np.zeros([10, 512, 512])

    for t in range(T):
        for i in range(numPoints):
            x = int(list(trackCoord[t][i])[0])
            y = int(list(trackCoord[t][i])[1])
            seeTracks[t, x, y] = trackCurvature[t][i] * 3 + 100
            # if trackCurvature[t][i] * 3 + 100 > 255:
            #     print(trackCurvature[t][i])

    seeTracks = np.asarray(seeTracks, "uint8")
    tifffile.imwrite(f"dat/{filename}/seeTracks{filename}.tif", seeTracks)

    radialVelocity = []
    for t in range(T - 1):
        dr = []
        for i in range(numPoints):

            [Cx, Cy] = centroid[t + 1]
            [x0, y0] = [list(trackCoord[t][i])[0], list(trackCoord[t][i])[1]]
            [x1, y1] = [list(trackCoord[t + 1][i])[0], list(trackCoord[t + 1][i])[1]]

            r0 = ((Cx - x0) ** 2 + (Cy - y0) ** 2) ** 0.5
            r1 = ((Cx - x1) ** 2 + (Cy - y1) ** 2) ** 0.5

            dr.append(r0 - r1)

        radialVelocity.append(dr)

    seeVelocity = np.zeros([9, 512, 512])

    for t in range(T - 1):
        for i in range(numPoints):
            x = int(list(trackCoord[t + 1][i])[0])
            y = int(list(trackCoord[t + 1][i])[1])
            seeVelocity[t, x, y] = radialVelocity[t][i] * 9 + 100
            if radialVelocity[t][i] * 9 + 100 > 255:
                print(radialVelocity[t][i])

    seeVelocity = np.asarray(seeVelocity, "uint8")
    tifffile.imwrite(f"dat/{filename}/seeSpeed{filename}.tif", seeVelocity)

    # curvatureContours = []
    # for t in range(T):
    #     curvatureContours.append(
    #         contourHeatmap(trackCoord[t], trackCurvature[t], trackContour[t])
    #     )

