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


folder = "dat/datBinaryBoundary"

cwd = os.getcwd()

files = os.listdir(cwd + f"/{folder}")

for vidFile in files:

    filename = vidFile

    filename = filename.replace("binary", "")
    filename = filename.replace(".tif", "")

    vidFile = f"{folder}/" + vidFile

    vid = sm.io.imread(vidFile).astype(int)

    T = len(vid)

    _dfShape = []

    for t in range(T):

        img = vid[t]

        img = 255 - img

        imgxy = fi.imgrcxy(img)

        imgLabel = sm.measure.label(imgxy, background=0, connectivity=1)
        imgLabels = np.unique(imgLabel)[1:]
        allPolys = []
        allContours = []

        for label in imgLabels:
            contour = sm.measure.find_contours(imgLabel == label, level=0)[0]
            poly = sm.measure.approximate_polygon(contour, tolerance=1)
            allContours.append(contour)
            allPolys.append(poly)

        allPolys, allContours = fi.removeCells(allPolys, allContours)
        for i in range(len(allPolys)):

            poly = allPolys[i]
            contours = allContours[i]
            polygon = Polygon(poly)
            _dfShape.append(
                {
                    "Time": t,
                    "Polygon": polygon,
                    "Contour": contour,
                    "Curvature": cell.findContourCurvature(contour, 11),
                    "Centroid": cell.centroid(polygon),
                    "Area": cell.area(polygon),
                    "Perimeter": cell.perimeter(polygon),
                    "Orientation": cell.orientation(polygon),
                    "Circularity": cell.circularity(polygon),
                    "Ellipticity": cell.ellipticity(polygon),
                    "Shape Factor": cell.shapeFactor(polygon),
                    "Q": cell.shapeTensor(polygon),
                    "Trace(S)": cell.traceS(polygon),
                    "Trace(QQ)": cell.traceQQ(polygon),
                    "Trace(q)": cell.traceqq(polygon),
                    "Polar_x": cell.mayorPolar(polygon),
                    "Polar_y": cell.minorPolar(polygon),
                    "Polarisation Orientation": cell.polarOri(polygon),
                    "Polarisation Magnitude": cell.polarMag(polygon),
                }
            )

    dfShape = pd.DataFrame(_dfShape)
    dfShape.to_pickle(f"dat/databases/BoundaryShape{filename}.pkl")
