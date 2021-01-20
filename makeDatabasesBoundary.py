import os
import shutil
from math import floor, log10

from collections import Counter
import cv2
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

filenames, fileType = cl.getFilesType()

for filename in filenames:

    if "Wound" in filename:
        wound = True
        dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")

    else:
        wound = False

    vidFile = f"dat/{filename}/binaryBoundary{filename}.tif"

    vid = sm.io.imread(vidFile).astype(int)

    T = len(vid)

    _dfShape = []

    for t in range(T):

        if wound != True:

            print(f"{filename} {t}")

            img = vid[t]

            img = 255 - img

            imgxy = fi.imgrcxy(img)

            # find and labels cells

            imgLabel = sm.measure.label(imgxy, background=0, connectivity=1)
            imgLabels = np.unique(imgLabel)[1:]
            allPolys = []
            allContours = []

            # converts cell boundary into a polygon

            for label in imgLabels:
                contour = sm.measure.find_contours(imgLabel == label, level=0)[0]
                poly = sm.measure.approximate_polygon(contour, tolerance=1)
                allContours.append(contour)
                allPolys.append(poly)

            allPolys, allContours = fi.removeCells(allPolys, allContours)

            # quantifly polygon properties and saves them
            for i in range(len(allPolys)):

                poly = allPolys[i]
                contour = allContours[i]
                polygon = Polygon(poly)
                _dfShape.append(
                    {
                        "Time": t,
                        "Polygon": polygon,
                        "Centroid": cell.centroid(polygon),
                        "Area": cell.area(polygon),
                        "Perimeter": cell.perimeter(polygon),
                        "Orientation": cell.orientation(polygon),
                        "Shape Factor": cell.shapeFactor(polygon),
                        "q": cell.qTensor(polygon),
                        "Trace(S)": cell.traceS(polygon),
                        "Polar": np.array(
                            cell.mayorPolar(polygon), cell.minorPolar(polygon)
                        ),
                    }
                )

        else:

            print(f"{filename} {t}")
            [wx, wy] = dfWound["Centroid"].iloc[t]

            img = vid[t]

            img = 255 - img

            imgxy = fi.imgrcxy(img)

            # find and labels cells

            imgLabel = sm.measure.label(imgxy, background=0, connectivity=1)
            imgLabels = np.unique(imgLabel)[1:]
            allPolys = []
            allContours = []

            # converts cell boundary into a polygon

            for label in imgLabels:
                contour = sm.measure.find_contours(imgLabel == label, level=0)[0]
                poly = sm.measure.approximate_polygon(contour, tolerance=1)
                allContours.append(contour)
                allPolys.append(poly)

            allPolys, allContours = fi.removeCells(allPolys, allContours)

            # quantifly polygon properties and saves them
            for i in range(len(allPolys)):

                poly = allPolys[i]
                contour = allContours[i]
                polygon = Polygon(poly)
                _dfShape.append(
                    {
                        "Time": t,
                        "Polygon": polygon,
                        "RWCo x": cell.centroid(polygon)[0] - wx,
                        "RWCo y": cell.centroid(polygon)[1] - wy,
                        "Centroid": cell.centroid(polygon),
                        "Area": cell.area(polygon),
                        "Perimeter": cell.perimeter(polygon),
                        "Orientation": cell.orientation(polygon),
                        "Shape Factor": cell.shapeFactor(polygon),
                        "q": cell.qTensor(polygon),
                        "Trace(S)": cell.traceS(polygon),
                        "Polar": np.array(
                            cell.mayorPolar(polygon), cell.minorPolar(polygon)
                        ),
                    }
                )

    dfShape = pd.DataFrame(_dfShape)
    dfShape.to_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")
