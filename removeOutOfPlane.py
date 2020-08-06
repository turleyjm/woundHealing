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


f = open("pythonText.txt", "r")

filenames = f.read()
filenames = filenames.split(", ")

for filename in filenames:

    if "Unwound" in filename:
        wound = False
    else:
        wound = True

    vidFile = f"dat/{filename}/outPlane{filename}.tif"

    vid = sm.io.imread(vidFile).astype(int)
    vidOutPlane = vid

    # removes all the regions that are out of plane

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

    vidOutPlane = np.asarray(vidOutPlane, "uint8")
    tifffile.imwrite(f"dat/{filename}/outPlane{filename}.tif", vidOutPlane)

    vidEcad = sm.io.imread(f"dat/{filename}/focusEcad{filename}.tif").astype(int)
    vidH2 = sm.io.imread(f"dat/{filename}/focusH2{filename}.tif").astype(int)

    vidEcad[vidOutPlane == 255] = 0

    vidEcad = np.asarray(vidEcad, "uint8")
    tifffile.imwrite(f"dat/{filename}/focusEcad{filename}.tif", vidEcad)

    vidH2[vidOutPlane == 255] = 0

    vidH2 = np.asarray(vidH2, "uint8")
    tifffile.imwrite(f"dat/{filename}/focusH2{filename}.tif", vidH2)
