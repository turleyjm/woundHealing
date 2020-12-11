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

    vid = sm.io.imread(f"dat/{filename}/track{filename}.tif").astype(float)

    [T, X, Y, rgb] = vid.shape
    binary = sm.io.imread(f"dat/{filename}/flash{filename}.tif").astype(float)

    vidLabel = []

    # vidTrack = vid

    for img in binary:

        imgLabel = sm.measure.label(255 - img, background=0, connectivity=1)
        vidLabel.append(imgLabel)

    uniqueVideo = []
    storageColourFrame = []

    uniqueFrames = np.unique(vid[0, 0, :, :], axis=0)
    for t in range(T):
        uniqueRows = np.unique(vid[t, 0, :, :], axis=0)
        for x in range(X):
            uniqueRow = np.unique(vid[t, x, :, :], axis=0)
            uniqueRows = np.concatenate((uniqueRows, uniqueRow))

        uniqueFrame = np.unique(uniqueRows, axis=0)
        storageColourFrame.append(uniqueFrame)
        uniqueFrames = np.concatenate((uniqueFrames, uniqueFrame))

    uniqueVideo = np.unique(uniqueFrames, axis=0)

    uniqueVideo

    _dfTracks = []

    for colour in uniqueVideo[0:400]:
        time = []
        for t in range(T):
            frame = storageColourFrame[t][
                np.all((storageColourFrame[t] - colour) == 0, axis=1)
            ]
            if len(frame) > 0:
                time.append(t)

        area = []
        centroid = []

        for t in time:
            a = vid[t][np.all((vid[t] - colour) == 0, axis=2)]
            area.append(len(a))
            label = vidLabel[t][np.all((vid[t] - colour) == 0, axis=2)]
            label = label[0]
            contour = sm.measure.find_contours(vidLabel[t] == label, level=0)[0]
            poly = sm.measure.approximate_polygon(contour, tolerance=1)
            polygon = Polygon(poly)
            centroid.append(cell.centroid(polygon))

        _dfTracks.append(
            {"Colour": colour, "Time": time, "Area": area, "centroid": centroid}
        )

    dfTracks = pd.DataFrame(_dfTracks)

