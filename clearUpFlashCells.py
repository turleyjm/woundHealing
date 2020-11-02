import os
from math import floor, log10
import xml.etree.ElementTree as et

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


f = open("pythonText.txt", "r")

filenames = f.read()
filenames = filenames.split(", ")


for filename in filenames:

    vid = sm.io.imread(f"dat/{filename}/track{filename}.tif").astype(float)
    binary = sm.io.imread(f"dat/{filename}/flash{filename}.tif").astype(float)

    vidLabel = []

    # vidTrack = vid

    for img in binary:

        imgLabel = sm.measure.label(255 - img, background=0, connectivity=1)
        vidLabel.append(imgLabel)

    [T, X, Y, rgb] = vid.shape

    uniqueFrame = []

    for t in range(T):
        uniqueRows = np.unique(vid[t, 0, :, :], axis=0)
        for x in range(X):
            uniqueRow = np.unique(vid[t, x, :, :], axis=0)
            uniqueRows = np.concatenate((uniqueRows, uniqueRow))

        uniqueFrame.append(np.unique(uniqueRows, axis=0))

    flashs = []
    for t in range(1, T - 1):

        for colour in uniqueFrame[t]:

            preFrame = uniqueFrame[t - 1][
                np.all((uniqueFrame[t - 1] - colour) == 0, axis=1)
            ]
            nextFrame = uniqueFrame[t + 1][
                np.all((uniqueFrame[t + 1] - colour) == 0, axis=1)
            ]

            if len(preFrame) == 1:
                preFlash = False
            else:
                preFlash = True

            if len(nextFrame) == 1:
                postFlash = False
            else:
                postFlash = True

            if preFlash and postFlash:
                flashs.append([t, colour])
    #             vidTrack[t][np.all((vidTrack[t] - colour) == 0, axis=2)] = np.array(
    #                 [0, 0, 0]
    #             )

    # vidTrack = np.asarray(vidTrack, "uint8")
    # tifffile.imwrite(f"dat/{filename}/removeLabel{filename}.tif", vidTrack)

    for t, colour in flashs:
        labels = vidLabel[t][np.all((vid[t] - colour) == 0, axis=2)]
        preLabel = vidLabel[t - 1][vidLabel[t] == labels[0]]
        postLabel = vidLabel[t + 1][vidLabel[t] == labels[0]]
        flashLabel = labels[0]

        uniqueLabels = set(list(preLabel))
        if 0 in uniqueLabels:
            uniqueLabels.remove(0)

        if len(uniqueLabels) == 0:
            binary[t][vidLabel[t] == flashLabel] = 255
        else:
            count = Counter(preLabel)
            c = []
            for l in uniqueLabels:
                c.append(count[l])

            uniqueLabels = list(uniqueLabels)
            preLabel = uniqueLabels[c.index(max(c))]

            realLabels = vidLabel[t][vidLabel[t - 1] == preLabel]

            uniqueLabels = set(list(realLabels))
            uniqueLabels.remove(flashLabel)
            if 0 in uniqueLabels:
                uniqueLabels.remove(0)

            if len(uniqueLabels) != 0:

                count = Counter(realLabels)
                c = []
                for l in uniqueLabels:
                    c.append(count[l])

                uniqueLabels = list(uniqueLabels)
                realLabel = uniqueLabels[c.index(max(c))]

                contourReal = sm.measure.find_contours(
                    vidLabel[t] == realLabel, level=0
                )[0]
                contourFlash = sm.measure.find_contours(
                    vidLabel[t] == flashLabel, level=0
                )[0]

                overlap = []

                if realLabel != flashLabel:

                    for con in contourFlash:
                        if (
                            len(contourReal[np.all((contourReal - con) == 0, axis=1)])
                            == 1
                        ):
                            overlap.append(
                                contourReal[np.all((contourReal - con) == 0, axis=1)][0]
                            )

                    for con in overlap:
                        binary[t][int(con[0]), int(con[1])] = 0

    binary = np.asarray(binary, "uint8")
    tifffile.imwrite(f"dat/{filename}/binaryBoundary{filename}.tif", binary)
