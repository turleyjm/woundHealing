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

    vid = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(float)

    [T, X, Y, rgb] = vid.shape

    highlightDivisions = np.zeros([T, 552, 552, 3])

    for x in range(X):
        for y in range(Y):
            highlightDivisions[:, 20 + x, 20 + y, :] = vid[:, x, y, :]

    df = pd.read_pickle(f"dat/{filename}/mitosisTracks{filename}.pkl")

    uniqueLabels = list(set(df["Label"]))

    for label in uniqueLabels:

        dfTrack = df.loc[lambda df: df["Label"] == label, :]

        t0 = dfTrack.iloc[0]["Time"][-1]
        [x, y] = dfTrack.iloc[0]["Position"][-1]
        x = int(x)
        y = int(y)

        [rr0, cc0] = circle_perimeter(y + 20, x + 20, 13)
        [rr1, cc1] = circle_perimeter(y + 20, x + 20, 14)
        [rr2, cc2] = circle_perimeter(y + 20, x + 20, 15)

        times = range(t0 - 5, t0 + 5)

        timeVid = []
        for t in times:
            if t >= 0 and t <= T - 1:
                timeVid.append(t)

        for t in timeVid:
            highlightDivisions[t][rr0, cc0, 2] = 200
            highlightDivisions[t][rr1, cc1, 2] = 200
            highlightDivisions[t][rr2, cc2, 2] = 200

    highlightDivisions = highlightDivisions[:, 20:532, 20:532]

    highlightDivisions = np.asarray(highlightDivisions, "uint8")
    tifffile.imwrite(f"dat/{filename}/divisions{filename}.tif", highlightDivisions)
