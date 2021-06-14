import os
import shutil
from math import floor, log10

from collections import Counter
import cv2
import matplotlib
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

f = open("pythonText.txt", "r")

filename = f.read()

highlightWound = sm.io.imread(f"dat/{filename}/highlightWound{filename}.tif").astype(
    int
)

dfDivisions = pd.read_pickle(f"dat/{filename}/mitosisTracks{filename}.pkl")

(T, X, Y, rgb) = highlightWound.shape

highlightDivisions = np.zeros([T, 552, 552, 3])

for x in range(X):
    for y in range(Y):
        highlightDivisions[:, 20 + x, 20 + y, :] = highlightWound[:, x, y, :]

uniqueLabels = list(set(dfDivisions["Label"]))

for label in uniqueLabels:

    dfTrack = dfDivisions.loc[lambda dfDivisions: dfDivisions["Label"] == label, :]

    t0 = dfTrack.iloc[0]["Time"][-1]
    [x, y] = dfTrack.iloc[0]["Position"][-1]
    x = int(x)
    y = int(y)

    rr0, cc0 = sm.draw.circle(551 - (y + 20), x + 20, 15)
    rr1, cc1 = sm.draw.circle(551 - (y + 20), x + 20, 12)

    times = range(t0 - 5, t0 + 5)

    timeVid = []
    for t in times:
        if t >= 0 and t <= T - 1:
            timeVid.append(t)

    for t in timeVid:
        highlightDivisions[t][rr0, cc0, 2] = 200
        highlightDivisions[t][rr1, cc1, 2] = 0

highlightDivisions = highlightDivisions[:, 20:532, 20:532]

highlightDivisions = np.asarray(highlightDivisions, "uint8")
tifffile.imwrite(f"results/divisionsWound{filename}.tif", highlightDivisions)