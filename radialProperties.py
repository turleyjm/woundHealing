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

T = 181

for filename in filenames:

    df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")

    wound = sm.io.imread(f"dat/{filename}/woundsite{filename}.tif").astype("uint8")

    dist = np.zeros([T, 512, 512])
    for t in range(T):
        dist[t] = sp.ndimage.morphology.distance_transform_edt(wound[t])

    mu = []
    err = []

    for t in range(T):
        prop = list(df["Q"][df["Time"] == t])

