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
import random

import cellProperties as cell
import findGoodCells as fi

f = open("pythonText.txt", "r")

filenames = f.read()
filenames = filenames.split(", ")

training = []

for filename in filenames:

    vidFile = f"dat/{filename}/focusH2{filename}.tif"
    vid = sm.io.imread(vidFile).astype(int)

    t0 = int(random.uniform(0, 1) * 176)

    for i in range(5):
        training.append(vid[t0 + i])

    t0 = int(random.uniform(0, 1) * 176)

    for i in range(5):
        training.append(vid[t0 + i])

training = np.asarray(training, "uint8")
tifffile.imwrite(f"dat/trainingData/trainingDataH2.tif", training)

training = []

for filename in filenames:

    vidFile = f"dat/{filename}/focusEcad{filename}.tif"
    vid = sm.io.imread(vidFile).astype(int)

    t0 = int(random.uniform(0, 1) * 176)
    # t0 = int(random.uniform(0, 1) * 56)  # change for woundsite

    for i in range(5):
        training.append(vid[t0 + i])

    t0 = int(random.uniform(0, 1) * 176)
    # t0 = int(random.uniform(0, 1) * 56)

    for i in range(5):
        training.append(vid[t0 + i])

training = np.asarray(training, "uint8")
tifffile.imwrite(f"dat/trainingData/trainingDataEcad.tif", training)

# training = np.asarray(training, "uint8")
# tifffile.imwrite(f"dat/trainingData/trainingDataWound.tif", training)

