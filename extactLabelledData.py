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

vidOverlay = sm.io.imread(f"dat/trainingData/deepOverlayBinary.tif").astype(int)
vidOverlayEdits = sm.io.imread(f"dat/trainingData/deepOverlayBinaryEdits.tif").astype(
    int
)
deepEcad = []
deepProb = []
deepBinary = []
deepLabelled = []
t = 0

binary = vidOverlay[:, :, :, 0]
binary[binary > 150] = 255

filename = filenames[0]
vidBinary = sm.io.imread(f"dat/{filename}/binaryBoundary{filename}.tif").astype(int)
vidEcad = sm.io.imread(f"dat/{filename}/focusEcad{filename}.tif").astype(int)
vidProb = sm.io.imread(f"dat/{filename}/probBoundary{filename}.tif").astype(float)

deepEcad.append(vidEcad[2])
deepProb.append(vidProb[2])
deepBinary.append(vidBinary[2])

for filename in filenames:

    vidBinary = sm.io.imread(f"dat/{filename}/binaryBoundary{filename}.tif").astype(int)
    vidEcad = sm.io.imread(f"dat/{filename}/focusEcad{filename}.tif").astype(int)
    vidProb = sm.io.imread(f"dat/{filename}/probBoundary{filename}.tif").astype(float)

    img = binary[3 * t + 1]

    for T in range(len(vidBinary)):
        if np.all((vidBinary[T] - img) == 0):
            deepEcad.append(vidEcad[T])
            deepProb.append(vidProb[T])
            deepBinary.append(vidBinary[T])
    t += 1

    img = binary[3 * t + 1]
    deepLabelled.append(img)

    for T in range(len(vidBinary)):
        if np.all((vidBinary[T] - img) == 0):
            deepEcad.append(vidEcad[T])
            deepProb.append(vidProb[T])
            deepBinary.append(vidBinary[T])
    t += 1

deepEcad = np.asarray(deepEcad, "uint8")
tifffile.imwrite(f"dat/trainingData/deepEcad.tif", deepEcad)
deepProb = np.asarray(deepProb, "float32")
tifffile.imwrite(f"dat/trainingData/deepProb.tif", deepProb)
deepBinary = np.asarray(deepBinary, "uint8")
tifffile.imwrite(f"dat/trainingData/deepBinary.tif", deepBinary)


vidOverlayEdits = sm.io.imread(f"dat/trainingData/deepOverlayBinaryEdits.tif").astype(
    int
)
deepLabelled = []
binary = vidOverlayEdits[:, :, :, 0]
binary[binary > 150] = 255
t = 0
for filename in filenames:
    deepLabelled.append(binary[3 * t + 1])
    t += 1
    deepLabelled.append(binary[3 * t + 1])
    t += 1

deepLabelled = np.asarray(deepLabelled, "uint8")
tifffile.imwrite(f"dat/trainingData/deepLabelled.tif", deepLabelled)
