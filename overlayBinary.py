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


f = open("pythonText.txt", "r")

filenames = f.read()
filenames = filenames.split(", ")


for filename in filenames:

    vidFile = f"dat/{filename}/focusEcad{filename}.tif"
    vid = sm.io.imread(vidFile).astype(int)
    binaryFile = f"dat/{filename}/binaryBoundary{filename}.tif"
    binary = sm.io.imread(binaryFile).astype(int)

    overlay = np.zeros([60, 512, 512, 3])

    overlay[:, :, :, 1] = vid
    overlay[:, :, :, 0] = binary * 0.7

    overlay = np.asarray(overlay, "uint8")
    tifffile.imwrite(f"dat/{filename}/overlayBinary{filename}.tif", overlay)
