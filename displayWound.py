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

    vidWound = sm.io.imread(f"dat/{filename}/woundsite{filename}.tif").astype(float)

    vid[:, :, :, 2][vidWound == 255] = 150

    vid = np.asarray(vid, "uint8")
    tifffile.imwrite(f"dat/{filename}/highlightWound{filename}.tif", vid)
