import os
from math import floor, log10

import cv2
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg as linalg
from scipy import ndimage
import shapely
import skimage as sm
import skimage.io
import skimage.measure
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
import tifffile
from skimage.draw import circle_perimeter

import cell_properties as cell
import find_good_cells as fi

filename = "HelenH2"

stack = sm.io.imread("dat_nucleus/nucleusTracksH2.tif").astype(float)

mirror = np.zeros([len(stack), len(stack[0]) + 4, 512, 512])

for t in range(len(stack[0])):
    mirror[:, 2 + t] = stack[:, t]

mirror[:, 0] = stack[:, 1]
mirror[:, 1] = stack[:, 0]
mirror[:, len(stack[0]) + 2] = stack[:, len(stack[0]) - 1]
mirror[:, len(stack[0]) + 3] = stack[:, len(stack[0]) - 2]

mirror = np.asarray(mirror, "uint8")
tifffile.imwrite(f"results/mitosis/mirror_{filename}.tif", mirror)
