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

filenames, fileType = cl.getFilesType()

for filename in filenames:

    vid = sm.io.imread(f"dat/{filename}/probBoundary{filename}.tif").astype(float)

    for t in range(len(vid)):
        if t > 99:
            T = f"{t}"
        elif t > 9:
            T = "0" + f"{t}"
        else:
            T = "00" + f"{t}"

        img = np.asarray(vid[t] * 255, "uint8")
        tifffile.imwrite(
            f"dat/{filename}/imagesForSeg/probBoundary{filename}_{T}.tif", img
        )

