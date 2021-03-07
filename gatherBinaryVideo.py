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

filenames = cl.getFiles()

for filename in filenames:

    if "Wound" in filename:

        vid = sm.io.imread(f"dat/{filename}/ecadFocus{filename}.tif").astype(float)

        T = len(vid)

        binary_vid = []
        track_vid = []

        for frame in range(T):

            if frame < 10:
                framenum = f"00{frame}"
            elif frame < 100:
                framenum = f"0{frame}"
            else:
                framenum = f"{frame}"

            foldername = (
                f"/Users/jt15004/Documents/Coding/Python/woundHealing/dat/{filename}/imagesForSeg"
                + f"/ecadWoundmask{filename}"
                + f"_{framenum}"
            )

            imgRGB = sm.io.imread(foldername + "/handCorrection.tif").astype(float)
            img = imgRGB[:, :, 0]
            binary_vid.append(img)

            img = sm.io.imread(foldername + "/tracked_cells_resized.tif").astype(float)
            track_vid.append(img)

        filename = filename.replace("probBoundary", "")

        data = np.asarray(binary_vid, "uint8")
        tifffile.imwrite(f"dat/{filename}/ecadBinary{filename}.tif", data)

        data = np.asarray(track_vid, "uint8")
        tifffile.imwrite(f"dat/{filename}/track{filename}.tif", data)

    else:

        vid = sm.io.imread(f"dat/{filename}/ecadFocus{filename}.tif").astype(float)

        T = len(vid)

        binary_vid = []
        track_vid = []

        for frame in range(T):

            if frame < 10:
                framenum = f"00{frame}"
            elif frame < 100:
                framenum = f"0{frame}"
            else:
                framenum = f"{frame}"

            foldername = (
                f"/Users/jt15004/Documents/Coding/Python/woundHealing/dat/{filename}/imagesForSeg"
                + f"/ecadProb{filename}"
                + f"_{framenum}"
            )

            imgRGB = sm.io.imread(foldername + "/handCorrection.tif").astype(float)
            img = imgRGB[:, :, 0]
            binary_vid.append(img)

            img = sm.io.imread(foldername + "/tracked_cells_resized.tif").astype(float)
            track_vid.append(img)

        filename = filename.replace("probBoundary", "")

        data = np.asarray(binary_vid, "uint8")
        tifffile.imwrite(f"dat/{filename}/ecadBinary{filename}.tif", data)

        data = np.asarray(track_vid, "uint8")
        tifffile.imwrite(f"dat/{filename}/track{filename}.tif", data)

