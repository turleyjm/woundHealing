import os
from math import floor, log10

import cv2
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg as linalg
from scipy import ndimage as ndi
import shapely
import skimage as sm
import skimage.io
import skimage.measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry.polygon import LinearRing
import findGoodCells as fi
import tifffile

folder1 = "dat/datProbBoundary"
folder2 = "dat/datBinaryBoundary"

cwd = os.getcwd()

files = os.listdir(cwd + f"/{folder1}")

for vidFile in files:

    filename = vidFile

    filename = filename.replace(".tif", "")

    vid = sm.io.imread(folder1 + "/" + vidFile).astype(float)

    T = len(vid)

    binary_vid = []
    # track_vid = []

    for frame in range(T):

        if frame < 10:
            framenum = f"00{frame}"
        elif frame < 100:
            framenum = f"0{frame}"
        else:
            framenum = f"{frame}"

        foldername = (
            "/Users/jt15004/Documents/Coding/Python/woundHealing/dat/imagesForSeg"
            + f"/{filename}"
            + f"_{framenum}"
        )

        imgRGB = sm.io.imread(foldername + "/handCorrection.tif").astype(float)
        img = imgRGB[:, :, 0]
        binary_vid.append(img)

        # img = sm.io.imread(foldername + "/tracked_cells_resized.tif").astype(float)
        # track_vid.append(img)

    filename = filename.replace("probBoundary", "")

    data = np.asarray(binary_vid, "uint8")
    tifffile.imwrite(f"{folder2}" + "/" + f"binary{filename}.tif", data)

    # data = np.asarray(track_vid, "uint8")
    # tifffile.imwrite(f"{folder}" + f"track_{filename}.tif", data)

