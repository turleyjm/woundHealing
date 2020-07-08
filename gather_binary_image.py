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
import find_good_cells as fi
import tifffile

folder = "My_Segmentation/from_fiji/"

filename = "prob_HelenEcad"  # imput the name of the file you what to be gathered
length = 60  # Number of frames in video

binary_vid = []
track_vid = []

for frame in range(length):

    if frame < 10:
        framenum = f"00{frame}"
    else:
        framenum = f"0{frame}"

    foldername = (
        "/Users/jt15004/Documents/Coding/Macros/ImagesForSeg"
        + f"/{filename}"
        + f"_{framenum}"
    )

    imgRGB = sm.io.imread(foldername + "/handCorrection.tif").astype(float)
    img = imgRGB[:, :, 0]
    binary_vid.append(img)

    img = sm.io.imread(foldername + "/tracked_cells_resized.tif").astype(float)
    track_vid.append(img)

data = np.asarray(binary_vid, "uint8")
tifffile.imwrite(f"{folder}" + f"binary_{filename}.tif", data)

data = np.asarray(track_vid, "uint8")
tifffile.imwrite(f"{folder}" + f"track_{filename}.tif", data)

