from ast import Break
import os
from os.path import exists
import shutil
from math import floor, log10, factorial

from collections import Counter
from trace import Trace
from turtle import position
import cv2
import matplotlib
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import random
import scipy as sp
import scipy.special as sc
import scipy.linalg as linalg
import shapely
import skimage as sm
import skimage.io
import skimage.measure
import skimage.feature
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
from mpl_toolkits.mplot3d import Axes3D
import tifffile
from skimage.draw import circle_perimeter
from scipy import optimize
import xml.etree.ElementTree as et
from scipy.optimize import leastsq
from datetime import datetime
import cellProperties as cell
import utils as util

pd.options.mode.chained_assignment = None
plt.rcParams.update({"font.size": 10})


# -------------------


filenames, fileType = util.getFilesType()
scale = 123.26 / 512


# display division tracks
if True:
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")

    util.createFolder("image/")
    for filename in filenames:
        df = dfDivisions[dfDivisions["Filename"] == filename]
        df = df[df["T"] > 20]

        h2Stack = sm.io.imread(f"dat/{filename}/{filename}.tif").astype(int)[:, :, 1]
        h2 = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)[:, :, :, 0]
        T, Z, X, Y = h2Stack.shape

        for i in range(len(df)):
            x = df["X"].iloc[i] / scale
            y = 512 - df["Y"].iloc[i] / scale
            t = int(df["T"].iloc[i] / 2)

            xMax = int(x + 30)
            xMin = int(x - 30)
            yMax = int(y + 30)
            yMin = int(y - 30)
            if xMax > 512:
                xMaxCrop = 512 - xMax
                xMax = 512
            else:
                xMaxCrop = 0
            if xMin < 0:
                xMinCrop = xMin
                xMin = 0
            else:
                xMinCrop = 0
            if yMax > 512:
                yMaxCrop = 512 - yMax
                yMax = 512
            else:
                yMaxCrop = 0
            if yMin < 0:
                yMinCrop = yMin
                yMin = 0
            else:
                yMinCrop = 0

            vidStack = np.zeros([10, Z, 60, 60])
            vid = np.zeros([10, 60, 60])
            for j in range(10):

                vidStack[j] = h2Stack[t - 9 + j, :, yMin:yMax, xMin:xMax]
                vid[j] = h2[t - 9 + j, yMin:yMax, xMin:xMax]

            vid = np.asarray(vid, "uint8")
            tifffile.imwrite(
                f"image/vidH2.tif", vid, imagej=True, metadata={"axes": "TYX"}
            )
            vidStack = np.asarray(vidStack, "uint8")
            tifffile.imwrite(
                f"image/vidStackH2.tif",
                vidStack,
                imagej=True,
                metadata={"axes": "TZYX"},
            )
            print(0)

    shutil.rmtree("image/")