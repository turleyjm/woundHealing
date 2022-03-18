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

plt.rcParams.update({"font.size": 20})

# -------------------


def getFiles():
    f = open("pythonText.txt", "r")
    filenames = f.read()
    filenames = filenames.split(", ")
    return filenames


def getFilesType():
    f = open("pythonText.txt", "r")
    fileType = f.read()

    if fileType == "All":
        cwd = os.getcwd()
        Fullfilenames = os.listdir(cwd + "/dat")
        filenames = []
        for filename in Fullfilenames:
            filenames.append(filename)

        if ".DS_Store" in filenames:
            filenames.remove(".DS_Store")

        if "confocalRawLocation.txt" in filenames:
            filenames.remove("confocalRawLocation.txt")

        if "woundDetails.xlsx" in filenames:
            filenames.remove("woundDetails.xlsx")
    else:
        cwd = os.getcwd()
        Fullfilenames = os.listdir(cwd + "/dat")
        filenames = []
        for filename in Fullfilenames:
            if fileType in filename:
                filenames.append(filename)

    filenames.sort()

    return filenames, fileType


def getFilesOfType(fileType):

    cwd = os.getcwd()
    Fullfilenames = os.listdir(cwd + "/dat")
    filenames = []
    for filename in Fullfilenames:
        if fileType in filename:
            filenames.append(filename)

    filenames.sort()

    return filenames


def ThreeD(a):
    lst = [[[] for col in range(a)] for col in range(a)]
    return lst


def sortTime(df, t):

    tMin = t[0]
    tMax = t[1]

    dftmin = df[df["Time"] >= tMin]
    df = dftmin[dftmin["Time"] < tMax]

    return df


def sortRadius(dfVelocity, t, r):

    rMin = r[0]
    rMax = r[1]
    tMin = t[0]
    tMax = t[1]

    dfrmin = dfVelocity[dfVelocity["R"] >= rMin]
    dfr = dfrmin[dfrmin["R"] < rMax]
    dftmin = dfr[dfr["Time"] >= tMin]
    df = dftmin[dftmin["Time"] < tMax]

    return df


def sortGrid(dfVelocity, x, y):

    xMin = x[0]
    xMax = x[1]
    yMin = y[0]
    yMax = y[1]

    dfxmin = dfVelocity[dfVelocity["X"] > xMin]
    dfx = dfxmin[dfxmin["X"] < xMax]

    dfymin = dfx[dfx["Y"] > yMin]
    df = dfymin[dfymin["Y"] < yMax]

    return df


def sortVolume(dfShape, x, y, t):

    xMin = x[0]
    xMax = x[1]
    yMin = y[0]
    yMax = y[1]
    tMin = t[0]
    tMax = t[1]

    dfxmin = dfShape[dfShape["X"] >= xMin]
    dfx = dfxmin[dfxmin["X"] < xMax]

    dfymin = dfx[dfx["Y"] >= yMin]
    dfy = dfymin[dfymin["Y"] < yMax]

    dftmin = dfy[dfy["T"] >= tMin]
    df = dftmin[dftmin["T"] < tMax]

    return df


def sortSection(dfVelocity, r, theta):

    rMin = r[0]
    rMax = r[1]
    thetaMin = theta[0]
    thetaMax = theta[1]

    dfxmin = dfVelocity[dfVelocity["R"] > rMin]
    dfx = dfxmin[dfxmin["R"] < rMax]

    dfymin = dfx[dfx["Theta"] > thetaMin]
    df = dfymin[dfymin["Theta"] < thetaMax]

    return df


def sortBand(dfRadial, band, pixelWidth):

    if band == 1:
        df = dfRadial[dfRadial["Wound Edge Distance"] < pixelWidth]
    else:
        df2 = dfRadial[dfRadial["Wound Edge Distance"] < band * pixelWidth]
        df = df2[df2["Wound Edge Distance"] >= (band - 1) * pixelWidth]

    return df


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


def rotation_matrix(theta):

    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    return R


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def rotation_matrix(theta):

    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    return R


def findStartTime(filename):
    if "Wound" in filename:
        dfwoundDetails = pd.read_excel(f"dat/woundDetails.xlsx")
        t0 = dfwoundDetails["Start Time"][dfwoundDetails["Filename"] == filename].iloc[
            0
        ]
    else:
        t0 = 0

    return t0