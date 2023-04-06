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


# ---- individual conditions ----


# fileType = "Unwound18h"
fileType = "WoundL18h"
# fileType = "WoundS18h"

# fileType = "UnwoundJNK"
# fileType = "WoundLJNK"
# fileType = "WoundSJNK"

# fileType = "UnwoundCa"
# fileType = "WoundLCa"
# fileType = "WoundSCa"

# fileType = "Unwoundrpr"
# fileType = "WoundLrpr"
# fileType = "WoundSrpr"

# ---- grouped conditions ----


# fileType = "AllTypes"
# fileType = "AllWound"

# fileType = "18h"
# fileType = "JNK"
# fileType = "Ca"
# fileType = "rpr"

# fileType = "Unwound"
# fileType = "WoundL"
# fileType = "WoundS"


# ---- fileType functions ----


def getFilesType(fileType=fileType):

    if fileType == "All":
        cwd = os.getcwd()
        Fullfilenames = os.listdir(cwd + "/dat")
        filenames = []
        for filename in Fullfilenames:
            filenames.append(filename)

        if ".DS_Store" in filenames:
            filenames.remove(".DS_Store")
        if "woundDetails.xls" in filenames:
            filenames.remove("woundDetails.xls")
        if "woundDetails.xlsx" in filenames:
            filenames.remove("woundDetails.xlsx")
        if "dat_pred" in filenames:
            filenames.remove("dat_pred")
        if "confocalRawLocation.txt" in filenames:
            filenames.remove("confocalRawLocation.txt")
        if "confocalRawLocation.txt" in filenames:
            filenames.remove("confocalRawLocationCa.txt")
        if "confocalRawLocation.txt" in filenames:
            filenames.remove("confocalRawLocationJNK.txt")
        if "confocalRawLocation.txt" in filenames:
            filenames.remove("confocalRawLocation_rpr.txt")

    else:
        cwd = os.getcwd()
        Fullfilenames = os.listdir(cwd + "/dat")
        filenames = []
        for filename in Fullfilenames:
            if fileType in filename:
                filenames.append(filename)

    filenames.sort()

    return filenames, fileType


def getFilesTypes(fileType=fileType):
    if fileType == "AllTypes":
        fileTypes = [
            "Unwound18h",
            "WoundL18h",
            "WoundS18h",
            "UnwoundJNK",
            "WoundLJNK",
            "WoundSJNK",
        ]
        # fileTypes = [
        #     "Unwound18h",
        #     "WoundL18h",
        #     "WoundS18h",
        #     "UnwoundJNK",
        #     "WoundLJNK",
        #     "WoundSJNK",
        #     "UnwoundCa",
        #     "WoundLCa",
        #     "WoundSCa",
        #     "Unwoundrpr",
        #     "WoundLrpr",
        #     "WoundSrpr",
        # ]
    elif fileType == "AllWound":
        fileTypes = ["WoundL18h", "WoundS18h", "WoundLJNK", "WoundSJNK"]

    elif fileType == "18h":
        fileTypes = ["Unwound18h", "WoundL18h", "WoundS18h"]
    elif fileType == "JNK":
        fileTypes = ["UnwoundJNK", "WoundLJNK", "WoundSJNK"]
    elif fileType == "Ca":
        fileTypes = ["UnwoundCa", "WoundLCa", "WoundSCa"]
    elif fileType == "rpr":
        fileTypes = ["Unwoundrpr", "WoundLrpr", "WoundSrpr"]

    elif fileType == "Unwound":
        # fileTypes = ["Unwound18h", "UnwoundJNK", "UnwoundCa", "Unwoundrpr"]
        fileTypes = ["Unwound18h", "UnwoundJNK", "Unwoundrpr"]
    elif fileType == "WoundL":
        # fileTypes = ["WoundL18h", "WoundLJNK", "WoundLCa", "WoundLrpr"]
        fileTypes = ["WoundL18h", "WoundLJNK"]
    elif fileType == "WoundS":
        # fileTypes = ["WoundS18h", "WoundSJNK", "WoundSCa", "WoundSrpr"]
        fileTypes = ["WoundS18h", "WoundSJNK"]

    else:
        fileTypes = [fileType]

    groupTitle = getgroupTitle(fileType)

    return fileTypes, groupTitle


def getFileTitle(fileType):

    if fileType == "WoundL18h":
        fileTitle = "large wound wt"
    elif fileType == "WoundS18h":
        fileTitle = "small wound wt"
    elif fileType == "Unwound18h":
        fileTitle = "unwounded wt"

    elif fileType == "WoundLJNK":
        fileTitle = "large wound JNK DN"
    elif fileType == "WoundSJNK":
        fileTitle = "small wound JNK DN"
    elif fileType == "UnwoundJNK":
        fileTitle = "unwounded JNK DN"

    elif fileType == "WoundLCa":
        fileTitle = "large wound Ca RNAi"
    elif fileType == "WoundSCa":
        fileTitle = "small wound Ca RNAi"
    elif fileType == "UnwoundCa":
        fileTitle = "unwounded Ca RNAi"

    elif fileType == "WoundLrpr":
        fileTitle = "large wound immune ablation"
    elif fileType == "WoundSrpr":
        fileTitle = "small wound immune ablation"
    elif fileType == "Unwoundrpr":
        fileTitle = "unwounded immune ablation"

    return fileTitle


def getgroupTitle(fileTypes):

    if fileTypes == "AllTypes":
        groupTitle = "all conditions"
    elif fileTypes == "AllWound":
        groupTitle = "all wounded conditions"

    elif fileTypes == "18h":
        groupTitle = "wild type"
    elif fileTypes == "JNK":
        groupTitle = "JNK DN"
    elif fileTypes == "Ca":
        groupTitle = "Ca RNAi"
    elif fileTypes == "rpr":
        groupTitle = "immune ablation"

    elif fileTypes == "Unwound":
        groupTitle = "unwounded"
    elif fileTypes == "WoundL":
        groupTitle = "large wound"
    elif fileTypes == "WoundS":
        groupTitle = "small wound"

    else:
        groupTitle = getFileTitle(fileTypes[0])

    return groupTitle


def getBoldTitle(fileTitle):

    if len(str(fileTitle).split(" ")) == 1:
        boldTitle = r"$\bf{" + fileTitle + "}$"
    elif len(str(fileTitle).split(" ")) == 2:
        boldTitle = (
            r"$\bf{"
            + fileTitle.split(" ")[0]
            + r"}$ $\bf{"
            + str(fileTitle).split(" ")[1]
            + "}$"
        )
    elif len(str(fileTitle).split(" ")) == 3:
        boldTitle = (
            r"$\bf{"
            + fileTitle.split(" ")[0]
            + r"}$ $\bf{"
            + str(fileTitle).split(" ")[1]
            + r"}$ $\bf{"
            + str(fileTitle).split(" ")[2]
            + "}$"
        )
    elif len(str(fileTitle).split(" ")) == 4:
        boldTitle = (
            r"$\bf{"
            + fileTitle.split(" ")[0]
            + r"}$ $\bf{"
            + str(fileTitle).split(" ")[1]
            + r"}$ $\bf{"
            + str(fileTitle).split(" ")[2]
            + r"}$ $\bf{"
            + str(fileTitle).split(" ")[3]
            + "}$"
        )

    return boldTitle


def compareType(groupTitle):
    if groupTitle == "wild type":
        compare = "Unwound18h"
    elif groupTitle == "JNK DN":
        compare = "UnwoundJNK"
    elif groupTitle == "Ca RNAi":
        compare = "UnwoundCa"
    elif groupTitle == "immune ablation":
        compare = "Unwoundrpr"

    return compare


def controlType(fileType):
    if "18h" in fileType:
        control = "Unwound18h"
    elif "JNK" in fileType:
        control = "UnwoundJNK"
    elif "Ca" in fileType:
        control = "UnwoundCa"
    elif "rpr" in fileType:
        control = "Unwoundrpr"

    return control


def getColorLineMarker(fileType, groupTitle):

    if (
        groupTitle == "wild type"
        or groupTitle == "JNK DN"
        or groupTitle == "Ca RNAi"
        or groupTitle == "immune ablation"
    ):
        colorDict = {
            "Unwound18h": [3, "o"],
            "WoundL18h": [10, "^"],
            "WoundS18h": [20, "s"],
            "UnwoundJNK": [3, ">"],
            "WoundLJNK": [10, "*"],
            "WoundSJNK": [20, "+"],
            "UnwoundCa": [3, "h"],
            "WoundLCa": [10, "d"],
            "WoundSCa": [20, "<"],
            "Unwoundrpr": [3, "v"],
            "WoundLrpr": [10, "H"],
            "WoundSrpr": [20, "p"],
        }
    else:
        colorDict = {
            "Unwound18h": [0, "o"],
            "WoundL18h": [0, "^"],
            "WoundS18h": [0, "s"],
            "UnwoundJNK": [8, ">"],
            "WoundLJNK": [8, "*"],
            "WoundSJNK": [8, "+"],
            "UnwoundCa": [16, "h"],
            "WoundLCa": [16, "d"],
            "WoundSCa": [16, "<"],
            "Unwoundrpr": [22, "v"],
            "WoundLrpr": [22, "H"],
            "WoundSrpr": [22, "p"],
        }

    n = 23
    cm = plt.get_cmap("gist_rainbow")
    i, mark = colorDict[fileType]

    return cm(1.0 * i / n), mark


# ---------------


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


def vidrcxyRGB(vid):

    T, X, Y, C = vid.shape

    vidxy = np.zeros(shape=(T, X, Y, C))

    for x in range(X):
        for y in range(Y):

            vidxy[:, x, y] = vid[:, (Y - 1) - y, x]

    return vidxy


def vidxyrcRGB(vid):

    T, X, Y, C = vid.shape

    vidrc = np.zeros(shape=(T, X, Y, C))

    for x in range(X):
        for y in range(Y):

            vidrc[:, (Y - 1) - y, x] = vid[:, x, y]

    return vidrc


def vidrcxy(vid):

    T, X, Y = vid.shape

    vidxy = np.zeros(shape=(T, X, Y))

    for x in range(X):
        for y in range(Y):

            vidxy[:, x, y] = vid[:, (Y - 1) - y, x]

    return vidxy


def vidxyrc(vid):

    T, X, Y = vid.shape

    vidrc = np.zeros(shape=(T, X, Y))

    for x in range(X):
        for y in range(Y):

            vidrc[:, (Y - 1) - y, x] = vid[:, x, y]

    return vidrc


def imgrcxy(img):

    X, Y = img.shape

    imgxy = np.zeros(shape=(X, Y))

    for x in range(X):
        for y in range(Y):

            imgxy[x, y] = img[(Y - 1) - y, x]

    return imgxy


def imgxyrc(img):

    X, Y = img.shape

    imgrc = np.zeros(shape=(X, Y))

    for x in range(X):
        for y in range(Y):

            imgrc[(Y - 1) - y, x] = img[x, y]

    return imgrc


def Prewound(filename):
    filenamePre = "Prew" + filename.split("W")[1]

    return filenamePre
