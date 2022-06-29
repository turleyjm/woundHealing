from ast import Break
import os
from os.path import exists
import shutil
from math import floor, log10, factorial

from collections import Counter
from trace import Trace
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


def dist(polygon, polygon0):
    [x1, y1] = cell.centroid(polygon)
    [x0, y0] = cell.centroid(polygon0)
    return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5


def angleDiff(theta, phi):

    diff = theta - phi

    if abs(diff) > 90:
        if diff > 0:
            diff = 180 - diff
        else:
            diff = 180 + diff

    return abs(diff)


def findtcj(polygon, img):

    centroid = cell.centroid(polygon)
    x, y = int(centroid[0]), int(centroid[1])
    img = 1 - img / 255
    img = np.asarray(img, "uint8")

    imgLabel = sm.measure.label(img, background=0, connectivity=1)
    label = imgLabel[x, y]
    contour = sm.measure.find_contours(imgLabel == label, level=0)[0]

    # imgLabelrc = util.imgxyrc(imgLabel)
    # imgLabelrc[imgLabelrc == label] = round(1.25 * imgLabelrc.max())
    # imgLabelrc = np.asarray(imgLabelrc, "uint16")
    # tifffile.imwrite(f"results/imgLabel{filename}.tif", imgLabelrc)

    if label == 0:
        print("label == 0")

    zeros = np.zeros([512, 512])

    zeros[imgLabel == label] = 1
    for con in contour:
        zeros[int(con[0]), int(con[1])] = 1

    struct2 = sp.ndimage.generate_binary_structure(2, 2)
    dilation = sp.ndimage.morphology.binary_dilation(zeros, structure=struct2).astype(
        zeros.dtype
    )
    dilation[zeros == 1] = 0
    # dilationrc = util.imgxyrc(dilation)
    # dilationrc = np.asarray(dilationrc, "uint16")
    # tifffile.imwrite(f"results/dilation{filename}.tif", dilationrc)

    tcj = np.zeros([512, 512])
    diff = img - dilation
    tcj[diff == -1] = 1
    tcj[tcj != 1] = 0

    outerTCJ = skimage.feature.peak_local_max(tcj)
    # tcjrc = util.imgxyrc(tcj)
    # tcjrc = np.asarray(tcjrc, "uint16")
    # tifffile.imwrite(f"results/tcj{filename}.tif", tcjrc)

    tcj = []
    for coord in outerTCJ:
        tcj.append(findtcjContour(coord, contour[0:-1]))

    if "False" in tcj:
        tcj.remove("False")
        print("removed")

    return tcj


def isBoundary(contour):

    boundary = False

    for con in contour:
        if con[0] == 0:
            boundary = True
        if con[1] == 0:
            boundary = True
        if con[0] == 511:
            boundary = True
        if con[1] == 511:
            boundary = True

    return boundary


def findtcjContour(coord, contour):

    close = []
    for con in contour:
        r = ((con[0] - coord[0]) ** 2 + (con[1] - coord[1]) ** 2) ** 0.5
        if r < 1.5:
            close.append(con)

    if len(close) == 1:
        tcj = close[0]
    elif len(close) == 0:
        tcj = "False"
    else:
        tcj = np.mean(close, axis=0)

    return tcj


def getSecondColour(track, colour):
    colours = track[np.all((track - colour) != 0, axis=1)]
    colours = colours[np.all((colours - np.array([255, 255, 255])) != 0, axis=1)]

    col = []
    count = []
    while len(colours) > 0:
        col.append(colours[0])
        count.append(len(colours[np.all((colours - colours[0]) == 0, axis=1)]))
        colours = colours[np.all((colours - colours[0]) != 0, axis=1)]

    maxm = np.max(count)
    colourD = col[count.index(maxm)]

    return colourD


# -------------------

filenames, fileType = util.getFilesType()
scale = 123.26 / 512


if True:
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    _df = []
    _dfTrack = []
    for filename in filenames:
        print(filename)

        df = dfDivisions[dfDivisions["Filename"] == filename]
        dfFileShape = dfShape[dfShape["Filename"] == filename]
        meanArea = np.mean(dfShape["Area"])
        stdArea = np.std(dfShape["Area"])
        t0 = util.findStartTime(filename)

        tracks = sm.io.imread(f"dat/{filename}/binaryTracks{filename}.tif").astype(int)
        binary = sm.io.imread(f"dat/{filename}/binary{filename}.tif").astype(int)
        T, X, Y, C = tracks.shape

        binary = util.vidrcxy(binary)
        tracks = util.vidrcxyRGB(tracks)
        tracksDivisions = tracks

        for i in range(len(df)):

            label = df["Label"].iloc[i]
            ori = df["Orientation"].iloc[i]
            tm = t = int((df["T"].iloc[i] - t0) / 2)
            x = df["X"].iloc[i] / scale
            y = df["Y"].iloc[i] / scale

            colour = tracks[t, int(x), int(y)]
            if np.all((colour - np.array([255, 255, 255])) == 0):
                continue

            track = tracks[t][np.all((tracks[t] - colour) == 0, axis=2)]
            if len(track) > 1500:
                continue

            finished = False
            t_i = t
            while finished == False:
                t_i += 1
                A0 = len(track)
                track = tracks[t_i][np.all((tracks[t_i - 1] - colour) == 0, axis=2)]
                A1 = len(track[np.all((track - colour) == 0, axis=1)])
                if A1 / A0 < 0.65:
                    finished = True
                    try:
                        colourD = getSecondColour(track, colour)
                    except:
                        continue

                if A1 > 1500:
                    finished = True
                    colourD = getSecondColour(track, colour)

                if t_i == T - 1:
                    finished = True
                    colourD = getSecondColour(track, colour)
            if A1 > 1500:
                continue
            tc = t_i - 1

            if tc - tm < 3:

                if tc > 30:
                    time = np.linspace(tc, tc - 30, 31)
                else:
                    time = np.linspace(tc, 0, tc + 1)
                polyList = []

                t = int(time[0])
                contour = sm.measure.find_contours(
                    np.all((tracks[t] - colour) == 0, axis=2), level=0
                )[0]
                poly = sm.measure.approximate_polygon(contour, tolerance=1)
                try:
                    polygon = Polygon(poly)
                    tcj = findtcj(polygon, binary[t])
                    theta = cell.orientation_tcj(tcj)
                    _dfTrack.append(
                        {
                            "Filename": filename,
                            "Label": label,
                            "Type": "parent",
                            "Colour": colour,
                            "Time": t,
                            "Division Time": int(t - tm),
                            "Polygon": polygon,
                            "Area": polygon.area,
                            "Shape Factor": cell.shapeFactor(polygon),
                            "Orientation": cell.orientation(polygon),
                            "Shape Factor tcj": cell.shapeFactor_tcj(tcj),
                            "Orientation tcj": cell.orientation_tcj(tcj),
                        }
                    )
                    polyList.append(polygon)
                    polygon0 = polygon
                except:
                    continue

                for t in time[1:]:
                    try:
                        contour = sm.measure.find_contours(
                            np.all((tracks[int(t)] - colour) == 0, axis=2), level=0
                        )[0]
                        poly = sm.measure.approximate_polygon(contour, tolerance=1)
                        polygon = Polygon(poly)

                        if polygon.area == 0:
                            break
                        elif dist(polygon, polygon0) > 10:
                            break
                        elif polygon0.area / polygon.area < 2 / 3:
                            break
                        elif polygon.area / polygon0.area < 2 / 3:
                            break
                        elif polygon.area > 1500:
                            break
                        else:
                            tcj = findtcj(polygon, binary[int(t)])
                            _dfTrack.append(
                                {
                                    "Filename": filename,
                                    "Label": label,
                                    "Type": "parent",
                                    "Colour": colour,
                                    "Time": int(t),
                                    "Division Time": int(t - tm),
                                    "Polygon": polygon,
                                    "Area": polygon.area,
                                    "Shape Factor": cell.shapeFactor(polygon),
                                    "Orientation": cell.orientation(polygon),
                                    "Shape Factor tcj": cell.shapeFactor_tcj(tcj),
                                    "Orientation tcj": cell.orientation_tcj(tcj),
                                }
                            )
                            polyList.append(polygon)
                            polygon0 = polygon

                    except:
                        break

                if tc < T - 31:
                    timeD = np.linspace(tc + 1, tc + 31, 31)
                else:
                    timeD = np.linspace(tc + 1, T - 1, T - tc - 1)
                polyListD1 = []

                t = int(timeD[0])
                try:
                    contour = sm.measure.find_contours(
                        np.all((tracks[t] - colour) == 0, axis=2), level=0
                    )[0]
                    poly = sm.measure.approximate_polygon(contour, tolerance=1)
                    polygon = Polygon(poly)
                    if polygon.area == 0:
                        continue
                    elif polygon.area > 1500:
                        continue
                    else:
                        tcj = findtcj(polygon, binary[t])
                        theta = cell.orientation_tcj(tcj)
                        _dfTrack.append(
                            {
                                "Filename": filename,
                                "Label": label,
                                "Type": "daughter1",
                                "Colour": colour,
                                "Time": t,
                                "Division Time": int(t - tm),
                                "Polygon": polygon,
                                "Area": polygon.area,
                                "Shape Factor": cell.shapeFactor(polygon),
                                "Orientation": cell.orientation(polygon),
                                "Shape Factor tcj": cell.shapeFactor_tcj(tcj),
                                "Orientation tcj": cell.orientation_tcj(tcj),
                            }
                        )
                        polyListD1.append(polygon)
                        polygon0 = polygon
                except:
                    continue

                for t in timeD[1:]:
                    try:
                        contour = sm.measure.find_contours(
                            np.all((tracks[int(t)] - colour) == 0, axis=2), level=0
                        )[0]
                        poly = sm.measure.approximate_polygon(contour, tolerance=1)
                        polygon = Polygon(poly)

                        if polygon.area == 0:
                            break
                        elif dist(polygon, polygon0) > 10:
                            break
                        elif polygon0.area / polygon.area < 2 / 3:
                            break
                        elif polygon.area / polygon0.area < 2 / 3:
                            break
                        elif polygon.area > 1500:
                            break
                        else:
                            tcj = findtcj(polygon, binary[int(t)])
                            _dfTrack.append(
                                {
                                    "Filename": filename,
                                    "Label": label,
                                    "Type": "daughter1",
                                    "Colour": colour,
                                    "Time": int(t),
                                    "Division Time": int(t - tm),
                                    "Polygon": polygon,
                                    "Area": polygon.area,
                                    "Shape Factor": cell.shapeFactor(polygon),
                                    "Orientation": cell.orientation(polygon),
                                    "Shape Factor tcj": cell.shapeFactor_tcj(tcj),
                                    "Orientation tcj": cell.orientation_tcj(tcj),
                                }
                            )
                            polyListD1.append(polygon)
                            polygon0 = polygon

                    except:
                        break

                polyListD2 = []

                t = int(timeD[0])
                try:
                    contour = sm.measure.find_contours(
                        np.all((tracks[t] - colourD) == 0, axis=2), level=0
                    )[0]
                    poly = sm.measure.approximate_polygon(contour, tolerance=1)
                    polygon = Polygon(poly)
                    if polygon.area == 0:
                        continue
                    elif polygon.area > 1500:
                        continue
                    else:
                        tcj = findtcj(polygon, binary[t])
                        theta = cell.orientation_tcj(tcj)
                        _dfTrack.append(
                            {
                                "Filename": filename,
                                "Label": label,
                                "Type": "daughter2",
                                "Colour": colourD,
                                "Time": t,
                                "Division Time": int(t - tm),
                                "Polygon": polygon,
                                "Area": polygon.area,
                                "Shape Factor": cell.shapeFactor(polygon),
                                "Orientation": cell.orientation(polygon),
                                "Shape Factor tcj": cell.shapeFactor_tcj(tcj),
                                "Orientation tcj": cell.orientation_tcj(tcj),
                            }
                        )
                        polyListD2.append(polygon)
                        polygon0 = polygon
                except:
                    continue

                for t in timeD[1:]:
                    try:
                        contour = sm.measure.find_contours(
                            np.all((tracks[int(t)] - colourD) == 0, axis=2), level=0
                        )[0]
                        poly = sm.measure.approximate_polygon(contour, tolerance=1)
                        polygon = Polygon(poly)

                        if polygon.area == 0:
                            break
                        elif dist(polygon, polygon0) > 10:
                            break
                        elif polygon0.area / polygon.area < 2 / 3:
                            break
                        elif polygon.area / polygon0.area < 2 / 3:
                            break
                        elif polygon.area > 1500:
                            break
                        else:
                            tcj = findtcj(polygon, binary[int(t)])
                            _dfTrack.append(
                                {
                                    "Filename": filename,
                                    "Label": label,
                                    "Type": "daughter2",
                                    "Colour": colourD,
                                    "Time": int(t),
                                    "Division Time": int(t - tm),
                                    "Polygon": polygon,
                                    "Area": polygon.area,
                                    "Shape Factor": cell.shapeFactor(polygon),
                                    "Orientation": cell.orientation(polygon),
                                    "Shape Factor tcj": cell.shapeFactor_tcj(tcj),
                                    "Orientation tcj": cell.orientation_tcj(tcj),
                                }
                            )
                            polyListD2.append(polygon)
                            polygon0 = polygon

                    except:
                        break

                _df.append(
                    {
                        "Filename": filename,
                        "Label": label,
                        "Orientation": ori,
                        "Times": np.array(time),
                        "Times daughters": np.array(timeD),
                        "Cytokineses Time": tc,
                        "Anaphase Time": tm,
                        "X": x,
                        "Y": y,
                        "Colour": colour,
                        "Daughter Colour": colourD,
                        "Polygons": polyList,
                        "Polygons daughter1": polyListD1,
                        "Polygons daughter2": polyListD2,
                        "Track length": len(polyList),
                        "Daughter length": np.min([len(polyListD1), len(polyListD2)]),
                    }
                )

    dfDivisionShape = pd.DataFrame(_df)
    dfDivisionShape.to_pickle(f"databases/dfDivisionShape{fileType}.pkl")
    dfDivisionTrack = pd.DataFrame(_dfTrack)
    dfDivisionTrack.to_pickle(f"databases/dfDivisionTrack{fileType}.pkl")


# display division tracks
if True:
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    for filename in filenames:
        focus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)
        tracks = sm.io.imread(f"dat/{filename}/binaryTracks{filename}.tif").astype(int)
        dfFile = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        df = dfFile[dfFile["Type"] == "parent"]

        (T, X, Y, rgb) = focus.shape

        for i in range(len(df)):

            colour = df["Colour"].iloc[i]
            t = df["Time"].iloc[i]
            focus[t, :, :, 2][np.all((tracks[t] - colour) == 0, axis=2)] = 255

        df = dfFile[dfFile["Type"] == "daughter1"]

        for i in range(len(df)):

            colour = df["Colour"].iloc[i]
            t = df["Time"].iloc[i]
            focus[t, :, :, 2][np.all((tracks[t] - colour) == 0, axis=2)] = 200
            focus[t, :, :, 0][np.all((tracks[t] - colour) == 0, axis=2)] += 150
            focus[t, :, :, 0][focus[t, :, :, 0] > 255] = 255

        df = dfFile[dfFile["Type"] == "daughter2"]

        for i in range(len(df)):

            colour = df["Colour"].iloc[i]
            t = df["Time"].iloc[i]
            focus[t, :, :, 2][np.all((tracks[t] - colour) == 0, axis=2)] = 200
            focus[t, :, :, 1][np.all((tracks[t] - colour) == 0, axis=2)] += 150
            focus[t, :, :, 1][focus[t, :, :, 1] > 255] = 255

        focus = np.asarray(focus, "uint8")
        tifffile.imwrite(f"results/divisionsTracksDisplay{filename}test.tif", focus)


# area of parent dividing cells
if True:
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfDivisionTrack = dfDivisionTrack[dfDivisionTrack["Type"] == "parent"]
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    T = np.max(dfShape["T"])
    time = list(
        np.linspace(
            np.min(dfDivisionTrack["Division Time"]),
            np.max(dfDivisionTrack["Division Time"]),
            int(
                np.max(dfDivisionTrack["Division Time"])
                - np.min(dfDivisionTrack["Division Time"])
                + 1
            ),
        )
    )
    area = [[] for col in range(len(time))]
    dArea = [[] for col in range(len(time))]
    for filename in filenames:
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        dfFileShape = dfShape[dfShape["Filename"] == filename]
        areaT = []
        for t in range(T):
            areaT.append(np.mean(dfFileShape["Area"][dfFileShape["T"] == t]))

        for i in range(len(df)):

            t = df["Time"].iloc[i]
            if t < T:
                divTime = df["Division Time"].iloc[i]
                index = time.index(divTime)
                area[index].append(df["Area"].iloc[i] * scale ** 2)
                dArea[index].append(df["Area"].iloc[i] * scale ** 2 - areaT[t])

    std = []
    dAreastd = []
    for i in range(len(area)):
        std.append(np.std(area[i]))
        area[i] = np.mean(area[i])
        dAreastd.append(np.std(dArea[i]))
        dArea[i] = np.mean(dArea[i])
    time = 2 * np.array(time)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].errorbar(time, area, std)
    ax[0].set(xlabel=r"Time (mins)", ylabel=r"$A$ $(\mu m^2$)")
    ax[0].title.set_text(r"$A$ during division")
    ax[0].set_ylim([-10, 55])

    ax[1].errorbar(time, dArea, dAreastd)
    ax[1].set(xlabel=r"Time (mins)", ylabel=r"$\delta A$ $(\mu m^2$)")
    ax[1].title.set_text(r"$\delta A$ during division")
    ax[1].set_ylim([-10, 55])

    fig.savefig(
        f"results/Area division {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


# shape of parent dividing cells
if True:
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfDivisionTrack = dfDivisionTrack[dfDivisionTrack["Type"] == "parent"]
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    T = np.max(dfShape["T"])
    time = list(
        np.linspace(
            np.min(dfDivisionTrack["Division Time"]),
            np.max(dfDivisionTrack["Division Time"]),
            int(
                np.max(dfDivisionTrack["Division Time"])
                - np.min(dfDivisionTrack["Division Time"])
                + 1
            ),
        )
    )
    sf = [[] for col in range(len(time))]
    dsf = [[] for col in range(len(time))]
    for filename in filenames:
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        dfFileShape = dfShape[dfShape["Filename"] == filename]
        sfT = []
        for t in range(T):
            sfT.append(np.mean(dfFileShape["Shape Factor"][dfFileShape["T"] == t]))

        for i in range(len(df)):

            t = df["Time"].iloc[i]
            if t < T:
                divTime = df["Division Time"].iloc[i]
                index = time.index(divTime)
                sf[index].append(df["Shape Factor"].iloc[i])
                dsf[index].append(df["Shape Factor"].iloc[i] - sfT[t])

    std = []
    dsfstd = []
    for i in range(len(sf)):
        std.append(np.std(sf[i]))
        sf[i] = np.mean(sf[i])
        dsfstd.append(np.std(dsf[i]))
        dsf[i] = np.mean(dsf[i])
    time = 2 * np.array(time)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].errorbar(time, sf, std)
    ax[0].set(xlabel=r"Time (mins)", ylabel=r"$S_f$")
    ax[0].title.set_text(r"$S_f$ during division")
    # ax[0].set_ylim([-5, 55])

    ax[1].errorbar(time, dsf, dsfstd)
    ax[1].set(xlabel=r"Time (mins)", ylabel=r"$\delta S_f$")
    ax[1].title.set_text(r"$\delta S_f$ during division")
    # ax[1].set_ylim([-5, 55])

    fig.savefig(
        f"results/Shape factor division {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


# orientation of parent dividing cells
if True:
    dfDivisionShape = pd.read_pickle(f"databases/dfDivisionShape{fileType}.pkl")
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfDivisionTrack = dfDivisionTrack[dfDivisionTrack["Type"] == "parent"]
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    T = np.max(dfShape["T"])
    time = list(
        np.linspace(
            np.min(dfDivisionTrack["Division Time"]),
            np.max(dfDivisionTrack["Division Time"]),
            int(
                np.max(dfDivisionTrack["Division Time"])
                - np.min(dfDivisionTrack["Division Time"])
                + 1
            ),
        )
    )
    diff = []
    sf = []
    for filename in filenames:
        dfFileShape = dfDivisionShape[dfDivisionShape["Filename"] == filename]
        dfFileShape = dfFileShape[dfFileShape["Track length"] > 18]
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        labels = list(dfFileShape["Label"])
        for label in labels:
            ori = dfFileShape["Orientation"][dfFileShape["Label"] == label].iloc[0]
            dfDiv = df[df["Label"] == label]
            oriPre = dfDiv["Orientation"][dfDiv["Division Time"] == -15].iloc[0]
            diff.append(angleDiff(ori, oriPre))
            sf.append(dfDiv["Shape Factor"][dfDiv["Division Time"] == -15].iloc[0])

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].hist(diff, 9)
    ax[0].set(xlabel=r"Difference in Orientaiton", ylabel=r"number")

    ax[1].scatter(diff, sf)
    ax[1].set(xlabel=r"Difference in Orientaiton", ylabel=r"$S_f$")
    # ax[1].title.set_text(r"$\delta S_f$ during division")

    fig.savefig(
        f"results/Orientation division {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


# orientation tcj of parent dividing cells
if True:
    dfDivisionShape = pd.read_pickle(f"databases/dfDivisionShape{fileType}.pkl")
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfDivisionTrack = dfDivisionTrack[dfDivisionTrack["Type"] == "parent"]
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    T = np.max(dfShape["T"])
    time = list(
        np.linspace(
            np.min(dfDivisionTrack["Division Time"]),
            np.max(dfDivisionTrack["Division Time"]),
            int(
                np.max(dfDivisionTrack["Division Time"])
                - np.min(dfDivisionTrack["Division Time"])
                + 1
            ),
        )
    )
    diff = []
    sf = []
    for filename in filenames:
        dfFileShape = dfDivisionShape[dfDivisionShape["Filename"] == filename]
        dfFileShape = dfFileShape[dfFileShape["Track length"] > 18]
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        labels = list(dfFileShape["Label"])
        for label in labels:
            ori = dfFileShape["Orientation"][dfFileShape["Label"] == label].iloc[0]
            dfDiv = df[df["Label"] == label]
            oriPre = dfDiv["Orientation tcj"][dfDiv["Division Time"] == -15].iloc[0]
            diff.append(angleDiff(ori, oriPre))
            sf.append(dfDiv["Shape Factor tcj"][dfDiv["Division Time"] == -15].iloc[0])

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].hist(diff, 9)
    ax[0].set(xlabel=r"Difference in Orientaiton tcj", ylabel=r"number")

    ax[1].scatter(diff, sf)
    ax[1].set(xlabel=r"Difference in Orientaiton tcj", ylabel=r"$S_f$ tcj")
    # ax[1].title.set_text(r"$\delta S_f$ during division")

    fig.savefig(
        f"results/Orientation tcj division {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


# Area of daughter cells
if True:
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfDivisionTrack = dfDivisionTrack[dfDivisionTrack["Type"] != "parent"]
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    T = np.max(dfShape["T"])
    time = list(
        np.linspace(
            np.min(dfDivisionTrack["Division Time"]),
            np.max(dfDivisionTrack["Division Time"]),
            int(
                np.max(dfDivisionTrack["Division Time"])
                - np.min(dfDivisionTrack["Division Time"])
                + 1
            ),
        )
    )
    area = [[] for col in range(len(time))]
    dArea = [[] for col in range(len(time))]
    for filename in filenames:
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        dfFileShape = dfShape[dfShape["Filename"] == filename]
        areaT = []
        for t in range(T):
            areaT.append(np.mean(dfFileShape["Area"][dfFileShape["T"] == t]))

        for i in range(len(df)):

            t = df["Time"].iloc[i]
            if t < T:
                divTime = df["Division Time"].iloc[i]
                index = time.index(divTime)
                area[index].append(df["Area"].iloc[i] * scale ** 2)
                dArea[index].append(df["Area"].iloc[i] * scale ** 2 - areaT[t])

    std = []
    dAreastd = []
    for i in range(len(area)):
        std.append(np.std(area[i]))
        area[i] = np.mean(area[i])
        dAreastd.append(np.std(dArea[i]))
        dArea[i] = np.mean(dArea[i])
    time = 2 * np.array(time)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].errorbar(time, area, std)
    ax[0].set(xlabel=r"Time (mins)", ylabel=r"$A$ $(\mu m^2$)")
    ax[0].title.set_text(r"$A$ after division")
    ax[0].set_ylim([-10, 55])

    ax[1].errorbar(time, dArea, dAreastd)
    ax[1].set(xlabel=r"Time (mins)", ylabel=r"$\delta A$ $(\mu m^2$)")
    ax[1].title.set_text(r"$\delta A$ after division")
    ax[1].set_ylim([-10, 55])

    fig.savefig(
        f"results/Area Daughter Cell {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


# shape of daughter cells
if True:
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfDivisionTrack = dfDivisionTrack[dfDivisionTrack["Type"] != "parent"]
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    T = np.max(dfShape["T"])
    time = list(
        np.linspace(
            np.min(dfDivisionTrack["Division Time"]),
            np.max(dfDivisionTrack["Division Time"]),
            int(
                np.max(dfDivisionTrack["Division Time"])
                - np.min(dfDivisionTrack["Division Time"])
                + 1
            ),
        )
    )
    sf = [[] for col in range(len(time))]
    dsf = [[] for col in range(len(time))]
    for filename in filenames:
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        dfFileShape = dfShape[dfShape["Filename"] == filename]
        sfT = []
        for t in range(T):
            sfT.append(np.mean(dfFileShape["Shape Factor"][dfFileShape["T"] == t]))

        for i in range(len(df)):

            t = df["Time"].iloc[i]
            if t < T:
                divTime = df["Division Time"].iloc[i]
                index = time.index(divTime)
                sf[index].append(df["Shape Factor"].iloc[i])
                dsf[index].append(df["Shape Factor"].iloc[i] - sfT[t])

    std = []
    dsfstd = []
    for i in range(len(sf)):
        std.append(np.std(sf[i]))
        sf[i] = np.mean(sf[i])
        dsfstd.append(np.std(dsf[i]))
        dsf[i] = np.mean(dsf[i])
    time = 2 * np.array(time)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].errorbar(time, sf, std)
    ax[0].set(xlabel=r"Time (mins)", ylabel=r"$S_f$")
    ax[0].title.set_text(r"$S_f$ during division")
    # ax[0].set_ylim([-5, 55])

    ax[1].errorbar(time, dsf, dsfstd)
    ax[1].set(xlabel=r"Time (mins)", ylabel=r"$\delta S_f$")
    ax[1].title.set_text(r"$\delta S_f$ during division")
    # ax[1].set_ylim([-5, 55])

    fig.savefig(
        f"results/Shape factor Daughter Cell {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


def maskQ(mask):
    S = np.zeros([2, 2])
    X, Y = mask.shape
    x = np.zeros([X, Y])
    y = np.zeros([X, Y])
    x += np.arange(X)
    y += (Y - 1 - np.arange(Y)).reshape(Y, 1)
    A = np.sum(mask)
    Cx = np.sum(x * mask) / A
    Cy = np.sum(y * mask) / A
    xx = (x - Cx) ** 2
    yy = (y - Cy) ** 2
    xy = (x - Cx) * (y - Cy)
    S[0, 0] = -np.sum(yy * mask) / A ** 2
    S[1, 0] = S[0, 1] = np.sum(xy * mask) / A ** 2
    S[1, 1] = -np.sum(xx * mask) / A ** 2
    TrS = S[0, 0] + S[1, 1]
    I = np.zeros(shape=(2, 2))
    I[0, 0] = 1
    I[1, 1] = 1
    q = S - TrS * I / 2

    return q


# orientation of daughter cells
if True:
    dfDivisionShape = pd.read_pickle(f"databases/dfDivisionShape{fileType}.pkl")
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    T = np.max(dfShape["T"])
    time = list(np.linspace(-10, 10, 21))
    dq = [[] for col in range(len(time))]
    for filename in filenames:
        tracks = sm.io.imread(f"dat/{filename}/binaryTracks{filename}.tif").astype(int)
        dfFileShape = dfDivisionShape[dfDivisionShape["Filename"] == filename]
        dfFileShape = dfFileShape[dfFileShape["Daughter length"] > 10]
        dfFileShape = dfFileShape[dfFileShape["Track length"] > 13]
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        labels = list(dfFileShape["Label"])
        for label in labels:
            dfDiv = df[df["Label"] == label]
            polygon = dfDiv["Polygon"][dfDiv["Division Time"] == time[0]].iloc[0]
            q0 = cell.qTensor(polygon)
            for t in time:
                if dfDiv["Type"][dfDiv["Division Time"] == t].iloc[0] == "parent":
                    polygon = dfDiv["Polygon"][dfDiv["Division Time"] == t].iloc[0]
                    q = cell.qTensor(polygon)
                    dq[time.index(t)].append(q - q0)
                else:
                    T = dfDiv["Time"][dfDiv["Division Time"] == t].iloc[0]
                    colour1 = dfDiv["Colour"][dfDiv["Division Time"] == t].iloc[0]
                    colour2 = dfDiv["Colour"][dfDiv["Division Time"] == t].iloc[1]
                    mask = np.zeros([512, 512])
                    mask[np.all((tracks[int(T)] - colour1) == 0, axis=2)] = 1
                    mask[np.all((tracks[int(T)] - colour2) == 0, axis=2)] = 1
                    q = maskQ(mask)
                    dq[time.index(t)].append(q - q0)

    dQ = []
    dQstd = []
    for i in range(len(dq)):
        dQ.append(np.mean(dq[i], axis=0))
        dQstd.append(np.std(dq[i], axis=0))

    dQ = np.array(dQ)
    dQstd = np.array(dQstd)
    time = 2 * np.array(time)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].errorbar(time, dQ[:, 0, 0], dQstd[:, 0, 0])
    ax[0].set(xlabel=r"Time (mins)", ylabel=r"$\delta Q^1$")
    ax[0].title.set_text(r"$S_f$ during division")
    ax[0].set_ylim([-0.07, 0.07])

    ax[1].errorbar(time, dQ[:, 1, 0], dQstd[:, 1, 0])
    ax[1].set(xlabel=r"Time (mins)", ylabel=r"$\delta Q^2$")
    ax[1].title.set_text(r"$\delta S_f$ during division")
    ax[1].set_ylim([-0.07, 0.07])

    fig.savefig(
        f"results/change in Q division {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")