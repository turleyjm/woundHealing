import os
from os.path import exists
import shutil
from math import floor, log10, factorial

from collections import Counter
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


# -------------------

filenames, fileType = util.getFilesType()
scale = 123.26 / 512


if False:
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    _df = []
    for filename in filenames:

        df = dfDivisions[dfDivisions["Filename"] == filename]
        dfFileShape = dfShape[dfShape["Filename"] == filename]
        meanArea = np.mean(dfShape["Area"])
        stdArea = np.std(dfShape["Area"])

        tracks = sm.io.imread(f"dat/{filename}/binaryTracks{filename}.tif").astype(int)
        T, X, Y, C = tracks.shape

        tracks = util.vidrcxyRGB(tracks)
        tracksDivisions = tracks

        for i in range(len(df)):

            label = df["Label"].iloc[i]
            ori = df["Orientation"].iloc[i]
            # if ori > 90:
            #     ori = 180 - ori
            tm = t = int(df["T"].iloc[i] / 2)
            x = df["X"].iloc[i] / scale
            y = df["Y"].iloc[i] / scale

            colour = tracks[t, int(x), int(y)]
            if np.all((colour - np.array([255, 255, 255])) == 0):
                continue

            track = tracks[t][np.all((tracks[t] - colour) == 0, axis=2)]

            finished = False
            t_i = t
            while finished == False:
                t_i += 1
                A0 = len(track)
                track = tracks[t_i][np.all((tracks[t_i - 1] - colour) == 0, axis=2)]
                A1 = len(track[np.all((track - colour) == 0, axis=1)])
                if A1 / A0 < 0.65:
                    finished = True
                    # tracksDivisions[t_i - 1][
                    #     np.all((tracks[t_i - 1] - colour) == 0, axis=2)
                    # ] = [0, 0, 0]

                if t_i == T - 1:
                    finished = True
                    # tracksDivisions[t_i - 1][
                    #     np.all((tracks[t_i - 1] - colour) == 0, axis=2)
                    # ] = [0, 0, 0]

            tc = t_i - 1

            if tc > 30:
                time = range(tc - 29, tc + 1)
            else:
                time = range(1, tc + 1)

            polyList = []
            for t in time:
                try:
                    contour = sm.measure.find_contours(
                        np.all((tracks[t] - colour) == 0, axis=2), level=0
                    )[0]
                    poly = sm.measure.approximate_polygon(contour, tolerance=1)
                    polygon = Polygon(poly)
                    contour = sm.measure.find_contours(
                        np.all((tracks[t - 1] - colour) == 0, axis=2), level=0
                    )[0]
                    poly = sm.measure.approximate_polygon(contour, tolerance=1)
                    polygon0 = Polygon(poly)

                    if polygon.area == 0:
                        polyList.append(False)
                    elif dist(polygon, polygon0) > 10:
                        # [t, cell.centroid(polygon)[0], 512 - cell.centroid(polygon)[1], filename, colour]
                        polyList.append(False)
                    else:
                        polyList.append(polygon)

                except:
                    continue

                # tracksDivisions[t][np.all((tracks[t] - colour) == 0, axis=2)] = [
                #     0,
                #     0,
                #     0,
                # ]

            if tc - tm < 7:
                _df.append(
                    {
                        "Filename": filename,
                        "Label": label,
                        "Orientation": ori,
                        "Times": np.array(time),
                        "Cytokineses Time": tc,
                        "Anaphase Time": tm,
                        "X": x,
                        "Y": y,
                        "Colour": colour,
                        "Polygons": polyList,
                        "Track Length": len(polyList),
                    }
                )

        # tracksDivisions = fi.imgxyrcRGB(tracksDivisions)
        # tracksDivisions = np.asarray(tracksDivisions, "uint8")
        # tifffile.imwrite(
        #     f"dat/{filename}/tracksDivisions{filename}.tif", tracksDivisions
        # )

    dfDivisionShape = pd.DataFrame(_df)
    dfDivisionShape.to_pickle(f"databases/dfDivisionShape{fileType}.pkl")


# display predivision tracks
if False:
    dfDivisionShape = pd.read_pickle(f"databases/dfDivisionShape{fileType}.pkl")
    for filename in filenames:
        focus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)
        tracks = sm.io.imread(f"dat/{filename}/binaryTracks{filename}.tif").astype(int)
        df = dfDivisionShape[dfDivisionShape["Filename"] == filename]

        (T, X, Y, rgb) = focus.shape

        for i in range(len(df)):

            colour = df["Colour"].iloc[i]
            time = df["Times"].iloc[i]
            for t in time:
                focus[t, :, :, 2][np.all((tracks[t] - colour) == 0, axis=2)] = 255

        focus = np.asarray(focus, "uint8")
        tifffile.imwrite(f"results/divisionsTracksDisplay{filename}.tif", focus)


if True:
    dfDivisionShape = pd.read_pickle(f"databases/dfDivisionShape{fileType}.pkl")
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    T = np.max(dfShape["T"])
    _dfArea = []
    for filename in filenames:
        df = dfDivisionShape[dfDivisionShape["Filename"] == filename]
        df = df[df["Track Length"] > 15]
        dfFileShape = dfShape[dfShape["Filename"] == filename]
        areaT = []
        for t in range(T):
            areaT.append(np.mean(dfFileShape["Area"][dfFileShape["T"] == t]))

        for i in range(len(df)):

            time = df["Times"].iloc[i]
            label = df["Label"].iloc[i]
            tm = df["Anaphase Time"].iloc[i]
            polyList = df["Polygons"].iloc[i]

            if len(time) == len(polyList):
                for j in range(len(polyList)):
                    if polyList[j] != False:
                        if time[j] < T:
                            _dfArea.append(
                                {
                                    "Filename": filename,
                                    "Label": label,
                                    "Delta Area": polyList[j].area - areaT[time[j]],
                                    "T": time[j] - tm,
                                }
                            )

    dfArea = pd.DataFrame(_dfArea)
    labels = dfArea["Label"][dfArea["T"] == np.min(dfArea["T"])]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].hist(2 * np.array(range(T)), rho)

    fig.savefig(
        f"results/Area division {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")