from ast import Break
import os
from os.path import exists
from re import A
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


def angleDiff(theta, phi):

    diff = theta - phi

    if abs(diff) > 90:
        if diff > 0:
            diff = 180 - diff
        else:
            diff = 180 + diff

    return abs(diff)


def filter_spots(spots, name, value, isabove):
    if isabove:
        spots = spots[spots[name] > value]
    else:
        spots = spots[spots[name] < value]

    return spots


def trackmate_vertices_import(trackmate_xml_path, get_tracks=False):
    """Import detected tracks with TrackMate Fiji plugin.

    ref: https://github.com/hadim/pytrackmate

    Parameters
    ----------
    trackmate_xml_path : str
        TrackMate XML file path.
    get_tracks : boolean
        Add tracks to label
    """

    root = et.fromstring(open(trackmate_xml_path).read())

    objects = []
    object_labels = {
        "FRAME": "t_stamp",
        "POSITION_T": "t",
        "POSITION_X": "x",
        "POSITION_Y": "y",
        "POSITION_Z": "z",
        "QUALITY": "q",
        "ID": "spot_id",
    }

    # features = root.find("Model").find("FeatureDeclarations").find("SpotFeatures")
    features = [
        "FRAME",
        "POSITION_T",
        "POSITION_X",
        "POSITION_Y",
        "POSITION_Z",
        "QUALITY",
        "ID",
    ]

    spots = root.find("Model").find("AllSpots")
    trajs = pd.DataFrame([])
    objects = []
    for frame in spots.findall("SpotsInFrame"):
        for spot in frame.findall("Spot"):
            single_object = []
            for label in features:
                single_object.append(spot.get(label))
            objects.append(single_object)

    trajs = pd.DataFrame(objects, columns=features)
    trajs = trajs.astype(float)

    # Apply initial filtering
    initial_filter = root.find("Settings").find("InitialSpotFilter")

    trajs = filter_spots(
        trajs,
        name=initial_filter.get("feature"),
        value=float(initial_filter.get("value")),
        isabove=True if initial_filter.get("isabove") == "true" else False,
    )

    # Apply filters
    spot_filters = root.find("Settings").find("SpotFilterCollection")

    for spot_filter in spot_filters.findall("Filter"):

        trajs = filter_spots(
            trajs,
            name=spot_filter.get("feature"),
            value=float(spot_filter.get("value")),
            isabove=True if spot_filter.get("isabove") == "true" else False,
        )

    trajs = trajs.loc[:, object_labels.keys()]
    trajs.columns = [object_labels[k] for k in object_labels.keys()]
    trajs["label"] = np.arange(trajs.shape[0])

    # Get tracks
    if get_tracks:
        filtered_track_ids = [
            int(track.get("TRACK_ID"))
            for track in root.find("Model").find("FilteredTracks").findall("TrackID")
        ]

        label_id = 0
        trajs["label"] = np.nan

        tracks = root.find("Model").find("AllTracks")
        for track in tracks.findall("Track"):

            track_id = int(track.get("TRACK_ID"))
            if track_id in filtered_track_ids:

                spot_ids = [
                    (
                        edge.get("SPOT_SOURCE_ID"),
                        edge.get("SPOT_TARGET_ID"),
                        edge.get("EDGE_TIME"),
                    )
                    for edge in track.findall("Edge")
                ]
                spot_ids = np.array(spot_ids).astype("float")[:, :2]
                spot_ids = set(spot_ids.flatten())

                trajs.loc[trajs["spot_id"].isin(spot_ids), "label"] = label_id
                label_id += 1

        # Label remaining columns
        single_track = trajs.loc[trajs["label"].isnull()]
        trajs.loc[trajs["label"].isnull(), "label"] = label_id + np.arange(
            0, len(single_track)
        )

    return trajs


# -------------------


filenames, fileType = util.getFilesType()
scale = 123.26 / 512

# display pre-division tracks Training dataset
if False:
    filename = "Unwound18h13"
    dfDivision = pd.read_pickle(f"dat/{filename}/dfDivision{filename}.pkl")

    util.createFolder("dat/Unwound18h13/image/")
    df = dfDivision[dfDivision["T"] > 10]
    df = df[df["X"] > 20]
    df = df[df["Y"] > 20]
    df = df[df["X"] < 492]
    df = df[df["Y"] < 492]
    df = df.sample(frac=1, random_state=1)

    h2Stack = sm.io.imread(f"dat/{filename}/{filename}.tif").astype(int)[:, :, 1]
    h2 = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)[:, :, :, 0]
    T, Z, X, Y = h2Stack.shape
    n = len(df)
    vidStackAll = np.zeros([11 * n, Z, 60, 60])
    vidAll = np.zeros([11 * n, 60, 60])

    for i in range(n):
        x = df["X"].iloc[i]
        y = 512 - df["Y"].iloc[i]
        t = int(df["T"].iloc[i])

        xMax = int(x + 30)
        xMin = int(x - 30)
        yMax = int(y + 30)
        yMin = int(y - 30)
        if xMax > 512:
            xMaxCrop = 60 - (xMax - 512)
            xMax = 512
        else:
            xMaxCrop = 60
        if xMin < 0:
            xMinCrop = -xMin
            xMin = 0
        else:
            xMinCrop = 0
        if yMax > 512:
            yMaxCrop = 60 - (yMax - 512)
            yMax = 512
        else:
            yMaxCrop = 60
        if yMin < 0:
            yMinCrop = -yMin
            yMin = 0
        else:
            yMinCrop = 0

        vidStack = np.zeros([10, Z, 60, 60])
        vid = np.zeros([10, 60, 60])
        for j in range(10):

            vid[j, yMinCrop:yMaxCrop, xMinCrop:xMaxCrop] = h2[
                t - 9 + j, yMin:yMax, xMin:xMax
            ]
            vidAll[11 * i + j, yMinCrop:yMaxCrop, xMinCrop:xMaxCrop] = h2[
                t - 9 + j, yMin:yMax, xMin:xMax
            ]
            vidStack[j, :, yMinCrop:yMaxCrop, xMinCrop:xMaxCrop] = h2Stack[
                t - 9 + j, :, yMin:yMax, xMin:xMax
            ]
            vidStackAll[11 * i + j, :, yMinCrop:yMaxCrop, xMinCrop:xMaxCrop] = h2Stack[
                t - 9 + j, :, yMin:yMax, xMin:xMax
            ]

        # vid = np.asarray(vid, "uint8")
        # tifffile.imwrite(
        #     f"dat/Unwound18h13/image/vidH2_{i}.tif",
        #     vid,
        #     imagej=True,
        #     metadata={"axes": "TYX"},
        # )
        # vidStack = np.asarray(vidStack, "uint8")
        # tifffile.imwrite(
        #     f"dat/Unwound18h13/image/vidStackH2_{i}.tif",
        #     vidStack,
        #     imagej=True,
        #     metadata={"axes": "TZYX"},
        # )

    vidAll = np.asarray(vidAll, "uint8")
    tifffile.imwrite(
        f"dat/Unwound18h13/image/vidH2_training.tif",
        vidAll,
        imagej=True,
        metadata={"axes": "TYX"},
    )

    vidStackAll = np.asarray(vidStackAll, "uint8")
    tifffile.imwrite(
        f"dat/Unwound18h13/image/vidStackH2_training.tif",
        vidStackAll,
        imagej=True,
        metadata={"axes": "TZYX"},
    )


# display pre-division tracks
if False:
    filename = "Unwound18h13"
    dfDivision = pd.read_pickle(f"dat/{filename}/dfDivision{filename}.pkl")

    util.createFolder("dat/Unwound18h13/image/")
    df = dfDivision[dfDivision["T"] > 10]
    df = df[df["X"] > 20]
    df = df[df["Y"] > 20]
    df = df[df["X"] < 492]
    df = df[df["Y"] < 492]

    h2Stack = sm.io.imread(f"dat/{filename}/{filename}.tif").astype(int)[:, :, 1]
    h2 = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)[:, :, :, 0]
    T, Z, X, Y = h2Stack.shape
    n = len(df)
    vidStackAll = np.zeros([11 * n, Z, 60, 60])
    vidAll = np.zeros([11 * n, 60, 60])

    for i in range(n):
        x = df["X"].iloc[i]
        y = 512 - df["Y"].iloc[i]
        t = int(df["T"].iloc[i])

        xMax = int(x + 30)
        xMin = int(x - 30)
        yMax = int(y + 30)
        yMin = int(y - 30)
        if xMax > 512:
            xMaxCrop = 60 - (xMax - 512)
            xMax = 512
        else:
            xMaxCrop = 60
        if xMin < 0:
            xMinCrop = -xMin
            xMin = 0
        else:
            xMinCrop = 0
        if yMax > 512:
            yMaxCrop = 60 - (yMax - 512)
            yMax = 512
        else:
            yMaxCrop = 60
        if yMin < 0:
            yMinCrop = -yMin
            yMin = 0
        else:
            yMinCrop = 0

        vidStack = np.zeros([10, Z, 60, 60])
        vid = np.zeros([10, 60, 60])
        for j in range(10):

            vid[j, yMinCrop:yMaxCrop, xMinCrop:xMaxCrop] = h2[
                t - 9 + j, yMin:yMax, xMin:xMax
            ]
            vidAll[11 * i + j, yMinCrop:yMaxCrop, xMinCrop:xMaxCrop] = h2[
                t - 9 + j, yMin:yMax, xMin:xMax
            ]
            vidStack[j, :, yMinCrop:yMaxCrop, xMinCrop:xMaxCrop] = h2Stack[
                t - 9 + j, :, yMin:yMax, xMin:xMax
            ]
            vidStackAll[11 * i + j, :, yMinCrop:yMaxCrop, xMinCrop:xMaxCrop] = h2Stack[
                t - 9 + j, :, yMin:yMax, xMin:xMax
            ]

        # vid = np.asarray(vid, "uint8")
        # tifffile.imwrite(
        #     f"dat/Unwound18h13/image/vidH2_{i}.tif",
        #     vid,
        #     imagej=True,
        #     metadata={"axes": "TYX"},
        # )
        # vidStack = np.asarray(vidStack, "uint8")
        # tifffile.imwrite(
        #     f"dat/Unwound18h13/image/vidStackH2_{i}.tif",
        #     vidStack,
        #     imagej=True,
        #     metadata={"axes": "TZYX"},
        # )

    vidAll = np.asarray(vidAll, "uint8")
    tifffile.imwrite(
        f"dat/Unwound18h13/image/vidH2.tif",
        vidAll,
        imagej=True,
        metadata={"axes": "TYX"},
    )

    vidStackAll = np.asarray(vidStackAll, "uint8")
    tifffile.imwrite(
        f"dat/Unwound18h13/image/vidStackH2.tif",
        vidStackAll,
        imagej=True,
        metadata={"axes": "TZYX"},
    )

if True:
    filename = "Unwound18h13"

    vidBinary = sm.io.imread(f"dat/{filename}/probDivNuclei{filename}.tif").astype(
        float
    )
    vidBinary[vidBinary < 0.3] = 0
    vidBinary[vidBinary >= 0.3] = 255
    vidBinary = np.asarray(vidBinary, "uint8")
    tifffile.imwrite(
        f"dat/Unwound18h13/image/vidBinary.tif",
        vidBinary,
        imagej=True,
        metadata={"axes": "TYX"},
    )

    dfNuclei = trackmate_vertices_import(
        f"dat/{filename}/divisionNucleiTracks{filename}.xml", get_tracks=True
    )
    dfDivision = pd.read_pickle(f"dat/{filename}/dfDivision{filename}.pkl")
    df = dfDivision[dfDivision["T"] > 10]
    df = df[df["X"] > 20]
    df = df[df["Y"] > 20]
    df = df[df["X"] < 492]
    df = df[df["Y"] < 492]
    n = len(df)

    _df = []

    for i in range(n):
        dft = dfNuclei[dfNuclei["t"] == i * 11 + 9]
        dft["R"] = 0
        dft["R"] = ((30 - dft["x"]) ** 2 + (30 - dft["y"]) ** 2) ** 0.5

        dfR = dft[dft["R"] < 12]
        label = dfR["label"][dfR["z"] == np.min(dfR["z"])].iloc[0]

        df2 = dfNuclei[dfNuclei["label"] == label]

        if len(df2) == 10:
            for j in range(10):
                img = vidBinary[i * 11 + j]
                img = sp.ndimage.binary_fill_holes(img).astype(int) * 255
                # img = np.asarray(img, "uint8")
                # tifffile.imwrite(
                #     f"dat/Unwound18h13/image/img.tif",
                #     img,
                # )
                imgLabel = sm.measure.label(img, background=0, connectivity=1)
                imgLabel = np.asarray(imgLabel, "uint8")
                # tifffile.imwrite(
                #     f"dat/Unwound18h13/image/imgLabel.tif",
                #     imgLabel,
                # )
                x, y = df2["x"].iloc[j], df2["y"].iloc[j]
                shapeLabel = imgLabel[int(y), int(x)]
                if shapeLabel != 0:
                    # convert to row-col
                    imgLabel = util.imgrcxy(imgLabel)
                    contour = sm.measure.find_contours(imgLabel == shapeLabel, level=0)[
                        0
                    ]
                    poly = sm.measure.approximate_polygon(contour, tolerance=1)
                    try:
                        polygon = Polygon(poly)
                        if j == 9:
                            a = 0
                        _df.append(
                            {
                                "Filename": filename,
                                "Div Label": df["Label"].iloc[i],
                                "Div Orientation": df["Orientation"].iloc[i] % 180,
                                "Track Label": label,
                                "X": x,
                                "Y": y,
                                "Z": df2["z"].iloc[j],
                                "T": df2["t"].iloc[j],
                                "Polygon": polygon,
                                "Area": cell.area(polygon) * scale ** 2,
                                "Shape Orientation": (
                                    cell.orientation(polygon) * 180 / np.pi - 90
                                )
                                % 180,
                                "Shape Factor": cell.shapeFactor(polygon),
                                "q": cell.qTensor(polygon),
                                "Time Before Division": j - 10,
                            }
                        )
                    except:
                        print(i * 11 + j)
                        continue

    dfDivNucleus = pd.DataFrame(_df)

    dfDivNucleus.to_pickle(f"dat/{filename}/dfDivNucleus{filename}.pkl")

if True:
    filename = "Unwound18h13"
    dfDivNucleus = pd.read_pickle(f"dat/{filename}/dfDivNucleus{filename}.pkl")

    Sf = []
    Sf_std = []
    area = []
    area_std = []
    time = []
    for t in range(10):
        Sf.append(
            np.mean(
                dfDivNucleus["Shape Factor"][
                    dfDivNucleus["Time Before Division"] == -10 + t
                ]
            )
        )
        Sf_std.append(
            np.std(
                dfDivNucleus["Shape Factor"][
                    dfDivNucleus["Time Before Division"] == -10 + t
                ]
            )
        )

        area.append(
            np.mean(
                dfDivNucleus["Area"][dfDivNucleus["Time Before Division"] == -10 + t]
            )
        )
        area_std.append(
            np.std(
                dfDivNucleus["Area"][dfDivNucleus["Time Before Division"] == -10 + t]
            )
        )
        time.append((-10 + t) * 2)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].errorbar(time, Sf, Sf_std)
    ax[0].set(xlabel=r"Time before anaphase (mins)", ylabel=r"$S_f$")
    ax[0].title.set_text(r"$S_f$ before anaphase")
    ax[0].set_ylim([0, 1])
    ax[0].set_xlim([-21, 0])

    ax[1].errorbar(time, area, area_std)
    ax[1].set(xlabel=r"Time before anaphase (mins)", ylabel=r"$A_n$")
    ax[1].title.set_text(r"nucleus area before anaphase")
    ax[1].set_ylim([0, 30])
    ax[1].set_xlim([-21, 0])

    fig.savefig(
        f"results/Nuclei shape factor and area before anaphase {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


if True:
    filename = "Unwound18h13"
    dfDivNucleus = pd.read_pickle(f"dat/{filename}/dfDivNucleus{filename}.pkl")

    diff = []
    diff_std = []
    time = []
    for t in range(10):
        df = dfDivNucleus[dfDivNucleus["Time Before Division"] == -10 + t]
        dtheta = []
        for i in range(len(df)):
            dtheta.append(
                angleDiff(
                    df["Shape Orientation"].iloc[i], df["Div Orientation"].iloc[i]
                )
                * np.sign(
                    df["Shape Orientation"].iloc[i] - df["Div Orientation"].iloc[i]
                )
            )
            # if (
            #     angleDiff(
            #         df["Shape Orientation"].iloc[i], df["Div Orientation"].iloc[i]
            #     )
            #     > 40
            # ):
            #     print(0)

        diff.append(np.mean(dtheta))
        diff_std.append(np.std(dtheta))
        time.append((-10 + t) * 2)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.errorbar(time, diff, diff_std)
    ax.set(xlabel=r"Time before anaphase (mins)", ylabel=r"$|\theta_d-\theta_n|$")
    ax.title.set_text(r"Nuclei alinement with division orientation")
    ax.set_ylim([-65, 65])
    ax.set_xlim([-21, 0])

    fig.savefig(
        f"results/Nuclei alining with division orientation {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")
