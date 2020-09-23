import os
from math import floor, log10
import xml.etree.ElementTree as et

import cv2
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from collections import Counter

import cellProperties as cell
import findGoodCells as fi

plt.rcParams.update({"font.size": 20})
plt.ioff()
pd.set_option("display.width", 1000)


def trackmate_vertices_import(trackmate_xml_path, get_tracks=False):
    """Import detected tracks with TrackMate Fiji plugin.

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
        "ESTIMATED_DIAMETER": "w",
        "QUALITY": "q",
        "ID": "spot_id",
        "MEAN_INTENSITY": "mean_intensity",
        "MEDIAN_INTENSITY": "median_intensity",
        "MIN_INTENSITY": "min_intensity",
        "MAX_INTENSITY": "max_intensity",
        "TOTAL_INTENSITY": "total_intensity",
        "STANDARD_DEVIATION": "std_intensity",
        "CONTRAST": "contrast",
        "SNR": "snr",
    }

    features = root.find("Model").find("FeatureDeclarations").find("SpotFeatures")
    features = [c.get("feature") for c in features.getchildren()] + ["ID"]

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
    trajs = trajs.astype(np.float)

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


def filter_spots(spots, name, value, isabove):
    if isabove:
        spots = spots[spots[name] > value]
    else:
        spots = spots[spots[name] < value]

    return spots


def trackmate_edges_import(trackmate_xml_path):

    root = et.fromstring(open(trackmate_xml_path).read())

    features = root.find("Model").find("FeatureDeclarations").find("EdgeFeatures")
    features = [c.get("feature") for c in features.getchildren()] + ["ID"]

    tracks = root.find("Model").find("AllTracks")
    trajs = pd.DataFrame([])
    objects = []
    for track in tracks:
        for edge in track:
            single_object = []
            for label in features:
                single_object.append(edge.get(label))
            objects.append(single_object)

    trajs = pd.DataFrame(objects, columns=features)
    trajs = trajs.astype(np.float)

    return trajs


f = open("pythonText.txt", "r")

filenames = f.read()
filenames = filenames.split(", ")

for filename in filenames:

    if "Unwound" in filename:
        wound = False
    else:
        wound = True

    # gather xml files

    dfVertice = trackmate_vertices_import(
        f"dat/{filename}/mitosisTracks{filename}.xml", get_tracks=True
    )
    dfEdge = trackmate_edges_import(f"dat/{filename}/mitosisTracks{filename}.xml")

    uniqueLabel = list(set(dfVertice["label"]))
