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


def costFunction(u0, u1):

    cost = np.linalg.norm(u0 + u1) / (np.linalg.norm(u0) + np.linalg.norm(u1))

    return cost


def splitGraph(graph, spot):

    keys = list(graph.keys())
    values = list(graph.values())
    val = []

    # finds starting spot
    for value in values:
        if len(value) == 1:
            val.append(value[0])
        else:
            val.append(value[0])
            val.append(value[1])

    keySpot = int(np.setdiff1d(keys, val))

    graph0 = {}
    graph1 = {}

    branch = []
    source = []
    notFinished = True

    # makes tree and avoids the break point
    while notFinished:

        if keySpot == spot:

            if branch == []:
                notFinished = False

            elif branch[0] == spot:
                falseSpot = branch[0]
                branch.remove(falseSpot)
                falseSpot = source[0]
                source.remove(falseSpot)

            else:
                keySpot = branch[0]
                branch.remove(keySpot)
                newStart = source[0]
                source.remove(newStart)
                out = graph0[newStart]
                if out[0] == spot:
                    graph0[newStart] = [keySpot]
                else:
                    graph0[newStart] = [keySpot, out[0]]

        else:

            try:
                output = graph[keySpot]
                n = len(output)

                if n > 1:
                    branch.append(output[1])
                    source.append(keySpot)

                graph0[keySpot] = [output[0]]
                keySpot = output[0]

            except KeyError:
                if branch == []:
                    notFinished = False
                elif branch[0] == spot:
                    falseSpot = branch[0]
                    branch.remove(falseSpot)
                    falseSpot = source[0]
                    source.remove(falseSpot)
                else:
                    keySpot = branch[0]
                    branch.remove(keySpot)
                    newStart = source[0]
                    source.remove(newStart)
                    out = graph0[newStart]
                    if out[0] == spot:
                        graph0[newStart] = [keySpot]
                    else:
                        graph0[newStart] = [keySpot, out[0]]

    keySpot = spot

    branch = []
    source = []
    notFinished = True

    # makes second tree starting from the brake point
    while notFinished:
        try:
            output = graph[keySpot]
            n = len(output)

            if n > 1:
                branch.append(output[1])
                source.append(keySpot)

            graph1[keySpot] = [output[0]]
            keySpot = output[0]

        except KeyError:
            if branch == []:
                notFinished = False
            else:
                keySpot = branch[0]
                branch.remove(keySpot)
                newStart = source[0]
                source.remove(newStart)
                out = graph1[newStart]
                graph1[newStart] = [keySpot, out[0]]

    return [graph0, graph1]


def importDividingTracks(trackmate_xml_path):

    root = et.fromstring(open(trackmate_xml_path).read())

    # get spots

    features = ["name", "POSITION_X", "POSITION_Y", "POSITION_T"]

    spots = root.find("Model").find("AllSpots")
    objects = []
    for frame in spots.findall("SpotsInFrame"):
        for spot in frame.findall("Spot"):
            single_object = []
            name = list(spot.get("name"))[2:]
            n = ""
            for i in name:
                n += i

            single_object.append(int(n))
            single_object.append(spot.get("POSITION_X"))
            single_object.append(spot.get("POSITION_Y"))
            single_object.append(spot.get("POSITION_T"))

            objects.append(single_object)

    spotDat = pd.DataFrame(objects, columns=features)

    # get tracks

    features = root.find("Model").find("FeatureDeclarations").find("EdgeFeatures")
    features = [c.get("feature") for c in features.getchildren()] + ["ID"]
    features = features[0:2]

    tracks = root.find("Model").find("AllTracks")
    objects = []
    _trackDat = []
    nextLabel = 0
    for track in tracks:
        label = int(track.get("TRACK_ID"))
        divisions = int(track.get("NUMBER_SPLITS"))
        objects = []

        if divisions >= 1:
            for edge in track:
                single_object = []
                for feat in features:
                    single_object.append(int(edge.get(feat)))
                objects.append(single_object)

            start = []
            end = []
            for obj in objects:
                start.append(obj[0])
                end.append(obj[1])

            if divisions == 1:

                uniqueLabels = set(list(start))
                count = Counter(start)
                c = []
                for l in uniqueLabels:
                    c.append(count[l])

                uniqueLabels = list(uniqueLabels)
                parentLabel = uniqueLabels[c.index(max(c))]

                daughter = []

                for i in range(len(start)):
                    if start[i] == parentLabel:
                        daughter.append(end[i])

                parent = spotDat[spotDat["name"] == parentLabel]
                daughter0 = spotDat[spotDat["name"] == daughter[0]]
                daughter1 = spotDat[spotDat["name"] == daughter[1]]

                u0 = np.array(
                    [
                        float(daughter0["POSITION_X"]) - float(parent["POSITION_X"]),
                        float(daughter0["POSITION_Y"]) - float(parent["POSITION_Y"]),
                    ]
                )

                u1 = np.array(
                    [
                        float(daughter1["POSITION_X"]) - float(parent["POSITION_X"]),
                        float(daughter1["POSITION_Y"]) - float(parent["POSITION_Y"]),
                    ]
                )

                cost = costFunction(u0, u1)
                print(cost)
                print(label)

                if cost < 0.5:
                    spots = start
                    for spot in end:
                        spots.append(spot)

                    uniqueSpots = set(spots)
                    for spot in uniqueSpots:
                        df = spotDat[spotDat["name"] == spot]

                        time = int(float(df["POSITION_T"]))
                        centroid = [float(df["POSITION_X"]), float(df["POSITION_Y"])]

                        _trackDat.append(
                            {
                                "Label": label,
                                "Time": time,
                                "Position": centroid,
                                "Sub Label": label,
                            }
                        )

            else:
                graph = {}
                start = []
                end = []
                for obj in objects:
                    start.append(obj[0])
                    end.append(obj[1])
                uniqueLabels = set(list(start))

                for spot in uniqueLabels:
                    link = []
                    for i in range(len(start)):
                        if start[i] == spot:
                            link.append(end[i])

                    graph[spot] = link

                keys = list(graph.keys())

                falseLink = []
                for key in keys:
                    if len(graph[key]) == 2:
                        daughter0 = graph[key][0]
                        daughter1 = graph[key][1]
                        df0 = spotDat[spotDat["name"] == daughter0]
                        df1 = spotDat[spotDat["name"] == daughter1]
                        dfp = spotDat[spotDat["name"] == key]

                        u0 = np.array(
                            [
                                float(df0["POSITION_X"]) - float(dfp["POSITION_X"]),
                                float(df0["POSITION_Y"]) - float(dfp["POSITION_Y"]),
                            ]
                        )

                        u1 = np.array(
                            [
                                float(df1["POSITION_X"]) - float(dfp["POSITION_X"]),
                                float(df1["POSITION_Y"]) - float(dfp["POSITION_Y"]),
                            ]
                        )

                        cost = costFunction(u0, u1)
                        print(cost)
                        print(label)

                        if cost > 0.5:

                            maxDis = max(np.linalg.norm(u0), np.linalg.norm(u1))
                            if np.linalg.norm(u0) == maxDis:
                                falseLink.append(daughter0)
                            else:
                                falseLink.append(daughter1)

                dictGraph = {}
                dictGraph["graph0"] = graph
                a = 1
                for falseSpot in falseLink:
                    keys = list(dictGraph.keys())
                    for key in keys:
                        graph = dictGraph[key]
                        values = list(graph.values())
                        val = []
                        for value in values:
                            if len(value) == 1:
                                val.append(value[0])
                            else:
                                val.append(value[0])
                                val.append(value[1])
                        values = val

                        if falseSpot in values:
                            [graph, graph1] = splitGraph(graph, falseSpot)
                            dictGraph[key] = graph
                            dictGraph[f"graph{a}"] = graph1
                            a += 1

                keys = list(dictGraph.keys())
                for key in keys:
                    graph = dictGraph[key]
                    graphKeys = list(graph.keys())

                    divisionNum = 0
                    for graphKey in graphKeys:
                        if len(graph[graphKey]) == 2:
                            divisionNum += 0

                    if divisionNum > 1:
                        print("Fuck")

                keys = list(dictGraph.keys())
                for key in keys:
                    graph = dictGraph[key]

                    graphKeys = list(graph.keys())
                    values = list(graph.values())
                    val = []
                    divisions = False
                    for value in values:
                        if len(value) == 1:
                            val.append(value[0])
                        else:
                            val.append(value[0])
                            val.append(value[1])
                            divisions = True
                    values = val
                    spots = values + graphKeys
                    uniqueSpots = set(graphKeys)

                    if divisions:
                        for spot in uniqueSpots:
                            df = spotDat[spotDat["name"] == spot]

                            time = int(float(df["POSITION_T"]))
                            centroid = [
                                float(df["POSITION_X"]),
                                float(df["POSITION_Y"]),
                            ]

                            _trackDat.append(
                                {
                                    "Label": label,
                                    "Time": time,
                                    "Position": centroid,
                                    "Sub Label": nextLabel,
                                }
                            )

                        nextLabel += 1

        else:
            continue

    trackDat = pd.DataFrame(_trackDat)

    trackDat = trackDat.sort_values(["Label", "Time"], ascending=[True, True])

    return trackDat


f = open("pythonText.txt", "r")

filenames = f.read()
filenames = filenames.split(", ")

for filename in filenames:

    if "Unwound" in filename:
        wound = False
    else:
        wound = True

    # gather xml files

    df = importDividingTracks(f"dat/{filename}/mitosisHeightTracks{filename}.xml")

    print("done")

