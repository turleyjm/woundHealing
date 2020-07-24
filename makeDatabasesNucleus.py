import xml.etree.ElementTree as et

import os
import numpy as np
import pandas as pd
import skimage as sm
import scipy as sp
from shapely.geometry import Polygon
import tifffile

import cellProperties as cell


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


# -------------------

folder = "dat/datNucleusTracks"

cwd = os.getcwd()

files = os.listdir(cwd + f"/{folder}")

for track in files:

    filename = track
    dfNucleus = trackmate_vertices_import(folder + "/" + filename, get_tracks=True)

    filename = filename.replace("nuclei", "")
    filename = filename.replace(".xml", "")

    uniqueLabel = list(set(dfNucleus["label"]))

    _dfTracks = []
    for label in uniqueLabel:

        df = dfNucleus.loc[lambda dfNucleus: dfNucleus["label"] == label, :]

        x = []
        y = []
        z = []
        t = []

        for i in range(len(df)):
            x.append(df.iloc[i, 2])
            y.append(df.iloc[i, 3])
            z.append(df.iloc[i, 4])
            t.append(df.iloc[i, 1])

        X = []
        Y = []
        Z = []
        T = []

        X.append(x[0])
        Y.append(y[0])
        Z.append(z[0])
        T.append(t[0])

        for i in range(len(df) - 1):
            t0 = t[i]
            t1 = t[i + 1]

            if t1 - t0 > 1:
                X.append((x[i] + x[i + 1]) / 2)
                Y.append((y[i] + y[i + 1]) / 2)
                Z.append((z[i] + z[i + 1]) / 2)
                T.append((t[i] + t[i + 1]) / 2)

            X.append(x[i + 1])
            Y.append(y[i + 1])
            Z.append(z[i + 1])
            T.append(t[i + 1])

        _dfTracks.append({"Label": label, "x": X, "y": Y, "z": Z, "t": T})

    dfTracks = pd.DataFrame(_dfTracks)

    dfTracks.to_pickle(f"dat/databases/nucleusTracks{filename}.pkl")
