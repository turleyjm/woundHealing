import xml.etree.ElementTree as et

import numpy as np
import pandas as pd
import skimage as sm
import scipy as sp
from shapely.geometry import Polygon
import tifffile

import cell_properties as cell

trackmate_xml_path = "dat_nucleus/mitosisHighlight.xml"

filename = "wound16h01"

vid = sm.io.imread("dat_nucleus/mitosisH2.tif").astype(float)

(T, X, Y) = vid.shape
vid_binary = []
vid_label = []

for t in range(T):

    img = vid[t]
    img = sp.signal.medfilt(img, 5)
    img[img < 0.15] = 0
    img[img >= 0.15] = 255
    img = sp.signal.medfilt(img, 3)
    img_label = sm.measure.label(img, background=0, connectivity=1)

    vid_binary.append(img)
    vid_label.append(img_label)

vid_binary = np.asarray(vid_binary, "uint8")
vid_label = np.asarray(vid_label, "uint8")
tifffile.imwrite(f"dat_nucleus/vid_binary_{filename}.tif", vid_binary)


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


# ----------------------------------------------

df_vertice = trackmate_vertices_import(trackmate_xml_path, get_tracks=True)
df_edge = trackmate_edges_import(trackmate_xml_path)

unique_label = list(set(df_vertice["label"]))

_df_divisions = []f
for label in unique_label:

    df = df_vertice.loc[lambda df_vertice: df_vertice["label"] == label, :]

    t = list(df.iloc[:, 0])
    unique = list(set(t))

    if len(unique) < len(t):
        division = True
        spotID = df.iloc[:, 7]
    else:
        division = False

    if division == True:
        _df2 = []
        for spot in spotID:
            _df2.append(
                df_edge.loc[lambda df_edge: df_edge["SPOT_SOURCE_ID"] == spot, :]
            )

        df2 = pd.concat(_df2)

        for spot in spotID:

            df3 = df2.loc[lambda df2: df2["SPOT_SOURCE_ID"] == spot, :]
            n = len(df3)
            if n == 2:
                divID = spot
                daughter0 = list(df3["SPOT_TARGET_ID"])[0]
                daughter1 = list(df3["SPOT_TARGET_ID"])[1]

        divID = list([0, divID])
        con = True
        while con == True:

            try:
                con = False
                divID.append(
                    list(
                        df2.loc[lambda df2: df2["SPOT_TARGET_ID"] == divID[-1], :][
                            "SPOT_SOURCE_ID"
                        ]
                    )[0]
                )
                con = True
            except:
                pass

        parent = divID[1:]
        parent.reverse()

        daughter0 = list([0, daughter0])
        con = True
        while con == True:

            try:
                con = False
                daughter0.append(
                    list(
                        df2.loc[lambda df2: df2["SPOT_SOURCE_ID"] == daughter0[-1], :][
                            "SPOT_TARGET_ID"
                        ]
                    )[0]
                )
                con = True
            except:
                pass

        daughter0 = daughter0[1:]

        daughter1 = list([0, daughter1])
        con = True
        while con == True:

            try:
                con = False
                daughter1.append(
                    list(
                        df2.loc[lambda df2: df2["SPOT_SOURCE_ID"] == daughter1[-1], :][
                            "SPOT_TARGET_ID"
                        ]
                    )[0]
                )
                con = True
            except:
                pass

        daughter1 = daughter1[1:]

        _df3 = []
        for spot in parent:
            _df3.append(
                df_vertice.loc[lambda df_vertice: df_vertice["spot_id"] == spot, :]
            )

        df3 = pd.concat(_df3)
        time_list = []
        C_list = []

        for i in range(len(df3)):
            t = int(df3.iloc[i, 1])
            x = df3.iloc[i, 2]
            y = df3.iloc[i, 3]

            time_list.append(t)
            C_list.append([x, y])

            if i < len(df3) - 1:
                t0 = int(df3.iloc[i, 1])
                t1 = int(df3.iloc[i + 1, 1])
                gap = t1 - t0
                if gap > 1:
                    for j in range(gap - 1):
                        time_list.append(t + 1 + j)

                        x1 = df3.iloc[i + 1, 2]
                        y1 = df3.iloc[i + 1, 3]

                        cx = x + ((x1 - x) * (j + 1)) / (gap)
                        cy = y + ((y1 - y) * (j + 1)) / (gap)
                        C_list.append([cx, cy])

        _df_divisions.append(
            {"Label": label, "Time": time_list, "Position": C_list, "Chain": "parent",}
        )
        divisionTime = time_list[-1]
        divisionPlace = C_list[-1]

        _df3 = []
        for spot in daughter0:
            _df3.append(
                df_vertice.loc[lambda df_vertice: df_vertice["spot_id"] == spot, :]
            )

        df3 = pd.concat(_df3)
        time_list = []
        C_list = []

        t = int(df3.iloc[0, 1])
        if t - 1 != divisionTime:
            gap = t - divisionTime
            for j in range(gap - 1):
                time_list.append(divisionTime + 1 + j)

                [x, y] = divisionPlace
                x1 = df3.iloc[0, 2]
                y1 = df3.iloc[0, 3]

                cx = x + ((x1 - x) * (j + 1)) / (gap)
                cy = y + ((y1 - y) * (j + 1)) / (gap)
                C_list.append([cx, cy])

        for i in range(len(df3)):
            t = int(df3.iloc[i, 1])
            x = df3.iloc[i, 2]
            y = df3.iloc[i, 3]

            time_list.append(t)
            C_list.append([x, y])

            if i < len(df3) - 1:
                t0 = int(df3.iloc[i, 1])
                t1 = int(df3.iloc[i + 1, 1])
                gap = t1 - t0
                if gap > 1:
                    for j in range(gap - 1):
                        time_list.append(t + 1 + j)

                        x1 = df3.iloc[i + 1, 2]
                        y1 = df3.iloc[i + 1, 3]

                        cx = x + ((x1 - x) * (j + 1)) / (gap)
                        cy = y + ((y1 - y) * (j + 1)) / (gap)
                        C_list.append([cx, cy])

        _df_divisions.append(
            {
                "Label": label,
                "Time": time_list,
                "Position": C_list,
                "Chain": "daughter0",
            }
        )

        _df3 = []
        for spot in daughter1:
            _df3.append(
                df_vertice.loc[lambda df_vertice: df_vertice["spot_id"] == spot, :]
            )

        df3 = pd.concat(_df3)
        time_list = []
        C_list = []

        t = int(df3.iloc[0, 1])
        if t - 1 != divisionTime:
            gap = t - divisionTime
            for j in range(gap - 1):
                time_list.append(divisionTime + 1 + j)

                [x, y] = divisionPlace
                x1 = df3.iloc[0, 2]
                y1 = df3.iloc[0, 3]

                cx = x + ((x1 - x) * (j + 1)) / (gap)
                cy = y + ((y1 - y) * (j + 1)) / (gap)
                C_list.append([cx, cy])

        for i in range(len(df3)):
            t = int(df3.iloc[i, 1])
            x = df3.iloc[i, 2]
            y = df3.iloc[i, 3]

            time_list.append(t)
            C_list.append([x, y])

            if i < len(df3) - 1:
                t0 = int(df3.iloc[i, 1])
                t1 = int(df3.iloc[i + 1, 1])
                gap = t1 - t0
                if gap > 1:
                    for j in range(gap - 1):
                        time_list.append(t + 1 + j)

                        x1 = df3.iloc[i + 1, 2]
                        y1 = df3.iloc[i + 1, 3]

                        cx = x + ((x1 - x) * (j + 1)) / (gap)
                        cy = y + ((y1 - y) * (j + 1)) / (gap)
                        C_list.append([cx, cy])

        _df_divisions.append(
            {
                "Label": label,
                "Time": time_list,
                "Position": C_list,
                "Chain": "daughter1",
            }
        )

df_divisions = pd.DataFrame(_df_divisions)
df_divisions.to_pickle(f"databases/mitosis_of_{filename}.pkl")
df_vertice.to_pickle(f"databases/vertice_of_{filename}.pkl")
df_edge.to_pickle(f"databases/edge_of_{filename}.pkl")

# --------------------------------------

trackmate_xml_path = "dat_nucleus/Wound16h01.xml"

df_vertice_nucleus = trackmate_vertices_import(trackmate_xml_path, get_tracks=True)

unique_label = list(set(df_vertice_nucleus["label"]))

_df_tracks = []
for label in unique_label:

    df = df_vertice_nucleus.loc[
        lambda df_vertice: df_vertice_nucleus["label"] == label, :
    ]

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

        if t1 - t0 > 180.01:
            X.append((x[i] + x[i + 1]) / 2)
            Y.append((y[i] + y[i + 1]) / 2)
            Z.append((z[i] + z[i + 1]) / 2)
            T.append((t[i] + t[i + 1]) / 2)

        X.append(x[i + 1])
        Y.append(y[i + 1])
        Z.append(z[i + 1])
        T.append(t[i + 1])

    _df_tracks.append({"Label": label, "x": X, "y": Y, "z": Z, "t": T})

df_tracks = pd.DataFrame(_df_tracks)

df_tracks.to_pickle(f"databases/nucleusVertice{filename}.pkl")
