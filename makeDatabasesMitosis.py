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


def heightOfMitosis(img, polygon):

    (Cx, Cy) = cell.centroid(polygon)

    rList = []
    zList = []

    # finds region local to the dividion

    if Cx > 462:
        xRange = range(int(Cx - 50), 512)
    elif Cx < 50:
        xRange = range(0, int(Cx + 50))
    else:
        xRange = range(int(Cx - 50), int(Cx + 50))

    if Cy > 462:
        yRange = range(int(Cy - 50), 512)
    elif Cy < 50:
        yRange = range(0, int(Cy + 50))
    else:
        yRange = range(int(Cy - 50), int(Cy + 50))

    # collects the height of the neclui local to the division

    for x in xRange:
        for y in yRange:
            z = img[y, x]  # change coord
            if z != 0:
                zList.append(z)
                r = ((Cx - x) ** 2 + (Cy - y) ** 2) ** 0.5
                rList.append(r)

    n = len(rList)

    # finds the change in hieght of mitosis

    height = []
    rMin = 50

    for i in range(n):

        r = rList[i]

        if r < 50:

            height.append(zList[i])

            if r < rMin:

                rMin = r
                mitosisHeight = zList[i]

    meanHeight = cell.mean(height)

    deltaHeight = mitosisHeight - meanHeight

    return deltaHeight


# ------------------

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
    dfEdge = trackmate_edges_import(f"dat/{filename}/mitosisHeightTracks{filename}.xml")

    uniqueLabel = list(set(dfVertice["label"]))

    _dfDivisions = []

    for label in uniqueLabel:

        df = dfVertice.loc[lambda dfVertice: dfVertice["label"] == label, :]

        t = list(df["t_stamp"])
        unique = list(set(t))

        if len(unique) < len(t):
            # compare number of spots with length of time of the track to see if there is a division
            division = True
            spotID = df["spot_id"]
        else:
            division = False

        if division == True:
            _df2 = []
            for spot in spotID:  # collect edges of the track label
                _df2.append(
                    dfEdge.loc[lambda dfEdge: dfEdge["SPOT_SOURCE_ID"] == spot, :]
                )

            df2 = pd.concat(_df2)

            for spot in spotID:

                df3 = df2.loc[lambda df2: df2["SPOT_SOURCE_ID"] == spot, :]
                n = len(df3)
                if n == 2:
                    divID = spot
                    # finds the spot correponding to the new daughter nuclui
                    daughter0 = list(df3["SPOT_TARGET_ID"])[0]
                    daughter1 = list(df3["SPOT_TARGET_ID"])[1]

            # track the links back from the spot just before division to get parent chain
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

            # track the links forward from the zth daughter spot just after division to get daughter0 chain

            daughter0 = list([0, daughter0])
            con = True
            while con == True:

                try:
                    con = False
                    daughter0.append(
                        list(
                            df2.loc[
                                lambda df2: df2["SPOT_SOURCE_ID"] == daughter0[-1], :
                            ]["SPOT_TARGET_ID"]
                        )[0]
                    )
                    con = True
                except:
                    pass

            daughter0 = daughter0[1:]

            # track the links forward from the 1st daughter spot just after division to get daughter1 chain

            daughter1 = list([0, daughter1])
            con = True
            while con == True:

                try:
                    con = False
                    daughter1.append(
                        list(
                            df2.loc[
                                lambda df2: df2["SPOT_SOURCE_ID"] == daughter1[-1], :
                            ]["SPOT_TARGET_ID"]
                        )[0]
                    )
                    con = True
                except:
                    pass

            daughter1 = daughter1[1:]

            # Now we have the chains of spots we can collect to postion and time of each point.
            _df3 = []
            for spot in parent:
                _df3.append(
                    dfVertice.loc[lambda dfVertice: dfVertice["spot_id"] == spot, :]
                )

            df3 = pd.concat(_df3)
            timeList = []
            cList = []

            for i in range(len(df3)):  # fill in any stop gaps
                t = int(df3["t"].iloc[i])
                x = df3["x"].iloc[i]
                y = df3["y"].iloc[i]

                timeList.append(t)
                cList.append([x, y])

                if i < len(df3) - 1:
                    t0 = int(df3["t"].iloc[i])
                    t1 = int(df3["t"].iloc[i + 1])
                    gap = t1 - t0
                    if gap > 1:
                        for j in range(gap - 1):
                            timeList.append(t + 1 + j)

                            x1 = df3["x"].iloc[i + 1]
                            y1 = df3["y"].iloc[i + 1]

                            cx = x + ((x1 - x) * (j + 1)) / (gap)
                            cy = y + ((y1 - y) * (j + 1)) / (gap)
                            cList.append([cx, cy])
            # save chains with data of there position and time
            _dfDivisions.append(
                {
                    "Label": label,
                    "Time": timeList,
                    "Position": cList,
                    "Chain": "parent",
                }
            )
            divisionTime = timeList[-1]
            divisionPlace = cList[-1]

            _df3 = []
            for spot in daughter0:
                _df3.append(
                    dfVertice.loc[lambda dfVertice: dfVertice["spot_id"] == spot, :]
                )

            df3 = pd.concat(_df3)
            timeList = []
            cList = []

            t = int(df3.iloc[0, 1])
            if t - 1 != divisionTime:  # fill in any stop gaps
                gap = t - divisionTime
                for j in range(gap - 1):
                    timeList.append(divisionTime + 1 + j)

                    [x, y] = divisionPlace
                    x1 = df3["x"].iloc[0]
                    y1 = df3["y"].iloc[0]

                    cx = x + ((x1 - x) * (j + 1)) / (gap)
                    cy = y + ((y1 - y) * (j + 1)) / (gap)
                    cList.append([cx, cy])

            for i in range(len(df3)):
                t = int(df3["t"].iloc[i])
                x = df3["x"].iloc[i]
                y = df3["y"].iloc[i]

                timeList.append(t)
                cList.append([x, y])

                if i < len(df3) - 1:
                    t0 = int(df3["t"].iloc[i])
                    t1 = int(df3["t"].iloc[i + 1])
                    gap = t1 - t0
                    if gap > 1:
                        for j in range(gap - 1):
                            timeList.append(t + 1 + j)

                            x1 = df3.iloc[i + 1, 2]
                            y1 = df3.iloc[i + 1, 3]

                            cx = x + ((x1 - x) * (j + 1)) / (gap)
                            cy = y + ((y1 - y) * (j + 1)) / (gap)
                            cList.append([cx, cy])

            # save chains with data of there position and time
            _dfDivisions.append(
                {
                    "Label": label,
                    "Time": timeList,
                    "Position": cList,
                    "Chain": "daughter0",
                }
            )

            _df3 = []
            for spot in daughter1:
                _df3.append(
                    dfVertice.loc[lambda dfVertice: dfVertice["spot_id"] == spot, :]
                )

            df3 = pd.concat(_df3)
            timeList = []
            cList = []

            t = int(df3["t"].iloc[0])
            if t - 1 != divisionTime:  # fill in any stop gaps
                gap = t - divisionTime
                for j in range(gap - 1):
                    timeList.append(divisionTime + 1 + j)

                    [x, y] = divisionPlace
                    x1 = df3["x"].iloc[0]
                    y1 = df3["y"].iloc[0]

                    cx = x + ((x1 - x) * (j + 1)) / (gap)
                    cy = y + ((y1 - y) * (j + 1)) / (gap)
                    cList.append([cx, cy])

            for i in range(len(df3)):
                t = int(df3["t"].iloc[i])
                x = df3["x"].iloc[i]
                y = df3["y"].iloc[i]

                timeList.append(t)
                cList.append([x, y])

                if i < len(df3) - 1:
                    t0 = int(df3["t"].iloc[i])
                    t1 = int(df3["t"].iloc[i + 1])
                    gap = t1 - t0
                    if gap > 1:
                        for j in range(gap - 1):
                            timeList.append(t + 1 + j)

                            x1 = df3["x"].iloc[i + 1]
                            y1 = df3["y"].iloc[i + 1]

                            cx = x + ((x1 - x) * (j + 1)) / (gap)
                            cy = y + ((y1 - y) * (j + 1)) / (gap)
                            cList.append([cx, cy])
            # save chains with data of there position and time
            _dfDivisions.append(
                {
                    "Label": label,
                    "Time": timeList,
                    "Position": cList,
                    "Chain": "daughter1",
                }
            )

    dfDivisions = pd.DataFrame(_dfDivisions)

    vidBinary = (
        sm.io.imread(f"dat/{filename}/probMitosis{filename}.tif").astype(float) * 255
    )

    T = len(vidBinary)

    for t in range(T):
        vidBinary[t] = sp.signal.medfilt(vidBinary[t], kernel_size=5)

    vidBinary[vidBinary > 100] = 255
    vidBinary[vidBinary <= 100] = 0

    vidBinary = np.asarray(vidBinary, "uint8")
    tifffile.imwrite(f"dat/{filename}/binaryMitosis{filename}.tif", vidBinary)

    vidHeight = (
        sm.io.imread(f"dat/{filename}/height{filename}.tif").astype("float") * 25
    )

    vidLabels = []

    _dfDivisions2 = []

    T = len(vidBinary)

    # use the boundary of the nuclei and convents to polygons to find there orientation and shape factor

    for t in range(T):
        # labels all the mitosic nuclei
        vidLabels.append(sm.measure.label(vidBinary[t], background=0, connectivity=1))

    for i in range(len(dfDivisions)):

        Tm = len((dfDivisions["Time"][i]))

        polygons = []
        sf = []
        ori = []
        h = []

        for t in range(Tm):

            frame = dfDivisions["Time"][i][t]
            imgLabel = vidLabels[frame]
            [x, y] = dfDivisions["Position"][i][t]

            x = int(x)
            y = int(y)

            label = imgLabel[y, x]  # coordenate change

            if label != 0:

                contour = sm.measure.find_contours(imgLabel == label, level=0)[0]
                poly = sm.measure.approximate_polygon(contour, tolerance=1)

                polygon = Polygon(poly)

                sf.append(cell.shapeFactor(polygon))
                ori.append(cell.orientation(polygon))
                polygons.append(polygon)
                h.append(heightOfMitosis(vidHeight[t], polygon))

            else:

                sf.append(False)
                ori.append(False)
                polygons.append(False)
                h.append(False)

        label = dfDivisions["Label"][i]
        timeList = dfDivisions["Time"][i]
        cList = dfDivisions["Position"][i]
        chain = dfDivisions["Chain"][i]

        _dfDivisions2.append(
            {
                "Label": label,
                "Time": timeList,
                "Position": cList,
                "Chain": chain,
                "Shape Factor": sf,
                "Height": h,
                "Necleus Orientation": ori,
                "Polygons": polygons,
            }
        )

    dfDivisions2 = pd.DataFrame(_dfDivisions2)

    # -------

    # finds the orientation of division

    divOri = []

    _dfDivisions3 = []

    uniqueLabel = list(set(dfDivisions2["Label"]))

    j = 0

    if wound == True:
        dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")

        dist = []

        for label in uniqueLabel:
            df3 = dfDivisions2.loc[
                lambda dfDivisions2: dfDivisions2["Label"] == label, :
            ]

            if len(df3["Time"][3 * j + 1]) > 2:
                delta_t0 = 2
                t0 = df3["Time"][3 * j + 1][2]
            elif len(df3["Time"][3 * j + 1]) > 1:
                delta_t0 = 1
                t0 = df3["Time"][3 * j + 1][1]
            else:
                delta_t0 = 0
                t0 = df3["Time"][3 * j + 1][0]

            if len(df3["Time"][3 * j + 2]) > 2:
                delta_t1 = 2
                t0 = df3["Time"][3 * j + 2][2]
            elif len(df3["Time"][3 * j + 2]) > 1:
                delta_t1 = 1
                t0 = df3["Time"][3 * j + 2][1]
            else:
                delta_t1 = 0
                t0 = df3["Time"][3 * j + 2][0]

            (Cy, Cx) = dfWound["centriod"][t0]  # change ord
            woundPolygon = dfWound["polygon"][t0]
            r = (woundPolygon.area / np.pi) ** 0.5
            [x0, y0] = df3["Position"][3 * j + 1][delta_t0]
            [x1, y1] = df3["Position"][3 * j + 2][delta_t1]

            xm = (x0 + x1) / 2
            ym = (y0 + y1) / 2
            v = np.array([x0 - x1, y0 - y1])
            w = np.array([xm - Cx, ym - Cy])

            phi = np.arccos(np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w)))

            if phi > np.pi / 2:
                theta = np.pi - phi
            else:
                theta = phi
            divOri = theta * (180 / np.pi)
            dist = np.linalg.norm(w) - r

            for i in range(3):
                _dfDivisions3.append(
                    {
                        "Label": df3["Label"][3 * j + i],
                        "Time": df3["Time"][3 * j + i],
                        "Position": df3["Position"][3 * j + i],
                        "Chain": df3["Chain"][3 * j + i],
                        "Shape Factor": df3["Shape Factor"][3 * j + i],
                        "Height": df3["Height"][3 * j + i],
                        "Necleus Orientation": df3["Necleus Orientation"][3 * j + i],
                        "Polygons": df3["Polygons"][3 * j + i],
                        "Division Orientation": divOri,
                    }
                )

            j += 1

    else:

        for label in uniqueLabel:
            df3 = dfDivisions2.loc[
                lambda dfDivisions2: dfDivisions2["Label"] == label, :
            ]

            if len(df3["Time"][3 * j + 1]) > 2:
                delta_t0 = 2
            elif len(df3["Time"][3 * j + 1]) > 1:
                delta_t0 = 1
            else:
                delta_t0 = 0

            if len(df3["Time"][3 * j + 2]) > 2:
                delta_t1 = 2
            elif len(df3["Time"][3 * j + 2]) > 1:
                delta_t1 = 1
            else:
                delta_t1 = 0

            (Cx, Cy) = (0, 1)  # change coord
            [x0, y0] = df3["Position"][3 * j + 1][delta_t0]
            [x1, y1] = df3["Position"][3 * j + 2][delta_t1]

            xm = (x0 + x1) / 2
            ym = (y0 + y1) / 2
            v = np.array([x0 - x1, y0 - y1])
            w = np.array([xm - Cx, ym - Cy])

            phi = np.arccos(np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w)))

            if phi > np.pi / 2:
                theta = np.pi - phi
            else:
                theta = phi
            divOri.append(theta * (180 / np.pi))

            for i in range(3):
                _dfDivisions3.append(
                    {
                        "Label": df3["Label"][3 * j + i],
                        "Time": df3["Time"][3 * j + i],
                        "Position": df3["Position"][3 * j + i],
                        "Chain": df3["Chain"][3 * j + i],
                        "Shape Factor": df3["Shape Factor"][3 * j + i],
                        "Height": df3["Height"][3 * j + i],
                        "Necleus Orientation": df3["Necleus Orientation"][3 * j + i],
                        "Polygons": df3["Polygons"][3 * j + i],
                        "Division Orientation": divOri,
                    }
                )

            j += 1

    dfDivisions3 = pd.DataFrame(_dfDivisions3)

    # links the mitosic cells to there cell boundarys and tracks these cells

    binary = sm.io.imread(f"dat/{filename}/binaryBoundary{filename}.tif").astype(
        "uint8"
    )
    trackBinary = binary

    vidLabels = []
    T = len(binary)

    for t in range(T):
        img = binary[t]
        img = 255 - img
        vidLabels.append(sm.measure.label(img, background=0, connectivity=1))

    vidLabels = np.asarray(vidLabels, "uint16")
    tifffile.imwrite(f"dat/{filename}/vidLabel{filename}.tif", vidLabels)

    _dfDivisions4 = []

    j = 0

    for label in uniqueLabel:
        df4 = dfDivisions3.loc[lambda dfDivisions3: dfDivisions3["Label"] == label, :]

        polygonsParent = []
        polygonsDaughter1 = []
        polygonsDaughter2 = []

        (Cx, Cy) = df4["Position"][3 * j][-1]
        tm = df4["Time"][3 * j][-1]

        Cx = int(Cx)
        Cy = int(Cy)
        t = tm

        parentLabel = vidLabels[tm][Cy, Cx]  # change coord

        divided = False

        # finds the time and position of cytokinesis

        while divided == False and (t + 1 < T):

            labels = vidLabels[t + 1][vidLabels[t] == parentLabel]

            uniqueLabels = set(list(labels))
            if 0 in uniqueLabels:
                uniqueLabels.remove(0)

            count = Counter(labels)
            c = []
            for l in uniqueLabels:
                c.append(count[l])

            uniqueLabels = list(uniqueLabels)
            mostLabel = uniqueLabels[c.index(max(c))]
            C = max(c)

            c.remove(max(c))
            uniqueLabels.remove(mostLabel)

            if c == []:
                Cdash = 0
            else:
                mostLabel2nd = uniqueLabels[c.index(max(c))]
                Cdash = max(c)

            if Cdash / C > 0.5:
                divided = True
                daughterLabel1 = mostLabel
                daughterLabel2 = mostLabel2nd
            else:
                t += 1
                parentLabel = mostLabel

        # tracks the cell forwards and backwards in time

        if divided == True:

            tc = t  # time of cytokinesis

            if len(vidLabels) > tc + 11:
                tFinal = 9
            else:
                tFinal = len(vidLabels) - tc - 2

            trackBinary[tc + 1][vidLabels[tc + 1] == daughterLabel1] = 200

            contour = sm.measure.find_contours(
                vidLabels[tc + 1] == daughterLabel1, level=0
            )[0]
            poly = sm.measure.approximate_polygon(contour, tolerance=1)
            polygonsDaughter1.append(poly)

            # --

            trackBinary[tc + 1][vidLabels[tc + 1] == daughterLabel2] = 150

            contour = sm.measure.find_contours(
                vidLabels[tc + 1] == daughterLabel2, level=0
            )[0]
            poly = sm.measure.approximate_polygon(contour, tolerance=1)
            polygonsDaughter2.append(poly)

            for i in range(tFinal):

                labels = vidLabels[tc + 2 + i][vidLabels[tc + 1 + i] == daughterLabel1]

                uniqueLabels = set(list(labels))
                if 0 in uniqueLabels:
                    uniqueLabels.remove(0)

                count = Counter(labels)
                c = []
                for l in uniqueLabels:
                    c.append(count[l])

                uniqueLabels = list(uniqueLabels)
                daughterLabel1 = uniqueLabels[c.index(max(c))]

                trackBinary[tc + 2 + i][vidLabels[tc + 2 + i] == daughterLabel1] = 200

                contour = sm.measure.find_contours(
                    vidLabels[tc + 2 + i] == daughterLabel1, level=0
                )[0]
                poly = sm.measure.approximate_polygon(contour, tolerance=1)
                polygonsDaughter1.append(poly)

                # ----

                labels = vidLabels[tc + 2 + i][vidLabels[tc + 1 + i] == daughterLabel2]

                uniqueLabels = set(list(labels))
                if 0 in uniqueLabels:
                    uniqueLabels.remove(0)

                count = Counter(labels)
                c = []
                for l in uniqueLabels:
                    c.append(count[l])

                uniqueLabels = list(uniqueLabels)
                daughterLabel2 = uniqueLabels[c.index(max(c))]

                trackBinary[tc + 2 + i][vidLabels[tc + 2 + i] == daughterLabel2] = 150

                contour = sm.measure.find_contours(
                    vidLabels[tc + 2 + i] == daughterLabel2, level=0
                )[0]
                poly = sm.measure.approximate_polygon(contour, tolerance=1)
                polygonsDaughter2.append(poly)

            if 0 < tc - 10:
                tMitosis = 9
            else:
                tMitosis = tc

            trackBinary[tc][vidLabels[tc] == parentLabel] = 100
            contour = sm.measure.find_contours(vidLabels[tc] == parentLabel, level=0)[0]
            poly = sm.measure.approximate_polygon(contour, tolerance=1)
            polygonsParent.append(poly)

            for i in range(tMitosis):
                labels = vidLabels[tc - i - 1][vidLabels[tc - i] == parentLabel]

                uniqueLabels = set(list(labels))
                if 0 in uniqueLabels:
                    uniqueLabels.remove(0)

                count = Counter(labels)
                c = []
                for l in uniqueLabels:
                    c.append(count[l])

                uniqueLabels = list(uniqueLabels)
                parentLabel = uniqueLabels[c.index(max(c))]

                trackBinary[tc - i - 1][vidLabels[tc - i - 1] == parentLabel] = 100

                contour = sm.measure.find_contours(
                    vidLabels[tc - i - 1] == parentLabel, level=0
                )[0]
                poly = sm.measure.approximate_polygon(contour, tolerance=1)
                polygonsParent.append(poly)

            _dfDivisions4.append(
                {
                    "Label": df4["Label"][3 * j],
                    "Time": df4["Time"][3 * j],
                    "Position": df4["Position"][3 * j],
                    "Chain": df4["Chain"][3 * j],
                    "Shape Factor": df4["Shape Factor"][3 * j],
                    "Height": df4["Height"][3 * j],
                    "Necleus Orientation": df4["Necleus Orientation"][3 * j],
                    "Polygons": df4["Polygons"][3 * j],
                    "Division Orientation": df4["Division Orientation"][3 * j],
                    "Boundary Polygons": polygonsParent,
                    "Cytokineses time": tc,
                    "Time difference": tc - tm,
                }
            )

            _dfDivisions4.append(
                {
                    "Label": df4["Label"][3 * j + 1],
                    "Time": df4["Time"][3 * j + 1],
                    "Position": df4["Position"][3 * j + 1],
                    "Chain": df4["Chain"][3 * j + 1],
                    "Shape Factor": df4["Shape Factor"][3 * j + 1],
                    "Height": df4["Height"][3 * j + 1],
                    "Necleus Orientation": df4["Necleus Orientation"][3 * j + 1],
                    "Polygons": df4["Polygons"][3 * j + 1],
                    "Division Orientation": df4["Division Orientation"][3 * j + 1],
                    "Boundary Polygons": polygonsDaughter1,
                    "Cytokineses time": tc,
                    "Time difference": tc - tm,
                }
            )

            _dfDivisions4.append(
                {
                    "Label": df4["Label"][3 * j + 2],
                    "Time": df4["Time"][3 * j + 2],
                    "Position": df4["Position"][3 * j + 2],
                    "Chain": df4["Chain"][3 * j + 2],
                    "Shape Factor": df4["Shape Factor"][3 * j + 2],
                    "Height": df4["Height"][3 * j + 2],
                    "Necleus Orientation": df4["Necleus Orientation"][3 * j + 2],
                    "Polygons": df4["Polygons"][3 * j + 2],
                    "Division Orientation": df4["Division Orientation"][3 * j + 2],
                    "Boundary Polygons": polygonsDaughter2,
                    "Cytokineses time": tc,
                    "Time difference": tc - tm,
                }
            )

            j += 1

        else:
            for i in range(3):
                _dfDivisions4.append(
                    {
                        "Label": df4["Label"][3 * j + i],
                        "Time": df4["Time"][3 * j + i],
                        "Position": df4["Position"][3 * j + i],
                        "Chain": df4["Chain"][3 * j + i],
                        "Shape Factor": df4["Shape Factor"][3 * j + i],
                        "Height": df4["Height"][3 * j + i],
                        "Necleus Orientation": df4["Necleus Orientation"][3 * j + i],
                        "Polygons": df4["Polygons"][3 * j + i],
                        "Division Orientation": df4["Division Orientation"][3 * j + i],
                    }
                )

            j += 1

    dfDivisions4 = pd.DataFrame(_dfDivisions4)

    dfDivisions4.to_pickle(f"dat/{filename}/mitosisTracks{filename}.pkl")

    trackBinary = np.asarray(trackBinary, "uint8")
    tifffile.imwrite(f"dat/{filename}/tracks{filename}.tif", trackBinary)
