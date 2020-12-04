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

costLim = 0.5


def costFunction(u0, u1):

    cost = np.linalg.norm(u0 + u1) / (np.linalg.norm(u0) + np.linalg.norm(u1))

    return cost


def mapDivision(graph):

    graphD = {}
    graphKeys = list(graph.keys())

    divisionKeys = []
    for graphKey in graphKeys:
        if len(graph[graphKey]) == 2:
            divisionKeys.append(graphKey)

    origin = min(divisionKeys)
    pathStarts = list(graph[origin])

    while pathStarts != []:
        start = pathStarts[0]
        spot = start

        finished = False
        while finished != True:
            try:
                spot = graph[spot][0]
                if spot in divisionKeys:
                    graphD[start] = spot
                    pathStarts.append(graph[spot][0])
                    pathStarts.append(graph[spot][1])
                    finished = True
            except:
                finished = True

        pathStarts.remove(start)

    return graphD


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

    features = ["name", "POSITION_X", "POSITION_Y", "FRAME"]

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
            single_object.append(
                511 - float(spot.get("POSITION_Y"))
            )  # this makes coords xy
            single_object.append(spot.get("FRAME"))

            objects.append(single_object)

    spotDat = pd.DataFrame(objects, columns=features)

    # get tracks

    features = root.find("Model").find("FeatureDeclarations").find("EdgeFeatures")
    features = [c.get("feature") for c in features.getchildren()] + ["ID"]
    features = features[0:2]

    tracks = root.find("Model").find("AllTracks")
    track = tracks[-1]
    nextLabel = int(track.get("TRACK_ID")) + 1
    objects = []
    _trackDat = []
    for track in tracks:
        # looks for tracks with Divisions
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

            # if only one division just need to check if its a good one
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

                # check the division
                cost = costFunction(u0, u1)
                # print(cost)
                # print(label)

                if cost < costLim:
                    spots = start
                    for spot in end:
                        spots.append(spot)

                    uniqueSpots = set(spots)
                    for spot in uniqueSpots:
                        df = spotDat[spotDat["name"] == spot]

                        time = int(float(df["FRAME"]))
                        centroid = [float(df["POSITION_X"]), float(df["POSITION_Y"])]

                        # save the good tracks
                        _trackDat.append(
                            {
                                "Label": label,
                                "Time": time,
                                "Spot": spot,
                                "Position": centroid,
                                "Original Label": label,
                            }
                        )

            # if there is multi division tracks
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

                # check for false links to be unpicked later
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
                        # print(cost)
                        # print(label)

                        if cost > costLim:

                            maxDis = max(np.linalg.norm(u0), np.linalg.norm(u1))
                            if np.linalg.norm(u0) == maxDis:
                                falseLink.append(daughter0)
                            else:
                                falseLink.append(daughter1)

                # unpick the false division and save a Dictionary of the graphs
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
                    divisionKeys = []
                    for graphKey in graphKeys:
                        if len(graph[graphKey]) == 2:
                            divisionNum += 1
                            divisionKeys.append(graphKey)
                    # check if there are still multi division
                    # if divisionNum == 2:
                    #     key0 = divisionKeys[0]
                    #     key1 = divisionKeys[1]
                    #     if key0 > key1: #make key0 smallest
                    #         key = key1
                    #         key1 = key0
                    #         key0 = key

                    #     [graph, graph1] = splitGraph(graph, falseSpot)
                    #     dictGraph[key] = graph
                    #     dictGraph[f"graph{a}"] = graph1
                    #     a += 1

                    if divisionNum > 1:
                        graphD = mapDivision(graph)
                        divisionKeys = list(graphD.keys())
                        divisionKeys.reverse()

                        for key in divisionKeys:
                            end = graphD[key]
                            spot = key
                            spots = (
                                []
                            )  # note that the first 2 links in the track are missed out on perpese
                            finished = False
                            while finished != True:
                                try:
                                    spot = graph[spot][0]
                                    if end == spot:
                                        spots.append(spot)
                                        finished = True
                                    else:
                                        spots.append(spot)
                                except:
                                    finished = True

                            n = len(spots)
                            length = []
                            for i in range(n - 1):
                                spot0 = spots[i]
                                spot1 = spots[i + 1]
                                df0 = spotDat[spotDat["name"] == spot0]
                                df1 = spotDat[spotDat["name"] == spot1]
                                l = (
                                    (
                                        float(df0["POSITION_X"])
                                        - float(df1["POSITION_X"])
                                    )
                                    ** 2
                                    + (
                                        float(df0["POSITION_Y"])
                                        - float(df1["POSITION_Y"])
                                    )
                                    ** 2
                                ) ** 0.5
                                length.append(l)
                            m = max(length)
                            index = length.index(m) + 1
                            falseSpot = spots[index]
                            [graph, graph1] = splitGraph(graph, falseSpot)
                            dictGraph[key] = graph
                            dictGraph[f"graph{a}"] = graph1
                            a += 1

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
                    uniqueSpots = set(spots)

                    if divisions:
                        for spot in uniqueSpots:
                            df = spotDat[spotDat["name"] == spot]

                            time = int(float(df["FRAME"]))
                            centroid = [
                                float(df["POSITION_X"]),
                                float(df["POSITION_Y"]),
                            ]
                            # save divisions from the split up tracks in it
                            _trackDat.append(
                                {
                                    "Label": nextLabel,
                                    "Time": time,
                                    "Spot": spot,
                                    "Position": centroid,
                                    "Original Label": label,
                                }
                            )

                        nextLabel += 1

        else:
            continue

    trackDat = pd.DataFrame(_trackDat)

    if len(trackDat) > 0:
        trackDat = trackDat.sort_values(["Label", "Time"], ascending=[True, True])

    return trackDat


def heightOfMitosis(img, polygon):

    (Cy, Cx) = cell.centroid(polygon)

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


# ----------------------------------------------------

f = open("pythonText.txt", "r")

filenames = f.read()
filenames = filenames.split(", ")

for filename in filenames:

    if "Wound" in filename:
        wound = True
    else:
        wound = False

    # gather xml files

    df = importDividingTracks(f"dat/{filename}/mitosisTracks{filename}.xml")

    if len(df) > 0:

        uniqueLabels = list(set(df["Label"]))

        _dfDivisions = []

        for label in uniqueLabels:
            # pick out each division track
            dfTrack = df.loc[lambda df: df["Label"] == label, :]

            timeList = list(dfTrack["Time"])
            uniqueTime = list(set(timeList))
            count = Counter(timeList)

            num = []
            for t in uniqueTime:

                n = count[t]

                if n == 2:
                    num.append(t)

            divisionTime = min(num)
            parent = list(dfTrack["Spot"][dfTrack["Time"] < divisionTime])
            [daughter0, daughter1] = list(
                dfTrack["Spot"][dfTrack["Time"] == divisionTime]
            )
            [daughterPos0, daughterPos1] = list(
                dfTrack["Position"][dfTrack["Time"] == divisionTime]
            )
            daughterSpots = list(dfTrack["Spot"][dfTrack["Time"] > divisionTime])
            daughterPos = list(dfTrack["Position"][dfTrack["Time"] > divisionTime])

            originalLabel = int(dfTrack["Original Label"][dfTrack["Spot"] == daughter0])

            daughter0 = [daughter0]
            daughter1 = [daughter1]
            # build a database of the tracks
            for i in range(len(daughterSpots)):
                spot = daughterSpots[i]
                position = daughterPos[i]

                r0 = (
                    (daughterPos0[0] - position[0]) ** 2
                    + (daughterPos0[1] - position[1]) ** 2
                ) ** 0.5
                r1 = (
                    (daughterPos1[0] - position[0]) ** 2
                    + (daughterPos1[1] - position[1]) ** 2
                ) ** 0.5

                if r0 > r1:
                    daughter1.append(spot)
                    daughterPos1 = position
                else:
                    daughter0.append(spot)
                    daughterPos0 = position

            timeList = []
            cList = []

            for spot in parent:

                t = int(dfTrack["Time"][dfTrack["Spot"] == spot])
                timeList.append(t)
                [x, y] = list(dfTrack["Position"][dfTrack["Spot"] == spot])[0]
                cList.append([x, y])

                # save the parent part of the track in one database line
            _dfDivisions.append(
                {
                    "Label": label,
                    "Time": timeList,
                    "Position": cList,
                    "Chain": "parent",
                    "Original Label": originalLabel,
                    "Spot": parent,
                }
            )

            timeList = []
            cList = []

            for spot in daughter0:

                t = int(dfTrack["Time"][dfTrack["Spot"] == spot])
                timeList.append(t)
                [x, y] = list(dfTrack["Position"][dfTrack["Spot"] == spot])[0]
                cList.append([x, y])

            # save the daughter0 part of the track in one database line
            _dfDivisions.append(
                {
                    "Label": label,
                    "Time": timeList,
                    "Position": cList,
                    "Chain": "daughter0",
                    "Original Label": originalLabel,
                    "Spot": daughter0,
                }
            )

            timeList = []
            cList = []

            for spot in daughter1:

                t = int(dfTrack["Time"][dfTrack["Spot"] == spot])
                timeList.append(t)
                [x, y] = list(dfTrack["Position"][dfTrack["Spot"] == spot])[0]
                cList.append([x, y])

            # save the daughter1 part of the track in one database line
            _dfDivisions.append(
                {
                    "Label": label,
                    "Time": timeList,
                    "Position": cList,
                    "Chain": "daughter1",
                    "Original Label": originalLabel,
                    "Spot": daughter1,
                }
            )

        dfDivisions = pd.DataFrame(_dfDivisions)

        dfDivisions = dfDivisions.sort_values(
            ["Label", "Original Label"], ascending=[True, True]
        )

        vidBinary = (
            sm.io.imread(f"dat/{filename}/probMitosis{filename}.tif").astype(float)
            * 255
        )

        T = len(vidBinary)

        vidBinary[vidBinary > 100] = 255
        vidBinary[vidBinary <= 100] = 0

        for t in range(T):
            vidBinary[t] = sp.signal.medfilt(vidBinary[t], kernel_size=5)

        vidBinary = np.asarray(vidBinary, "uint8")
        tifffile.imwrite(f"dat/{filename}/binaryMitosis{filename}.tif", vidBinary)

        vidHeight = (
            sm.io.imread(f"dat/{filename}/height{filename}.tif").astype("float") * 30
        )

        vidLabels = []

        _dfDivisions2 = []

        T = len(vidBinary)

        # use the boundary of the nuclei and convents to polygons to find there orientation and shape factor

        for t in range(T):
            # labels all the mitosic nuclei
            img = sm.measure.label(vidBinary[t], background=0, connectivity=1)
            imgxy = fi.imgrcxy(img)
            vidLabels.append(imgxy)
            vidHeight[t] = fi.imgrcxy(vidHeight[t])  # changes both to xy coords

        for i in range(len(dfDivisions)):

            M = len((dfDivisions["Time"][i]))

            polygons = []
            sf = []
            ori = []
            h = []

            for j in range(M):

                frame = dfDivisions["Time"][i][j]
                imgLabel = vidLabels[frame]
                [x, y] = dfDivisions["Position"][i][j]

                x = int(x)
                y = int(y)

                label = imgLabel[x, y]

                if label != 0:

                    contour = sm.measure.find_contours(imgLabel == label, level=0)[0]
                    poly = sm.measure.approximate_polygon(contour, tolerance=1)

                    if LinearRing(poly).is_simple:
                        polygon = Polygon(poly)
                        sf.append(cell.shapeFactor(polygon))
                        ori.append(cell.orientation(polygon))
                        polygons.append(polygon)
                        h.append(heightOfMitosis(vidHeight[frame], polygon))
                    else:
                        sf.append(False)
                        ori.append(False)
                        polygons.append(False)
                        h.append(False)

                else:

                    sf.append(False)
                    ori.append(False)
                    polygons.append(False)
                    h.append(False)

            label = dfDivisions["Label"][i]
            timeList = dfDivisions["Time"][i]
            cList = dfDivisions["Position"][i]
            chain = dfDivisions["Chain"][i]
            originalLabel = dfDivisions["Original Label"][i]
            spots = dfDivisions["Spot"][i]

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
                    "Original Label": originalLabel,
                    "Spot": spots,
                }
            )

        dfDivisions2 = pd.DataFrame(_dfDivisions2)
        dfDivisions2 = dfDivisions2.sort_values(
            ["Label", "Original Label"], ascending=[True, True]
        )

        # -------

        # finds the orientation of division

        _dfDivisions3 = []

        uniqueLabel = list(set(dfDivisions2["Label"]))

        if wound == True:
            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")

            dist = []

            tw = len(dfWound)

            for label in uniqueLabel:
                df3 = dfDivisions2.loc[
                    lambda dfDivisions2: dfDivisions2["Label"] == label, :
                ]

                tm = df3["Time"].iloc[1][0]

                if len(df3["Time"].iloc[1]) > 2:
                    delta_t0 = 2
                    t0 = df3["Time"].iloc[1][2]
                elif len(df3["Time"].iloc[1]) > 1:
                    delta_t0 = 1
                    t0 = df3["Time"].iloc[1][1]
                else:
                    delta_t0 = 0
                    t0 = df3["Time"].iloc[1][0]

                if len(df3["Time"].iloc[2]) > 2:
                    delta_t1 = 2
                    t0 = df3["Time"].iloc[2][2]
                elif len(df3["Time"].iloc[2]) > 1:
                    delta_t1 = 1
                    t0 = df3["Time"].iloc[2][1]
                else:
                    delta_t1 = 0
                    t0 = df3["Time"].iloc[2][0]

                if tw > t0:

                    (Cy, Cx) = dfWound["Centriod"][t0]  # change ord
                    woundPolygon = dfWound["Polygon"][t0]
                    r = (woundPolygon.area / np.pi) ** 0.5
                    [x0, y0] = df3["Position"].iloc[1][delta_t0]
                    [x1, y1] = df3["Position"].iloc[2][delta_t1]

                    xm = (x0 + x1) / 2
                    ym = (y0 + y1) / 2
                    v = np.array([x0 - x1, y0 - y1])
                    w = np.array([xm - Cx, ym - Cy])

                    phi = np.arccos(
                        np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))
                    )

                    if phi > np.pi / 2:
                        theta = np.pi - phi
                    else:
                        theta = phi
                    divOri = theta * (180 / np.pi)
                    dist = np.linalg.norm(w) - r

                    for i in range(3):
                        _dfDivisions3.append(
                            {
                                "Label": df3["Label"].iloc[i],
                                "Time": df3["Time"].iloc[i],
                                "Position": df3["Position"].iloc[i],
                                "Chain": df3["Chain"].iloc[i],
                                "Shape Factor": df3["Shape Factor"].iloc[i],
                                "Height": df3["Height"].iloc[i],
                                "Necleus Orientation": df3["Necleus Orientation"].iloc[
                                    i
                                ],
                                "Polygons": df3["Polygons"].iloc[i],
                                "Division Orientation": divOri,
                                "Division While Wounded": "Y",
                                "Original Label": df3["Original Label"].iloc[i],
                                "Spot": df3["Spot"].iloc[i],
                            }
                        )

                else:
                    [x0, y0] = df3["Position"].iloc[1][delta_t0]
                    [x1, y1] = df3["Position"].iloc[2][delta_t1]

                    xm = (x0 + x1) / 2
                    ym = (y0 + y1) / 2
                    v = np.array([x0 - x1, y0 - y1])
                    w = np.array([1, 0])

                    phi = np.arccos(
                        np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))
                    )

                    if phi > np.pi:
                        theta = np.pi - phi
                    else:
                        theta = phi
                    divOri = theta * (180 / np.pi)

                    for i in range(3):
                        _dfDivisions3.append(
                            {
                                "Label": df3["Label"].iloc[i],
                                "Time": df3["Time"].iloc[i],
                                "Position": df3["Position"].iloc[i],
                                "Chain": df3["Chain"].iloc[i],
                                "Shape Factor": df3["Shape Factor"].iloc[i],
                                "Height": df3["Height"].iloc[i],
                                "Necleus Orientation": df3["Necleus Orientation"].iloc[
                                    i
                                ],
                                "Polygons": df3["Polygons"].iloc[i],
                                "Division Orientation": divOri,
                                "Division While Wounded": "N",
                                "Original Label": df3["Original Label"].iloc[i],
                                "Spot": df3["Spot"].iloc[i],
                            }
                        )

        else:

            for label in uniqueLabel:
                df3 = dfDivisions2.loc[
                    lambda dfDivisions2: dfDivisions2["Label"] == label, :
                ]

                if len(df3["Time"].iloc[1]) > 2:
                    delta_t0 = 2
                elif len(df3["Time"].iloc[1]) > 1:
                    delta_t0 = 1
                else:
                    delta_t0 = 0

                if len(df3["Time"].iloc[2]) > 2:
                    delta_t1 = 2
                elif len(df3["Time"].iloc[2]) > 1:
                    delta_t1 = 1
                else:
                    delta_t1 = 0

                [x0, y0] = df3["Position"].iloc[1][delta_t0]
                [x1, y1] = df3["Position"].iloc[2][delta_t1]

                v = np.array([x0 - x1, y0 - y1])
                w = np.array([1, 0])

                phi = np.arccos(np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w)))

                if phi > np.pi:
                    theta = np.pi - phi
                else:
                    theta = phi
                divOri = theta * (180 / np.pi)

                for i in range(3):
                    _dfDivisions3.append(
                        {
                            "Label": df3["Label"].iloc[i],
                            "Time": df3["Time"].iloc[i],
                            "Position": df3["Position"].iloc[i],
                            "Chain": df3["Chain"].iloc[i],
                            "Shape Factor": df3["Shape Factor"].iloc[i],
                            "Height": df3["Height"].iloc[i],
                            "Necleus Orientation": df3["Necleus Orientation"].iloc[i],
                            "Polygons": df3["Polygons"].iloc[i],
                            "Division Orientation": divOri,
                            "Division While Wounded": "N",
                            "Original Label": df3["Original Label"].iloc[i],
                            "Spot": df3["Spot"].iloc[i],
                        }
                    )

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

        # vidLabels = np.asarray(vidLabels, "uint16")
        # tifffile.imwrite(f"dat/{filename}/vidLabel{filename}.tif", vidLabels)

        _dfDivisions4 = []

        for label in uniqueLabel:
            df4 = dfDivisions3.loc[
                lambda dfDivisions3: dfDivisions3["Label"] == label, :
            ]

            polygonsParent = []
            polygonsDaughter1 = []
            polygonsDaughter2 = []

            (Cx, Cy) = df4["Position"].iloc[0][-1]
            tm = df4["Time"].iloc[0][-1]

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

                if Cdash / C > 0.3:
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

                    labels = vidLabels[tc + 2 + i][
                        vidLabels[tc + 1 + i] == daughterLabel1
                    ]

                    uniqueLabels = set(list(labels))
                    if 0 in uniqueLabels:
                        uniqueLabels.remove(0)

                    count = Counter(labels)
                    c = []
                    for l in uniqueLabels:
                        c.append(count[l])

                    uniqueLabels = list(uniqueLabels)
                    daughterLabel1 = uniqueLabels[c.index(max(c))]

                    trackBinary[tc + 2 + i][
                        vidLabels[tc + 2 + i] == daughterLabel1
                    ] = 200

                    contour = sm.measure.find_contours(
                        vidLabels[tc + 2 + i] == daughterLabel1, level=0
                    )[0]
                    poly = sm.measure.approximate_polygon(contour, tolerance=1)
                    polygonsDaughter1.append(poly)

                    # ----

                    labels = vidLabels[tc + 2 + i][
                        vidLabels[tc + 1 + i] == daughterLabel2
                    ]

                    uniqueLabels = set(list(labels))
                    if 0 in uniqueLabels:
                        uniqueLabels.remove(0)

                    count = Counter(labels)
                    c = []
                    for l in uniqueLabels:
                        c.append(count[l])

                    uniqueLabels = list(uniqueLabels)
                    daughterLabel2 = uniqueLabels[c.index(max(c))]

                    trackBinary[tc + 2 + i][
                        vidLabels[tc + 2 + i] == daughterLabel2
                    ] = 150

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
                contour = sm.measure.find_contours(
                    vidLabels[tc] == parentLabel, level=0
                )[0]
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
                        "Label": df4["Label"].iloc[0],
                        "Time": df4["Time"].iloc[0],
                        "Position": df4["Position"].iloc[0],
                        "Chain": df4["Chain"].iloc[0],
                        "Shape Factor": df4["Shape Factor"].iloc[0],
                        "Height": df4["Height"].iloc[0],
                        "Necleus Orientation": df4["Necleus Orientation"].iloc[0],
                        "Polygons": df4["Polygons"].iloc[0],
                        "Division Orientation": df4["Division Orientation"].iloc[0],
                        "Division While Wounded": df4["Division While Wounded"].iloc[0],
                        "Boundary Polygons": polygonsParent,
                        "Cytokineses time": tc,
                        "Time difference": tc - tm,
                        "Original Label": df4["Original Label"].iloc[0],
                        "Spot": df4["Spot"].iloc[0],
                    }
                )

                _dfDivisions4.append(
                    {
                        "Label": df4["Label"].iloc[1],
                        "Time": df4["Time"].iloc[1],
                        "Position": df4["Position"].iloc[1],
                        "Chain": df4["Chain"].iloc[1],
                        "Shape Factor": df4["Shape Factor"].iloc[1],
                        "Height": df4["Height"].iloc[1],
                        "Necleus Orientation": df4["Necleus Orientation"].iloc[1],
                        "Polygons": df4["Polygons"].iloc[1],
                        "Division Orientation": df4["Division Orientation"].iloc[1],
                        "Division While Wounded": df4["Division While Wounded"].iloc[1],
                        "Boundary Polygons": polygonsDaughter1,
                        "Cytokineses time": tc,
                        "Time difference": tc - tm,
                        "Original Label": df4["Original Label"].iloc[1],
                        "Spot": df4["Spot"].iloc[1],
                    }
                )

                _dfDivisions4.append(
                    {
                        "Label": df4["Label"].iloc[2],
                        "Time": df4["Time"].iloc[2],
                        "Position": df4["Position"].iloc[2],
                        "Chain": df4["Chain"].iloc[2],
                        "Shape Factor": df4["Shape Factor"].iloc[2],
                        "Height": df4["Height"].iloc[2],
                        "Necleus Orientation": df4["Necleus Orientation"].iloc[2],
                        "Polygons": df4["Polygons"].iloc[2],
                        "Division Orientation": df4["Division Orientation"].iloc[2],
                        "Division While Wounded": df4["Division While Wounded"].iloc[2],
                        "Boundary Polygons": polygonsDaughter2,
                        "Cytokineses time": tc,
                        "Time difference": tc - tm,
                        "Original Label": df4["Original Label"].iloc[2],
                        "Spot": df4["Spot"].iloc[2],
                    }
                )

            else:
                for i in range(3):
                    _dfDivisions4.append(
                        {
                            "Label": df4["Label"].iloc[i],
                            "Time": df4["Time"].iloc[i],
                            "Position": df4["Position"].iloc[i],
                            "Chain": df4["Chain"].iloc[i],
                            "Shape Factor": df4["Shape Factor"].iloc[i],
                            "Height": df4["Height"].iloc[i],
                            "Necleus Orientation": df4["Necleus Orientation"].iloc[i],
                            "Polygons": df4["Polygons"].iloc[i],
                            "Division Orientation": df4["Division Orientation"].iloc[i],
                            "Division While Wounded": df4[
                                "Division While Wounded"
                            ].iloc[i],
                            "Original Label": df4["Original Label"].iloc[i],
                            "Spot": df4["Spot"].iloc[i],
                        }
                    )

        dfDivisions4 = pd.DataFrame(_dfDivisions4)

        dfDivisions4 = dfDivisions4.sort_values(
            ["Label", "Original Label"], ascending=[True, True]
        )

        dfDivisions4.to_pickle(f"dat/{filename}/mitosisTracks{filename}.pkl")

        trackBinary = np.asarray(trackBinary, "uint8")
        tifffile.imwrite(f"dat/{filename}/tracks{filename}.tif", trackBinary)

        # display divisions

        vid = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(float)

        [T, X, Y, rgb] = vid.shape

        highlightDivisions = np.zeros([T, 552, 552, 3])

        for x in range(X):
            for y in range(Y):
                highlightDivisions[:, 20 + x, 20 + y, :] = vid[:, x, y, :]

        uniqueLabels = list(set(dfDivisions4["Label"]))

        for label in uniqueLabels:

            if label == 52:
                print("d")

            dfTrack = dfDivisions4.loc[
                lambda dfDivisions4: dfDivisions4["Label"] == label, :
            ]

            t0 = dfTrack.iloc[0]["Time"][-1]
            [x, y] = dfTrack.iloc[0]["Position"][-1]
            x = int(x)
            y = int(y)

            [rr0, cc0] = circle_perimeter(
                551 - (y + 20), x + 20, 13
            )  # change to column row coord
            [rr1, cc1] = circle_perimeter(551 - (y + 20), x + 20, 14)
            [rr2, cc2] = circle_perimeter(551 - (y + 20), x + 20, 15)

            times = range(t0 - 5, t0 + 5)

            timeVid = []
            for t in times:
                if t >= 0 and t <= T - 1:
                    timeVid.append(t)

            for t in timeVid:
                highlightDivisions[t][rr0, cc0, 2] = 200
                highlightDivisions[t][rr1, cc1, 2] = 200
                highlightDivisions[t][rr2, cc2, 2] = 200

        highlightDivisions = highlightDivisions[:, 20:532, 20:532]

        highlightDivisions = np.asarray(highlightDivisions, "uint8")
        tifffile.imwrite(f"dat/{filename}/divisions{filename}.tif", highlightDivisions)
