import os
import shutil
from math import dist, floor, log10

from collections import Counter
import cv2
import matplotlib
from matplotlib import markers
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy as sp
import scipy.linalg as linalg
from scipy.stats import pearsonr
import shapely
import skimage as sm
import skimage.io
import skimage.measure
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
from sympy import true
import tifffile
from skimage.draw import circle_perimeter
from scipy import optimize
import xml.etree.ElementTree as et
from pySTARMA import starma_model
from pySTARMA import stacf_stpacf
import matplotlib.colors as colors

import cellProperties as cell
import utils as util

plt.rcParams.update({"font.size": 12})

# -------------------

filenames, fileType = util.getFilesType()
scale = 123.26 / 512
T = 93

# Cell Behaviers

if False:
    _df = []
    for filename in filenames:
        dfDivision = pd.read_pickle(f"dat/{filename}/dfDivision{filename}.pkl")
        dfShape = pd.read_pickle(f"dat/{filename}/shape{filename}.pkl")
        Q = np.mean(dfShape["q"])
        theta0 = 0.5 * np.arctan2(Q[1, 0], Q[0, 0])

        t0 = util.findStartTime(filename)
        if "Wound" in filename:
            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
            dist = sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)
            for i in range(len(dfDivision)):
                t = dfDivision["T"].iloc[i]
                (x_w, y_w) = dfWound["Position"].iloc[t]
                x = dfDivision["X"].iloc[i]
                y = dfDivision["Y"].iloc[i]
                ori = (dfDivision["Orientation"].iloc[i] - theta0 * 180 / np.pi) % 180
                theta = (np.arctan2(y - y_w, x - x_w) - theta0) * 180 / np.pi
                ori_w = (ori - theta) % 180
                if ori > 90:
                    ori = 180 - ori
                if ori_w > 90:
                    ori_w = 180 - ori_w
                theta = (np.arctan2(y - y_w, x - x_w) - theta0) * 180 / np.pi
                r = dist[t, 512 - y, x]
                _df.append(
                    {
                        "Filename": filename,
                        "Label": dfDivision["Label"].iloc[i],
                        "T": int(t0 + t * 2),  # frames are taken every 2 minutes
                        "X": x * scale,
                        "Y": y * scale,
                        "R": r * scale,
                        "Theta": theta % 360,
                        "Orientation": ori,
                        "Orientation Wound": ori_w,
                    }
                )
        else:
            dfVelocityMean = pd.read_pickle(f"databases/dfVelocityMean{fileType}.pkl")
            dfFilename = dfVelocityMean[
                dfVelocityMean["Filename"] == filename
            ].reset_index()
            for i in range(len(dfDivision)):
                t = dfDivision["T"].iloc[i]
                mig = np.sum(
                    np.stack(np.array(dfFilename.loc[:t, "v"]), axis=0), axis=0
                )
                xc = 256 * scale + mig[0]
                yc = 256 * scale + mig[1]
                x = dfDivision["X"].iloc[i] * scale
                y = dfDivision["Y"].iloc[i] * scale
                r = ((xc - x) ** 2 + (yc - y) ** 2) ** 0.5
                ori = (dfDivision["Orientation"].iloc[i] - theta0 * 180 / np.pi) % 180
                if ori > 90:
                    ori = 180 - ori
                theta = (np.arctan2(y - yc, x - xc) - theta0) * 180 / np.pi
                ori_w = (ori - theta) % 180
                if ori_w > 90:
                    ori_w = 180 - ori_w
                theta = (np.arctan2(y - yc, x - xc) - theta0) * 180 / np.pi
                _df.append(
                    {
                        "Filename": filename,
                        "Label": dfDivision["Label"].iloc[i],
                        "T": int(t0 + t * 2),  # frames are taken every t2 minutes
                        "X": x,
                        "Y": y,
                        "R": r,
                        "Theta": theta % 360,
                        "Orientation": ori,
                        "Orientation Wound": ori_w,
                    }
                )

    dfDivisions = pd.DataFrame(_df)
    dfDivisions.to_pickle(f"databases/dfDivisions{fileType}.pkl")

if False:
    _df2 = []
    _df = []
    for filename in filenames:

        df = pd.read_pickle(f"dat/{filename}/shape{filename}.pkl")
        Q = np.mean(df["q"])
        theta0 = np.arccos(Q[0, 0] / (Q[0, 0] ** 2 + Q[0, 1] ** 2) ** 0.5) / 2
        R = util.rotation_matrix(-theta0)

        df = pd.read_pickle(f"dat/{filename}/nucleusVelocity{filename}.pkl")
        mig = np.zeros(2)

        for t in range(T):
            dft = df[df["T"] == t]
            v = np.mean(dft["Velocity"]) * scale
            v = np.matmul(R, v)
            _df.append(
                {
                    "Filename": filename,
                    "T": t,
                    "v": v,
                }
            )

            for i in range(len(dft)):
                x = dft["X"].iloc[i] * scale
                y = dft["Y"].iloc[i] * scale
                dv = np.matmul(R, dft["Velocity"].iloc[i] * scale) - v
                [x, y] = np.matmul(R, np.array([x, y]))

                _df2.append(
                    {
                        "Filename": filename,
                        "T": t,
                        "X": x - mig[0],
                        "Y": y - mig[1],
                        "dv": dv,
                    }
                )
            mig += v

    dfVelocityMean = pd.DataFrame(_df)
    dfVelocityMean.to_pickle(f"databases/dfVelocityMean{fileType}.pkl")
    dfVelocity = pd.DataFrame(_df2)
    dfVelocity.to_pickle(f"databases/dfVelocity{fileType}.pkl")


if False:
    _df2 = []
    dfVelocity = pd.read_pickle(f"databases/dfVelocity{fileType}.pkl")
    dfVelocityMean = pd.read_pickle(f"databases/dfVelocityMean{fileType}.pkl")
    for filename in filenames:

        df = pd.read_pickle(f"dat/{filename}/shape{filename}.pkl")
        dfFilename = dfVelocityMean[dfVelocityMean["Filename"] == filename]
        mig = np.zeros(2)
        Q = np.mean(df["q"])
        theta0 = np.arctan2(Q[0, 1], Q[0, 0]) / 2
        R = util.rotation_matrix(-theta0)

        for t in range(T):
            dft = df[df["Time"] == t]
            Q = np.matmul(R, np.matmul(np.mean(dft["q"]), np.matrix.transpose(R)))
            P = np.matmul(R, np.mean(dft["Polar"]))

            for i in range(len(dft)):
                [x, y] = [
                    dft["Centroid"].iloc[i][0] * scale,
                    dft["Centroid"].iloc[i][1] * scale,
                ]
                q = np.matmul(R, np.matmul(dft["q"].iloc[i], np.matrix.transpose(R)))
                dq = q - Q
                A = dft["Area"].iloc[i] * scale ** 2
                TrQdq = np.trace(np.matmul(Q, dq))
                dp = np.matmul(R, dft["Polar"].iloc[i]) - P
                [x, y] = np.matmul(R, np.array([x, y]))
                p = np.matmul(R, dft["Polar"].iloc[i])

                _df2.append(
                    {
                        "Filename": filename,
                        "T": t,
                        "X": x - mig[0],
                        "Y": y - mig[1],
                        "Centroid": np.array(dft["Centroid"].iloc[i]) * scale,
                        "dq": dq,
                        "q": q,
                        "TrQdq": TrQdq,
                        "Area": A,
                        "dp": dp,
                        "Polar": p,
                    }
                )

            mig += np.array(dfFilename["v"][dfFilename["T"] == t])[0]

    dfShape = pd.DataFrame(_df2)
    dfShape.to_pickle(f"databases/dfShape{fileType}.pkl")


# Cell Behaviers relative to wound

if False:
    _df2 = []
    for filename in filenames:

        if "Wound" in filename:
            df = pd.read_pickle(f"dat/{filename}/shape{filename}.pkl")
            Q = np.mean(df["q"])
            theta = np.arctan2(Q[0, 1], Q[0, 0]) / 2
            R = util.rotation_matrix(-theta)
            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
            dist = sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)
            t0 = util.findStartTime(filename)

            for t in range(T):
                dft = df[df["Time"] == t]
                Q = np.matmul(R, np.matmul(np.mean(dft["q"]), np.matrix.transpose(R)))
                P = np.matmul(R, np.mean(dft["Polar"]))
                xw, yw = dfWound["Position"].iloc[t]

                for i in range(len(dft)):
                    x = dft["Centroid"].iloc[i][0]
                    y = dft["Centroid"].iloc[i][1]
                    r = dist[t, int(512 - y), int(x)]
                    phi = np.arctan2(y - yw, x - xw)
                    Rw = util.rotation_matrix(-phi)

                    q = np.matmul(
                        R, np.matmul(dft["q"].iloc[i], np.matrix.transpose(R))
                    )
                    dq = q - Q
                    dq = np.matmul(Rw, np.matmul(dq, np.matrix.transpose(Rw)))

                    A = dft["Area"].iloc[i] * scale ** 2
                    dp = np.matmul(R, dft["Polar"].iloc[i]) - P
                    dp = np.matmul(Rw, np.matmul(dp, np.matrix.transpose(Rw)))

                    _df2.append(
                        {
                            "Filename": filename,
                            "T": int(2 * t + t0),  # frames are taken every 2 minutes
                            "X": x * scale,
                            "Y": y * scale,
                            "R": r * scale,
                            "Phi": phi,
                            "Area": A,
                            "dq": dq,
                            "dp": dp,
                        }
                    )
        else:
            t0 = 0
            dfVelocityMean = pd.read_pickle(f"databases/dfVelocityMean{fileType}.pkl")
            dfFilename = dfVelocityMean[
                dfVelocityMean["Filename"] == filename
            ].reset_index()

            df = pd.read_pickle(f"dat/{filename}/shape{filename}.pkl")
            Q = np.mean(df["q"])
            theta = np.arctan2(Q[0, 1], Q[0, 0]) / 2
            R = util.rotation_matrix(-theta)
            dist = sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)

            for t in range(T):
                dft = df[df["Time"] == t]
                Q = np.matmul(R, np.matmul(np.mean(dft["q"]), np.matrix.transpose(R)))
                P = np.matmul(R, np.mean(dft["Polar"]))
                mig = np.sum(
                    np.stack(np.array(dfFilename.loc[:t, "v"]), axis=0), axis=0
                )
                xw = 256 + mig[0] / scale
                yw = 256 + mig[1] / scale

                for i in range(len(dft)):
                    x = dft["Centroid"].iloc[i][0]
                    y = dft["Centroid"].iloc[i][1]
                    if t > 89:
                        tdash = 89
                    else:
                        tdash = t
                    r = dist[tdash, int(512 - y), int(x)]
                    phi = np.arctan2(y - yw, x - xw)
                    Rw = util.rotation_matrix(-phi)

                    q = np.matmul(
                        R, np.matmul(dft["q"].iloc[i], np.matrix.transpose(R))
                    )
                    dq = q - Q
                    dq = np.matmul(Rw, np.matmul(dq, np.matrix.transpose(Rw)))

                    A = dft["Area"].iloc[i] * scale ** 2
                    dp = np.matmul(R, dft["Polar"].iloc[i]) - P
                    dp = np.matmul(Rw, np.matmul(dp, np.matrix.transpose(Rw)))

                    _df2.append(
                        {
                            "Filename": filename,
                            "T": int(2 * t + t0),  # frames are taken every 2 minutes
                            "X": x * scale,
                            "Y": y * scale,
                            "R": r * scale,
                            "Phi": phi,
                            "Area": A,
                            "dq": dq,
                            "dp": dp,
                        }
                    )

    dfShape = pd.DataFrame(_df2)
    dfShape.to_pickle(f"databases/dfShapeWound{fileType}.pkl")


if True:
    _df2 = []
    for filename in filenames:

        if "Wound" in filename:
            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
            dist = sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)
            t0 = util.findStartTime(filename)
            df = pd.read_pickle(f"dat/{filename}/nucleusVelocity{filename}.pkl")

            for t in range(T):
                dft = df[df["T"] == t]
                xw, yw = dfWound["Position"].iloc[t]
                V = np.mean(dft["Velocity"])

                for i in range(len(dft)):
                    x = dft["X"].iloc[i]
                    y = dft["Y"].iloc[i]
                    r = dist[t, int(511 - y), int(x)]
                    phi = np.arctan2(y - yw, x - xw)
                    R = util.rotation_matrix(-phi)

                    v = np.matmul(R, dft["Velocity"].iloc[i]) / 2
                    dv = np.matmul(R, dft["Velocity"].iloc[i] - V) / 2

                    _df2.append(
                        {
                            "Filename": filename,
                            "T": int(2 * t + t0),  # frames are taken every 2 minutes
                            "X": x * scale,
                            "Y": y * scale,
                            "R": r * scale,
                            "Phi": phi,
                            "v": -v,
                            "dv": -dv,
                        }
                    )
        else:
            t0 = 0
            dfVelocityMean = pd.read_pickle(f"databases/dfVelocityMean{fileType}.pkl")
            df = pd.read_pickle(f"dat/{filename}/nucleusVelocity{filename}.pkl")
            dfFilename = dfVelocityMean[
                dfVelocityMean["Filename"] == filename
            ].reset_index()
            dist = sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)

            for t in range(T):
                dft = df[df["T"] == t]
                mig = np.sum(
                    np.stack(np.array(dfFilename.loc[:t, "v"]), axis=0), axis=0
                )
                xw = 256 + mig[0] / scale
                yw = 256 + mig[1] / scale
                V = np.mean(dft["Velocity"])

                for i in range(len(dft)):
                    x = dft["X"].iloc[i]
                    y = dft["Y"].iloc[i]
                    if t > 89:
                        tdash = 89
                    else:
                        tdash = t
                    r = dist[tdash, int(511 - y), int(x)]
                    phi = np.arctan2(y - yw, x - xw)
                    R = util.rotation_matrix(-phi)

                    v = np.matmul(R, dft["Velocity"].iloc[i]) / 2
                    dv = np.matmul(R, dft["Velocity"].iloc[i] - V) / 2

                    _df2.append(
                        {
                            "Filename": filename,
                            "T": int(2 * t + t0),  # frames are taken every 2 minutes
                            "X": x * scale,
                            "Y": y * scale,
                            "R": r * scale,
                            "Phi": phi,
                            "v": -v * scale,
                            "dv": -dv * scale,
                        }
                    )

    dfVelocity = pd.DataFrame(_df2)
    dfVelocity.to_pickle(f"databases/dfVelocityWound{fileType}.pkl")


# Cells Divsions and Shape changes


if False:
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


if False:
    _df2 = []
    for filename in filenames:

        dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
        dist = sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)
        t0 = util.findStartTime(filename)
        df = pd.read_pickle(f"dat/{filename}/nucleusVelocity{filename}.pkl")

        for t in range(T):
            dft = df[df["T"] == t]
            xw, yw = dfWound["Position"].iloc[t]
            V = np.mean(dft["Velocity"])

            for i in range(len(dft)):
                x = dft["X"].iloc[i]
                y = dft["Y"].iloc[i]
                r = dist[t, int(x), int(y)]
                phi = np.arctan2(y - yw, x - xw)
                R = util.rotation_matrix(-phi)

                v = np.matmul(R, dft["Velocity"].iloc[i]) / 2
                dv = np.matmul(R, dft["Velocity"].iloc[i] - V) / 2

                _df2.append(
                    {
                        "Filename": filename,
                        "T": int(2 * t + t0),  # frames are taken every 2 minutes
                        "X": x * scale,
                        "Y": y * scale,
                        "R": r * scale,
                        "Phi": phi,
                        "v": -v * scale,
                        "dv": -dv * scale,
                    }
                )

    dfVelocity = pd.DataFrame(_df2)
    dfVelocity.to_pickle(f"databases/dfVelocityWound{fileType}.pkl")