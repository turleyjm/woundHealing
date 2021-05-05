import os
import shutil
from math import floor, log10

from collections import Counter
import cv2
import matplotlib
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy as sp
import scipy.linalg as linalg
from scipy.stats import mannwhitneyu
import shapely
import skimage as sm
import skimage.feature
import skimage.io
import skimage.measure
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
import tifffile
from skimage.draw import circle_perimeter
from scipy import optimize
import xml.etree.ElementTree as et

import cellProperties as cell
import findGoodCells as fi
import commonLiberty as cl

# -------------------


def orientationMean(polygons):

    n = len(polygons)

    I = np.zeros(shape=(2, 2))
    for i in range(n):
        I += cell.inertia(polygons[i])

    D, V = linalg.eig(I / n)
    e1 = D[0]
    e2 = D[1]
    v1 = V[:, 0]
    v2 = V[:, 1]
    if e1 < e2:
        v = v1
    else:
        v = v2
    theta = np.arctan(v[1] / v[0])
    if theta < 0:
        theta = theta + np.pi
    if theta > np.pi:
        theta = theta - np.pi
    return theta


def shapeFactorMean(polygons):

    n = len(polygons)

    I = np.zeros(shape=(2, 2))
    for i in range(n):
        I += cell.inertia(polygons[i])

    D = linalg.eig(I / n)[0]
    e1 = D[0]
    e2 = D[1]
    SF = abs((e1 - e2) / (e1 + e2))
    return SF


def shapeFactor_tcj20Tensor(tcj20Tensor):

    D = linalg.eig(tcj20Tensor)[0]
    e1 = D[0]
    e2 = D[1]
    sf20_tcj = abs((e1 - e2) / (e1 + e2))

    return sf20_tcj


def orientation_tcj20Tensor(tcj20Tensor):

    D, V = linalg.eig(tcj20Tensor)
    e1 = D[0]
    e2 = D[1]
    v1 = V[:, 0]
    v2 = V[:, 1]
    if e1 < e2:
        v = v1
    else:
        v = v2
    theta = np.arctan(v[1] / v[0])
    if theta < 0:
        theta = theta + np.pi
    if theta > np.pi:
        theta = theta - np.pi

    return theta


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


def findtcj(polygon, img):

    centroid = cell.centroid(polygon)
    x, y = int(centroid[0]), int(centroid[1])

    imgLabel = sm.measure.label(img, background=0, connectivity=1)
    label = imgLabel[x, y]
    contour = sm.measure.find_contours(imgLabel == label, level=0)[0]

    # imgLabelrc = fi.imgxyrc(imgLabel)
    # imgLabelrc[imgLabelrc == label] = round(1.25 * imgLabelrc.max())
    # imgLabelrc = np.asarray(imgLabelrc, "uint16")
    # tifffile.imwrite(f"results/imgLabel{filename}.tif", imgLabelrc)

    if isBoundary(contour) == True:
        return "False"
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
    # dilationrc = fi.imgxyrc(dilation)
    # dilationrc = np.asarray(dilationrc, "uint16")
    # tifffile.imwrite(f"results/dilation{filename}.tif", dilationrc)

    tcj = np.zeros([512, 512])
    diff = img - dilation
    tcj[diff == -1] = 1
    tcj[tcj != 1] = 0

    outerTCJ = skimage.feature.peak_local_max(tcj)
    # tcjrc = fi.imgxyrc(tcj)
    # tcjrc = np.asarray(tcjrc, "uint16")
    # tifffile.imwrite(f"results/tcj{filename}.tif", tcjrc)

    tcj = []
    for coord in outerTCJ:
        tcj.append(findtcjContour(coord, contour[0:-1]))

    if "False" in tcj:
        tcj.remove("False")
        print("removed")

    return tcj


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


def angleDiff(theta, phi):

    diff = theta - phi

    if abs(diff) > 90:
        if diff > 0:
            diff = 180 - diff
        else:
            diff = -180 - diff

    return abs(diff)


def findtcjMean(polygons, binarys):

    n = len(polygons)

    I = np.zeros(shape=(2, 2))
    for i in range(n):
        tcj = findtcj(polygons[i], binarys[i])
        if "False" in tcj:
            return I, True

        I += cell.inertia_tcj(tcj)

    return I / n, False


def dist(polygon, polygon0):
    [x1, y1] = cell.centroid(polygon)
    [x0, y0] = cell.centroid(polygon0)
    return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5


# -------------------

plt.rcParams.update({"font.size": 8})

filenames, fileType = cl.getFilesType()
scale = 147.91 / 512

_dfOrientation = []
run = False
if run:
    for filename in filenames:

        dfDivisions = pd.read_pickle(f"dat/{filename}/mitosisTracks{filename}.pkl")
        dfShape = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")
        meanArea = np.mean(dfShape["Area"])
        stdArea = np.std(dfShape["Area"])

        df = dfDivisions[dfDivisions["Chain"] == "parent"]
        vidFile = f"dat/{filename}/track{filename}.tif"
        tracks = sm.io.imread(vidFile).astype(int)
        T, X, Y, C = tracks.shape

        tracks = fi.imgrcxyRGB(tracks)
        tracksDivisions = tracks

        for i in range(len(df)):

            label = df["Label"].iloc[i]
            ori = df["Division Orientation"].iloc[i]
            # if ori > 90:
            #     ori = 180 - ori
            T = df["Time"].iloc[i]
            t = T[-1]
            tm = t
            [x, y] = df["Position"].iloc[i][-1]

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

                if t_i == 180:
                    finished = True
                    # tracksDivisions[t_i - 1][
                    #     np.all((tracks[t_i - 1] - colour) == 0, axis=2)
                    # ] = [0, 0, 0]

            tc = t_i - 1

            if tc > 30:
                T = range(tc - 29, tc + 1)
            else:
                T = range(1, tc + 1)

            polyList = []
            for t in T:
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

            if tc - tm < 10:
                _dfOrientation.append(
                    {
                        "Filename": filename,
                        "Label": label,
                        "Orientation": ori,
                        "T": t,
                        "Cytokineses Time": tc,
                        "Anaphase Time": tm,
                        "X": x,
                        "Y": y,
                        "Colour": colour,
                        "Polygons": polyList,
                    }
                )

        # tracksDivisions = fi.imgxyrcRGB(tracksDivisions)
        # tracksDivisions = np.asarray(tracksDivisions, "uint8")
        # tifffile.imwrite(
        #     f"dat/{filename}/tracksDivisions{filename}.tif", tracksDivisions
        # )

    dfOrientation = pd.DataFrame(_dfOrientation)

    _dfPolgyons = []

    for i in range(len(dfOrientation)):

        polyList = dfOrientation["Polygons"].iloc[i]

        T = len(polyList)
        t = 1 - T
        if False in polyList:
            continue
        elif polyList == []:
            continue
        else:
            Area0 = polyList[0].area
            sf0 = cell.shapeFactor(polyList[0])
            for polygon in polyList:

                _dfPolgyons.append(
                    {
                        "Filename": dfOrientation["Filename"].iloc[i],
                        "Label": dfOrientation["Label"].iloc[i],
                        "Orientation": dfOrientation["Orientation"].iloc[i],
                        "Cytokineses Time": dfOrientation["Cytokineses Time"].iloc[i],
                        "Anaphase Time": dfOrientation["Anaphase Time"].iloc[i],
                        "Polygon": polygon,
                        "Precytokineses Time": t,
                        "Area": polygon.area - Area0,
                        "Shape Factor": cell.shapeFactor(polygon) - sf0,
                        "Area Start": Area0,
                        "Shape Factor Start": sf0,
                        "Colour": dfOrientation["Colour"].iloc[i],
                    }
                )
                t += 1

    dfPolgyons = pd.DataFrame(_dfPolgyons)
    dfPolgyons.to_pickle(f"databases/dfPolgyons{fileType}.pkl")
else:
    dfPolgyons = pd.read_pickle(f"databases/dfPolgyons{fileType}.pkl")

run = False
if run:

    _df30 = []
    for filename in filenames:
        df = dfPolgyons[dfPolgyons["Filename"] == filename]
        labels = list(set(df["Label"]))
        labels.sort()

        for label in labels:
            n = len(df[df["Label"] == label])
            if n == 30:
                _df30.append(
                    {
                        "Area": np.array(df["Area"][df["Label"] == label]),
                        "Shape Factor": np.array(
                            df["Shape Factor"][df["Label"] == label]
                        ),
                        "Cytokineses Time": int(
                            df["Cytokineses Time"][df["Label"] == label].iloc[0]
                        ),
                        "Anaphase Time": int(
                            df["Anaphase Time"][df["Label"] == label].iloc[0]
                        ),
                        "Shape Factor Start": df["Shape Factor Start"][
                            df["Label"] == label
                        ].iloc[0],
                    }
                )

    df30 = pd.DataFrame(_df30)

    area = np.zeros([len(df30), 30])
    sf = np.zeros([len(df30), 30])
    sfL = np.zeros([len(df30), 30])
    j = 0
    for i in range(len(df30)):
        area[i] = df30["Area"].iloc[i]
        sf[i] = df30["Shape Factor"].iloc[i]
        sf0 = df30["Shape Factor Start"].iloc[i]
        if sf0 > 0.5:
            sfL[j] = df30["Shape Factor"].iloc[i]
            j += 1
    sfL = sfL[:j]

    muA = np.mean(area, axis=0) * scale ** 2
    sdA = np.std(area, axis=0) * scale ** 2
    muSf = np.mean(sf, axis=0)
    sdSf = np.std(sf, axis=0)
    muSfL = np.mean(sfL, axis=0)
    sdSfL = np.std(sfL, axis=0)

    t = np.array(range(-30, 0))

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(wspace=0.3)

    ax[0].errorbar(t, muA, yerr=sdA, marker="o")
    ax[0].set(xlabel="Time", ylabel=r"Area change $\mu m^2$")

    ax[1].errorbar(t, muSf, yerr=sdSf, marker="o")
    ax[1].set(xlabel="Time", ylabel="shape factor change")
    ax[1].set_ylim([-0.5, 0.3])

    ax[2].errorbar(t, muSfL, yerr=sdSfL, marker="o")
    ax[2].set(
        xlabel="Time", ylabel="shape factor change",
    )
    ax[2].set_ylim([-0.5, 0.3])
    ax[2].title.set_text(r"$s_f  > 0.5$")

    fig.savefig(
        f"results/change in cell shape and area before cytokineses {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

    areaMax = []
    sfMin = []
    sfLMin = []
    timeDif = []
    for i in range(len(df30)):
        area = list(df30["Area"].iloc[i])
        sf = list(df30["Shape Factor"].iloc[i])

        areaMax.append(area.index(max(area)) - 30)
        sfMin.append(sf.index(min(sf)) - 30)
        timeDif.append(
            df30["Anaphase Time"].iloc[i] - df30["Cytokineses Time"].iloc[i] - 1
        )

        sf0 = df30["Shape Factor Start"].iloc[i]
        if sf0 > 0.5:
            sfLMin.append(sf.index(min(sf)) - 30)

    fig, ax = plt.subplots(2, 2, figsize=(6, 6))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    ax[0, 0].hist(areaMax, bins=20)
    ax[0, 0].set(xlabel="Time", ylabel="Max Area")

    ax[0, 1].hist(
        timeDif, bins=20,
    )
    ax[0, 1].set(xlabel="Time", ylabel="Anaphase")

    ax[1, 0].hist(sfMin, bins=20, range=(-30, -1), density=True)
    ax[1, 0].set(xlabel="Time", ylabel="Min shape factor")
    ax[1, 0].set_ylim([0, 0.2])

    ax[1, 1].hist(sfLMin, bins=20, range=(-30, -1), density=True)
    ax[1, 1].set(xlabel="Time", ylabel="Min shape factor")
    ax[1, 1].title.set_text(r"$s_f  > 0.5$")
    ax[1, 1].set_ylim([0, 0.2])

    fig.savefig(
        f"results/time properties cytokineses {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")

    # heatmapAS = np.zeros([30, 30])
    # heatmapAM = np.zeros([30, 30])
    # heatmapSM = np.zeros([30, 30])

    # for i in range(len(areaMax)):

    #     A = 30 + areaMax[i]
    #     S = 30 + sfMin[i]
    #     M = 30 + timeDif[i]

    #     heatmapAS[A, S] += 1
    #     heatmapAM[A, M] += 1
    #     heatmapSM[S, M] += 1

    # fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    # plt.subplots_adjust(wspace=0.3)
    # plt.gcf().subplots_adjust(bottom=0.15)

    # dx, dy = 1, 1
    # x, y = np.mgrid[-30:0:dx, -30:0:dy]

    # c = ax[0].pcolor(x, y, heatmapAS, cmap="Reds")
    # fig.colorbar(c, ax=ax[0])
    # ax[0].set(xlabel="Max Area", ylabel="Min shape factor")

    # c = ax[1].pcolor(x, y, heatmapAM, cmap="Reds")
    # fig.colorbar(c, ax=ax[1])
    # ax[1].set(xlabel="Max Area", ylabel="Anaphase")

    # c = ax[2].pcolor(x, y, heatmapSM, cmap="Reds")
    # fig.colorbar(c, ax=ax[2])
    # ax[2].set(xlabel="Min shape factor", ylabel="Anaphase")

    # fig.savefig(
    #     f"results/time properties cytokineses heatmap {fileType}",
    #     dpi=300,
    #     transparent=True,
    # )
    # plt.close("all")


_df20 = []
for filename in filenames:
    df = dfPolgyons[dfPolgyons["Filename"] == filename]
    labels = list(set(df["Label"]))
    labels.sort()

    for label in labels:
        n = len(df[df["Label"] == label])
        if n == 30:
            _df20.append(
                {
                    "Filename": filename,
                    "Label": label,
                    "Area": np.array(df["Area"][df["Label"] == label]),
                    "Shape Factor": np.array(df["Shape Factor"][df["Label"] == label]),
                    "Cytokineses Time": int(
                        df["Cytokineses Time"][df["Label"] == label].iloc[0]
                    ),
                    "Anaphase Time": int(
                        df["Anaphase Time"][df["Label"] == label].iloc[0]
                    ),
                    "Division Orientation": df["Orientation"][
                        df["Label"] == label
                    ].iloc[0],
                    "Polygon": list(df["Polygon"][df["Label"] == label]),
                }
            )

df20 = pd.DataFrame(_df20)


# All division
run = True
if run:
    _dftcj = []
    diffOri20 = []
    diffOriA = []
    diffOri20_tcj = []
    diffOriA_tcj = []
    for filename in filenames:
        df = df20[df20["Filename"] == filename]
        binary = sm.io.imread(f"dat/{filename}/ecadBinary{filename}.tif").astype(int)
        binary = fi.vidrcxy(255 - binary)
        for i in range(len(df)):
            divisionOri = df["Division Orientation"].iloc[i]

            polygons = df["Polygon"].iloc[i][0:10]
            t = [
                df["Cytokineses Time"].iloc[i] - 29,
                df["Cytokineses Time"].iloc[i] - 19,
            ]
            tcj20Tensor, boundary = findtcjMean(polygons, binary[t[0] : t[1]])

            shapeOri20 = orientationMean(polygons) * (180 / np.pi)
            sf20 = shapeFactorMean(polygons)

            polygon = df["Polygon"].iloc[i][
                29 - df["Cytokineses Time"].iloc[i] + df["Anaphase Time"].iloc[i]
            ]
            t0 = df["Anaphase Time"].iloc[i]
            tcjA = findtcj(polygon, binary[t0])

            shapeOriA = cell.orientation(polygon) * (180 / np.pi)
            sfA = cell.shapeFactor(polygon)

            if True != boundary:
                if "False" not in tcjA:
                    diffOri20.append(angleDiff(divisionOri, shapeOri20))
                    diffOriA.append(angleDiff(divisionOri, shapeOriA))

                    sf20_tcj = shapeFactor_tcj20Tensor(tcj20Tensor)
                    shapeOri20_tcj = orientation_tcj20Tensor(tcj20Tensor) * (
                        180 / np.pi
                    )
                    diffOri20_tcj.append(angleDiff(divisionOri, shapeOri20_tcj))

                    sfA_tcj = cell.shapeFactor_tcj(tcjA)
                    shapeOriA_tcj = cell.orientation_tcj(tcjA) * (180 / np.pi)
                    diffOriA_tcj.append(angleDiff(divisionOri, shapeOriA_tcj))

                    _dftcj.append(
                        {
                            "Filename": filename,
                            "Label": df["Label"].iloc[i],
                            "Division Orientation": divisionOri,
                            "Polygon": df["Polygon"].iloc[i],
                            "Anaphase Time": df["Anaphase Time"].iloc[i],
                            "Cytokineses Time": df["Cytokineses Time"],
                            "Pre-rounded up shape Orinentation": shapeOri20,
                            "Anaphase shape Orinentation": shapeOriA,
                            "Pre-rounded up tcj Orinentation": shapeOri20_tcj,
                            "Anaphase tcj Orinentation": shapeOriA_tcj,
                            "Pre-rounded up Shape Factor": sf20,
                            "Anaphase Shape Factor": sfA,
                            "Pre-rounded up tcj Shape Factor": sf20_tcj,
                            "Anaphase tcj Shape Factor": sfA_tcj,
                        }
                    )

    dftcj = pd.DataFrame(_dftcj)
    dftcj.to_pickle(f"databases/dftcj{fileType}.pkl")

    yMax = (
        max(
            [
                max(plt.hist(diffOri20, bins=9, range=[0, 90])[0]),
                max(plt.hist(diffOriA, bins=9, range=[0, 90])[0]),
                max(plt.hist(diffOri20_tcj, bins=9, range=[0, 90])[0]),
                max(plt.hist(diffOriA_tcj, bins=9, range=[0, 90])[0]),
            ]
        )
        * 1.1
    )

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    plt.suptitle(f"divsion orientation predictors {fileType}")

    ax[0, 0].hist(diffOri20, bins=9, range=[0, 90])
    ax[0, 0].set(xlabel="Orientation Diffence", ylabel="freqency")
    ax[0, 0].title.set_text(f"Pre round up orientation")
    ax[0, 0].set_ylim([0, yMax])

    ax[0, 1].hist(diffOriA, bins=9, range=[0, 90])
    ax[0, 1].set(xlabel="Orientation Diffence", ylabel="freqency")
    ax[0, 1].title.set_text(f"Anaphase orientation")
    ax[0, 1].set_ylim([0, yMax])

    ax[1, 0].hist(diffOri20_tcj, bins=9, range=[0, 90])
    ax[1, 0].set(xlabel="Orientation Diffence", ylabel="freqency")
    ax[1, 0].title.set_text(
        f"Pre round up TCJ orientation P={round(mannwhitneyu(diffOri20, diffOri20_tcj)[1],3)}"
    )
    ax[1, 0].set_ylim([0, yMax])

    ax[1, 1].hist(diffOriA_tcj, bins=9, range=[0, 90])
    ax[1, 1].set(xlabel="Orientation Diffence", ylabel="freqency")
    ax[1, 1].title.set_text(
        f"Anaphase TCJ orientation P={round(mannwhitneyu(diffOriA, diffOriA_tcj)[1],3)}"
    )
    ax[1, 1].set_ylim([0, yMax])

    fig.savefig(
        f"results/divsion orientation predictor {fileType}", dpi=300, transparent=True,
    )
    plt.close("all")
else:
    dftcj = pd.read_pickle(f"databases/dftcj{fileType}.pkl")


# low and high shape factor
run = True
if run:
    diffOri20 = []
    diffOriA = []
    diffOri20_tcj = []
    diffOriA_tcj = []
    for i in range(len(dftcj)):

        sf20 = dftcj["Pre-rounded up Shape Factor"].iloc[i]
        if sf20 < 0.15:
            divisionOri = dftcj["Division Orientation"].iloc[i]
            shapeOri20 = dftcj["Pre-rounded up shape Orinentation"].iloc[i]
            shapeOriA = dftcj["Anaphase shape Orinentation"].iloc[i]
            shapeOri20_tcj = dftcj["Pre-rounded up tcj Orinentation"].iloc[i]
            shapeOriA_tcj = dftcj["Anaphase tcj Orinentation"].iloc[i]

            diffOri20.append(angleDiff(divisionOri, shapeOri20))
            diffOriA.append(angleDiff(divisionOri, shapeOriA))
            diffOri20_tcj.append(angleDiff(divisionOri, shapeOri20_tcj))
            diffOriA_tcj.append(angleDiff(divisionOri, shapeOriA_tcj))

    yMax = (
        max(
            [
                max(plt.hist(diffOri20, bins=9, range=[0, 90])[0]),
                max(plt.hist(diffOriA, bins=9, range=[0, 90])[0]),
                max(plt.hist(diffOri20_tcj, bins=9, range=[0, 90])[0]),
                max(plt.hist(diffOriA_tcj, bins=9, range=[0, 90])[0]),
            ]
        )
        * 1.1
    )

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    plt.suptitle(f"divsion orientation predictors sf < 0.15 {fileType}")

    ax[0, 0].hist(diffOri20, bins=9, range=[0, 90])
    ax[0, 0].set(xlabel="Orientation Diffence", ylabel="freqency")
    ax[0, 0].title.set_text(f"Pre round up orientation")
    ax[0, 0].set_ylim([0, yMax])

    ax[0, 1].hist(diffOriA, bins=9, range=[0, 90])
    ax[0, 1].set(xlabel="Orientation Diffence", ylabel="freqency")
    ax[0, 1].title.set_text(f"Anaphase orientation")
    ax[0, 1].set_ylim([0, yMax])

    ax[1, 0].hist(diffOri20_tcj, bins=9, range=[0, 90])
    ax[1, 0].set(xlabel="Orientation Diffence", ylabel="freqency")
    ax[1, 0].title.set_text(
        f"Pre round up TCJ orientation P={round(mannwhitneyu(diffOri20, diffOri20_tcj)[1],3)}"
    )
    ax[1, 0].set_ylim([0, yMax])

    ax[1, 1].hist(diffOriA_tcj, bins=9, range=[0, 90])
    ax[1, 1].set(xlabel="Orientation Diffence", ylabel="freqency")
    ax[1, 1].title.set_text(
        f"Anaphase TCJ orientation P={round(mannwhitneyu(diffOriA, diffOriA_tcj)[1],3)}"
    )
    ax[1, 1].set_ylim([0, yMax])

    fig.savefig(
        f"results/divsion orientation predictor low sf {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

    diffOri20 = []
    diffOriA = []
    diffOri20_tcj = []
    diffOriA_tcj = []
    for i in range(len(dftcj)):

        sf20 = dftcj["Pre-rounded up Shape Factor"].iloc[i]
        if sf20 > 0.5:
            divisionOri = dftcj["Division Orientation"].iloc[i]
            shapeOri20 = dftcj["Pre-rounded up shape Orinentation"].iloc[i]
            shapeOriA = dftcj["Anaphase shape Orinentation"].iloc[i]
            shapeOri20_tcj = dftcj["Pre-rounded up tcj Orinentation"].iloc[i]
            shapeOriA_tcj = dftcj["Anaphase tcj Orinentation"].iloc[i]

            diffOri20.append(angleDiff(divisionOri, shapeOri20))
            diffOriA.append(angleDiff(divisionOri, shapeOriA))
            diffOri20_tcj.append(angleDiff(divisionOri, shapeOri20_tcj))
            diffOriA_tcj.append(angleDiff(divisionOri, shapeOriA_tcj))

    yMax = (
        max(
            [
                max(plt.hist(diffOri20, bins=9, range=[0, 90])[0]),
                max(plt.hist(diffOriA, bins=9, range=[0, 90])[0]),
                max(plt.hist(diffOri20_tcj, bins=9, range=[0, 90])[0]),
                max(plt.hist(diffOriA_tcj, bins=9, range=[0, 90])[0]),
            ]
        )
        * 1.1
    )

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    plt.suptitle(f"divsion orientation predictors sf > 0.5 {fileType}")

    ax[0, 0].hist(diffOri20, bins=9, range=[0, 90])
    ax[0, 0].set(xlabel="Orientation Diffence", ylabel="freqency")
    ax[0, 0].title.set_text(f"Pre round up orientation")
    ax[0, 0].set_ylim([0, yMax])

    ax[0, 1].hist(diffOriA, bins=9, range=[0, 90])
    ax[0, 1].set(xlabel="Orientation Diffence", ylabel="freqency")
    ax[0, 1].title.set_text(f"Anaphase orientation")
    ax[0, 1].set_ylim([0, yMax])

    ax[1, 0].hist(diffOri20_tcj, bins=9, range=[0, 90])
    ax[1, 0].set(xlabel="Orientation Diffence", ylabel="freqency")
    ax[1, 0].title.set_text(
        f"Pre round up TCJ orientation P={round(mannwhitneyu(diffOri20, diffOri20_tcj)[1],3)}"
    )
    ax[1, 0].set_ylim([0, yMax])

    ax[1, 1].hist(diffOriA_tcj, bins=9, range=[0, 90])
    ax[1, 1].set(xlabel="Orientation Diffence", ylabel="freqency")
    ax[1, 1].title.set_text(
        f"Anaphase TCJ orientation P={round(mannwhitneyu(diffOriA, diffOriA_tcj)[1],3)}"
    )
    ax[1, 1].set_ylim([0, yMax])

    fig.savefig(
        f"results/divsion orientation predictor high sf {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


run = True
if run:
    diffOri20 = []
    diffOriA = []
    diffOri20_tcj = []
    diffOriA_tcj = []
    for i in range(len(dftcj)):

        shapeOri20 = dftcj["Pre-rounded up shape Orinentation"].iloc[i]
        shapeOri20_tcj = dftcj["Pre-rounded up tcj Orinentation"].iloc[i]
        if angleDiff(shapeOri20, shapeOri20_tcj) > 15:
            divisionOri = dftcj["Division Orientation"].iloc[i]
            shapeOriA = dftcj["Anaphase shape Orinentation"].iloc[i]
            shapeOriA_tcj = dftcj["Anaphase tcj Orinentation"].iloc[i]

            diffOri20.append(angleDiff(divisionOri, shapeOri20))
            diffOriA.append(angleDiff(divisionOri, shapeOriA))
            diffOri20_tcj.append(angleDiff(divisionOri, shapeOri20_tcj))
            diffOriA_tcj.append(angleDiff(divisionOri, shapeOriA_tcj))

    yMax = (
        max(
            [
                max(plt.hist(diffOri20, bins=9, range=[0, 90])[0]),
                max(plt.hist(diffOriA, bins=9, range=[0, 90])[0]),
                max(plt.hist(diffOri20_tcj, bins=9, range=[0, 90])[0]),
                max(plt.hist(diffOriA_tcj, bins=9, range=[0, 90])[0]),
            ]
        )
        * 1.1
    )

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    plt.suptitle(
        f"divsion orientation predictors where predictors differ by 15 {fileType}"
    )

    ax[0, 0].hist(diffOri20, bins=9, range=[0, 90])
    ax[0, 0].set(xlabel="Orientation Diffence", ylabel="freqency")
    ax[0, 0].title.set_text(f"Pre round up orientation")
    ax[0, 0].set_ylim([0, yMax])

    ax[0, 1].hist(diffOriA, bins=9, range=[0, 90])
    ax[0, 1].set(xlabel="Orientation Diffence", ylabel="freqency")
    ax[0, 1].title.set_text(f"Anaphase orientation")
    ax[0, 1].set_ylim([0, yMax])

    ax[1, 0].hist(diffOri20_tcj, bins=9, range=[0, 90])
    ax[1, 0].set(xlabel="Orientation Diffence", ylabel="freqency")
    ax[1, 0].title.set_text(f"Pre round up TCJ orientation")
    ax[1, 0].set_ylim([0, yMax])

    ax[1, 1].hist(diffOriA_tcj, bins=9, range=[0, 90])
    ax[1, 1].set(xlabel="Orientation Diffence", ylabel="freqency")
    ax[1, 1].title.set_text(f"Anaphase TCJ orientation")
    ax[1, 1].set_ylim([0, yMax])

    fig.savefig(
        f"results/divsion orientation predictor high differ in prediction {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")
