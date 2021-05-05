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
from PIL import Image
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
import plotly.graph_objects as go

import cellProperties as cell
import findGoodCells as fi
import commonLiberty as cl

# -------------------


def sphere(shape, radius, position):
    # https://stackoverflow.com/questions/46626267/how-to-generate-a-sphere-in-3d-numpy-array
    # assume shape and position are both a 3-tuple of int or float
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    semisizes = (radius,) * 3

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        arr += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below 1
    return arr <= 1.0


def correlationFunction(x, y, t, r):

    count = 0

    n = len(x)
    for i in range(n):
        for j in range(n):
            if (
                (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (t[i] - t[j]) ** 0.5
            ) ** 0.5 > r and (
                (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (t[i] - t[j]) ** 0.5
            ) ** 0.5 <= r + 5:
                count += 1

    corr = 3 * count / (2 * np.pi * (3 * r ** 2 + 3 * r + 1))
    return corr


def inPlaneShell(t, x, y, r0, r1, outPlane):

    if r0 == 0:
        r0 = 1

    sphere0 = sphere((181, 148, 148), r0, (t, x, y)).astype("int")
    sphere1 = sphere((181, 148, 148), r1, (t, x, y)).astype("int")

    sphere1[sphere0 == 1] = 0
    sphere1[outPlane == 255] = 0

    return sphere1


plt.rcParams.update({"font.size": 16})

filenames, fileType = cl.getFilesType()
scale = 147.91 / 512

_dfSpaceTime = []

if True:
    for filename in filenames:

        dfDivisions = pd.read_pickle(f"dat/{filename}/mitosisTracks{filename}.pkl")
        df = dfDivisions[dfDivisions["Chain"] == "parent"]

        for i in range(len(df)):

            label = df["Label"].iloc[i]
            t = df["Time"].iloc[i][-1]
            [x, y] = df["Position"].iloc[i][-1]
            ori = df["Division Orientation"].iloc[i]

            _dfSpaceTime.append(
                {
                    "Filename": filename,
                    "Label": label,
                    "Orientation": ori,
                    "T": t,
                    "X": x * scale,
                    "Y": y * scale,
                }
            )

    dfSpaceTime = pd.DataFrame(_dfSpaceTime)
    dfSpaceTime.to_pickle(f"databases/dfSpaceTime{fileType}.pkl")
else:
    dfSpaceTime = pd.read_pickle(f"databases/dfSpaceTime{fileType}.pkl")

if False:
    for filename in filenames:
        df = dfSpaceTime[dfSpaceTime["Filename"] == filename]
        x = np.array(df["X"])
        y = np.array(df["Y"])
        t = np.array(df["T"])
        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=t, mode="markers")])
        fig.show()
        plt.close("all")


if False:
    x = np.array(dfSpaceTime["X"])
    y = np.array(dfSpaceTime["Y"])

    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    ax[0].hist(x, bins=10)
    ax[0].set(xlabel="x")

    ax[1].hist(y, bins=10)
    ax[1].set(xlabel="y")

    fig.savefig(
        f"results/xy distributions {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


# correction of divisions
if True:
    volume = [[[] for col in range(len(filenames))] for col in range(30)]
    count = [[[] for col in range(len(filenames))] for col in range(30)]
    thetaCorrelation = [[[] for col in range(len(filenames))] for col in range(30)]

    for k in range(len(filenames)):
        filename = filenames[k]
        divisions = np.zeros([181, 148, 148])
        orientations = np.zeros([181, 148, 148])
        outPlanePixel = sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype(
            float
        )
        outPlane = []
        for t in range(len(outPlanePixel)):
            img = Image.fromarray(outPlanePixel[t])
            outPlane.append(np.array(img.resize((148, 148))))
        outPlane = np.array(outPlane)
        outPlane[outPlane > 50] = 255
        outPlane[outPlane < 0] = 0

        df = dfSpaceTime[dfSpaceTime["Filename"] == filename]
        x = np.array(df["X"])
        y = np.array(df["Y"])
        t = np.array(df["T"])
        ori = np.array(df["Orientation"])

        for i in range(len(x)):
            divisions[int(t[i]), int(x[i]), int(y[i])] = 1
            orientations[int(t[i]), int(x[i]), int(y[i])] = ori[i]

        R = np.array(range(30)) * 5

        for i in range(len(R)):
            r0 = R[i]
            r1 = r0 + 5
            for j in range(len(x)):
                shell = inPlaneShell(int(t[j]), int(x[j]), int(y[j]), r0, r1, outPlane)
                volume[i][k].append(np.sum(shell))
                count[i][k].append(np.sum(divisions[shell == 1]))
                thetas = orientations[shell == 1]
                thetas = thetas[thetas != 0]
                if len(thetas) != 0:
                    corr = []
                    v = np.array(
                        [np.cos(np.pi * ori[j] / 90), np.sin(np.pi * ori[j] / 90)]
                    )
                    for theta in thetas:
                        u = np.array(
                            [np.cos(np.pi * theta / 90), np.sin(np.pi * theta / 90)]
                        )
                        corr.append(np.dot(v, u))
                    thetaCorrelation[i][k].append(np.mean(corr))

    correlation = []
    oriCorrelation = []
    for i in range(len(R)):
        correlation.append(np.sum(np.sum(count[i])) / np.sum(np.sum(volume[i])))
        oriCorrelation.append(np.mean(np.mean(thetaCorrelation[i])))

    fig = plt.figure(1, figsize=(9, 8))
    plt.plot(R + 2.5, correlation)
    plt.gcf().subplots_adjust(left=0.15)
    plt.xlabel(r"$R = ((t_i-t)^2 + (x_i-x)^2 + (y_i-y)^2)^{1/2}$")
    plt.ylabel(f"Correction")
    plt.ylim(0, max(correlation) * 1.1)
    plt.title("Division Correction of Data")
    fig.savefig(
        f"results/Division Correction",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

    fig = plt.figure(1, figsize=(9, 8))
    plt.plot(R + 2.5, oriCorrelation)
    plt.gcf().subplots_adjust(left=0.15)
    plt.xlabel(r"$R = ((t_i-t)^2 + (x_i-x)^2 + (y_i-y)^2)^{1/2}$")
    plt.ylabel(f"Correction")
    plt.ylim(-1, 1)
    plt.title("Division Orientation Correction of Data")
    fig.savefig(
        f"results/Division Orientation Correction",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

    volume = [[[] for col in range(len(filenames))] for col in range(30)]
    count = [[[] for col in range(len(filenames))] for col in range(30)]

    for k in range(len(filenames)):
        divisions = np.zeros([181, 148, 148])
        outPlane = np.zeros([181, 148, 148])
        df = dfSpaceTime[dfSpaceTime["Filename"] == filenames[k]]
        m = len(df)

        x = 148 * np.random.random_sample(m)
        y = 148 * np.random.random_sample(m)
        t = np.array(df["T"])

        for i in range(m):
            divisions[int(t[i]), int(x[i]), int(y[i])] = 1

        R = np.array(range(30)) * 5

        for i in range(len(R)):
            r0 = R[i]
            r1 = r0 + 5
            for j in range(m):
                shell = inPlaneShell(int(t[j]), int(x[j]), int(y[j]), r0, r1, outPlane)
                volume[i][k].append(np.sum(shell))
                count[i][k].append(np.sum(divisions[shell == 1]))

    correlationUni = []
    for i in range(len(R)):
        correlationUni.append(np.sum(np.sum(count[i])) / np.sum(np.sum(volume[i])))

    fig = plt.figure(1, figsize=(9, 8))
    plt.plot(R + 2.5, correlation, label="Data")
    plt.plot(R + 2.5, correlationUni, label="Uniform")
    plt.gcf().subplots_adjust(left=0.15)
    plt.xlabel(r"$R = ((t_i-t)^2 + (x_i-x)^2 + (y_i-y)^2)^{1/2}$")
    plt.ylabel(f"Correction")
    plt.ylim(0, max(correlation) * 1.1)
    plt.title("Division Correction")
    plt.legend()
    fig.savefig(
        f"results/Division Correction compared",
        dpi=300,
        transparent=True,
    )
    plt.close("all")
