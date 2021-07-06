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
import shapely
import skimage as sm
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

plt.rcParams.update({"font.size": 16})

# -------------------

filenames, fileType = cl.getFilesType()

T = 181
scale = 147.91 / 512

count = 0

if False:
    fig = plt.figure(1, figsize=(9, 8))
    time = range(T)
    for filename in filenames:

        df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")
        count += len(df)

        sf0 = np.mean(list(df["Shape Factor"][df["Time"] == 0]))

        mu = []
        err = []

        for t in time:
            prop = list(df["Shape Factor"][df["Time"] == t])
            mu.append(np.mean(prop))
            err.append(np.std(prop) / len(prop) ** 0.5)

        plt.plot(time, mu)

    plt.xlabel("Time")
    plt.ylabel(f"Shape Factor relative")
    fig.savefig(
        f"results/Shape Factor",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

# ----------------------------


if False:
    fig = plt.figure(1, figsize=(9, 8))
    time = range(T)
    for filename in filenames:

        df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")
        A0 = np.mean(list(df["Area"][df["Time"] == 0] * scale ** 2))

        mu = []
        err = []

        for t in time:
            prop = list(df["Area"][df["Time"] == t] * scale ** 2)
            mu.append(np.mean(prop) / A0)
            err.append(np.std(prop) / len(prop) ** 0.5)

        plt.plot(time, mu)
        mu = np.array(mu)

    plt.xlabel("Time")
    plt.ylabel("Area relative")
    fig.savefig(
        f"results/Area",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


# ----------------------------


if False:
    df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")

    iso = []
    for t in range(T):
        prop = list(df[f"Q"][df["Time"] == t])
        Ori = []
        for i in range(len(prop)):

            Q = prop[i]
            v1 = Q[0]
            x = v1[0]
            y = v1[1]

            c = (x ** 2 + y ** 2) ** 0.5

            if x == 0 and y == 0:
                continue
            else:
                Ori.append(np.array([x, y]) / c)

        n = len(Ori)

        OriDash = sum(Ori) / n

        rho = ((OriDash[0]) ** 2 + (OriDash[1]) ** 2) ** 0.5

        OriSigma = sum(((Ori - OriDash) ** 2) / n) ** 0.5

        OriSigma = sum(OriSigma)

        iso.append(rho / OriSigma)

    x = range(T)
    fig = plt.figure(1, figsize=(9, 8))
    plt.gcf().subplots_adjust(left=0.2)
    plt.plot(x, iso)

    plt.xlabel("Time")
    plt.ylabel(f"isotopy of the tissue")
    plt.gcf().subplots_adjust(bottom=0.2)
    fig.savefig(
        f"results/Orientation of {filename}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


_dfEntropy = []


def rotation_matrix(theta):

    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    return R


if False:
    q0 = []
    dQr12 = []
    dQr22 = []
    mseq0 = []
    msedQr12 = []
    msedQr22 = []
    _df = []
    for filename in filenames:
        df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")

        T = np.linspace(0, 170, 18)
        for t in T:

            dft = cl.sortTime(df, [t, t + 10])
            m = len(dft)

            q = []
            for i in range(m):
                q.append(df["q"].iloc[i])

            Q = np.mean(q, axis=0)
            thetastar = np.arctan2(Q[0, 1], Q[0, 0])
            q0 = (2 * Q[0, 0] ** 2 + 2 * Q[0, 1] ** 2) ** 0.5

            R = rotation_matrix(thetastar)

            qr = np.matmul(R.transpose(), q)
            Qr = np.matmul(R.transpose(), Q)

            dQr = qr - Qr

            dQr1 = []
            dQr2 = []
            for i in range(m):
                dQr1.append(dQr[i][0, 0])
                dQr2.append(dQr[i][0, 1])
            dQr1 = np.array(dQr1)
            dQr2 = np.array(dQr2)

            _df.append(
                {
                    "filename": filename,
                    "q0": q0,
                    "dQr1": np.mean(dQr1 ** 2),
                    "dQr2": np.mean(dQr2 ** 2),
                    "T": t,
                }
            )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/dfq0_dq1_dq2{fileType}.pkl")
else:
    df = pd.read_pickle(f"databases/dfq0_dq1_dq2{fileType}.pkl")


if True:

    T = np.linspace(0, 170, 18)
    q0 = []
    dQr1 = []
    dQr2 = []
    mseq0 = []
    msedQr1 = []
    msedQr2 = []
    n = len(filenames)
    for t in T:
        q0.append(np.mean(df["q0"][df["T"] == t]))
        dQr1.append(np.mean(df["dQr1"][df["T"] == t]))
        dQr2.append(np.mean(df["dQr2"][df["T"] == t]))
        mseq0.append(np.std(df["q0"][df["T"] == t]) / n ** 0.5)
        msedQr1.append(np.std(df["dQr1"][df["T"] == t]) / n ** 0.5)
        msedQr2.append(np.std(df["dQr2"][df["T"] == t]) / n ** 0.5)

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    ax[0].errorbar(T, q0, yerr=mseq0)
    ax[0].set_xlabel("Time (mins)")
    ax[0].set_ylabel(r"$q_0$")
    ax[0].set_ylim([0, 0.04])

    ax[1].errorbar(T, dQr1, yerr=msedQr1, label=r"$(\delta q_1)^2$")
    ax[1].errorbar(T, dQr2, yerr=msedQr2, label=r"$(\delta q_2)^2$")
    ax[1].legend()
    ax[1].set_xlabel("Time (mins)")
    ax[1].set_ylabel(r"$(\delta q_i)^2$")
    ax[1].set_ylim([0.0005, 0.001])

    plt.suptitle(f"Shape with time {fileType}")

    fig.savefig(
        f"results/q0_dq1_dq2 {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")