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
import utils as util

plt.rcParams.update({"font.size": 12})

# -------------------

filenames, fileType = util.getFilesType()

T = 90
scale = 123.26 / 512


# -------------------

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

else:
    dfVelocity = pd.read_pickle(f"databases/dfVelocity{fileType}.pkl")
    dfVelocityMean = pd.read_pickle(f"databases/dfVelocityMean{fileType}.pkl")

if False:
    _df2 = []
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

else:
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")


# typical cell length
if False:
    A = []
    for t in range(T):
        A.append(np.mean(dfShape["Area"][dfShape["T"] == t] ** 0.5))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(2 * np.array(range(T)), A)
    ax.set(xlabel=r"Time", ylabel=r"Typical cell length $(\mu m)$")
    ax.title.set_text("Typical Cell Length")
    fig.savefig(
        f"results/Typical cell length {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

if False:
    Q1 = []
    Q1std = []
    for t in range(T):
        Q1.append(np.mean(dfShape["q"][dfShape["T"] == t])[0, 0])
        Q1std.append(np.std(np.stack(dfShape["q"][dfShape["T"] == t], axis=0)[:, 0, 0]))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(wspace=0.3)
    ax[0].errorbar(2 * np.array(range(T)), Q1, yerr=Q1std)
    ax[0].set(xlabel=r"Time", ylabel=r"$\bar{Q}^{(1)}$")
    ax[0].title.set_text(r"Mean of $Q^{(1)}$")
    ax[0].set_ylim([-0.03, 0.05])

    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        Q1 = []
        for t in range(T):
            Q1.append(np.mean(df["q"][df["T"] == t])[0, 0])

        ax[1].plot(2 * np.array(range(T)), Q1)

    ax[1].set(xlabel=r"Time", ylabel=r"$\bar{Q}^{(1)}$")
    ax[1].title.set_text(r"Mean of $Q^{(1)}$ indivial videos")
    ax[1].set_ylim([-0.03, 0.05])

    fig.savefig(
        f"results/mean Q1 {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

if False:
    Q2 = []
    Q2std = []
    for t in range(T):
        Q2.append(np.mean(dfShape["q"][dfShape["T"] == t])[0, 1])
        Q2std.append(np.std(np.stack(dfShape["q"][dfShape["T"] == t], axis=0)[:, 0, 1]))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(wspace=0.3)
    ax[0].errorbar(2 * np.array(range(T)), Q2, yerr=Q2std)
    ax[0].set(xlabel=r"Time", ylabel=r"$\bar{Q}^{(2)}$")
    ax[0].title.set_text(r"Mean of $Q^{(2)}$")
    ax[0].set_ylim([-0.03, 0.05])

    ax[1].set_ylim([-0.025, 0.05])
    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        Q2 = []
        for t in range(T):
            Q2.append(np.mean(df["q"][df["T"] == t])[0, 1])

        ax[1].plot(2 * np.array(range(T)), Q2)

    ax[1].set(xlabel=r"Time", ylabel=r"$\bar{Q}^{(2)}$")
    ax[1].title.set_text(r"Mean of $Q^{(2)}$ indivial videos")
    ax[1].set_ylim([-0.03, 0.05])

    fig.savefig(
        f"results/mean Q2 {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


if False:
    Q1 = []
    Q2 = []
    Q2std = []
    for t in range(T):
        Q1.append(np.mean(dfShape["q"][dfShape["T"] == t])[0, 0])
        Q2.append(np.mean(dfShape["q"][dfShape["T"] == t])[0, 1])
        Q2std.append(np.std(np.stack(dfShape["q"][dfShape["T"] == t], axis=0)[:, 0, 1]))

    Q1max = np.max(Q1)
    Q2 = np.array(Q2) / Q1max
    Q2std = np.array(Q2std) / Q1max

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(wspace=0.3)
    ax[0].errorbar(2 * np.array(range(T)), Q2, yerr=Q2std)
    ax[0].set(xlabel=r"Time", ylabel=r"$\bar{Q}^{(2)}/\bar{Q}^{(1)}$")
    ax[0].title.set_text(r"Mean of $Q^{(2)}$")
    ax[0].set_ylim([-0.5, 1])

    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        Q2 = []
        for t in range(T):
            Q2.append(np.mean(df["q"][df["T"] == t])[0, 1])

        Q2 = np.array(Q2) / Q1max
        ax[1].plot(2 * np.array(range(T)), Q2)

    ax[1].set(xlabel=r"Time", ylabel=r"$\bar{Q}^{(2)}/\bar{Q}^{(1)}$")
    ax[1].title.set_text(r"Mean of $Q^{(2)}$ indivial videos")
    ax[1].set_ylim([-0.5, 1])

    fig.savefig(
        f"results/mean Q2 over Q1 {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


if False:
    P1 = []
    P1std = []
    for t in range(T):
        P1.append(np.mean(dfShape["Polar"][dfShape["T"] == t])[0])
        P1std.append(
            np.std(np.stack(dfShape["Polar"][dfShape["T"] == t], axis=0)[:, 0])
        )

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(wspace=0.3)
    ax[0].errorbar(2 * np.array(range(T)), P1, yerr=P1std)
    ax[0].set(xlabel=r"Time", ylabel=r"$\bar{P}_1$")
    ax[0].title.set_text(r"Mean of $P_1$")
    ax[0].set_ylim([-0.01, 0.01])

    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        P1 = []
        for t in range(T):
            P1.append(np.mean(df["Polar"][df["T"] == t])[0])

        ax[1].plot(2 * np.array(range(T)), P1)

    ax[1].set(xlabel=r"Time", ylabel=r"$\bar{P}_1$")
    ax[1].title.set_text(r"Mean of $P_1$ indivial videos")
    ax[1].set_ylim([-0.01, 0.01])

    fig.savefig(
        f"results/mean P1 {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


if False:
    P2 = []
    P2std = []
    for t in range(T):
        P2.append(np.mean(dfShape["Polar"][dfShape["T"] == t])[1])
        P2std.append(
            np.std(np.stack(dfShape["Polar"][dfShape["T"] == t], axis=0)[:, 1])
        )

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(wspace=0.3)
    ax[0].errorbar(2 * np.array(range(T)), P2, yerr=P2std)
    ax[0].set(xlabel=r"Time", ylabel=r"$\bar{P}_2$")
    ax[0].title.set_text(r"Mean of $P_2$")
    ax[0].set_ylim([-0.01, 0.01])

    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        P2 = []
        for t in range(T):
            P2.append(np.mean(df["Polar"][df["T"] == t])[1])

        ax[1].plot(2 * np.array(range(T)), P2)

    ax[1].set(xlabel=r"Time", ylabel=r"$\bar{P}_2$")
    ax[1].title.set_text(r"Mean of $P_2$ indivial videos")
    ax[1].set_ylim([-0.01, 0.01])

    fig.savefig(
        f"results/mean P2 {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


if True:
    rho = []
    for t in range(T):
        rho.append(1 / np.mean(dfShape["Area"][dfShape["T"] == t]))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(wspace=0.3)
    ax[0].plot(2 * np.array(range(T)), rho)
    ax[0].set(xlabel=r"Time", ylabel=r"$\rho$")
    ax[0].title.set_text(r"$\rho$")
    ax[0].set_ylim([0.048, 0.1])

    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        rho = []
        for t in range(T):
            rho.append(1 / np.mean(df["Area"][df["T"] == t]))

        ax[1].plot(2 * np.array(range(T)), rho)

    ax[1].set(xlabel=r"Time", ylabel=r"$\rho$")
    ax[1].title.set_text(r"$\rho$ of indivial videos")
    ax[1].set_ylim([0.048, 0.1])

    fig.savefig(
        f"results/mean rho {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")