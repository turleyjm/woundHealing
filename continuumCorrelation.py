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
import shapely
import skimage as sm
import skimage.io
import skimage.measure
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
from mpl_toolkits.mplot3d import Axes3D
import tifffile
from skimage.draw import circle_perimeter
from scipy import optimize
import xml.etree.ElementTree as et
from scipy.optimize import leastsq

import cellProperties as cell
import findGoodCells as fi
import commonLiberty as cl

plt.rcParams.update({"font.size": 16})

# -------------------

filenames, fileType = cl.getFilesType()

T = 93
scale = 123.26 / 512
L = 123.26
grid = 11
timeGrid = 9

# -------------------


def exponential(x, coeffs):
    A = coeffs[0]
    c = coeffs[1]
    return A * np.exp(c * x)


def residualsExponential(coeffs, y, x):
    return y - exponential(x, coeffs)


# -------------------

_df2 = []
if False:
    for filename in filenames:

        df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")

        for t in range(T):
            dft = df[df["Time"] == t]
            Q = np.mean(dft["q"])
            W = np.mean(dft["uhat"])

            for i in range(len(dft)):
                [x, y] = [
                    dft["Centroid"].iloc[i][0] * scale,
                    dft["Centroid"].iloc[i][1] * scale,
                ]
                dQ = dft["q"].iloc[i] - Q
                A = dft["Area"].iloc[i] * scale ** 2
                TrdQ = np.trace(np.matmul(Q, dQ))
                Pol = dft["Polar"].iloc[i]
                deltauhat = dft["uhat"].iloc[i] - W

                _df2.append(
                    {
                        "Filename": filename,
                        "T": t,
                        "X": x,
                        "Y": y,
                        "dQ": dQ,
                        "Q": Q,
                        "TrdQ": TrdQ,
                        "Area": A,
                        "Polar": Pol,
                        "deltauhat": deltauhat,
                    }
                )

    dfShape = pd.DataFrame(_df2)
    dfShape.to_pickle(f"databases/dfContinuum{fileType}.pkl")

else:
    dfShape = pd.read_pickle(f"databases/dfContinuum{fileType}.pkl")

_df = []
if True:
    for filename in filenames:

        df = pd.read_pickle(f"dat/{filename}/nucleusTracks{filename}.pkl")

        for t in range(T):
            dft = df[df["Time"] == t]
            v = np.mean(dft["Velocity"]) * scale

            for i in range(len(dft)):
                x = dft["X"].iloc[i] * scale
                y = dft["Y"].iloc[i] * scale
                dv = dft["Velocity"].iloc[i] * scale - v

                _df2.append(
                    {
                        "Filename": filename,
                        "T": t,
                        "X": x,
                        "Y": y,
                        "dv": dv,
                    }
                )

    dfVelocity = pd.DataFrame(_df2)
    dfVelocity.to_pickle(f"databases/dfContinuumVelocity{fileType}.pkl")

else:
    dfVelocity = pd.read_pickle(f"databases/dfContinuumVelocity{fileType}.pkl")

# delta rho Q and uhat space time correlation
if True:
    T = np.array(range(timeGrid)) * 10
    R = np.array(range(grid)) * 10
    v = [[[] for col in range(len(R))] for col in range(len(T))]
    for filename in filenames:

        df = dfVelocity[dfVelocity["Filename"] == filename]
        heatmapdv1 = np.zeros([90, grid, grid])
        heatmapdv2 = np.zeros([90, grid, grid])

        # outPlanePixel = sm.io.imread(
        #         f"dat/{filename}/outPlane{filename}.tif"
        #     ).astype(float)
        # outPlane = []
        # for t in range(90):
        #     img = Image.fromarray(outPlanePixel[t])
        #     outPlane.append(np.array(img.resize((124, 124)))[7:117, 7:117])
        # outPlane = np.array(outPlane)
        # outPlane[outPlane > 50] = 255
        # outPlane[outPlane < 0] = 0
        # outPlane[outPlane == 255] = 1

        outPlane = np.ones([90, 124, 124])

        for t in range(90):
            dft = df[df["T"] == t]
            for i in range(grid):
                for j in range(grid):
                    x = [
                        (L - 110) / 2 + i * 110 / grid,
                        (L - 110) / 2 + (i + 1) * 110 / grid,
                    ]
                    y = [
                        (L - 110) / 2 + j * 110 / grid,
                        (L - 110) / 2 + (j + 1) * 110 / grid,
                    ]
                    dfg = cl.sortGrid(dft, x, y)
                    if list(dfg["dv"]) != []:
                        heatmapdv1[t, i, j] = np.mean(dfg["dv"], axis=0)[0]
                        heatmapdv2[t, i, j] = np.mean(dfg["dv"], axis=0)[1]

            heatmapdv1[t] = heatmapdv1[t] - np.mean(heatmapdv1[t])
            heatmapdv2[t] = heatmapdv2[t] - np.mean(heatmapdv2[t])

        for i in range(grid):
            for j in range(grid):
                for t in T:
                    dv1 = np.mean(heatmapdv1[t : t + 10, i, j])
                    dv2 = np.mean(heatmapdv2[t : t + 10, i, j])
                    for idash in range(grid):
                        for jdash in range(grid):
                            for tdash in T:
                                deltaT = int((tdash - t) / 10)
                                deltaR = int(
                                    ((i - idash) ** 2 + (j - jdash) ** 2) ** 0.5
                                )
                                if deltaR < grid:
                                    if deltaT >= 0 and deltaT < timeGrid:
                                        dv1dash = np.mean(
                                            heatmapdv1[tdash : tdash + 10, idash, jdash]
                                        )
                                        dv2dash = np.mean(
                                            heatmapdv2[tdash : tdash + 10, idash, jdash]
                                        )
                                        v[deltaT][deltaR].append(
                                            (dv1 * dv1dash) + (dv2 * dv2dash)
                                        )

    vCorrelation = [[] for col in range(len(T))]
    for i in range(len(T)):
        for j in range(len(R)):
            vCorrelation[i].append(np.mean(v[i][j]))

    vCorrelation = np.array(vCorrelation)
    vCorrelation = np.nan_to_num(vCorrelation)

    deltavStd = np.mean(heatmapdv1 ** 2 + heatmapdv2 ** 2)

    _df = []

    _df.append(
        {
            "v": v,
            "vCorrelation": vCorrelation,
            "deltavStd": deltavStd,
        }
    )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/continuumCorrelationVelocity{fileType}.pkl")

    t, r = np.mgrid[0:180:20, 0:110:10]
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    maxCorr = np.max([vCorrelation, -vCorrelation])

    c = ax.pcolor(
        t, r, vCorrelation, cmap="RdBu_r", vmin=-maxCorr, vmax=maxCorr, shading="auto"
    )
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"$R (\mu m)$ ")
    ax.title.set_text(r"Correlation of $\delta v$" + f" {fileType}")

    fig.savefig(
        f"results/Correlation velocity {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

    # -------------------

    t, r = np.mgrid[0:180:20, 0:110:10]
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    c = ax.pcolor(
        t, r, vCorrelation / deltavStd, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto"
    )
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"$R (\mu m)$ ")
    ax.title.set_text(r"Correlation of $\delta v$" + f" {fileType}")

    fig.savefig(
        f"results/Correlation velocity {fileType} Norm",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


if False:
    df = pd.read_pickle(f"databases/continuumCorrelation{fileType}.pkl")

    rhoCorrelation = df["rhoCorrelation"].iloc[0]
    deltaQCorrelation = df["deltaQCorrelation"].iloc[0]

    R0 = rhoCorrelation[0]
    T0 = rhoCorrelation[:, 0]
    T = np.array(range(timeGrid)) * 20
    R = np.array(range(grid)) * 10

    mR = leastsq(
        residualsExponential,
        x0=(1, -1),
        args=(R0, R),
    )[0]

    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    plt.subplots_adjust(wspace=0.3)
    ax[0, 0].plot(R + 10, R0, label="Data")
    ax[0, 0].plot(R + 10, exponential(R, mR), label="Fit Curve")
    ax[0, 0].set(xlabel=r"Distance $(\mu m)$", ylabel="Correlation")
    ax[0, 0].title.set_text(r"$\delta \rho$, $\alpha$ = " + f"{round(mR[1],3)}")
    ax[0, 0].legend()

    mT = leastsq(
        residualsExponential,
        x0=(1, -1),
        args=(T0, T),
    )[0]

    ax[0, 1].plot(T + 10, T0)
    ax[0, 1].plot(T + 10, exponential(T, mT))
    ax[0, 1].set(xlabel="Time (mins)", ylabel="Correlation")
    ax[0, 1].title.set_text(r"$\delta \rho$, $\beta$ = " + f"{round(mT[1],3)}")

    R0 = deltaQCorrelation[0]
    T0 = deltaQCorrelation[:, 0]
    T = np.array(range(timeGrid)) * 20
    R = np.array(range(grid)) * 10

    mR = leastsq(
        residualsExponential,
        x0=(1, -1),
        args=(R0, R),
    )[0]

    ax[1, 0].plot(R + 10, R0)
    ax[1, 0].plot(R + 10, exponential(R, mR))
    ax[1, 0].set(xlabel=r"Distance $(\mu m)$", ylabel="Correlation")
    ax[1, 0].title.set_text(r"$\delta Q$, $\alpha$ = " + f"{round(mR[1],3)}")

    mT = leastsq(
        residualsExponential,
        x0=(1, -1),
        args=(T0, T),
    )[0]

    ax[1, 1].plot(T + 10, T0)
    ax[1, 1].plot(T + 10, exponential(T, mT))
    ax[1, 1].set(xlabel="Time (mins)", ylabel="Correlation")
    ax[1, 1].title.set_text(r"$\delta Q$, $\beta$ = " + f"{round(mT[1],3)}")

    fig.savefig(
        f"results/continuumCorrelation Fit {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


# delta rho Q and uhat direction space time correlation
if False:
    T = np.array(range(timeGrid)) * 10
    R = np.linspace(0, 10 * (grid - 1), grid)
    theta = np.linspace(0, 2 * np.pi, 21)
    rho = [
        [[[] for col in range(len(theta))] for col in range(len(R))]
        for col in range(len(T))
    ]
    deltaQ = [
        [[[] for col in range(len(theta))] for col in range(len(R))]
        for col in range(len(T))
    ]
    deltauhat = [
        [[[] for col in range(len(theta))] for col in range(len(R))]
        for col in range(len(T))
    ]
    for filename in filenames:

        df = dfShape[dfShape["Filename"] == filename]
        Q = np.mean(df["Q"])
        theta0 = np.arccos(Q[0, 0] / (Q[0, 0] ** 2 + Q[0, 1] ** 2) ** 0.5) / 2
        heatmapdrho = np.zeros([90, grid, grid])
        heatmapdQ1 = np.zeros([90, grid, grid])
        heatmapdQ2 = np.zeros([90, grid, grid])
        heatmapuhat1 = np.zeros([90, grid, grid])
        heatmapuhat2 = np.zeros([90, grid, grid])

        # outPlanePixel = sm.io.imread(
        #         f"dat/{filename}/outPlane{filename}.tif"
        #     ).astype(float)
        # outPlane = []
        # for t in range(90):
        #     img = Image.fromarray(outPlanePixel[t])
        #     outPlane.append(np.array(img.resize((124, 124)))[7:117, 7:117])
        # outPlane = np.array(outPlane)
        # outPlane[outPlane > 50] = 255
        # outPlane[outPlane < 0] = 0
        # outPlane[outPlane == 255] = 1

        outPlane = np.ones([90, 124, 124])

        for t in range(90):
            dft = df[df["T"] == t]
            for i in range(grid):
                for j in range(grid):
                    x = [
                        (L - 110) / 2 + i * 110 / grid,
                        (L - 110) / 2 + (i + 1) * 110 / grid,
                    ]
                    y = [
                        (L - 110) / 2 + j * 110 / grid,
                        (L - 110) / 2 + (j + 1) * 110 / grid,
                    ]
                    area = np.sum(
                        outPlane[
                            t, round(x[0]) : round(x[1]), round(y[0]) : round(y[1])
                        ]
                    )
                    dfg = cl.sortGrid(dft, x, y)
                    if list(dfg["Area"]) != []:
                        heatmapdrho[t, i, j] = len(dfg["Area"]) / area
                        heatmapdQ1[t, i, j] = np.mean(dfg["dQ"], axis=0)[0, 0]
                        heatmapdQ2[t, i, j] = np.mean(dfg["dQ"], axis=0)[1, 0]
                        heatmapuhat1[t, i, j] = np.mean(dfg["deltauhat"], axis=0)[0]
                        heatmapuhat2[t, i, j] = np.mean(dfg["deltauhat"], axis=0)[1]

            heatmapdrho[t] = heatmapdrho[t] - np.mean(heatmapdrho[t])

            if False:
                dx, dy = 110 / grid, 110 / grid
                xdash, ydash = np.mgrid[0:110:dx, 0:110:dy]

                fig, ax = plt.subplots()
                c = ax.pcolor(
                    xdash,
                    ydash,
                    heatmapdrho[t],
                    cmap="RdBu_r",
                    vmax=0.1,
                    vmin=-0.1,
                    shading="auto",
                )
                fig.colorbar(c, ax=ax)
                plt.xlabel(r"x $(\mu m)$")
                plt.ylabel(r"y $(\mu m)$")
                plt.title(r"$\delta \rho_0$ " + f"{filename}")
                fig.savefig(
                    f"results/P0 heatmap {t*2} {filename}",
                    dpi=300,
                    transparent=True,
                )
                plt.close("all")

        for i in range(grid):
            for j in range(grid):
                for t in T:
                    deltarho = np.mean(heatmapdrho[t : t + 10, i, j])
                    dQ1 = np.mean(heatmapdQ1[t : t + 10, i, j])
                    dQ2 = np.mean(heatmapdQ2[t : t + 10, i, j])
                    duhat1 = np.mean(heatmapuhat1[t : t + 10, i, j])
                    duhat2 = np.mean(heatmapuhat2[t : t + 10, i, j])
                    for idash in range(grid):
                        for jdash in range(grid):
                            for tdash in T:
                                deltaT = int((tdash - t) / 10)
                                deltaR = int(
                                    ((i - idash) ** 2 + (j - jdash) ** 2) ** 0.5
                                )
                                deltatheta = int(
                                    (
                                        20
                                        * (np.arctan2(jdash - j, idash - i) - theta0)
                                        / (2 * np.pi)
                                    )
                                    % 20
                                )
                                if deltaR < grid:
                                    if deltaT >= 0 and deltaT < timeGrid:
                                        rho[deltaT][deltaR][deltatheta].append(
                                            deltarho
                                            * np.mean(
                                                heatmapdrho[
                                                    tdash : tdash + 10, idash, jdash
                                                ]
                                            )
                                        )

                                        dQ1dash = np.mean(
                                            heatmapdQ1[tdash : tdash + 10, idash, jdash]
                                        )
                                        dQ2dash = np.mean(
                                            heatmapdQ2[tdash : tdash + 10, idash, jdash]
                                        )
                                        deltaQ[deltaT][deltaR][deltatheta].append(
                                            2 * (dQ1 * dQ1dash) + 2 * (dQ2 * dQ2dash)
                                        )

                                        duhat1dash = np.mean(
                                            heatmapuhat1[
                                                tdash : tdash + 10, idash, jdash
                                            ]
                                        )
                                        duhat2dash = np.mean(
                                            heatmapuhat2[
                                                tdash : tdash + 10, idash, jdash
                                            ]
                                        )
                                        deltauhat[deltaT][deltaR][deltatheta].append(
                                            (duhat1 * duhat1dash)
                                            + (duhat2 * duhat2dash)
                                        )

    rhoCorrelation = [[[] for col in range(len(theta))] for col in range(len(T))]
    deltaQCorrelation = [[[] for col in range(len(theta))] for col in range(len(T))]
    deltauhatCorrelation = [[[] for col in range(len(theta))] for col in range(len(T))]
    for i in range(len(T)):
        for j in range(len(R)):
            for th in range(len(theta)):

                rhoCorrelation[i][th].append(np.mean(rho[i][j][th]))
                deltaQCorrelation[i][th].append(np.mean(deltaQ[i][j][th]))
                deltauhatCorrelation[i][th].append(np.mean(deltauhat[i][j][th]))

    rhoCorrelation = np.array(rhoCorrelation)
    deltaQCorrelation = np.array(deltaQCorrelation)
    deltauhatCorrelation = np.array(deltauhatCorrelation)
    rhoCorrelation = np.nan_to_num(rhoCorrelation)
    deltaQCorrelation = np.nan_to_num(deltaQCorrelation)
    deltauhatCorrelation = np.nan_to_num(deltauhatCorrelation)

    _df = []

    _df.append(
        {
            "rho": rho,
            "deltaQ": deltaQ,
            "deltauhat": deltauhat,
            "rhoCorrelation": rhoCorrelation,
            "deltaQCorrelation": deltaQCorrelation,
            "deltauhatCorrelation": deltauhatCorrelation,
        }
    )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/continuumDirectionCorrelation{fileType}.pkl")
else:
    df = pd.read_pickle(f"databases/continuumDirectionCorrelation{fileType}.pkl")
    rhoCorrelation = df["rhoCorrelation"].iloc[0]
    deltaQCorrelation = df["deltaQCorrelation"].iloc[0]
    deltauhatCorrelation = df["deltauhatCorrelation"].iloc[0]


if False:
    T = np.array(range(timeGrid)) * 10
    R = np.linspace(0, 10 * (grid - 1), grid)
    theta = np.linspace(0, 2 * np.pi, 21)
    rad = np.linspace(0, 10 * (grid - 1), grid)
    azm = np.linspace(0, 2 * np.pi, 21)

    cl.createFolder("results/video/")
    maxCorr = np.max([rhoCorrelation, -rhoCorrelation])
    for t in range(len(T)):

        ra2, th2 = np.meshgrid(rad, azm)

        fig = plt.figure()
        ax = Axes3D(fig)

        plt.subplot(projection="polar")

        pc = plt.pcolormesh(th2, ra2, rhoCorrelation[t], vmin=-maxCorr, vmax=maxCorr)

        plt.colorbar(pc)
        plt.grid()
        fig.savefig(
            "results/video/"
            + f"Directional correlation delta rho {fileType} at T={t}.png",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

    # make video
    img_array = []

    for t in range(len(T)):
        img = cv2.imread(
            f"results/video/Directional correlation delta rho {fileType} at T={t}.png"
        )
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(
        f"results/Directional correlation delta rho {fileType}.mp4",
        cv2.VideoWriter_fourcc(*"DIVX"),
        3,
        size,
    )
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
    cv2.destroyAllWindows()

    shutil.rmtree("results/video")

    # -------------------

    cl.createFolder("results/video/")
    maxCorr = np.max([deltauhatCorrelation, -deltauhatCorrelation])
    for t in range(len(T)):
        ra2, th2 = np.meshgrid(R, theta)

        fig = plt.figure()
        ax = Axes3D(fig)

        plt.subplot(projection="polar")

        pc = plt.pcolormesh(
            th2, ra2, deltauhatCorrelation[t], vmin=-maxCorr, vmax=maxCorr
        )

        plt.colorbar(pc)
        plt.grid()
        fig.savefig(
            "results/video/"
            + f"Directional correlation deltauhat {fileType} at T={t}.png",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

    # make video
    img_array = []

    for t in range(len(T)):
        img = cv2.imread(
            f"results/video/Directional correlation deltauhat {fileType} at T={t}.png"
        )
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(
        f"results/Directional correlation deltauhat {fileType}.mp4",
        cv2.VideoWriter_fourcc(*"DIVX"),
        3,
        size,
    )
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
    cv2.destroyAllWindows()

    shutil.rmtree("results/video")

    # -------------------

    cl.createFolder("results/video/")
    maxCorr = np.max([deltaQCorrelation, -deltaQCorrelation])
    for t in range(len(T)):
        ra2, th2 = np.meshgrid(R, theta)

        fig = plt.figure()
        ax = Axes3D(fig)

        plt.subplot(projection="polar")

        pc = plt.pcolormesh(th2, ra2, deltaQCorrelation[t], vmin=-maxCorr, vmax=maxCorr)

        plt.colorbar(pc)
        plt.grid()
        fig.savefig(
            "results/video/"
            + f"Directional correlation deltaQ {fileType} at T={t}.png",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

    # make video
    img_array = []

    for t in range(len(T)):
        img = cv2.imread(
            f"results/video/Directional correlation deltaQ {fileType} at T={t}.png"
        )
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(
        f"results/Directional correlation deltaQ {fileType}.mp4",
        cv2.VideoWriter_fourcc(*"DIVX"),
        3,
        size,
    )
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
    cv2.destroyAllWindows()

    shutil.rmtree("results/video")
