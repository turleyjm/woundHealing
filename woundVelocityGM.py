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
import tifffile
from skimage.draw import circle_perimeter
from scipy import optimize
import xml.etree.ElementTree as et

import cellProperties as cell
import utils as util

plt.rcParams.update({"font.size": 16})

# -------------------


def weighted_avg_and_std(values, weight, axis=0):
    average = np.average(values, weights=weight, axis=axis)
    variance = np.average((values - average) ** 2, weights=weight, axis=axis)
    return average, np.sqrt(variance)


# -------------------

fileTypes, groupTitle = util.getFilesTypes()
scale = 123.26 / 512
T = 84
timeStep = 4
R = 50
rStep = 10

# -------------------

# Individual: v with distance from wound edge and time
if True:
    for fileType in fileTypes:
        filenames = util.getFilesType(fileType)[0]
        v1 = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        v1Cont = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        area = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        dfVelocity = pd.read_pickle(f"databases/dfVelocityWound{fileType}.pkl")
        for k in range(len(filenames)):
            filename = filenames[k]
            dfFile = dfVelocity[dfVelocity["Filename"] == filename]
            dv1Cont = np.mean(dfFile["dv"] ** 2, axis=0)[0] ** 0.5
            if "Wound" in filename:
                t0 = util.findStartTime(filename)
            else:
                t0 = 0
            t2 = int(timeStep / 2 * (int(T / timeStep) + 1) - t0 / 2)

            for r in range(v1.shape[2]):
                for t in range(v1.shape[1]):
                    df1 = dfFile[dfFile["T"] > timeStep * t]
                    df2 = df1[df1["T"] <= timeStep * (t + 1)]
                    df3 = df2[df2["R"] > rStep * r]
                    df = df3[df3["R"] <= rStep * (r + 1)]
                    if len(df) > 0:
                        v1[k, t, r] = np.mean(df["dv"], axis=0)[0]
                        v1Cont[k, t, r] = np.mean(df["dv"], axis=0)[0] / dv1Cont

            inPlane = 1 - (
                sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype(int)[:t2]
                / 255
            )
            dist = (
                sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)[:t2]
                * scale
            )

            for r in range(area.shape[2]):
                for t in range(area.shape[1]):
                    t1 = int(timeStep / 2 * t - t0 / 2)
                    t2 = int(timeStep / 2 * (t + 1) - t0 / 2)
                    if t1 < 0:
                        t1 = 0
                    if t2 < 0:
                        t2 = 0
                    area[k, t, r] = (
                        np.sum(
                            inPlane[t1:t2][
                                (dist[t1:t2] > rStep * r)
                                & (dist[t1:t2] <= rStep * (r + 1))
                            ]
                        )
                        * scale**2
                    )

        V1 = np.zeros([int(T / timeStep), int(R / rStep)])
        V1Cont = np.zeros([int(T / timeStep), int(R / rStep)])
        std = np.zeros([int(T / timeStep), int(R / rStep)])
        meanArea = np.zeros([int(T / timeStep), int(R / rStep)])

        for r in range(area.shape[2]):
            for t in range(area.shape[1]):
                _V1 = v1[:, t, r][v1[:, t, r] != 0]
                _area = area[:, t, r][v1[:, t, r] != 0]
                _V1Cont = v1Cont[:, t, r][v1Cont[:, t, r] != 0]
                if (len(_area) > 0) & (np.sum(_area) > 0):
                    _dd, _std = weighted_avg_and_std(_V1, _area)
                    V1[t, r] = _dd
                    std[t, r] = _std
                    meanArea[t, r] = np.mean(_area)
                    _dd, _std = weighted_avg_and_std(_V1Cont, _area)
                    V1Cont[t, r] = _dd
                else:
                    V1[t, r] = np.nan
                    std[t, r] = np.nan
                    V1Cont[t, r] = np.nan

        V1[meanArea < 500] = np.nan
        V1Cont[meanArea < 500] = np.nan

        t, r = np.mgrid[
            timeStep / 2 : T + timeStep / 2 : timeStep,
            rStep / 2 : R + rStep / 2 : rStep,
        ]
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        c = ax.pcolor(
            t,
            r,
            V1,
            vmin=-0.3,
            vmax=0.3,
            cmap="RdBu_r",
        )
        fig.colorbar(c, ax=ax)
        ax.set(
            xlabel="Time after wounding (mins)",
            ylabel=f"Distance from \n wound edge" + r"$(\mu m)$",
        )
        fileTitle = util.getFileTitle(fileType)
        boldTitle = util.getBoldTitle(fileTitle)
        if "wt" in boldTitle:
            ax.title.set_text(r"$\delta V_1$ distance and time" + f"\n{boldTitle}")
        else:
            colour, mark = util.getColorLineMarker(fileType, groupTitle)
            plt.title(
                r"$\delta V_1$ distance and time" + f"\n{boldTitle}", color=colour
            )

        fig.savefig(
            f"results/v heatmap {fileTitle}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")

        if False:
            controlType = util.controlType(fileType)
            dfCont = pd.read_pickle(f"databases/dfVelocityWound{controlType}.pkl")
            dV1Cont = np.mean(dfCont["dv"] ** 2, axis=0)[0] ** 0.5

            fig, ax = plt.subplots(2, 1, figsize=(6, 6))
            c = ax[0].pcolor(
                t,
                r,
                V1 / dV1Cont,
                vmin=-1.2,
                vmax=1.2,
                cmap="RdBu_r",
            )
            fig.colorbar(c, ax=ax[0])
            ax[0].set(
                xlabel="Time after wounding (mins)",
                ylabel=f"Distance from \n wound edge" + r"$(\mu m)$",
            )
            fileTitle = util.getFileTitle(fileType)
            boldTitle = util.getBoldTitle(fileTitle)
            controlTitle = util.getFileTitle(controlType)
            controlTitle = util.getBoldTitle(controlTitle)
            ax[0].title.set_text(
                r"$\delta v_1$"
                + f"{boldTitle} heterogeneity \n compared to {controlTitle}"
            )

            c = ax[1].pcolor(
                t,
                r,
                V1Cont,
                vmin=-1.2,
                vmax=1.2,
                cmap="RdBu_r",
            )
            fig.colorbar(c, ax=ax[1])
            ax[1].set(
                xlabel="Time after wounding (mins)",
                ylabel=f"Distance from \n wound edge" + r"$(\mu m)$",
            )
            fileTitle = util.getFileTitle(fileType)
            boldTitle = util.getBoldTitle(fileTitle)
            ax[1].title.set_text(r"$\delta v_1$" + f"{boldTitle} heterogeneity")

            plt.subplots_adjust(
                left=0.12, bottom=0.11, right=0.9, top=0.88, wspace=0.2, hspace=0.55
            )

            fig.savefig(
                f"results/v heatmap heterogeneity {fileTitle}",
                transparent=True,
                bbox_inches="tight",
                dpi=300,
            )
            plt.close("all")

# Individual: compare from large wound v with distance from wound edge and time
if True:
    fileType = "WoundL18h"
    filenames = util.getFilesType(fileType)[0]
    v1 = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
    area = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
    dfVelocity = pd.read_pickle(f"databases/dfVelocityWound{fileType}.pkl")
    for k in range(len(filenames)):
        filename = filenames[k]
        dfFile = dfVelocity[dfVelocity["Filename"] == filename]
        dv1Cont = np.mean(dfFile["dv"] ** 2, axis=0)[0] ** 0.5
        if "Wound" in filename:
            t0 = util.findStartTime(filename)
        else:
            t0 = 0
        t2 = int(timeStep / 2 * (int(T / timeStep) + 1) - t0 / 2)

        for r in range(v1.shape[2]):
            for t in range(v1.shape[1]):
                df1 = dfFile[dfFile["T"] > timeStep * t]
                df2 = df1[df1["T"] <= timeStep * (t + 1)]
                df3 = df2[df2["R"] > rStep * r]
                df = df3[df3["R"] <= rStep * (r + 1)]
                if len(df) > 0:
                    v1[k, t, r] = np.mean(df["dv"], axis=0)[0]

        inPlane = 1 - (
            sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype(int)[:t2]
            / 255
        )
        dist = (
            sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)[:t2]
            * scale
        )

        for r in range(area.shape[2]):
            for t in range(area.shape[1]):
                t1 = int(timeStep / 2 * t - t0 / 2)
                t2 = int(timeStep / 2 * (t + 1) - t0 / 2)
                if t1 < 0:
                    t1 = 0
                if t2 < 0:
                    t2 = 0
                area[k, t, r] = (
                    np.sum(
                        inPlane[t1:t2][
                            (dist[t1:t2] > rStep * r) & (dist[t1:t2] <= rStep * (r + 1))
                        ]
                    )
                    * scale**2
                )

    V1Large = np.zeros([int(T / timeStep), int(R / rStep)])
    std = np.zeros([int(T / timeStep), int(R / rStep)])
    meanArea = np.zeros([int(T / timeStep), int(R / rStep)])

    for r in range(area.shape[2]):
        for t in range(area.shape[1]):
            _V1 = v1[:, t, r][v1[:, t, r] != 0]
            _area = area[:, t, r][v1[:, t, r] != 0]
            if (len(_area) > 0) & (np.sum(_area) > 0):
                _dd, _std = weighted_avg_and_std(_V1, _area)
                V1Large[t, r] = _dd
                std[t, r] = _std
                meanArea[t, r] = np.mean(_area)
            else:
                V1Large[t, r] = np.nan
                std[t, r] = np.nan

    V1Large[meanArea < 500] = np.nan

    for fileType in fileTypes[1:]:
        filenames = util.getFilesType(fileType)[0]
        v1 = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        v1Cont = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        area = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
        dfVelocity = pd.read_pickle(f"databases/dfVelocityWound{fileType}.pkl")
        for k in range(len(filenames)):
            filename = filenames[k]
            dfFile = dfVelocity[dfVelocity["Filename"] == filename]
            dv1Cont = np.mean(dfFile["dv"] ** 2, axis=0)[0] ** 0.5
            if "Wound" in filename:
                t0 = util.findStartTime(filename)
            else:
                t0 = 0
            t2 = int(timeStep / 2 * (int(T / timeStep) + 1) - t0 / 2)

            for r in range(v1.shape[2]):
                for t in range(v1.shape[1]):
                    df1 = dfFile[dfFile["T"] > timeStep * t]
                    df2 = df1[df1["T"] <= timeStep * (t + 1)]
                    df3 = df2[df2["R"] > rStep * r]
                    df = df3[df3["R"] <= rStep * (r + 1)]
                    if len(df) > 0:
                        v1[k, t, r] = np.mean(df["dv"], axis=0)[0]
                        v1Cont[k, t, r] = np.mean(df["dv"], axis=0)[0] / dv1Cont

            inPlane = 1 - (
                sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype(int)[:t2]
                / 255
            )
            dist = (
                sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)[:t2]
                * scale
            )

            for r in range(area.shape[2]):
                for t in range(area.shape[1]):
                    t1 = int(timeStep / 2 * t - t0 / 2)
                    t2 = int(timeStep / 2 * (t + 1) - t0 / 2)
                    if t1 < 0:
                        t1 = 0
                    if t2 < 0:
                        t2 = 0
                    area[k, t, r] = (
                        np.sum(
                            inPlane[t1:t2][
                                (dist[t1:t2] > rStep * r)
                                & (dist[t1:t2] <= rStep * (r + 1))
                            ]
                        )
                        * scale**2
                    )

        V1 = np.zeros([int(T / timeStep), int(R / rStep)])
        V1Cont = np.zeros([int(T / timeStep), int(R / rStep)])
        std = np.zeros([int(T / timeStep), int(R / rStep)])
        meanArea = np.zeros([int(T / timeStep), int(R / rStep)])

        for r in range(area.shape[2]):
            for t in range(area.shape[1]):
                _V1 = v1[:, t, r][v1[:, t, r] != 0]
                _area = area[:, t, r][v1[:, t, r] != 0]
                _V1Cont = v1Cont[:, t, r][v1Cont[:, t, r] != 0]
                if (len(_area) > 0) & (np.sum(_area) > 0):
                    _dd, _std = weighted_avg_and_std(_V1, _area)
                    V1[t, r] = _dd
                    std[t, r] = _std
                    meanArea[t, r] = np.mean(_area)
                    _dd, _std = weighted_avg_and_std(_V1Cont, _area)
                    V1Cont[t, r] = _dd
                else:
                    V1[t, r] = np.nan
                    std[t, r] = np.nan
                    V1Cont[t, r] = np.nan

        V1[meanArea < 500] = np.nan
        V1Cont[meanArea < 500] = np.nan

        t, r = np.mgrid[
            timeStep / 2 : T + timeStep / 2 : timeStep,
            rStep / 2 : R + rStep / 2 : rStep,
        ]
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        c = ax.pcolor(
            t,
            r,
            V1 - V1Large,
            vmin=-0.2,
            vmax=0.2,
            cmap="RdBu_r",
        )
        fig.colorbar(c, ax=ax)
        ax.set(
            xlabel="Time after wounding (mins)",
            ylabel=f"Distance from \n wound edge" + r"$(\mu m)$",
        )
        fileTitle = util.getFileTitle(fileType)
        boldTitle = util.getBoldTitle(fileTitle)
        if "large" in boldTitle:
            colour, mark = util.getColorLineMarker(fileType, groupTitle)
            plt.title(
                r"Wild type difference in $\delta V_1$" + f"\nfrom {boldTitle}",
                color=colour,
            )
        else:
            ax.title.set_text(
                r"Wild type difference in $\delta V_1$" + f"\nfrom {boldTitle}"
            )

        fig.savefig(
            f"results/v heatmap change large wound {fileTitle}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")
