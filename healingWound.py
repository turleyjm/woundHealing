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

filenames, fileType = util.getFilesType()
scale = 123.26 / 512
T = 93

# Mean Wound Area
if False:
    fig = plt.figure(1, figsize=(9, 8))
    sf = []
    endTime = []
    divisions = []
    R = [[] for col in range(T)]
    _df = []
    for filename in filenames:
        t0 = util.findStartTime(filename)

        dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
        sf.append(dfWound["Shape Factor"].iloc[0])
        time = np.array(dfWound["Time"])
        area = np.array(dfWound["Area"]) * (scale) ** 2

        tf = sum(area > 0)
        endTime.append(tf)
        for t in range(T):
            if pd.isnull(area[t]):
                area[t] = 0

        # df = dfMitosis[dfMitosis["Chain"] == "parent"]
        # count = 0
        # for i in range(len(df)):
        #     if df["Time"].iloc[i][-1] < tf:
        #         count += 1
        # divisions.append(count)

        for t in range(T):
            if area[t] > area[0] * 0.2:
                R[t].append(area[t])
                _df.append({"Area": area[t], "Time": int(t0 / 2) * 2 + 2 * t})

        A = area[area > area[0] * 0.2]
        print(f"{filename} {area[0]}")

        plt.plot(t0 + np.arange(0, len(A) * 2, 2), A)

    plt.xlabel("Time")
    plt.ylabel(r"Area ($\mu m ^2$)")
    plt.title(f"Area {fileType}")
    fig.savefig(
        f"results/Wound Area {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

    df = pd.DataFrame(_df)
    A = []
    Time = []
    std = []
    T = set(df["Time"])
    N = len(filenames)
    for t in T:
        if len(df[df["Time"] == t]) > N / 3:
            Time.append(t)
            A.append(np.mean(df["Area"][df["Time"] == t]))
            std.append(np.std(df["Area"][df["Time"] == t]))

    fig = plt.figure(1, figsize=(9, 8))
    plt.errorbar(Time, A, yerr=std)
    plt.xlabel("Time")
    plt.ylabel(r" Mean Area ($\mu m ^2$)")
    plt.title(f"Mean Area {fileType}")
    fig.savefig(
        f"results/Mean Wound Area {fileType}",
        bbox_inches="tight",
    )
    plt.close("all")

# Mean Wound Area with invial
if True:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sf = []
    endTime = []
    divisions = []
    R = [[] for col in range(T)]
    _df = []
    for filename in filenames:
        t0 = util.findStartTime(filename)
        if filename == "WoundL18h18":
            continue

        dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
        sf.append(dfWound["Shape Factor"].iloc[0])
        time = np.array(dfWound["Time"])
        area = np.array(dfWound["Area"]) * (scale) ** 2

        tf = sum(area > 0)
        endTime.append(tf)
        for t in range(T):
            if pd.isnull(area[t]):
                area[t] = 0

        # df = dfMitosis[dfMitosis["Chain"] == "parent"]
        # count = 0
        # for i in range(len(df)):
        #     if df["Time"].iloc[i][-1] < tf:
        #         count += 1
        # divisions.append(count)

        for t in range(T):
            if area[t] > area[0] * 0.2:
                R[t].append(area[t])
                _df.append(
                    {
                        "Filename": filename,
                        "Area": area[t],
                        "Time": int(t0 / 2 + 0.01) * 2 + 2 * t,
                    }
                )

        A = area[area > area[0] * 0.2]
        # print(f"{filename} {area[0]}")

        plt.plot(t0 + np.arange(0, len(A) * 2, 2), A)

    df = pd.DataFrame(_df)
    A = []
    Time = []
    std = []
    T = set(df["Time"])
    N = len(filenames)
    for t in T:
        if len(df[df["Time"] == t]) > N / 3:
            Time.append(t)
            A.append(np.mean(df["Area"][df["Time"] == t]))
            std.append(np.std(df["Area"][df["Time"] == t]))
    A = np.array(A)
    std = np.array(std)

    plt.plot(Time, A, linewidth=3, color="k")
    ax.fill_between(Time, A - std, A + std, alpha=0.2, color="k")
    plt.xlabel("Time")
    plt.ylabel(r" Mean Area ($\mu m ^2$)")
    plt.title(f"Mean Area {fileType}")

    if "WoundL" in fileType:
        ax.set_xlim([0, 120])
        ax.set_ylim([190, 1300])
    else:
        ax.set_xlim([0, 45])
        ax.set_ylim([40, 550])
    fig.savefig(
        f"results/Mean and inval Wound Area {fileType}",
        bbox_inches="tight",
    )
    plt.close("all")

# Compare Mean Wound Area
if False:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    labels = ["WoundS18h", "WoundS18h", "WoundL18h"]
    i = 0
    for fileType in labels:
        filenames = util.getFilesType(fileType)[0]
        _df = []
        Area0 = []

        for filename in filenames:
            t0 = util.findStartTime(filename)
            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
            T = len(dfWound)
            area = np.array(dfWound["Area"]) * (scale) ** 2
            Area0.append(area[0])

            for t in range(T):
                if area[t] > area[0] * 0.2:
                    _df.append({"Area": area[t], "Time": int(t0 / 2) * 2 + 2 * t})
                else:
                    _df.append({"Area": 0, "Time": int(t0 / 2) * 2 + 2 * t})

        df = pd.DataFrame(_df)
        A = []
        Time = []
        std = []
        T = set(df["Time"])
        N = len(filenames)
        Area0 = np.mean(Area0)
        for t in T:
            if len(df[df["Time"] == t]) > N / 3:
                if np.mean(df["Area"][df["Time"] == t]) > 0.2 * Area0:
                    Time.append(t)
                    A.append(np.mean(df["Area"][df["Time"] == t]))
                    std.append(np.std(df["Area"][df["Time"] == t]))

        if i > 0:
            typeName = util.getFileTitle(fileType)
            plt.errorbar(Time, A, yerr=std, marker="o", label=f"{typeName}")
        else:
            plt.errorbar(Time, A, yerr=std, marker="o")
        i += 1

    plt.xlabel("Time after wounding (mins)")
    plt.ylabel(r"Area ($\mu m ^2$)")
    plt.title(f"Mean area of wound")
    plt.legend()
    fig.savefig(
        f"results/Mean Wound Area",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

# Normalise by start size
if False:
    T = 93
    fig = plt.figure(1, figsize=(5, 5))
    sf = []
    endTime = []
    divisions = []
    R = [[] for col in range(T)]
    _df = []
    for filename in filenames:
        t0 = util.findStartTime(filename)

        dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
        sf.append(dfWound["Shape Factor"].iloc[0])
        time = np.array(dfWound["Time"])
        area = np.array(dfWound["Area"]) * (scale) ** 2

        tf = sum(area > 0)
        endTime.append(tf)
        for t in range(T):
            if pd.isnull(area[t]):
                area[t] = 0

        # df = dfMitosis[dfMitosis["Chain"] == "parent"]
        # count = 0
        # for i in range(len(df)):
        #     if df["Time"].iloc[i][-1] < tf:
        #         count += 1
        # divisions.append(count)

        for t in range(T):
            if area[t] > area[0] * 0.2:
                R[t].append(area[t] / area[0])
                _df.append({"Area": area[t] / area[0], "Time": int(t0 / 2) * 2 + 2 * t})

        A = area[area > area[0] * 0.2] / area[0]
        # print(f"{filename} {area[0]}")

        plt.plot(t0 + np.arange(0, len(A) * 2, 2), A, marker="o")

    plt.xlabel("Time")
    plt.ylabel(r"Area/$A_0$")
    plt.title(f"Normalise Area {fileType}")
    fig.savefig(
        f"results/Wound Area norm 1 {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

    df = pd.DataFrame(_df)
    A = []
    Time = []
    std = []
    T = set(df["Time"])
    N = len(filenames)
    for t in T:
        if len(df[df["Time"] == t]) > N / 3:
            Time.append(t)
            A.append(np.mean(df["Area"][df["Time"] == t]))
            std.append(np.std(df["Area"][df["Time"] == t]))

    fig = plt.figure(1, figsize=(5, 5))
    plt.errorbar(Time, A, yerr=std, marker="o")
    plt.xlabel("Time")
    plt.ylabel(r"Area/$A_0$")
    plt.title(f"Normalise Mean Area {fileType}")
    fig.savefig(
        f"results/Mean Wound Area norm 1 {fileType}",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

# Normalise by start size
if False:
    T = 93
    fig = plt.figure(1, figsize=(5, 5))
    sf = []
    endTime = []
    divisions = []
    R = [[] for col in range(T)]
    _df = []
    for filename in filenames:
        t0 = util.findStartTime(filename)

        dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
        sf.append(dfWound["Shape Factor"].iloc[0])
        time = np.array(dfWound["Time"])
        area = np.array(dfWound["Area"]) * (scale) ** 2

        tf = len(area[area > area[0] * 0.2]) * 2 + t0
        endTime.append(tf)
        for t in range(T):
            if pd.isnull(area[t]):
                area[t] = 0

        # df = dfMitosis[dfMitosis["Chain"] == "parent"]
        # count = 0
        # for i in range(len(df)):
        #     if df["Time"].iloc[i][-1] < tf:
        #         count += 1
        # divisions.append(count)
        time = []
        for t in range(T):
            if area[t] > area[0] * 0.2:
                _df.append(
                    {"Area": area[t] / area[0], "Time": (int(t0 / 2) * 2 + 2 * t) / tf}
                )
                time.append((int(t0 / 2) * 2 + 2 * t) / tf)

        A = area[area > area[0] * 0.2] / area[0]
        # print(f"{filename} {area[0]}")

        plt.plot(time, A, marker="o")

    plt.xlabel("Time")
    plt.ylabel(r"Area ($\mu m ^2$)")
    plt.title(f"Normalise Area {fileType}")
    fig.savefig(
        f"results/Wound Area norm 2 {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

    df = pd.DataFrame(_df)
    A = []
    Time = []
    std = []
    T = np.linspace(0, 1, 20)
    N = len(filenames)
    for t in T:
        df1 = df[df["Time"] > t]
        df2 = df1[df1["Time"] < t + 0.05]
        if len(df2) > N / 3:
            Time.append(t)
            A.append(np.mean(df2["Area"]))
            std.append(np.std(df2["Area"]))

    fig = plt.figure(1, figsize=(5, 5))
    plt.errorbar(Time, A, yerr=std, marker="o")
    plt.xlabel("Time")
    plt.ylabel(r" Mean Area ($\mu m ^2$)")
    plt.title(f"Normalise Mean Area {fileType}")
    fig.savefig(
        f"results/Mean Wound Area norm 2 {fileType}",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

# Wound Radius Mean
if False:
    fig = plt.figure(1, figsize=(9, 8))
    plt.errorbar(time, R, yerr=err)
    plt.gcf().subplots_adjust(left=0.2)
    plt.title(f"Mean finish time = {meanFinish}")
    plt.suptitle("Wound Area")
    plt.xlabel("Time (mins)")
    plt.ylabel(r"Area ($\mu m ^2$)")
    fig.savefig(
        f"results/Wound Area Mean {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

    R = np.array(R)
    err = np.array(err)

    fig = plt.figure(1, figsize=(9, 8))
    plt.errorbar(time, (R / np.pi) ** 0.5, yerr=(err / np.pi) ** 0.5)
    plt.gcf().subplots_adjust(left=0.2)
    plt.title(f"Mean finish time = {meanFinish}")
    plt.suptitle("Wound Radius")
    plt.xlabel("Time (mins)")
    plt.ylabel(r"Radius ($\mu m$)")
    fig.savefig(
        f"results/Wound Radius Mean {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

#  ------------------- Radius around wound thats fully in frame

# max distance from wound
if False:
    rList = [[] for col in range(T)]
    for filename in filenames:
        dist = sm.io.imread(f"dat/{filename}/distanceWound{filename}.tif").astype(
            "uint16"
        )
        for t in range(T):
            Max = dist[t].max()
            dist[t][1:511] = Max
            rList[t].append(dist[t].min())

    for t in range(T):
        rList[t] = np.mean(rList[t])

    t = range(T)
    rList = np.array(rList) * scale

    fig = plt.figure(1, figsize=(9, 8))
    plt.plot(t, rList)
    plt.ylim([0, 80])

    plt.xlabel(r"Time (mins)")
    plt.ylabel(r"distance from wound edge to frame edge ($\mu m$)")

    fig.savefig(
        f"results/max distance from wound edge {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

# correlation shape factor and healing time
if False:
    fig = plt.figure(1, figsize=(9, 8))
    plt.scatter(sf, endTime)
    plt.xlabel(r"sf")
    plt.ylabel(r"end time")

    fig.savefig(
        f"results/correlation {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

# correlation healing time and mitosis
if False:
    cor, a = pearsonr(endTime, divisions)
    fig = plt.figure(1, figsize=(9, 8))
    plt.scatter(endTime, divisions)
    plt.xlabel(r"end time")
    plt.ylabel(r"divisions")
    plt.title(f"Pearsons Correlation = {cor}")

    fig.savefig(
        f"results/correlation healing time and mitosis {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

# end time
if False:
    x = np.array(range(len(endTime))) + 1
    fig = plt.figure(1, figsize=(9, 8))
    plt.scatter(x, endTime)
    plt.xlabel(r"video")
    plt.ylabel(r"end time")

    fig.savefig(
        f"results/end time {fileType}",
        dpi=300,
        transparent=True,
    )
    plt.close("all")

# Radial Wound
if False:

    for filename in filenames:
        dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
        vid = sm.io.imread(f"dat/{filename}/ecadFocus{filename}.tif").astype(int)
        woundsite = sm.io.imread(f"dat/{filename}/woundsite{filename}.tif").astype(int)

        wound = np.zeros([181, 20])
        intWound = np.zeros([181, 20])
        for t in range(T):
            x, y = dfWound["Position"].iloc[t]
            x = int(x)
            y = 512 - int(y)
            img = np.zeros([512, 512])
            img[y, x] = 1
            img = sp.ndimage.morphology.distance_transform_edt(1 - img)
            intensity = []
            intensityWound = []
            for r in range(20):
                intensity.append(
                    np.mean(vid[t][(5 * r / scale < img) & (img < 5 * (r + 1) / scale)])
                )
                intensityWound.append(
                    np.mean(
                        woundsite[t][
                            (5 * r / scale < img) & (img < 5 * (r + 1) / scale)
                        ]
                    )
                )

            wound[t] = intensity
            intWound[t] = intensityWound

        t, r = np.mgrid[0:181:1, 1:100:5]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        c = ax[0].pcolor(t, r, wound)
        fig.colorbar(c, ax=ax[0])
        ax[0].set_xlabel("Time (mins)")
        ax[0].set_ylabel(r"Distance from wound center $(\mu m)$")
        ax[0].title.set_text(f"Intensity {filename}")

        c = ax[1].pcolor(t, r, 255 - intWound)
        fig.colorbar(c, ax=ax[1])
        ax[1].set_xlabel("Time (mins)")
        ax[1].set_ylabel(r"Distance from wound center $(\mu m)$")
        ax[1].title.set_text(f"Wound {filename}")
        fig.savefig(
            f"results/Radial Wound {filename}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

# run all Mean Wound Area
if False:
    fileTypes = [
        "WoundS18h",
        "WoundL18h",
        "WoundXL18h",
        "WoundSJNK",
        "WoundLJNK",
        "WoundXLJNK",
    ]
    for fileType in fileTypes:
        filenames, fileType = util.getFilesType(fileType)

        T = 93
        fig = plt.figure(1, figsize=(9, 8))
        sf = []
        endTime = []
        divisions = []
        R = [[] for col in range(T)]
        _df = []
        for filename in filenames:
            t0 = util.findStartTime(filename)

            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
            sf.append(dfWound["Shape Factor"].iloc[0])
            time = np.array(dfWound["Time"])
            area = np.array(dfWound["Area"]) * (scale) ** 2

            tf = sum(area > 0)
            endTime.append(tf)
            for t in range(T):
                if pd.isnull(area[t]):
                    area[t] = 0

            for t in range(T):
                if area[t] > area[0] * 0.2:
                    R[t].append(area[t])
                    _df.append({"Area": area[t], "Time": int(t0 / 2) * 2 + 2 * t})

            A = area[area > area[0] * 0.2]
            # print(f"{filename} {area[0]}")

            plt.plot(t0 + np.arange(0, len(A) * 2, 2), A)

        plt.xlabel("Time")
        plt.ylabel(r"Area ($\mu m ^2$)")
        plt.title(f"Area {fileType}")
        fig.savefig(
            f"results/Wound Area {fileType}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

        df = pd.DataFrame(_df)
        A = []
        Time = []
        std = []
        T = set(df["Time"])
        N = len(filenames)
        for t in T:
            if len(df[df["Time"] == t]) > N / 3:
                Time.append(t)
                A.append(np.mean(df["Area"][df["Time"] == t]))
                std.append(np.std(df["Area"][df["Time"] == t]))

        fig = plt.figure(1, figsize=(9, 8))
        plt.errorbar(Time, A, yerr=std)
        plt.xlabel("Time")
        plt.ylabel(r" Mean Area ($\mu m ^2$)")
        plt.title(f"Mean Area {fileType}")
        fig.savefig(
            f"results/Mean Wound Area {fileType}",
            bbox_inches="tight",
        )
        plt.close("all")

# run all compare Wt and KD
if False:
    fileTypes = ["WoundS18h", "WoundSJNK"]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for fileType in fileTypes:
        filenames, fileType = util.getFilesType(fileType)

        T = 93
        R = [[] for col in range(T)]
        _df = []
        for filename in filenames:
            t0 = util.findStartTime(filename)

            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
            time = np.array(dfWound["Time"])
            area = np.array(dfWound["Area"]) * (scale) ** 2

            for t in range(T):
                if pd.isnull(area[t]):
                    area[t] = 0

            for t in range(T):
                if area[t] > area[0] * 0.2:
                    R[t].append(area[t])
                    _df.append({"Area": area[t], "Time": int(t0 / 2) * 2 + 2 * t})

        df = pd.DataFrame(_df)
        A = []
        Time = []
        std = []
        T = set(df["Time"])
        N = len(filenames)
        for t in T:
            if len(df[df["Time"] == t]) > N / 3:
                Time.append(t)
                A.append(np.mean(df["Area"][df["Time"] == t]))
                std.append(np.std(df["Area"][df["Time"] == t]))

        typeName = util.getFileTitle(fileType)
        ax.errorbar(Time, A, yerr=std, label=f"{typeName}")

    ax.set(xlabel="Time", ylabel=r"Mean Area ($\mu m ^2$)")
    ax.title.set_text(f"Mean Area WoundS")
    ax.legend()
    fig.savefig(
        f"results/Compare Mean Wound Area WoundS",
        bbox_inches="tight",
    )
    plt.close("all")

    fileTypes = ["WoundL18h", "WoundLJNK"]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for fileType in fileTypes:
        filenames, fileType = util.getFilesType(fileType)

        T = 93
        R = [[] for col in range(T)]
        _df = []
        for filename in filenames:
            t0 = util.findStartTime(filename)

            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
            time = np.array(dfWound["Time"])
            area = np.array(dfWound["Area"]) * (scale) ** 2

            for t in range(T):
                if pd.isnull(area[t]):
                    area[t] = 0

            for t in range(T):
                if area[t] > area[0] * 0.2:
                    R[t].append(area[t])
                    _df.append({"Area": area[t], "Time": int(t0 / 2) * 2 + 2 * t})

        df = pd.DataFrame(_df)
        A = []
        Time = []
        std = []
        T = set(df["Time"])
        N = len(filenames)
        for t in T:
            if len(df[df["Time"] == t]) > N / 3:
                Time.append(t)
                A.append(np.mean(df["Area"][df["Time"] == t]))
                std.append(np.std(df["Area"][df["Time"] == t]))

        typeName = util.getFileTitle(fileType)
        ax.errorbar(Time, A, yerr=std, label=f"{typeName}")

    ax.set(xlabel="Time", ylabel=r"Mean Area ($\mu m ^2$)")
    ax.title.set_text(f"Mean Area WoundL")
    ax.legend()
    fig.savefig(
        f"results/Compare Mean Wound Area WoundL",
        bbox_inches="tight",
    )
    plt.close("all")

    fileTypes = ["WoundXL18h", "WoundXLJNK"]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for fileType in fileTypes:
        filenames, fileType = util.getFilesType(fileType)

        T = 93
        R = [[] for col in range(T)]
        _df = []
        for filename in filenames:
            t0 = util.findStartTime(filename)

            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
            time = np.array(dfWound["Time"])
            area = np.array(dfWound["Area"]) * (scale) ** 2

            for t in range(T):
                if pd.isnull(area[t]):
                    area[t] = 0

            for t in range(T):
                if area[t] > area[0] * 0.2:
                    R[t].append(area[t])
                    _df.append({"Area": area[t], "Time": int(t0 / 2) * 2 + 2 * t})

        df = pd.DataFrame(_df)
        A = []
        Time = []
        std = []
        T = set(df["Time"])
        N = len(filenames)
        for t in T:
            if len(df[df["Time"] == t]) > N / 3:
                Time.append(t)
                A.append(np.mean(df["Area"][df["Time"] == t]))
                std.append(np.std(df["Area"][df["Time"] == t]))

        typeName = util.getFileTitle(fileType)
        ax.errorbar(Time, A, yerr=std, label=f"{typeName}")

    ax.set(xlabel="Time", ylabel=r"Mean Area ($\mu m ^2$)")
    ax.title.set_text(f"Mean Area WoundXL")
    ax.legend()
    fig.savefig(
        f"results/Compare Mean Wound Area WoundXL",
        bbox_inches="tight",
    )
    plt.close("all")
