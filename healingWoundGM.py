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

fileTypes, groupTitle = util.getFilesTypes()
scale = 123.26 / 512
T = 93

# -------------------

# Compare wounds: Mean wound area
if True:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.vlines(10, -100, 1500, colors="r", linestyles="dashed")
    ax.set_ylim([0, 1250])
    i = 0
    for fileType in fileTypes:
        if "Unw" in fileType:
            continue
        filenames, fileType = util.getFilesType(fileType)
        _df = []
        Area0 = []

        for filename in filenames:
            t0 = util.findStartTime(filename)
            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
            T = len(dfWound)
            area = np.array(dfWound["Area"]) * (scale) ** 2
            t10 = 5 - int(t0 / 2)
            Area0.append(area[t10])

            for t in range(T):
                if area[t] > area[0] * 0.2:
                    _df.append({"Area": area[t], "Time": int(t0 / 2) * 2 + 2 * t})
                else:
                    _df.append({"Area": 0, "Time": int(t0 / 2) * 2 + 2 * t})

        df = pd.DataFrame(_df)
        A = []
        time = []
        std = []
        T = set(df["Time"])
        N = len(filenames)
        Area0 = np.mean(Area0)
        for t in T:
            if len(df[df["Time"] == t]) > N / 3:
                if np.mean(df["Area"][df["Time"] == t]) > 0.2 * Area0:
                    time.append(t)
                    A.append(np.mean(df["Area"][df["Time"] == t]))
                    std.append(np.std(df["Area"][df["Time"] == t]))

        A = np.array(A)
        std = np.array(std)
        colour, mark = util.getColorLineMarker(fileType, groupTitle)
        fileTitle = util.getFileTitle(fileType)
        ax.plot(time, A, label=fileTitle, marker=mark, color=colour)
        ax.fill_between(time, A - std, A + std, alpha=0.15, color=colour)

    plt.xlabel("Time after wounding (mins)")
    plt.ylabel(r"Area ($\mu m ^2$)")
    boldTitle = util.getBoldTitle(groupTitle)
    plt.title(f"Mean area of wound \n {boldTitle}")
    plt.legend(loc="upper right", fontsize=12)
    fig.savefig(
        f"results/Compared mean wound area {groupTitle}",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")
