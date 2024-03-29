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
import skimage.feature
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
from sympy import true
import tifffile
from skimage.draw import circle_perimeter
from scipy import optimize
import xml.etree.ElementTree as et
import matplotlib.colors as colors
import seaborn as sns

import cellProperties as cell
import utils as util

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

filenames, fileType = util.getFilesType()
scale = 123.26 / 512
T = 8
plt.rcParams.update({"font.size": 22})

# -------------------

# Q_0 for prewound and early unwounded
if True:
    _df = []
    fileType = "Unwound18h"
    filenames, fileType = util.getFilesType(fileType)
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        _df.append(
            {
                "Type": "Unwound18h wt",
                "Filename": filename,
                "q": np.mean(df["q"][df["T"] < 8])[0, 0],
            }
        )
        # q = np.mean(df["q"][df["T"] < 8])[0, 0]
        # print(f"{filename}: {q}")

    fileType = "Unwound15h"
    filenames, fileType = util.getFilesType(fileType)
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        _df.append(
            {
                "Type": "Unwound14.75h wt",
                "Filename": filename,
                "q": np.mean(df["q"][df["T"] < 8])[0, 0],
            }
        )
        # q = np.mean(df["q"][df["T"] < 8])[0, 0]
        # print(f"{filename}: {q}")

    fileType = "Unwound26h"
    filenames, fileType = util.getFilesType(fileType)
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        _df.append(
            {
                "Type": "Unwound26h wt",
                "Filename": filename,
                "q": np.mean(df["q"][df["T"] < 8])[0, 0],
            }
        )
        # q = np.mean(df["q"][df["T"] < 8])[0, 0]
        # print(f"{filename}: {q}")

    fileType = "WoundS18h"
    filenames, fileType = util.getFilesType(fileType)
    dfShape = pd.read_pickle(f"databases/dfShape{util.Prewound(fileType)}.pkl")
    for filename in filenames:
        filenamePre = util.Prewound(filename)
        df = dfShape[dfShape["Filename"] == filenamePre]
        if len(df) > 0:
            _df.append(
                {
                    "Type": "Pre-wound wt",
                    "Filename": filenamePre,
                    "q": np.mean(df["q"], axis=0)[0, 0],
                }
            )
            # q = np.mean(df["q"])[0, 0]
            # print(f"{filenamePre}: {q}")

    fileType = "WoundL18h"
    filenames, fileType = util.getFilesType(fileType)
    dfShape = pd.read_pickle(f"databases/dfShape{util.Prewound(fileType)}.pkl")
    for filename in filenames:
        filenamePre = util.Prewound(filename)
        df = dfShape[dfShape["Filename"] == filenamePre]
        if len(df) > 0:
            _df.append(
                {
                    "Type": "Pre-wound wt",
                    "Filename": filenamePre,
                    "q": np.mean(df["q"])[0, 0],
                }
            )
            # q = np.mean(df["q"])[0, 0]
            # print(f"{filenamePre}: {q}")

    fileType = "UnwoundJNK"
    filenames, fileType = util.getFilesType(fileType)
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        _df.append(
            {
                "Type": "Unwound JNK",
                "Filename": filename,
                "q": np.mean(df["q"][df["T"] < 8])[0, 0],
            }
        )
        # q = np.mean(df["q"][df["T"] < 8])[0, 0]
        # print(f"{filename}: {q}")

    fileType = "WoundSJNK"
    filenames, fileType = util.getFilesType(fileType)
    dfShape = pd.read_pickle(f"databases/dfShape{util.Prewound(fileType)}.pkl")
    for filename in filenames:
        filenamePre = util.Prewound(filename)
        df = dfShape[dfShape["Filename"] == filenamePre]
        if len(df) > 0:
            _df.append(
                {
                    "Type": "Pre-wound JNK",
                    "Filename": filenamePre,
                    "q": np.mean(df["q"])[0, 0],
                }
            )
            # q = np.mean(df["q"])[0, 0]
            # print(f"{filenamePre}: {q}")

    fileType = "WoundLJNK"
    filenames, fileType = util.getFilesType(fileType)
    dfShape = pd.read_pickle(f"databases/dfShape{util.Prewound(fileType)}.pkl")
    for filename in filenames:
        filenamePre = util.Prewound(filename)
        df = dfShape[dfShape["Filename"] == filenamePre]
        if len(df) > 0:
            _df.append(
                {
                    "Type": "Pre-wound JNK",
                    "Filename": filenamePre,
                    "q": np.mean(df["q"])[0, 0],
                }
            )
            # q = np.mean(df["q"])[0, 0]
            # print(f"{filenamePre}: {q}")

    fileType = "Unwoundrpr"
    filenames, fileType = util.getFilesType(fileType)
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        _df.append(
            {
                "Type": "Unwound Immune ab.",
                "Filename": filename,
                "q": np.mean(df["q"][df["T"] < 8])[0, 0],
            }
        )
        # q = np.mean(df["q"][df["T"] < 8])[0, 0]
        # print(f"{filename}: {q}")

    fileType = "WoundSrpr"
    filenames, fileType = util.getFilesType(fileType)
    dfShape = pd.read_pickle(f"databases/dfShape{util.Prewound(fileType)}.pkl")
    for filename in filenames:
        filenamePre = util.Prewound(filename)
        df = dfShape[dfShape["Filename"] == filenamePre]
        if len(df) > 0:
            _df.append(
                {
                    "Type": "Pre-wound Immune ab.",
                    "Filename": filenamePre,
                    "q": np.mean(df["q"])[0, 0],
                }
            )
            # q = np.mean(df["q"])[0, 0]
            # print(f"{filenamePre}: {q}")

    fileType = "WoundLrpr"
    filenames, fileType = util.getFilesType(fileType)
    dfShape = pd.read_pickle(f"databases/dfShape{util.Prewound(fileType)}.pkl")
    for filename in filenames:
        filenamePre = util.Prewound(filename)
        df = dfShape[dfShape["Filename"] == filenamePre]
        if len(df) > 0:
            _df.append(
                {
                    "Type": "Pre-wound Immune ab.",
                    "Filename": filenamePre,
                    "q": np.mean(df["q"])[0, 0],
                }
            )
            # q = np.mean(df["q"])[0, 0]
            # print(f"{filenamePre}: {q}")

    fileType = "UnwoundCa"
    filenames, fileType = util.getFilesType(fileType)
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    for filename in filenames:
        df = dfShape[dfShape["Filename"] == filename]
        _df.append(
            {
                "Type": "Unwound Ca",
                "Filename": filename,
                "q": np.mean(df["q"][df["T"] < 8])[0, 0],
            }
        )
        # q = np.mean(df["q"][df["T"] < 8])[0, 0]
        # print(f"{filename}: {q}")

    fileType = "WoundLCa"
    filenames, fileType = util.getFilesType(fileType)
    dfShape = pd.read_pickle(f"databases/dfShape{util.Prewound(fileType)}.pkl")
    for filename in filenames:
        filenamePre = util.Prewound(filename)
        df = dfShape[dfShape["Filename"] == filenamePre]
        if len(df) > 0:
            _df.append(
                {
                    "Type": "Pre-wound Ca",
                    "Filename": filenamePre,
                    "q": np.mean(df["q"])[0, 0],
                }
            )
            # q = np.mean(df["q"])[0, 0]
            # print(f"{filenamePre}: {q}")

    sns.set(font_scale=1.5)
    df = pd.DataFrame(_df)
    ax = sns.boxplot(
        y="q",
        x="Type",
        data=df,
        boxprops={"facecolor": "None"},
    )
    plt.draw()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_ylim([-0.005, 0.03])
    sns_plot = sns.swarmplot(data=df, y="q", x="Type")
    fig = sns_plot.get_figure()
    fig.savefig(
        f"results/compare q prewound",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")
    # sp.stats.ttest_ind(df["q"][df["Type"]=="Unwound18h wt"], df["q"][df["Type"]=="Prewound rpr"])

# division density for prewound and early unwounded
if False:
    _df = []
    fileType = "Unwound18h"
    filenames, fileType = util.getFilesType(fileType)
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    for filename in filenames:
        df = dfDivisions[dfDivisions["Filename"] == filename]
        outPlane = sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype(int)
        area = np.sum(255 - outPlane[1:5]) / 255 * scale**2
        if area == 0:
            print(0)
        _df.append(
            {
                "Type": "Unwound wt",
                "Filename": filename,
                "Divsion density": len(df[(df["T"] >= 1 * 2) & (df["T"] < 5 * 2)])
                * 5000
                / area,
            }
        )

    fileType = "WoundS18h"
    filenames, fileType = util.getFilesType(fileType)
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{util.Prewound(fileType)}.pkl")
    for filename in filenames:
        filenamePre = util.Prewound(filename)
        df = dfDivisions[dfDivisions["Filename"] == filenamePre]
        if len(df) > 0:
            outPlane = sm.io.imread(
                f"dat/{filename}/{filenamePre}/outPlane{filenamePre}.tif"
            ).astype(int)
            area = np.sum(255 - outPlane[1:5]) / 255 * scale**2
            _df.append(
                {
                    "Type": "Prewound wt",
                    "Filename": filenamePre,
                    "Divsion density": len(df) * 5000 / area,
                }
            )
    fileType = "WoundL18h"
    filenames, fileType = util.getFilesType(fileType)
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{util.Prewound(fileType)}.pkl")
    for filename in filenames:
        filenamePre = util.Prewound(filename)
        df = dfDivisions[dfDivisions["Filename"] == filenamePre]
        if len(df) > 0:
            outPlane = sm.io.imread(
                f"dat/{filename}/{filenamePre}/outPlane{filenamePre}.tif"
            ).astype(int)
            area = np.sum(255 - outPlane[1:5]) / 255 * scale**2
            _df.append(
                {
                    "Type": "Prewound wt",
                    "Filename": filenamePre,
                    "Divsion density": len(df) * 5000 / area,
                }
            )

    fileType = "UnwoundJNK"
    filenames, fileType = util.getFilesType(fileType)
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    for filename in filenames:
        df = dfDivisions[dfDivisions["Filename"] == filename]
        outPlane = sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype(int)
        area = np.sum(255 - outPlane[1:5]) / 255 * scale**2
        _df.append(
            {
                "Type": "Unwound JNK",
                "Filename": filename,
                "Divsion density": len(df[(df["T"] >= 1 * 2) & (df["T"] < 5 * 2)])
                * 5000
                / area,
            }
        )

    fileType = "WoundSJNK"
    filenames, fileType = util.getFilesType(fileType)
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{util.Prewound(fileType)}.pkl")
    for filename in filenames:
        filenamePre = util.Prewound(filename)
        df = dfDivisions[dfDivisions["Filename"] == filenamePre]
        if len(df) > 0:
            outPlane = sm.io.imread(
                f"dat/{filename}/{filenamePre}/outPlane{filenamePre}.tif"
            ).astype(int)
            area = np.sum(255 - outPlane[1:5]) / 255 * scale**2
            _df.append(
                {
                    "Type": "Prewound JNK",
                    "Filename": filenamePre,
                    "Divsion density": len(df) * 5000 / area,
                }
            )
    fileType = "WoundLJNK"
    filenames, fileType = util.getFilesType(fileType)
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{util.Prewound(fileType)}.pkl")
    for filename in filenames:
        filenamePre = util.Prewound(filename)
        df = dfDivisions[dfDivisions["Filename"] == filenamePre]
        if len(df) > 0:
            outPlane = sm.io.imread(
                f"dat/{filename}/{filenamePre}/outPlane{filenamePre}.tif"
            ).astype(int)
            area = np.sum(255 - outPlane[1:5]) / 255 * scale**2
            _df.append(
                {
                    "Type": "Prewound JNK",
                    "Filename": filenamePre,
                    "Divsion density": len(df) * 5000 / area,
                }
            )

    df = pd.DataFrame(_df)
    sns.boxplot(y="Divsion density", x="Type", data=df, boxprops={"facecolor": "None"})
    sns_plot = sns.swarmplot(data=df, y="Divsion density", x="Type")
    fig = sns_plot.get_figure()
    fig.savefig(
        f"results/compare divsion density prewound",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")
    # sp.stats.ttest_ind(df["Divsion density"][df["Type"]=="Prewound wt"], df["Divsion density"][df["Type"]=="Prewound JNK"])

# cross lethality
if False:

    height = [
        63,
        81,
        0,
        117,
        116,
    ]
    bars = ("1x4", "2x4", "3x5", "6x4", "6x5")

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    # Create bars
    ax.bar(bars, height, color=("green", "magenta", "cyan", "orange", "orange"))
    ax.set(ylabel="Normalised number of F1 progeny")
    ax.title.set_text(f"Lethality of F1 progeny")

    fig.savefig(
        f"results/Lethality of F1 progeny",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")
