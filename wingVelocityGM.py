from ast import Break
import os
from os.path import exists
import shutil
from math import floor, log10, factorial

from collections import Counter
from trace import Trace
from turtle import position
import cv2
import matplotlib
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import random
import scipy as sp
import scipy.special as sc
import scipy.linalg as linalg
import shapely
import skimage as sm
import skimage.io
import skimage.measure
import skimage.feature
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
from mpl_toolkits.mplot3d import Axes3D
import tifffile
from skimage.draw import circle_perimeter
from scipy import optimize
import xml.etree.ElementTree as et
from scipy.optimize import leastsq
from datetime import datetime
import cellProperties as cell
import utils as util

plt.rcParams.update({"font.size": 16})

# -------------------

fileTypes, groupTitle = util.getFilesTypes()
T = 90
scale = 123.26 / 512

# -------------------

# compare: Mean migration of cells in tissue in x snd y
if True:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for fileType in fileTypes:
        filenames = util.getFilesType(fileType)[0]
        dfVelocityMean = pd.read_pickle(f"databases/dfVelocityMean{fileType}.pkl")
        df = dfVelocityMean[dfVelocityMean["Filename"] == filenames[0]]
        n = len(df)
        V = np.zeros([len(filenames), n // 5, 2])
        for i in range(len(filenames)):
            filename = filenames[i]
            df = dfVelocityMean[dfVelocityMean["Filename"] == filename]
            mig = 0
            for t in range(n):
                mig += np.mean(df["V"][df["T"] == t], axis=0)
                if t % 5 == 0:
                    V[i, int(t / 5)] = mig

        std = np.std(V, axis=0)
        V = np.mean(V, axis=0)
        colour, mark = util.getColorLineMarker(fileType, groupTitle)
        fileTitle = util.getFileTitle(fileType)
        ax.errorbar(
            V[:, 0],
            V[:, 1],
            std[:, 0],
            std[:, 1],
            label=fileTitle,
            color=colour,
            marker=mark,
        )

    ax.set(xlabel=r"x ($\mu m$)", ylabel=r"y ($\mu m$)")
    boldTitle = util.getBoldTitle(groupTitle)
    ax.title.set_text(f"Migration of cells in tissue \n {boldTitle}")
    ax.set_xlim([-55, 5])
    ax.set_ylim([-10, 30])
    ax.legend(loc="upper right", fontsize=12)

    fig.savefig(
        f"results/Migration of cells in tissue {groupTitle}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

# compare: Mean migration of cells in tissue in r
if True:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for fileType in fileTypes:
        filenames = util.getFilesType(fileType)[0]
        dfVelocityMean = pd.read_pickle(f"databases/dfVelocityMean{fileType}.pkl")
        df = dfVelocityMean[dfVelocityMean["Filename"] == filenames[0]]
        n = len(df)
        V = np.zeros([len(filenames), n])
        for i in range(len(filenames)):
            filename = filenames[i]
            df = dfVelocityMean[dfVelocityMean["Filename"] == filename]
            mig = 0
            for t in range(n):
                mig += np.mean(df["V"][df["T"] == t], axis=0)
                V[i, int(t)] = (mig[0] ** 2 + mig[1] ** 2) ** 0.5

        time = 2 * np.array(range(int(T)))

        std = np.std(V, axis=0)
        V = np.mean(V, axis=0)
        colour, mark = util.getColorLineMarker(fileType, groupTitle)
        fileTitle = util.getFileTitle(fileType)
        ax.plot(time, V, label=fileTitle, color=colour, marker=mark, markevery=10)
        ax.fill_between(time, V - std, V + std, alpha=0.15, color=colour)

    ax.set(xlabel="Time (mins)", ylabel=r"r ($\mu m$)")
    boldTitle = util.getBoldTitle(groupTitle)
    ax.title.set_text(f"Migration magnitude of cells in tissue \n {boldTitle}")
    ax.set_ylim([0, 60])
    ax.legend(loc="upper left", fontsize=12)

    fig.savefig(
        f"results/Migration magnitude of cells in tissue {groupTitle}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")
