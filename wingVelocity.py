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

pd.options.mode.chained_assignment = None
plt.rcParams.update({"font.size": 10})

# -------------------

filenames, fileType = util.getFilesType()
scale = 123.26 / 512

if False:
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    dfVelocityMean = pd.read_pickle(f"databases/dfVelocityMean{fileType}.pkl")
    for filename in filenames:
        df = dfVelocityMean[dfVelocityMean["Filename"] == filename]
        n = len(df)
        migration = []
        mig = 0
        for t in range(n):
            mig += np.mean(df["V"][df["T"] == t], axis=0)
            migration.append(list(mig))

        migration = np.array(migration)
        ax.plot(migration[:, 0], migration[:, 1])

    migration = []
    mig = 0
    for t in range(n):
        mig += np.mean(dfVelocityMean["V"][dfVelocityMean["T"] == t], axis=0)
        if t % 5 == 0:
            migration.append(list(mig))

    migration = np.array(migration)
    ax.plot(migration[:, 0], migration[:, 1], marker="o")
    ax.set(xlabel=r"x ($\mu m$)", ylabel=r"y ($\mu m$)")
    ax.title.set_text(f"Migration of tissue")
    ax.set_xlim([-65, 5])
    ax.set_ylim([-5, 40])

    fig.savefig(
        f"results/migration of tissue {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")


if True:
    dfVelocity = pd.read_pickle(f"databases/dfVelocity{fileType}.pkl")
    dv = np.stack(dfVelocity["dv"])

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.hist2d(
        dv[:, 0], dv[:, 1], bins=20, range=np.array([[-1, 1], [-1, 1]]), density=False
    )
    # plt.colorbar()
    ax.set(xlabel=r"x ($\mu m$)", ylabel=r"y ($\mu m$)")
    ax.title.set_text(r"Histagram of $\delta v$")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    fig.savefig(
        f"results/distribution of dv {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

    print(f"x std {np.std(dv[:, 0])}")
    print(f"y std {np.std(dv[:, 1])}")
    print(f"x skew {sp.stats.skew(dv[:, 0])}")
    print(f"y skew {sp.stats.skew(dv[:, 1])}")
    print(f"x kurtosis {sp.stats.kurtosis(dv[:, 0], fisher=True)}")
    print(f"y kurtosis {sp.stats.kurtosis(dv[:, 1], fisher=True)}")
