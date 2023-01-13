import os
import shutil
from math import floor, log10

from collections import Counter
import cv2
import matplotlib
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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

plt.rcParams.update({"font.size": 16})

# -------------------

fileTypes, groupTitle = util.getFilesTypes()

T = 90
scale = 123.26 / 512

# -------------------

# compare mean sf
if True:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for fileType in fileTypes:
        filenames = util.getFilesType(fileType)[0]
        dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
        sf = np.zeros([len(filenames), T])
        for i in range(len(filenames)):
            filename = filenames[i]
            df = dfShape[dfShape["Filename"] == filename]
            for t in range(T):
                sf[i, t] = np.mean(df["Shape Factor"][df["T"] == t])

        time = 2 * np.array(range(T))

        std = np.std(sf, axis=0)
        sf = np.mean(sf, axis=0)
        color = util.getColor(fileType)
        fileTitle = util.getFileTitle(fileType)
        ax.plot(time, sf, label=fileTitle, color=color)
        ax.fill_between(time, sf - std, sf + std, alpha=0.15, color=color)

    # ax.set_ylim([0.3, 0.46])
    ax.legend(loc="upper left", fontsize=12)
    ax.set(xlabel="Time after wounding (mins)", ylabel=r"$S_f$")
    boldTitle = util.getBoldTitle(groupTitle)
    ax.title.set_text(f"Mean shape factor with \n time " + boldTitle)
    fig.savefig(
        f"results/mean sf {groupTitle}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")
