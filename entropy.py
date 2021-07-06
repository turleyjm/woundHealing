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
from scipy.stats import mannwhitneyu
import shapely
import skimage as sm
import skimage.feature
import skimage.io
import skimage.measure
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
import tifffile
from skimage.draw import circle_perimeter
from scipy import optimize
import xml.etree.ElementTree as et
import plotly.graph_objects as go
from scipy.optimize import leastsq

import cellProperties as cell
import findGoodCells as fi
import commonLiberty as cl


plt.rcParams.update({"font.size": 16})

filenames, fileType = cl.getFilesType()

scale = 147.91 / 512
T = 181

_dfEntropy = []

if False:
    for filename in filenames:
        print(filename)
        df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")

        T = np.linspace(0, 170, 18)
        for t in T:

            dft = cl.sortTime(df, [t, t + 10])
            m = len(dft)

            Q = np.zeros([m, 2])
            for i in range(m):
                Q[i] = dft["q"].iloc[i][0]

            heatmap, xedges, yedges = np.histogram2d(
                Q[:, 0], Q[:, 1], range=[[-0.3, 0.3], [-0.3, 0.3]], bins=60
            )

            if False:
                x, y = np.mgrid[-0.3:0.3:0.01, -0.3:0.3:0.01]
                fig, ax = plt.subplots(figsize=(8, 8))

                c = ax.pcolor(x, y, heatmap, cmap="Reds")
                fig.colorbar(c, ax=ax)
                ax.set_xlabel("Time (min)")
                ax.set_ylabel(r"$R (\mu m)$ ")
                ax.title.set_text(f"Correlation {filename}")

                fig.savefig(
                    f"results/entropy q1 {filename}",
                    dpi=300,
                    transparent=True,
                )
                plt.close("all")

            prob = heatmap / m
            prob[prob != 0]
            p = prob[prob != 0]

            entropy = p * np.log(p)
            entropyQ = -sum(entropy)

            P = np.zeros([m, 2])
            for i in range(m):
                xp = dft["Polar"].iloc[i][0]
                yp = dft["Polar"].iloc[i][1]
                theta = dft["Orientation"].iloc[i]

                x = xp * np.cos(theta) - yp * np.sin(theta)
                y = xp * np.sin(theta) + yp * np.cos(theta)

                P[i, 0] = x
                P[i, 1] = y

            heatmap, xedges, yedges = np.histogram2d(
                P[:, 0], P[:, 1], range=[[-0.15, 0.15], [-0.15, 0.15]], bins=60
            )

            if False:
                x, y = np.mgrid[-0.3:0.3:0.01, -0.3:0.3:0.01]
                fig, ax = plt.subplots(figsize=(8, 8))

                c = ax.pcolor(x, y, heatmap, cmap="Reds")
                fig.colorbar(c, ax=ax)
                ax.set_xlabel("Time (min)")
                ax.set_ylabel(r"$R (\mu m)$ ")
                ax.title.set_text(f"Correlation {filename}")

                fig.savefig(
                    f"results/entropy polar {filename}",
                    dpi=300,
                    transparent=True,
                )
                plt.close("all")

            prob = heatmap / m
            prob[prob != 0]
            p = prob[prob != 0]

            entropy = p * np.log(p)
            entropyP = -sum(entropy)

            _dfEntropy.append(
                {
                    "filename": filename,
                    "Entropy Q": entropyQ,
                    "Entropy Polar": entropyP,
                    "T": t,
                }
            )

    dfEntropy = pd.DataFrame(_dfEntropy)
    dfEntropy.to_pickle(f"databases/dfEntropy{fileType}.pkl")
else:
    dfEntropy = pd.read_pickle(f"databases/dfEntropy{fileType}.pkl")

if fileType != "Unwound":
    finish = []
    for filename in filenames:

        dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")

        area = np.array(dfWound["Area"]) * (scale) ** 2
        t = 0
        while pd.notnull(area[t]):
            t += 1

        finish.append(t - 1)

    medianFinish = int(np.median(finish))

T = np.linspace(0, 170, 18)
entropyQ = []
entropyP = []
mseEntropyQ = []
mseEntropyP = []
n = len(filenames)
for t in T:
    entropyQ.append(np.mean(dfEntropy["Entropy Q"][dfEntropy["T"] == t]))
    entropyP.append(np.mean(dfEntropy["Entropy Polar"][dfEntropy["T"] == t]))
    mseEntropyQ.append(np.std(dfEntropy["Entropy Q"][dfEntropy["T"] == t]) / n ** 0.5)
    mseEntropyP.append(
        np.std(dfEntropy["Entropy Polar"][dfEntropy["T"] == t]) / n ** 0.5
    )


fig, ax = plt.subplots(2, 1, figsize=(9, 8))
ax[0].errorbar(T, entropyQ, yerr=mseEntropyQ)
ax[0].set_xlabel("Time (mins)")
ax[0].set_ylabel("Entropy of q1")
ax[0].set_ylim([4.3, 5])

ax[1].errorbar(T, entropyP, yerr=mseEntropyP)
ax[1].set_xlabel("Time (mins)")
ax[1].set_ylabel(r"Entropy of polar")
ax[1].set_ylim([2.5, 3])

plt.suptitle(f"Shape entropy {fileType}")
if fileType != "Unwound":
    ax[0].axvline(x=medianFinish, color="Red")
    ax[1].axvline(x=medianFinish, color="Red")

fig.savefig(
    f"results/entropy change over time {fileType}",
    dpi=300,
    transparent=True,
)
plt.close("all")