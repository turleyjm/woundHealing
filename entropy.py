import os
from math import floor, log10
from scipy.special import gamma

import cv2
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg as linalg
import shapely
import skimage as sm
import skimage.io
import skimage.measure
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing

import cellProperties as cell
import findGoodCells as fi

plt.rcParams.update({"font.size": 28})

plt.ioff()
pd.set_option("display.width", 1000)

cwd = os.getcwd()
filenames = os.listdir(cwd + "/dat_databases")
filenames.sort()
filenames = filenames[1:]

n = len(filenames)
num_frames = 14
num_videos = int(n / 14)

_dfEntropy = []

for video in range(num_videos):

    for frame in range(num_frames):
        filename = filenames[14 * video + frame]

        df = pd.read_pickle(f"dat_databases/{filename}")
        m = len(df)

        Q = np.zeros([m, 2])
        for i in range(m):
            Q[i] = df["Q"].iloc[i][0]

        heatmap, xedges, yedges = np.histogram2d(
            Q[:, 0], Q[:, 1], range=[[-0.2, 0.2], [-0.2, 0.2]], bins=25
        )

        prob = heatmap / m
        prob[prob != 0]
        p = prob[prob != 0]

        entropy = p * np.log(p)
        entropy = -sum(entropy)

        _dfEntropy.append(
            {"filename": filename, "Time": frame, "Entropy": entropy,}
        )

dfEntropy = pd.DataFrame(_dfEntropy)

mu = []
err = []

for frame in range(num_frames):
    prop = list(dfEntropy["Entropy"][dfEntropy["Time"] == frame])
    mu.append(cell.mean(prop))
    err.append(cell.sd(prop) / (len(prop) ** 0.5))

x = range(14)

fig = plt.figure(1, figsize=(9, 8))
plt.gcf().subplots_adjust(left=0.2)
plt.errorbar(x, mu, yerr=err, fmt="o")

plt.xlabel("Time")
plt.ylabel(f"Entropy")
plt.gcf().subplots_adjust(bottom=0.2)
fig.savefig(
    f"results/Entropy", dpi=300, transparent=True,
)
plt.close("all")

