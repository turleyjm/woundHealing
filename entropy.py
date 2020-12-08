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
import tifffile

import cellProperties as cell
import findGoodCells as fi

plt.rcParams.update({"font.size": 20})

plt.ioff()
pd.set_option("display.width", 1000)


def best_fit_slope_and_intercept(xs, ys):
    m = ((cell.mean(xs) * cell.mean(ys)) - cell.mean(xs * ys)) / (
        (cell.mean(xs) * cell.mean(xs)) - cell.mean(xs * xs)
    )

    b = cell.mean(ys) - m * cell.mean(xs)

    return (m, b)


def round_sig(x, sig=2):

    return round(x, sig - int(floor(log10(abs(x)))) - 1)


cwd = os.getcwd()
filenames = os.listdir(cwd + "/dat_databases")
filenames.sort()
filenames = filenames[1:]

n = len(filenames)
num_frames = 14
num_videos = int(n / 14)

_dfEntropy = []

j = 0

for video in range(num_videos):

    for frame in range(num_frames):
        filename = filenames[14 * video + frame]

        df = pd.read_pickle(f"dat_databases/{filename}")
        m = len(df)

        Q = np.zeros([m, 2])
        for i in range(m):
            Q[i] = df["Q"].iloc[i][0]

        heatmap, xedges, yedges = np.histogram2d(
            Q[:, 0], Q[:, 1], range=[[-0.3, 0.3], [-0.3, 0.3]], bins=60
        )

        # if filename == "df_of_sample06_01.pkl":
        #     data = np.asarray(heatmap, "uint8")
        #     tifffile.imwrite(f"results/entropy histogram sample06_01.tif", data)
        # if filename == "df_of_sample06_07.pkl":
        #     data = np.asarray(heatmap, "uint8")
        #     tifffile.imwrite(f"results/entropy histogram sample06_07.tif", data)
        # if filename == "df_of_sample06_14.pkl":
        #     data = np.asarray(heatmap, "uint8")
        #     tifffile.imwrite(f"results/entropy histogram sample06_14.tif", data)

        prob = heatmap / m
        prob[prob != 0]
        p = prob[prob != 0]

        entropy = p * np.log(p)
        entropyQ = -sum(entropy)

        Q = np.zeros([m, 2])
        for i in range(m):
            xp = df["Polar_x"][i]
            yp = df["Polar_y"][i]
            theta = df["Orientation"][i]

            x = xp * np.cos(theta) - yp * np.sin(theta)
            y = xp * np.sin(theta) + yp * np.cos(theta)

            Q[i, 0] = x
            Q[i, 1] = y

        heatmap, xedges, yedges = np.histogram2d(
            Q[:, 0], Q[:, 1], range=[[-0.15, 0.15], [-0.15, 0.15]], bins=60
        )

        prob = heatmap / m
        prob[prob != 0]
        p = prob[prob != 0]

        entropy = p * np.log(p)
        entropyP = -sum(entropy)

        _dfEntropy.append(
            {
                "filename": filename,
                "Time": frame,
                "Entropy Q": entropyQ,
                "Entropy Polar": entropyP,
                "video": j,
            }
        )
    j += 1

dfEntropy = pd.DataFrame(_dfEntropy)

# ---------------------------

mu = []
err = []

for frame in range(num_frames):
    prop = list(dfEntropy["Entropy Q"][dfEntropy["Time"] == frame])
    mu.append(cell.mean(prop))
    err.append(cell.sd(prop) / (len(prop) ** 0.5))

x = range(14)

fig = plt.figure(1, figsize=(9, 8))
plt.gcf().subplots_adjust(left=0.2)
plt.errorbar(x, mu, yerr=err, fmt="o")

plt.xlabel("Time")
plt.ylabel(f"Entropy Q")
plt.gcf().subplots_adjust(bottom=0.2)
fig.savefig(
    f"results/EntropyQ", dpi=300, transparent=True,
)
plt.close("all")

# ---------------------------

mu = []
err = []

for frame in range(num_frames):
    prop = list(dfEntropy["Entropy Polar"][dfEntropy["Time"] == frame])
    mu.append(cell.mean(prop))
    err.append(cell.sd(prop) / (len(prop) ** 0.5))

x = range(14)

fig = plt.figure(1, figsize=(9, 8))
plt.gcf().subplots_adjust(left=0.2)
plt.errorbar(x, mu, yerr=err, fmt="o")

plt.xlabel("Time")
plt.ylabel(f"Entropy Polar")
plt.gcf().subplots_adjust(bottom=0.2)
fig.savefig(
    f"results/EntropyPolar", dpi=300, transparent=True,
)
plt.close("all")

# ---------------------------

mag = []
for video in range(num_videos):
    df = dfEntropy[dfEntropy["video"] == video]
    mu = []
    for frame in range(num_frames):
        prop = list(df["Entropy Q"][df["Time"] == frame])
        mu.append(cell.mean(prop))
    mag.append(best_fit_slope_and_intercept(np.array(x), np.array(mu))[0])

mu = cell.mean(mag)
sigma = cell.sd(mag)
mu = round_sig(mu, sig=2)
sigma = round_sig(sigma, sig=2)

fig, ax = plt.subplots()
plt.gcf().subplots_adjust(bottom=0.15)
plt.title(f"mean={mu}, sd={sigma}")
ax.hist(mag, density=True, bins=10)
ax.axvline(mu, c="k", label="mean")
ax.axvline(mu + sigma, c="k", label=r"$\sigma$", ls="--")
ax.axvline(mu - sigma, c="k", ls="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(
    f"results/gradient of Entropy Q", dpi=200, transparent=True,
)

# ---------------------------

mag = []
for video in range(num_videos):
    df = dfEntropy[dfEntropy["video"] == video]
    mu = []
    for frame in range(num_frames):
        prop = list(df["Entropy Polar"][df["Time"] == frame])
        mu.append(cell.mean(prop))
    mag.append(best_fit_slope_and_intercept(np.array(x), np.array(mu))[0])

mu = cell.mean(mag)
sigma = cell.sd(mag)
mu = round_sig(mu, sig=2)
sigma = round_sig(sigma, sig=2)

fig, ax = plt.subplots()
plt.gcf().subplots_adjust(bottom=0.15)
plt.title(f"mean={mu}, sd={sigma}")
ax.hist(mag, density=True, bins=10)
ax.axvline(mu, c="k", label="mean")
ax.axvline(mu + sigma, c="k", label=r"$\sigma$", ls="--")
ax.axvline(mu - sigma, c="k", ls="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.legend(loc="upper right", fontsize=18, bbox_to_anchor=(0.9, 0.85))
fig.savefig(
    f"results/gradient of Entropy Polar", dpi=200, transparent=True,
)
plt.close("all")

