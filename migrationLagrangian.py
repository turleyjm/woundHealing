import os
import shutil
from math import floor, log10

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
from skimage.draw import circle_perimeter
from scipy import optimize

import cellProperties as cell
import findGoodCells as fi

plt.rcParams.update({"font.size": 20})
plt.ioff()
pd.set_option("display.width", 1000)

f = open("pythonText.txt", "r")

fileType = f.read()
cwd = os.getcwd()
Fullfilenames = os.listdir(cwd + "/dat")
filenames = []
for filename in Fullfilenames:
    if fileType in filename:
        filenames.append(filename)

filenames.sort()

T = 181
scale = 147.91 / 512

meanFinish = []
for filename in filenames:

    dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")

    area = np.array(dfWound["Area"]) * (scale) ** 2
    t = 0
    while pd.notnull(area[t]):
        t += 1

    meanFinish.append(t - 1)

meanFinish = 20

_df = []
for filename in filenames:
    dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
    dfNucleus = pd.read_pickle(f"dat/{filename}/nucleusTracks{filename}.pkl")
    centroid = dfWound["Centroid"]
    startWound = centroid[0]
    endWound = centroid[meanFinish]

    for i in range(len(dfNucleus)):
        t = dfNucleus["t"][i]
        x = dfNucleus["x"][i]
        y = dfNucleus["y"][i]
        label = dfNucleus["Label"][i]

        if t[0] == 0 and t[-1] > meanFinish:

            r0 = (
                ((startWound[0] - x[0]) ** 2 + (startWound[1] - y[0]) ** 2) ** 0.5
            ) * scale
            rt = (
                (
                    (endWound[0] - x[meanFinish]) ** 2
                    + (endWound[1] - y[meanFinish]) ** 2
                )
                ** 0.5
            ) * scale

            _df.append(
                {"Label": label, "r0": r0, "rt": rt, "migration": r0 - rt,}
            )

dfMigration = pd.DataFrame(_df)

R = np.linspace(25, 100, num=16)
D = [[] for col in range(16)]

start = dfMigration["r0"]
displacement = dfMigration["migration"]

for i in range(len(dfMigration)):
    r0 = start[i]
    if 20 < r0 < 100:
        D[int(r0 / 5 - 4)].append(displacement[i])

err = []
for i in range(16):
    err.append(cell.sd(D[i]) / (len(D[i]) ** 0.5))
    D[i] = cell.mean(D[i])

fig = plt.figure(1, figsize=(8, 8))
plt.gcf().subplots_adjust(left=0.20)
plt.errorbar(R, D, yerr=err)
plt.xlabel(r"Start distance ($\mu m$)")
plt.ylabel(r"Migation ($\mu m$)")
plt.title("Displacement Towards Wound in 20mins")
fig.savefig(
    f"results/Migation Lagrangian {fileType}", dpi=300, transparent=True,
)
plt.close("all")
