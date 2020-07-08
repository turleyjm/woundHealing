import os
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

import cell_properties as cell
import find_good_cells as fi

plt.rcParams.update({"font.size": 14})
plt.ioff()
pd.set_option("display.width", 1000)

cwd = os.getcwd()

# ------------------------------------------


def round_sig(x, sig=2):

    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def plot_dist(prop, filename, bins=40, xlim="None"):
    """produces a bar plot with mean line from the colume col of table df"""

    plt.rcParams.update({"font.size": 14})
    # mu = prop.mean()
    # sigma = cell.sd(prop)
    # sigma = float(sigma)
    # sigma = round_sig(sigma, 3)
    fig, ax = plt.subplots()
    ax.hist(prop, density=False, bins=bins)
    # ax.axvline(mu, c="k", label="mean")
    # ax.axvline(mu + sigma, c="k", label=r"$\sigma$", ls="--")
    # ax.axvline(mu - sigma, c="k", ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlim != "None":
        ax.set_xlim(xlim)
    # fig.legend(loc="upper right", fontsize=14, bbox_to_anchor=(0.9, 0.85))
    fig.savefig(
        "results/mitosis/" + f"fig_dist_{filename}.png", dpi=200, transparent=True,
    )

    plt.close("all")


# ---------------------------

wound = sm.io.imread("dat_nucleus/Helen_wound.tif").astype(float)
C = []
for t in range(len(wound)):
    img_wound = wound[t]
    img_wound[img_wound == 0] = 1
    img_wound[img_wound == 255] = 0
    img_wound[img_wound == 1] = 255
    img_label = sm.measure.label(img_wound, background=0, connectivity=1)
    contour = sm.measure.find_contours(img_label == 1, level=0)[0]
    poly = sm.measure.approximate_polygon(contour, tolerance=1)
    wound_polygon = Polygon(poly)
    c = cell.centroid(wound_polygon)
    c = np.array(c)
    C.append(c)

filename = "HelenH2"

vid = sm.io.imread(f"dat_nucleus/vid_binary_{filename}.tif").astype(float)

(T, X, Y) = vid.shape

df = pd.read_pickle(f"databases/mitosis_of_{filename}.pkl")

df0 = df.loc[lambda df: df["Chain"] == "daughter0", :]
df1 = df.loc[lambda df: df["Chain"] == "daughter1", :]

df = pd.concat([df0, df1])

n = len(df)

prop = []
for i in range(n):
    prop.append(len(df.iloc[i][2]))

prop = np.array(prop)
M = prop.max() - 1
t = []
for j in range(M):
    t.append([])

c = []
for i in range(len(prop)):
    if M == prop[i]:
        c.append(i)

for i in range(n):
    m = len(df.iloc[i][2]) - 1
    for j in range(m):
        [x0, y0] = df.iloc[i, 2][j]
        [x1, y1] = df.iloc[i, 2][j + 1]
        t0 = df.iloc[i, 1][j]
        t1 = df.iloc[i, 1][j + 1]
        r = ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
        t[j].append(r)

d = t[0:6]

sigma = []
for j in range(6):
    sigma.append(cell.sd(d[j]) / (len(d[j])) ** 0.5)
    d[j] = cell.mean(d[j])

x = np.array(range(6)) + 1
fig = plt.figure(1, figsize=(8, 8))
plt.errorbar(x, d, sigma)
fig.savefig(
    "results/mitosis/" + f"daughter_velocity", dpi=300, transparent=True,
)
plt.close("all")


wound = sm.io.imread("dat_nucleus/Helen_wound.tif").astype(float)
unique = set(list(df.iloc[:, 0]))
div_ori = []
dist = []
time = []

for label in unique:
    df3 = df.loc[lambda df: df["Label"] == label, :][0:2]

    if len(df3.iloc[0, 1]) > 2:
        delta_t0 = 2
        t0 = df3.iloc[0, 1][2]

    elif len(df3.iloc[0, 1]) > 1:
        delta_t0 = 1
        t0 = df3.iloc[0, 1][1]
    else:
        delta_t0 = 0
        t0 = df3.iloc[0, 1][0]

    if len(df3.iloc[1, 1]) > 2:
        delta_t1 = 2
        t0 = df3.iloc[1, 1][2]
    elif len(df3.iloc[1, 1]) > 1:
        delta_t1 = 1
        t0 = df3.iloc[1, 1][1]
    else:
        delta_t1 = 0
        t0 = df3.iloc[1, 1][0]

    (Cx, Cy) = C[t0]
    r = (wound_polygon.area / np.pi) ** 0.5
    [x0, y0] = df3.iloc[0, 2][delta_t0]
    [x1, y1] = df3.iloc[1, 2][delta_t1]

    xm = (x0 + x1) / 2
    ym = (y0 + y1) / 2
    v = np.array([x0 - x1, y0 - y1])
    w = np.array([xm - Cx, ym - Cy])

    phi = np.arccos(np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w)))

    if phi > np.pi / 2:
        theta = np.pi - phi
    else:
        theta = phi
    div_ori.append(theta * (180 / np.pi))
    dist.append(np.linalg.norm(w) - r)

plot_dist(div_ori, "div_ori" + filename, bins=15, xlim=[0, 90])
plot_dist(dist, "dist" + filename, bins=10)

vid = sm.io.imread(f"../dat_videos/HelenMerged.tif").astype(float)
color = (0, 0, 255)
df = pd.read_pickle(f"databases/mitosis_of_{filename}.pkl")

Times = list(range(8))
Times = np.array(Times) - 4

mask = np.zeros([T, X + 30, Y + 30,])

for label in unique:
    df3 = df.loc[lambda df: df["Label"] == label, :][0:1]

    t0 = df3.iloc[0, 1][-1]
    [x0, y0] = df3.iloc[0, 2][-1]
    x0 = int(x0 + 15)
    y0 = int(y0 + 15)

    rr, cc = circle_perimeter(x0, y0, 15)
    rr1, cc1 = circle_perimeter(x0, y0, 14)
    for t in Times:

        if -1 < t0 + t < T:
            mask[t0 + t][cc, rr] = 255
            mask[t0 + t][cc1, rr1] = 255

maskcrop = mask[:, 15:527, 15:527]

for t in range(T):

    vid[t][maskcrop[t] == 255] = (0, 0, 255)

vid = np.asarray(vid, "uint8")
tifffile.imwrite(f"results/mitosis/vid_mitosis_{filename}.tif", vid)

# wound mask

wound_prob = sm.io.imread("dat_nucleus/wound_prob.tif").astype(float)

for t in range(len(wound_prob)):

    img = wound_prob[t]
    binary = np.zeros([514, 514])

    mu = cell.mean(cell.mean(img))

    for x in range(512):
        for y in range(512):
            if img[x, y] == 255:
                binary[x + 1, y + 1] = 255

    img_label = sm.measure.label(binary, background=0, connectivity=1)
    img_labels = np.unique(img_label)[1:]

    for label in img_labels:
        contour = sm.measure.find_contours(img_label == label, level=0)[0]
        poly = sm.measure.approximate_polygon(contour, tolerance=1)
        try:
            polygon = Polygon(poly)
            a = cell.area(polygon)

            if a < 600:
                binary[img_label == label] = 0
        except:
            continue

    binary = binary[1:513, 1:513]

    wound_prob[t] = binary

wound_prob = np.asarray(wound_prob, "uint8")
tifffile.imwrite(f"results/mitosis/binary_wound_{filename}.tif", wound_prob)

vid = sm.io.imread(f"../dat_videos/HelenMerged.tif").astype(float)

for t in range(T):
    for x in range(512):
        for y in range(512):

            if wound_prob[t][x][y] == 255:
                vid[t][x][y][2] = 155

vid = np.asarray(vid, "uint8")
tifffile.imwrite(f"results/mitosis/vid_woundmask_{filename}.tif", vid)

# Correction fuction

radius = []
unique = list(unique)
c = 0

for label1 in unique:
    c += 1
    df3 = df.loc[lambda df: df["Label"] == label1, :][0:1]

    [x1, y1] = df3.iloc[0, 2][-1]

    for label2 in unique[c:]:
        df4 = df.loc[lambda df: df["Label"] == label2, :][0:1]

        [x2, y2] = df4.iloc[0, 2][-1]

        radius.append(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)

plot_dist(radius, "correction" + filename, bins=30)

# height of nuclei

