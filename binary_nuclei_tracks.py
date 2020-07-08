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

plt.rcParams.update({"font.size": 20})
plt.ioff()
pd.set_option("display.width", 1000)

filename = "HelenH2"

binary = sm.io.imread(f"dat_nucleus/vid_binary_HelenH2.tif").astype(float)
height = sm.io.imread(f"results/mitosis/height_{filename}.tif").astype(float)

(T, X, Y) = binary.shape

df = pd.read_pickle(f"databases/mitosis_of_{filename}.pkl")

cell_length = 20

z_step = 0.75

#----------------

def round_sig(x, sig=2):

    return round(x, sig - int(floor(log10(abs(x)))) - 1)

def plot_dist(prop, function_name, function_title, filename, bins=40, xlim="None"):
    """produces a bar plot with mean line from the colume col of table df"""

    # mu = cell.mean(prop)
    # sigma = cell.sd(prop)
    # sigma = float(sigma)
    # sigma = round_sig(sigma, 3)
    fig, ax = plt.subplots()
    plt.gcf().subplots_adjust(bottom=0.15)
    ax.hist(prop, density=False, bins=bins)
    ax.set_xlabel(function_name, y=0.13)
    # ax.axvline(mu, c="k", label="mean")
    # ax.axvline(mu + sigma, c="k", label=r"$\sigma$", ls="--")
    # ax.axvline(mu - sigma, c="k", ls="--")
    # ax.axvline(med, c='r', label='median')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlim != "None":
        ax.set_xlim(xlim)
    # plt.suptitle(f"Distribution of {function_name}", y=1)
    # plt.suptitle(r"$\sigma$" + f" = {sigma}", y=0.95)
    # fig.legend(loc="upper right", fontsize=18, bbox_to_anchor=(0.9, 0.85))
    fig.savefig(
        "results/bar_graphs/" + f"_dist_{function_title}_{filename}.png",
        dpi=200,
        transparent=True,
    )

#---------------

vid_labels = []

for t in range(T):

    vid_labels.append(sm.measure.label(binary[t], background=0, connectivity=1))

_df2 = []

for i in range(len(df)):

    Tm = len((df.iloc[i][2]))

    polygons = []
    sf = []
    ori = []
    h = []

    for t in range(Tm):

        frame = df.iloc[i][1][t]
        img_label = vid_labels[frame]
        [x, y] = df.iloc[i][2][t]

        x = int(x)
        y = int(y)

        label = img_label[y, x]  # coordenate change

        if label != 0:

            contour = sm.measure.find_contours(img_label == label, level=0)[0]
            poly = sm.measure.approximate_polygon(contour, tolerance=1)

            polygon = Polygon(poly)

            h_dist = []
            for r in range(int(cell_length * 3 / 10)):

                rr1, cc1 = circle_perimeter(
                    x + 2 * cell_length, y + 2 * cell_length, 5 * r
                )
                rr2, cc2 = circle_perimeter(
                    x + 2 * cell_length, y + 2 * cell_length, 5 * r + 1
                )
                rr3, cc3 = circle_perimeter(
                    x + 2 * cell_length, y + 2 * cell_length, 5 * r + 2
                )
                rr4, cc4 = circle_perimeter(
                    x + 2 * cell_length, y + 2 * cell_length, 5 * r + 3
                )
                rr5, cc5 = circle_perimeter(
                    x + 2 * cell_length, y + 2 * cell_length, 5 * r + 4
                )

                mask = np.zeros([512 + 4 * cell_length, 512 + 4 * cell_length])
                mask[cc1, rr1] = 255  # change coord
                mask[cc2, rr2] = 255
                mask[cc3, rr3] = 255
                mask[cc4, rr4] = 255
                mask[cc5, rr5] = 255

                h0 = (
                    cell.mean(
                        height[frame][
                            mask[
                                2 * cell_length : 512 + 2 * cell_length,
                                2 * cell_length : 512 + 2 * cell_length,
                            ]
                            == 255
                        ]
                    )
                    * z_step
                )
                h_dist.append(h0)

            h_dist = np.array(h_dist)

            h_dist = (-h_dist) + h_dist[-1]

            sf.append(cell.shape_factor(polygon))
            h.append(h_dist)  # swapped because stacks are taken in -z direction
            ori.append(cell.orientation(polygon))
            polygons.append(polygon)

        else:
            h_dist = []
            for r in range(int(cell_length * 3 / 10)):
                h_dist.append(False)

            sf.append(False)
            h.append(h_dist)
            ori.append(False)
            polygons.append(False)

    Label = df.iloc[i][0]
    time_list = df.iloc[i][1]
    C_list = df.iloc[i][2]
    chain = df.iloc[i][3]

    _df2.append(
        {
            "Label": Label,
            "Time": time_list,
            "Position": C_list,
            "Chain": chain,
            "Shape Factor": sf,
            "Height": h,
            "Orientation": ori,
            "Polygons": polygons,
        }
    )

df2 = pd.DataFrame(_df2)

df2.to_pickle(f"databases/mitosisPoly{filename}.pkl")


df0 = df2.loc[lambda df2: df2["Chain"] == "daughter0", :]
df1 = df2.loc[lambda df2: df2["Chain"] == "daughter1", :]

df_d = pd.concat([df0, df1])

df_p = df2.loc[lambda df2: df2["Chain"] == "parent", :]

# mitosis height kymograph

H = []
for r in range(int(cell_length * 3 / 10)):
    H.append([])
    for t in range(10):
        H[r].append([])

for i in range(len(df_p)):

    h = df_p.iloc[i][5]

    n = len(h)

    if n > 4:
        for j in range(5):
            for r in range(int(cell_length * 3 / 10)):
                H[r][4 - j].append(h[n - j - 1][r])

    else:
        for j in range(n):
            for r in range(int(cell_length * 3 / 10)):
                H[r][4 - j].append(h[n - j - 1][r])

for i in range(len(df_d)):

    h = df_d.iloc[i][5]

    n = len(h)

    if n > 4:
        for j in range(5):
            for r in range(int(cell_length * 3 / 10)):
                H[r][5 + j].append(h[j][r])

    else:
        for j in range(n):
            for r in range(int(cell_length * 3 / 10)):
                H[r][5 + j].append(h[j][r])


t = []
err = []
for j in range(10):
    err.append(cell.sd(H[0][j]) / (len(H[0][j])) ** 0.5)
    t.append(cell.mean(H[0][j]))

x = np.array(range(10)) - 4
fig = plt.figure(1, figsize=(8, 8))
plt.errorbar(x, t, err)
fig.savefig(
    "results/mitosis/" + f"mitosis_height", dpi=300, transparent=True,
)
plt.close("all")


for r in range(int(cell_length * 3 / 10)):
    for t in range(10):
        m = len(H[r][t])
        if m > 0:
            H[r][t] = cell.mean(H[r][t])
        else:
            H[r][t] = 0

fig, ax = plt.subplots()

pos = ax.imshow(H, origin=["lower"])

fig.colorbar(pos)
fig.savefig(
    "results/mitosis/heightKymograph.png", dpi=300, transparent=True,
)
plt.close()

# shape factor of nuclei

t = []
for i in range(10):
    t.append([])

for i in range(len(df_p)):

    h = df_p.iloc[i][4]

    n = len(h)

    if n > 4:
        for j in range(5):
            t[4 - j].append(h[n - j - 1])

    else:
        for j in range(n):
            t[4 - j].append(h[n - j - 1])


for i in range(len(df_d)):

    h = df_d.iloc[i][4]

    n = len(h)

    if n > 4:
        for j in range(5):
            t[5 + j].append(h[j])

    else:
        for j in range(n):
            t[5 + j].append(h[j])


err = []
for j in range(len(t)):
    err.append(cell.sd(t[j]) / (len(t[j])) ** 0.5)
    t[j] = cell.mean(t[j])

x = np.array(range(10)) - 4
fig = plt.figure(1, figsize=(8, 8))
plt.errorbar(x, t, err)
fig.savefig(
    "results/mitosis/" + f"mitosis_sf", dpi=300, transparent=True,
)
plt.close("all")

# orientation change

ori_change = []
ori_drift = []

for i in range(len(df_p)):
    ori = df_p.iloc[i][6]

    v = []
    for j in range(len(ori)):
        v.append(np.array([np.cos(2*ori[j]), np.sin(2*ori[j])]))

    dtheta = []

    for j in range(len(ori) - 1):
        costheta = np.dot(v[j], v[j+1])

        if np.cross(v[j], v[j+1]) > 0:
            dtheta.append(np.arccos(costheta)/2)
        else:
            dtheta.append(-np.arccos(costheta)/2)

    dtheta = np.array(dtheta) * (180/np.pi) #change radions

    ori_change.append(sum(abs(dtheta)))
    ori_drift.append(sum(dtheta))

plot_dist(
            ori_change,
            'Ori Change',
            'Orientation Change',
            filename,
            bins = 20
        )

plot_dist(
            ori_drift,
            'Ori drift',
            'Orientation drift',
            filename,
            bins = 20
        )
