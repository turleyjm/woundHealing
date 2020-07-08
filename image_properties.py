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

import cell_properties as cell
import find_good_cells as fi
import get_functions as gf
import standard_deviation_orientation as sdori

plt.rcParams.update({"font.size": 20})

plt.ioff()
pd.set_option("display.width", 1000)

folder = "dat_binary"
FIGDIR = "fig"

cwd = os.getcwd()


def round_sig(x, sig=2):

    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def plot_dist(prop, function_name, function_title, filename, bins=40, xlim="None"):
    """produces a bar plot with mean line from the colume col of table df"""

    mu = prop.mean()
    sigma = cell.sd(prop)
    sigma = float(sigma)
    sigma = round_sig(sigma, 3)
    fig, ax = plt.subplots()
    plt.gcf().subplots_adjust(bottom=0.15)
    ax.hist(prop, density=False, bins=bins)
    ax.set_xlabel(function_name, y=0.13)
    ax.axvline(mu, c="k", label="mean")
    # ax.axvline(mu + sigma, c="k", label=r"$\sigma$", ls="--")
    # ax.axvline(mu - sigma, c="k", ls="--")
    # ax.axvline(med, c='r', label='median')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlim != "None":
        ax.set_xlim(xlim)
    # plt.suptitle(f"Distribution of {function_name}", y=1)
    plt.suptitle(r"$\sigma$" + f" = {sigma}", y=0.95)
    fig.legend(loc="upper right", fontsize=18, bbox_to_anchor=(0.9, 0.85))
    fig.savefig(
        "results/bar_graphs/" + FIGDIR + f"_dist_{function_title}_{filename}.png",
        dpi=200,
        transparent=True,
    )

    plt.close("all")


(function, function_title, function_name, lim, radians) = gf.Get_Functions()

files = os.listdir(cwd + f"/{folder}")
img_list = []

n = len(files) - 1  # because for the .DS_Store file

files = sorted(files)


for i in range(n):
    img_file = f"{folder}/" + files[1 + i]  # because for the .DS_Store file
    img_list.append(img_file)
img_list = sorted(img_list)

p = len(function)

for i in range(n):

    img_file = img_list[i]
    filename = img_file.replace("dat_binary/", "")
    filename = filename.replace(".tiff", "")
    df = pd.read_pickle(f"databases/df_of_{filename}.pkl")
    m = len(df)

    for k in range(p):

        prop = df[function_title[k]]
        # prop = sdori.shift_mu_to_half_pi(prop)
        # prop = sdori.shift_by(prop, np.pi / 2)
        mu = cell.mean(prop)
        sigma = cell.sd(prop)
        dd = min(prop)
        uu = max(prop)
        d = mu - 3 * sigma
        u = mu + 3 * sigma

        if function_title == "Trace(S)":
            d = 1 / (2 * np.pi)
        if function_title == "Shape Factor":
            d = 0
            u = 1
        if function_title == "Circularity":
            u = 1
        if function_title == "Ellipticity":
            u = 1
        if d < 0:
            d = 0

        img_bin = int(20 * (uu - dd) / (u - d))

        if lim[k][0] == "None":
            img_bin = 20
        if lim[k] == "None":
            lim[k] = (d, u)
        plot_dist(
            prop,
            function_name[k],
            function_title[k],
            filename,
            bins=img_bin,
            xlim=lim[k],
        )

