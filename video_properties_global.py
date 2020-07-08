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

plt.rcParams.update({"font.size": 36})

plt.ioff()
pd.set_option("display.width", 1000)

folder = "dat_binary"
FIGDIR = "fig"

cwd = os.getcwd()


def best_fit_slope_and_intercept(xs, ys):
    m = ((cell.mean(xs) * cell.mean(ys)) - cell.mean(xs * ys)) / (
        (cell.mean(xs) * cell.mean(xs)) - cell.mean(xs * xs)
    )

    b = cell.mean(ys) - m * cell.mean(xs)

    return (m, b)


yes_num = "N"
yes_sdori = "Y"


def video_prop(function, folder, function_name, function_title, img_list):

    n = len(img_list)

    mu1 = []
    sigma1 = []
    sigma_mu1 = []
    num = []

    for i in range(n):

        img_file = img_list[i]
        filename = img_file.replace("dat_binary/", "")
        filename = filename.replace(".tiff", "")
        df = pd.read_pickle(f"databases/df_of_{filename}.pkl")
        m = len(df)
        prop = df[function_title]

        mu1.append(cell.mean(prop))
        sigma1.append(cell.sd(prop))
        sigma_mu1.append(cell.sd(prop) / cell.mean(prop))
        num.append(m)

    mu = []
    sigma = []
    sigma_mu = []
    error = []

    for i in range(14):
        mu2 = []
        sigma2 = []
        sigma_mu2 = []
        num2 = []

        for j in range(int(n / 14)):
            mu2.append(mu1[i + 14 * j] * num[i + 14 * j])
            sigma2.append(sigma1[i + 14 * j] * num[i + 14 * j])
            num2.append(num[i + 14 * j])
            sigma_mu2.append(sigma_mu1[i + 14 * j] * num[i + 14 * j])

        mu.append(sum(mu2) / sum(num2))
        sigma.append(sum(sigma2) / sum(num2))
        sigma_mu.append(sum(sigma_mu2) / sum(num2))
        error.append(sigma[i] / (sum(num2)) ** 0.5)

    x = range(14)
    xs = np.array(list(x), dtype=np.float64)
    ys = np.array(mu, dtype=np.float64)
    (m, b) = best_fit_slope_and_intercept(xs, ys)

    fig = plt.figure(1, figsize=(9, 8))
    plt.gcf().subplots_adjust(left=0.2)
    plt.errorbar(x, mu, yerr=error, fmt="o")
    plt.plot([0, 13], [b, 13 * m + b], "k-", lw=2)

    plt.xlabel("Time")
    plt.ylabel(f"{function_name}")
    plt.gcf().subplots_adjust(bottom=0.2)
    # plt.title(f"{function_name} with time")
    fig.savefig(
        "results/video_prop/" + FIGDIR + f"_Video_{function_title}",
        dpi=300,
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


p = len(function)

for i in range(p):
    video_prop(function[i], folder, function_name[i], function_title[i], img_list)


if yes_num == "Y":

    num = []

    for i in range(n):

        img_file = img_list[i]
        filename = img_file.replace("dat_binary/", "")
        filename = filename.replace(".tiff", "")
        df = pd.read_pickle(f"databases/df_of_{filename}.pkl")
        m = len(df)
        num.append(m)

    mu = []
    sigma = []
    sigma_mu = []
    error = []

    for i in range(14):
        mu2 = []

        for j in range(int(n / 14)):
            mu2.append(num[i + 14 * j])

        mu.append(cell.mean(mu2))
        sigma.append(cell.sd(mu2))
        sigma_mu.append(cell.sd(mu2) / cell.mean(mu2))
        error.append(sigma[i] / (n) ** 0.5)

    x = range(14)
    xs = np.array(list(x), dtype=np.float64)
    ys = np.array(mu, dtype=np.float64)
    (m, b) = best_fit_slope_and_intercept(xs, ys)

    fig = plt.figure(1, figsize=(8, 8))
    plt.plot([0, 13], [b, 13 * m + b], "k-", lw=2)
    plt.errorbar(x, mu, yerr=error, fmt="o")
    plt.xlabel("Frame of videos")
    plt.ylabel(f"Mean of Number of cells")
    plt.title(f"Number of cells with time")
    fig.savefig(
        "results/video_prop/" + FIGDIR + f"_Video_Number_of_Cells",
        dpi=300,
        transparent=True,
    )
    plt.close("all")


if yes_sdori == "Y":

    sigma_ori = []

    for i in range(n):

        img_file = img_list[i]
        sigma_ori.append(sdori.ori_sd_from_shift(img_file))

    mu = []
    sigma = []
    sigma_mu = []
    error = []

    for i in range(14):
        mu2 = []

        for j in range(int(n / 14)):
            mu2.append(sigma_ori[i + 14 * j])

        mu.append(cell.mean(mu2))
        sigma.append(cell.sd(mu2))
        sigma_mu.append(cell.sd(mu2) / cell.mean(mu2))
        error.append(sigma[i] / (n) ** 0.5)

    x = range(14)
    xs = np.array(list(x), dtype=np.float64)
    ys = np.array(mu, dtype=np.float64)
    (m, b) = best_fit_slope_and_intercept(xs, ys)

    fig = plt.figure(1, figsize=(8, 8))
    plt.plot([0, 13], [b, 13 * m + b], "k-", lw=2)
    plt.errorbar(x, mu, yerr=error, fmt="o")
    plt.xlabel("Time")
    plt.ylabel(r"$\langle \sigma \rangle$")
    # plt.title(f"Standard Deviation of Orientation with time")
    fig.savefig(
        "results/video_prop/" + FIGDIR + f"_Video_sd_of_Orientation",
        dpi=300,
        transparent=True,
    )
    plt.close("all")
