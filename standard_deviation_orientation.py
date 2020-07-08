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

plt.rcParams.update({"font.size": 14})

plt.ioff()
pd.set_option("display.width", 1000)


folder = "dat_binary"
FIGDIR = "fig"

cwd = os.getcwd()


def find_mean_shift(df):
    """returns the shift need fo the mean of the orientation to be pi/2"""

    ori = list(df)
    ori = sorted(ori)
    n = len(ori)
    mu = cell.mean(ori)
    i = 0
    if mu > np.pi / 2:
        while mu - np.pi / 2 > ori[i] - np.pi * i * 1 / n:
            i += 1
    else:
        while mu - np.pi / 2 < ori[i] - np.pi * i * 1 / n:
            i += 1

    t1 = mu - np.pi / 2 - (ori[i] - np.pi * i * 1 / n)
    i += -1
    t2 = mu - np.pi / 2 - (ori[i] - np.pi * i * 1 / n)
    i += 2
    t3 = mu - np.pi / 2 - (ori[i] - np.pi * i * 1 / n)

    x = min(abs(t1), abs(t2), abs(t3))

    if x == abs(t3):
        return ori[i]
    elif x == abs(t1):
        return ori[i - 1]
    else:
        return ori[i - 2]


def shift_mu_to_half_pi(df):
    """returns the orientation of the polygons shifted such that the mean is pi/2"""

    shift = find_mean_shift(df)

    df = df - shift

    for i in range(len(df)):
        if df[i] < 0:
            df[i] = df[i] + np.pi
        if df[i] > np.pi:
            df[i] = df[i] - np.pi

    mu = cell.mean(list(df))
    shift = mu - np.pi / 2
    df = shift_by(df, shift)

    mu = cell.mean(list(df))
    shift = mu - np.pi / 2
    df = shift_by(df, shift)

    return df


def shift_by(df, shift):
    """shifts the data by shift then mods pi the data"""

    df = df - shift

    for i in range(len(df)):
        if df[i] < 0:
            df[i] = df[i] + np.pi
        if df[i] > np.pi:
            df[i] = df[i] - np.pi

    return df


def ori_sd_from_shift(img_file):

    filename = img_file.replace("dat_binary/", "")
    filename = filename.replace(".tiff", "")

    df = pd.read_pickle(f"databases/df_of_{filename}.pkl")

    df3 = df[f"Orientation"]
    df3 = shift_mu_to_half_pi(df3)

    sd1 = cell.sd(df3)

    df3 = shift_by(df3, np.pi / 2)

    sd2 = cell.sd(df3)

    if sd2 > sd1:
        df3 = shift_by(df3, np.pi / 2)
        sigma = sd1
    else:
        sigma = sd2
    return sigma


def return_shift(df):

    df3 = df[f"Orientation"]
    shift1 = find_mean_shift(df3)
    df3 = shift_mu_to_half_pi(df3)

    while shift1 < 0:
        shift1 = shift1 + np.pi

    sd1 = cell.sd(df3)

    df3 = shift_by(df3, np.pi / 2)
    shift2 = shift1 + np.pi / 2
    if shift2 > np.pi:
        shift2 = shift2 - np.pi

    sd2 = cell.sd(df3)

    if sd2 > sd1:
        shift = shift1
    else:
        shift = shift2

    return shift


def shift_test(df):

    shift = return_shift(df)

    m = len(df)

    Q = []

    for i in range(m):
        polygon = df["Polygon"][i]
        Q.append(cell.shape_tensor(polygon))

    S = cell.mean(Q)

    thetastar = np.arctan(S[0, 1] / S[0, 0]) / 2

    return (shift, thetastar)

