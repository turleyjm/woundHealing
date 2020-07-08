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

plt.rcParams.update({"font.size": 14})

plt.ioff()
pd.set_option("display.width", 1000)


folder = "dat_binary"
FIGDIR = "fig"

cwd = os.getcwd()


# ----------------------------------------------------


frame_num = 14  # Number of frames per video?
frame_num = int(frame_num)
band_num = 5  # Number of bands?
band_num = int(band_num)


# ----------------------------------------------------


(function, function_title, function_name, lim, radians) = gf.Get_Functions()

files = os.listdir(cwd + f"/{folder}")
img_list = []

n = len(files) - 1  # because for the .DS_Store file

files = sorted(files)

for i in range(n):
    img_file = f"{folder}/" + files[1 + i]  # because for the .DS_Store file
    img_list.append(img_file)
img_list = sorted(img_list)

m = int(n / frame_num)
p = len(function)

length = (260 * (2) ** 0.5) / band_num

count = 0

for fun in range(len(function_title)):

    H = []
    for frame in range(frame_num):
        H.append([])
        for band in range(band_num):
            H[frame].append([])

    for image in range(n):
        img_file = img_list[image]

        filename = img_file.replace("dat_binary/", "")
        filename = filename.replace(".tiff", "")
        (u, v, w) = filename.partition("sample")
        (x, y, z) = w.partition("_")
        videoname = u + v + x
        frame = int(z) - 1
        df = pd.read_pickle(f"databases/df_of_{filename}.pkl")
        df2 = pd.read_pickle(f"databases/bands_of_{videoname}.pkl")

        cell_num = len(df)
        R = df2["Wound radius"][frame]
        wound_mid = df2["Wound midpoint"][frame]
        (x, y) = wound_mid

        for poly in range(cell_num):

            (Cx, Cy) = df["Centroid"][poly]
            fun_value = df[f"{function_title[fun]}"][poly]

            r = ((Cx - x) ** 2 + (Cy - y) ** 2) ** 0.5

            band = int((r - R) / length)
            if band < 0:
                band = 0

            H[frame][band].append(fun_value)
            count += 1

    count = 0

    C = []  # count of cells in each frame and band
    for frame in range(frame_num):
        C.append([])
        for band in range(band_num):
            C[frame].append([])

    M = []  # mean of cells in each frame and band
    for frame in range(frame_num):
        M.append([])
        for band in range(band_num):
            M[frame].append([])

    SD = []  # sd of cells in each frame and band
    for frame in range(frame_num):
        SD.append([])
        for band in range(band_num):
            SD[frame].append([])

    for frame in range(frame_num):
        for band in range(band_num):
            C[frame][band].append(len(H[frame][band]))
            M[frame][band].append(cell.mean(H[frame][band]))
            SD[frame][band].append(cell.sd(H[frame][band]))

    for band in range(band_num):

        mu = []
        error = []

        for frame in range(frame_num):
            mu.append(M[frame][band][0])
            error.append(SD[frame][band][0] / ((C[frame][band][0]) ** 0.5))

        x = range(frame_num)

        fig = plt.figure(1, figsize=(8, 8))
        plt.errorbar(x, mu, yerr=error, fmt="o")
        plt.xlabel("Frame of videos")
        plt.ylabel(f"Mean of {function_name[fun]}")
        plt.title(f"{function_name[fun]} with time")
        fig.savefig(
            "results/video_prop_bands/"
            + FIGDIR
            + f"_Video_{function_title[fun]}_in_band_{band}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

