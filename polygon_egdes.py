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

folder = "dat_binary"
FIGDIR = "fig"

cwd = os.getcwd()


files = os.listdir(cwd + f"/{folder}")
img_list = []

n = len(files)  # because for the .DS_Store file

files = sorted(files)

for i in range(n):
    img_file = f"{folder}/" + files[i]  # because for the .DS_Store file
    img_list.append(img_file)

if True:
    num_frames = 14
    num_videos = int(n / 14)
    edges = []

    for video in range(num_videos):

        img_video = img_list[video * num_frames : video * num_frames + num_frames]
        img_file = img_video[0]
        videoname = img_file.replace("dat_binary/", "")
        videoname = videoname.replace("01.tiff", "")

        for frame in range(num_frames):

            img_file = img_video[frame]
            filename = img_file.replace("dat_binary/", "")
            filename = filename.replace(".tiff", "")
            df = pd.read_pickle(f"databases/df_of_{filename}.pkl")

            m = len(df)

            Q = []

            for i in range(m):
                polygon = df["Polygon"][i]
                polygon = shapely.geometry.polygon.orient(polygon, sign=1.0)
                pts = list(polygon.exterior.coords)
                edges.append(len(pts))

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    ax.hist(edges, bins=20, density=True)
    ax.set(xlabel="Number of edges")
    ax.set_xlim([0, 20])

    fig.savefig(
        f"results/polygons edges",
        dpi=300,
        transparent=True,
    )
    plt.close("all")