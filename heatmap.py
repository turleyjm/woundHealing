import os
import shutil
from math import floor, log10

from collections import Counter
import cv2
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
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
import xml.etree.ElementTree as et

import cellProperties as cell
import findGoodCells as fi
import commonLiberty as cl

plt.rcParams.update({"font.size": 20})

# -------------------

scale = 147.91 / 512

cl.createFolder("results/video/")

filenames, fileType = cl.getFilesType()

for filename in filenames:

    vidFile = f"dat/{filename}/binaryBoundary{filename}.tif"

    vid = sm.io.imread(vidFile).astype(int)

    T = len(vid)

    for t in range(T):

        img_rc = 255 - vid[t]

        img = fi.imgrcxy(img_rc)

        img_label = sm.measure.label(img, background=0, connectivity=1)
        img_label = img_label.astype("float16")
        img_labels = np.unique(img_label)[1:]
        all_polys = []
        all_contours = []

        for label in img_labels:
            contour = sm.measure.find_contours(img_label == label, level=0)[0]
            polygon = sm.measure.approximate_polygon(contour, tolerance=1)
            all_contours.append(contour)
            all_polys.append(polygon)

        q = []
        for poly in all_polys:
            try:
                polygon = Polygon(poly)
                q.append(cell.qTensor(polygon))
            except:
                continue

        Q = np.mean(q)

        Q = Q / (Q[0, 0] ** 2 + Q[0, 1] ** 2) ** 0.5

        for label in img_labels:
            contour = sm.measure.find_contours(img_label == label, level=0)[0]
            poly = sm.measure.approximate_polygon(contour, tolerance=1)
            polygon = Polygon(poly)
            try:
                q = cell.qTensor(polygon)
                q = q / (q[0, 0] ** 2 + q[0, 1] ** 2) ** 0.5

                cor = np.dot(Q[0], q[0])
                img_label[img_label == label] = cor
            except:
                img_label[img_label == label] = 0
                continue

        x, y = np.mgrid[0 : 512 * scale : 1 * scale, 0 : 512 * scale : 1 * scale]

        z_min, z_max = -1, 1

        heatmap = img_label

        fig, ax = plt.subplots()
        c = ax.pcolor(x, y, heatmap, cmap="coolwarm", vmin=z_min, vmax=z_max)
        fig.colorbar(c, ax=ax)
        fig.savefig(
            f"results/video/{filename} t={t}", dpi=300, transparent=True,
        )
        plt.close("all")

    img_array = []
    for t in range(T):
        img = cv2.imread(f"results/video/{filename} t={t}.png")
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(
        f"results/heatmapOriCor{filename}.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 5, size
    )
    for t in range(T):
        out.write(img_array[t])

    out.release()
    cv2.destroyAllWindows()

    shutil.rmtree("results/video")
