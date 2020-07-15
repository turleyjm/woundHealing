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

plt.rcParams.update({"font.size": 18})

plt.ioff()
pd.set_option("display.width", 1000)

folder = "dat_binary"
FIGDIR = "fig"

cwd = os.getcwd()


# ----------------------------------------------------


def image_to_poly_labels(img_file):
    """Takes a image file and produces an arrey of vertex add the polygons labels to be used 
    in the making of the heat plot"""

    img = sm.io.imread(img_file).astype(int)
    # mu = cell.mean(cell.mean(img))
    # if mu < 130:
    #     img = img
    #     img = np.invert(img - 255) + 1

    img_xy = fi.img_rc_to_xy(img)

    img_label = sm.measure.label(img_xy, background=0, connectivity=1)
    img_labels = np.unique(img_label)[1:]
    all_polys = []
    all_contours = []

    for label in img_labels:
        contour = sm.measure.find_contours(img_label == label, level=0)[0]
        polygon = sm.measure.approximate_polygon(contour, tolerance=1)
        all_contours.append(contour)
        all_polys.append(polygon)

    return (all_polys, img_label, img_labels)


def heat_map(img_file, function, function_title, lim, radians):

    img = sm.io.imread(img_file).astype(int)
    mu = cell.mean(cell.mean(img))
    if mu < 130:
        img = np.invert(img - 255) + 1

    (all_polys, img_label, img_labels) = image_to_poly_labels(img_file)
    filename = img_file.replace("dat_binary/", "")
    filename = filename.replace(".tiff", "")

    _all_polys = fi.remove_cells(all_polys)
    _df = []
    for poly in _all_polys:
        try:
            polygon = Polygon(poly)
            _df.append({"QQ": function(polygon)})

        except:
            continue
    df = pd.DataFrame(_df)

    mu = cell.mean(df.QQ)
    sigma = cell.sd(df.QQ)

    if 0 > mu - 2 * sigma:
        y = 0
    else:
        y = mu - 2 * sigma
    z = mu + 2 * sigma

    if radians == "Y":
        y = lim[0]
        z = lim[1]
    n = len(img)
    img_xy = np.zeros(shape=(n, n)) + y

    for idx, label in enumerate(img_labels):

        try:
            poly = all_polys[label - 1]
            polygon = Polygon(poly)
            x = function(polygon)

            if y < x < z:
                img_xy[img_label == label] = x
            elif x > z:
                img_xy[img_label == label] = z
            else:
                img_xy[img_label == label] = y
        except:
            continue

    img = fi.img_xy_to_rc(img_xy)
    img_x = fi.img_x_axis(img)

    fig, ax = plt.subplots()

    if radians == "Y":
        pos = ax.imshow(img_x, origin=["lower"], cmap=plt.get_cmap("hsv"))
    else:
        pos = ax.imshow(img_x, origin=["lower"])

    fig.colorbar(pos)
    fig.suptitle(f"")
    # plt.title(f"Heat map of {function_title}")
    fig.savefig(
        "results/heatmaps/" + FIGDIR + f"_{function_title}_{filename}.png",
        dpi=300,
        transparent=True,
    )
    plt.close()


# ----------------------------------------------------


(function, function_title, function_name, lim, radians) = gf.Get_Functions()

files = os.listdir(cwd + f"/{folder}")
img_list = []
files = sorted(files)
n = len(files) - 1  # because for the .DS_Store file

for i in range(n):
    img_file = f"{folder}/" + files[1 + i]  # because for the .DS_Store file
    img_list.append(img_file)


m = len(function)


for i in range(n):
    img_file = img_list[i]

    for j in range(m):
        heat_map(img_file, function[j], function_title[j], lim[j], radians[j])

# ----------------------------------------------------
