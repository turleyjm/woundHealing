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


# ----------------------------------------------------


def image_to_polygons(img_file):
    """Takes a image file and produces an arrey of vertex add the polygons labels to be used 
    in the making of the heat plot"""

    img = sm.io.imread(img_file).astype(int)
    mu = cell.mean(cell.mean(img))
    if mu < 130:
        img = np.invert(img - 255) + 1

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

    return all_polys


def vector(polygon, img_x):
    """Adds a line in the image from the certre of a polygon in the direction of the 
    orientation. Used in the vector field function"""

    (x, y) = cell.centroid(polygon)
    theta = cell.orientation(polygon)

    xhat = x + 5 * np.cos(theta)
    yhat = y + 5 * np.sin(theta)
    xhat = int(xhat)
    yhat = int(yhat)
    xbar = x - 5 * np.cos(theta)
    ybar = y - 5 * np.sin(theta)
    xbar = int(xbar)
    ybar = int(ybar)
    cv2.line(img_x, (xbar, ybar), (xhat, yhat), (0, 255, 0), 2)
    return img_x


def p_vector(polygon, img_xy):
    """Adds a line in the image from the certre of a polygon in the direction of the 
    orientation. Used in the vector field function"""

    (x, y) = cell.centroid(polygon)
    theta = cell.polar_ori(polygon)

    xhat = x + 7 * np.cos(theta)
    yhat = y + 7 * np.sin(theta)
    x = int(x)
    y = int(y)
    xhat = int(xhat)
    yhat = int(yhat)
    cv2.line(img_xy, (x, y), (xhat, yhat), (0, 255, 0), 2)
    return img_xy


def vector_field(img_file):
    """Takes a image file and products a vector field of the orientations of each polygon 
    on the image"""

    img = sm.io.imread(img_file).astype(int)
    mu = cell.mean(cell.mean(img))
    if mu < 130:
        img = np.invert(img - 255) + 1

    img_x = fi.img_x_axis(img)

    all_polys = image_to_polygons(img_file)
    all_polys = fi.remove_cells(all_polys)
    filename = img_file.replace("dat_binary/", "")
    filename = filename.replace(".tiff", "")

    img_x = np.ascontiguousarray(img_x, dtype=np.uint8)

    for poly in all_polys:

        polygon = Polygon(poly)
        vector(polygon, img_x)

    fig, ax = plt.subplots()
    ax.imshow(img_x, origin=["lower"])
    fig.savefig(
        "results/vectormaps/" + FIGDIR + f"_vector_field_{filename}.png",
        transparent=True,
        dpi=300,
    )
    plt.close()


def p_vector_field(img_file):
    """Takes a image file and products a vector field of the orientations of each polygon 
    on the image"""

    img = sm.io.imread(img_file).astype(int)
    mu = cell.mean(cell.mean(img))
    if mu < 130:
        img = np.invert(img - 255) + 1

    img_x = fi.img_x_axis(img)

    all_polys = image_to_polygons(img_file)
    all_polys = fi.remove_cells(all_polys)
    filename = img_file.replace("dat_binary/", "")
    filename = filename.replace(".tiff", "")

    img_x = np.ascontiguousarray(img_x, dtype=np.uint8)

    for poly in all_polys:

        polygon = Polygon(poly)
        p_vector(polygon, img_x)

    fig, ax = plt.subplots()
    ax.imshow(img_x, origin=["lower"])
    fig.savefig(
        "results/vectormaps/" + FIGDIR + f"_pvector_field_{filename}.png",
        transparent=True,
        dpi=300,
    )
    plt.close()


# ----------------------------------------------------


files = os.listdir(cwd + f"/{folder}")
img_list = []

n = len(files) - 1  # because for the .DS_Store file

files = sorted(files)


for i in range(n):
    img_file = f"{folder}/" + files[1 + i]  # because for the .DS_Store file
    img_list.append(img_file)
img_list = sorted(img_list)


for i in range(n):
    img_file = img_list[i]
    vector_field(img_file)
    p_vector_field(img_file)
