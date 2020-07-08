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


def image_to_poly_labels(img_file):
    """Takes a image file and produces an arrey of vertex add the polygons labels to be used 
    in the making of the heat plot"""

    img = sm.io.imread(img_file).astype(int)
    mu = cell.mean(cell.mean(img))
    if mu < 130:
        img = img
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

    return (all_polys, img_label, img_labels)


def partition_matrix(img_file, n):

    H = np.zeros(shape=(n, n))
    C = np.zeros(shape=(n, n))
    mod = 512 / n

    filename = img_file.replace("dat_binary/", "")
    filename = filename.replace(".tiff", "")
    df = pd.read_pickle(f"databases/df_of_{filename}.pkl")
    b = len(df)

    for a in range(b):
        (cx, cy) = df["Centroid"][a]
        p_mag = df["Polarisation Magnitude"][a]

        x = cx // mod
        y = cy // mod

        x = int(x)
        y = int(y)

        H[x, y] += p_mag
        C[x, y] += 1

    for i in range(n):
        for j in range(n):
            H[i, j] = H[i, j] / C[i, j]

            if H[i, j] > 0:
                H[i, j] = H[i, j]
            else:
                H[i, j] = 0

    H_rc = fi.img_xy_to_rc(H)
    H = fi.img_x_axis(H_rc)

    fig, ax = plt.subplots()
    pos = ax.imshow(H, origin=["lower"])
    plt.title(f"Grid Heatmap of Polaristion")
    fig.colorbar(pos)
    fig.savefig(
        "results/heat-vectormaps/" + FIGDIR + f"Grid_Polarised_{filename}.png",
        dpi=300,
        transparent=True,
    )
    plt.close()


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

# ----------------------------------------------------
for img_file in img_list:

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
            _df.append({"QQ": cell.polar_mag(polygon)})

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

    n = len(img)
    img_xy = img * 0.00000001 + y

    for idx, label in enumerate(img_labels):

        try:
            poly = all_polys[label - 1]
            polygon = Polygon(poly)
            x = cell.polar_mag(polygon)

            if y < x < z:
                img_xy[img_label == label] = x
            elif x > z:
                img_xy[img_label == label] = z
            else:
                img_xy[img_label == label] = y
        except:
            continue

    fig, ax = plt.subplots()

    img = fi.img_xy_to_rc(img_xy)
    img_x = fi.img_x_axis(img)

    for poly in _all_polys:

        polygon = Polygon(poly)
        p_vector(polygon, img_x)

    fig, ax = plt.subplots()
    pos = ax.imshow(img_x, origin=["lower"])
    plt.title(f"Heat-vectormap of Polaristion")
    fig.colorbar(pos)
    fig.savefig(
        "results/heat-vectormaps/" + FIGDIR + f"_Polarised_{filename}.png",
        dpi=300,
        transparent=True,
    )
    plt.close()


for img_file in img_list:
    partition_matrix(img_file, 9)
