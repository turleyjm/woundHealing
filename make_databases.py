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

cwd = os.getcwd()


# ----------------------------------------------------


def image_to_polygons(img_file):
    """Takes a image file and produces an arrey of vertex"""

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
        poly = sm.measure.approximate_polygon(contour, tolerance=1)
        all_contours.append(contour)
        all_polys.append(poly)

    return all_polys


def save_dataframe(img_file):
    """Gives the bar plots of different properties of the image"""

    all_polys = image_to_polygons(img_file)
    all_polys = fi.remove_cells(all_polys)
    filename = img_file.replace("dat_binary/", "")
    filename = filename.replace(".tiff", "")
    _df = []
    for poly in all_polys:

        polygon = Polygon(poly)
        _df.append(
            {
                "Polygon": polygon,
                "Centroid": cell.centroid(polygon),
                "Area": cell.area(polygon),
                "Perimeter": cell.perimeter(polygon),
                "Orientation": cell.orientation(polygon),
                "Circularity": cell.circularity(polygon),
                "Ellipticity": cell.ellipticity(polygon),
                "Shape Factor": cell.shape_factor(polygon),
                "Q": cell.shape_tensor(polygon),
                "Trace(S)": cell.trace_S(polygon),
                "Trace(QQ)": cell.trace_QQ(polygon),
                "Trace(q)": cell.trace_qq(polygon),
                "Polar_x": cell.mayor_x_polar(polygon),
                "Polar_y": cell.minor_y_polar(polygon),
                "Polarisation Orientation": cell.polar_ori(polygon),
                "Polarisation Magnitude": cell.polar_mag(polygon),
            }
        )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/df_of_{filename}.pkl")


# ----------------------------------------------------


files = os.listdir(cwd + f"/{folder}")
img_list = []

n = len(files) - 1  # because for the .DS_Store file

files = sorted(files)


for i in range(n):
    img_file = f"{folder}/" + files[1 + i]  # because for the .DS_Store file
    img_list.append(img_file)


for i in range(n):
    img_file = img_list[i]
    save_dataframe(img_file)
    print(img_file)
