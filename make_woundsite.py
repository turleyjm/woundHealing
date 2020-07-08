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
import tifffile
from skimage.draw import circle_perimeter

import cell_properties as cell
import find_good_cells as fi

plt.rcParams.update({"font.size": 20})
plt.ioff()
pd.set_option("display.width", 1000)

filename = "wound16h01"

binary = sm.io.imread(f"dat_nucleus/{filename}" + "_woundsite" + ".tif").astype(float)

start = (250, 310)  # change coords

(T, X, Y) = binary.shape

_df = []
polygons = []
centriods = []

img_wound = binary[0]
img_label = sm.measure.label(img_wound, background=0, connectivity=1)
label = img_label[start]
contour = sm.measure.find_contours(img_label == label, level=0)[0]
poly = sm.measure.approximate_polygon(contour, tolerance=1)
polygon = Polygon(poly)
polygons.append(polygon)
(Cx, Cy) = cell.centroid(polygon)
centriods.append([Cx, Cy])
binary[0][img_label != label] = 0

for t in range(T - 1):
    img_wound = binary[t + 1]

    img_label = sm.measure.label(img_wound, background=0, connectivity=1)

    label = img_label[int(Cx), int(Cy)]
    contour = sm.measure.find_contours(img_label == label, level=0)[0]
    poly = sm.measure.approximate_polygon(contour, tolerance=1)
    polygon = Polygon(poly)
    polygons.append(polygon)
    (Cx, Cy) = cell.centroid(polygon)
    binary[t + 1][img_label != label] = 0

_df.append({"name": filename, "polygons": polygons, "centriods": centriods})

df = pd.DataFrame(_df)
df.to_pickle(f"databases/woundsites.pkl")

binary = np.asarray(binary, "uint8")
tifffile.imwrite(f"results/mitosis/binary_wound_{filename}.tif", binary)
