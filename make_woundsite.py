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
from scipy import optimize

import cell_properties as cell
import find_good_cells as fi

plt.rcParams.update({"font.size": 20})
plt.ioff()
pd.set_option("display.width", 1000)

# ------------


def best_fit_slope_and_intercept(xs, ys):
    m = ((cell.mean(xs) * cell.mean(ys)) - cell.mean(xs * ys)) / (
        (cell.mean(xs) * cell.mean(xs)) - cell.mean(xs * xs)
    )

    b = cell.mean(ys) - m * cell.mean(xs)

    return (m, b)


def isStraight(x, y):

    (m, b) = best_fit_slope_and_intercept(x, y)

    err = 0
    for i in range(len(x)):
        err += (y[i] - m * x[i] + b) ** 2

    if err > 0:
        return False
    else:
        return True


class ComputeCurvature:
    def __init__(self):
        """ Initialize some variables """
        self.xc = 0  # X-coordinate of circle center
        self.yc = 0  # Y-coordinate of circle center
        self.r = 0  # Radius of the circle
        self.xx = np.array([])  # Data points
        self.yy = np.array([])  # Data points

    def calc_r(self, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((self.xx - xc) ** 2 + (self.yy - yc) ** 2)

    def f(self, c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        ri = self.calc_r(*c)
        return ri - ri.mean()

    def df(self, c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc = c
        df_dc = np.empty((len(c), x.size))

        ri = self.calc_r(xc, yc)
        df_dc[0] = (xc - x) / ri  # dR/dxc
        df_dc[1] = (yc - y) / ri  # dR/dyc
        df_dc = df_dc - df_dc.mean(axis=1)[:, np.newaxis]
        return df_dc

    def fit(self, xx, yy):
        self.xx = xx
        self.yy = yy
        center_estimate = np.r_[np.mean(xx), np.mean(yy)]
        center = optimize.leastsq(
            self.f, center_estimate, Dfun=self.df, col_deriv=True
        )[0]

        self.xc, self.yc = center
        ri = self.calc_r(*center)
        self.r = ri.mean()

        return 1 / self.r  # Return the curvature


# Apply code for an example
# x = np.r_[36, 36, 19, 18, 33, 26]
# y = np.r_[14, 10, 28, 31, 18, 26]
# comp_curv = ComputeCurvature()
# curvature = comp_curv.fit(x, y)

# theta_fit = np.linspace(-np.pi, np.pi, 180)
# x_fit = comp_curv.xc + comp_curv.r*np.cos(theta_fit)
# y_fit = comp_curv.yc + comp_curv.r*np.sin(theta_fit)
# plt.plot(x_fit, y_fit, 'k--', label='fit', lw=2)
# plt.plot(x, y, 'ro', label='data', ms=8, mec='b', mew=1)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('curvature = {:.3e}'.format(curvature))
# plt.show()


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

m = 21
n = len(contour)

contourPlus = contour[n - int((m - 1) / 2) : n]
contourMinus = contour[0 : int((m - 1) / 2)]

con = np.concatenate([contourPlus, contour, contourMinus])

curvature = []

for i in range(n):
    x = np.r_[con[i : i + m][:, 0]]
    y = np.r_[con[i : i + m][:, 1]]
    comp_curv = ComputeCurvature()
    curvature.append(comp_curv.fit(x, y))

_df.append(
    {
        "name": filename,
        "polygons": polygons,
        "centriods": centriods,
        "curvature": curvature,
    }
)

for i in range(n):
    binary[0][int(contour[i][0]), int(contour[i][1])] = curvature[i] * 750


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

    m = 21
    n = len(contour)

    contourPlus = contour[n - int((m - 1) / 2) : n]
    contourMinus = contour[0 : int((m - 1) / 2)]

    con = np.concatenate([contourPlus, contour, contourMinus])

    curvature = []

    for i in range(n):
        x = np.r_[con[i : i + m][:, 0]]
        y = np.r_[con[i : i + m][:, 1]]
        comp_curv = ComputeCurvature()
        curvature.append(comp_curv.fit(x, y))

    _df.append(
        {
            "name": filename,
            "polygons": polygons,
            "centriods": centriods,
            "curvature": curvature,
        }
    )

    for i in range(n):
        binary[t + 1][int(contour[i][0]), int(contour[i][1])] = curvature[i] * 750

df = pd.DataFrame(_df)
df.to_pickle(f"databases/woundsites.pkl")

binary = np.asarray(binary, "uint8")
tifffile.imwrite(f"results/mitosis/binary_wound_{filename}.tif", binary)
