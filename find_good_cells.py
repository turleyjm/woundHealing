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


def boundary(all_polys):

    _all_polys = []
    for poly in all_polys:
        polygon = Polygon(poly)
        pts = list(polygon.exterior.coords)

        test = vector_boundary(pts)

        if test:
            _all_polys.append(poly)

    all_polys = _all_polys

    return all_polys


def convex(all_polys):

    allpolys = []
    for i in range(len(all_polys)):
        try:
            polys = all_polys[i]
            if is_not_convex(polys):
                allpolys.append(all_polys[i])
        except:
            continue

    all_polys = allpolys
    return all_polys


def is_not_convex(polys):

    polygon = Polygon(polys)
    angle_list = angles(polygon)
    m = len(angle_list)
    for j in range(m):
        if angle_list[j] > 4.5:
            return False
    return True


def angles(polygon):

    polygon = shapely.geometry.polygon.orient(polygon, sign=1.0)
    pts = list(polygon.exterior.coords)
    n = len(pts)
    pt = np.zeros(shape=(n + 1, 2))
    angles = []
    for i in range(n):
        pt[i][0] = pts[i][0]
        pt[i][1] = pts[i][1]
    pt[n] = pt[1]
    for i in range(n - 1):
        x = pt[i]
        y = pt[i + 1]
        z = pt[i + 2]
        u = x - y
        v = z - y
        theta = angle(u[0], u[1], v[0], v[1])
        angles.append(theta)
    return angles


def angle(u1, u2, v1, v2):

    top = u1 * v1 + u2 * v2
    bot = (u1 ** 2 + u2 ** 2) ** 0.5 * (v1 ** 2 + v2 ** 2) ** 0.5
    theta = np.arccos(top / bot)
    if u1 * v2 < u2 * v1:
        return theta
    else:
        return 2 * np.pi - theta


def is_in_area(polys, mu_area, sd_area):
    polygon = Polygon(polys)
    area = polygon.area
    if area > mu_area + 3 * sd_area:
        return False
    else:
        return True


def mean_sd_of_area(all_polys):

    area = []
    n = len(all_polys)
    for i in range(n):
        polygon = Polygon(all_polys[i])
        area.append(polygon.area)
    mu_area = cell.mean(area)
    sd_area = cell.sd(area)
    return (mu_area, sd_area)


def too_big(all_polys):

    (mu_area, sd_area) = mean_sd_of_area(all_polys)
    allpolys = []
    for i in range(len(all_polys)):
        try:
            polys = all_polys[i]
            if is_in_area(polys, mu_area, sd_area):
                allpolys.append(all_polys[i])
        except:
            continue
    all_polys = allpolys
    return all_polys


def remove_cells(all_polys):

    all_polys = no_area(all_polys)
    all_polys = convex(all_polys)
    all_polys = too_big(all_polys)
    all_polys = simple(all_polys)
    all_polys = boundary(all_polys)
    return all_polys


def simple(all_polys):

    allpolys = []
    for i in range(len(all_polys)):
        try:
            polys = all_polys[i]
            if LinearRing(polys).is_simple:
                allpolys.append(all_polys[i])
        except:
            continue
    all_polys = allpolys
    return all_polys


def no_area(all_polys):

    allpolys = []
    for i in range(len(all_polys)):
        try:
            poly = all_polys[i]
            polygon = Polygon(poly)
            if polygon.area != 0:
                allpolys.append(all_polys[i])
        except:
            continue
    all_polys = allpolys
    return all_polys


def img_rc_to_xy(img):

    n = len(img)

    img_xy = np.zeros(shape=(n, n))

    for x in range(n):
        for y in range(n):

            img_xy[x, y] = img[(n - 1) - y, x]

    return img_xy


def img_xy_to_rc(img_xy):

    n = len(img_xy)

    img = np.zeros(shape=(n, n))

    for i in range(n):
        for j in range(n):

            img[(n - 1) - j, i] = img_xy[i, j]

    return img


def img_x_axis(img):

    n = len(img)

    img_x = np.zeros(shape=(n, n))

    for x in range(n):
        for y in range(n):

            img_x[x, y] = img[(n - 1) - x, y]

    return img_x


def vector_boundary(pts):

    n = len(pts)

    test = True

    for i in range(n):
        if pts[i][0] == 0 or pts[i][0] == 511:
            test = False
        elif pts[i][1] == 0 or pts[i][1] == 511:
            test = False
        else:
            continue
    return test


def remove_wound(all_polys, img_xy, img_wound_xy):
    _all_polys = []

    for poly in all_polys:

        polygon = Polygon(poly)
        label = cell.find_label(polygon, img_xy)

        binary_wound = img_wound_xy[img_xy == label]

        mu = cell.mean(binary_wound)

        if mu == 255:
            _all_polys.append(poly)

    all_polys = _all_polys

    return all_polys
