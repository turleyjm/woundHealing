import os
import shutil
from math import floor, log10

from collections import Counter
import cv2
import matplotlib
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

import cellProperties as cell


def boundary(allPolys, allContours):

    _allPolys = []
    _allContours = []
    for i in range(len(allPolys)):
        poly = allPolys[i]
        polygon = Polygon(poly)
        pts = list(polygon.exterior.coords)

        test = vectorBoundary(pts)

        if test:
            _allPolys.append(poly)
            _allContours.append(allContours[i])

    allPolys = _allPolys
    allContours = _allContours

    return allPolys, allContours


def convex(allPolys, allContours):

    _allPolys = []
    _allContours = []
    for i in range(len(allPolys)):
        try:
            polys = allPolys[i]
            if notConvex(polys):
                _allPolys.append(allPolys[i])
                _allContours.append(allContours[i])
        except:
            continue

    allPolys = _allPolys
    allContours = _allContours
    return allPolys, allContours


def notConvex(polys):

    polygon = Polygon(polys)
    angleList = angles(polygon)
    m = len(angleList)
    for j in range(m):
        if angleList[j] > 4.5:
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


def inArea(polys, muArea, sdArea):
    polygon = Polygon(polys)
    area = polygon.area
    if area > muArea + 5 * sdArea:
        return False
    else:
        return True


def meanSdArea(allPolys):

    area = []
    n = len(allPolys)
    for i in range(n):
        polygon = Polygon(allPolys[i])
        area.append(polygon.area)
    muArea = np.mean(area)
    sdArea = np.std(area)
    return (muArea, sdArea)


def tooBig(allPolys, allContours):

    (muArea, sdArea) = meanSdArea(allPolys)
    _allPolys = []
    _allContours = []
    for i in range(len(allPolys)):
        try:
            polys = allPolys[i]
            if inArea(polys, muArea, sdArea):
                _allPolys.append(allPolys[i])
                _allContours.append(allContours[i])
        except:
            continue
    allPolys = _allPolys
    allContours = _allContours
    return allPolys, allContours


def removeCells(allPolys, allContours):

    allPolys, allContours = noArea(allPolys, allContours)
    allPolys, allContours = convex(allPolys, allContours)
    allPolys, allContours = tooBig(allPolys, allContours)
    allPolys, allContours = simple(allPolys, allContours)
    allPolys, allContours = boundary(allPolys, allContours)
    return allPolys, allContours


def simple(allPolys, allContours):

    _allPolys = []
    _allContours = []
    for i in range(len(allPolys)):
        try:
            polys = allPolys[i]
            if LinearRing(polys).is_simple:
                _allPolys.append(allPolys[i])
                _allContours.append(allContours[i])
        except:
            continue
    allPolys = _allPolys
    allContours = _allContours
    return allPolys, allContours


def noArea(allPolys, allContours):

    _allPolys = []
    _allContours = []
    for i in range(len(allPolys)):
        try:
            poly = allPolys[i]
            polygon = Polygon(poly)
            if polygon.area != 0:
                _allPolys.append(allPolys[i])
                _allContours.append(allContours[i])
        except:
            continue
    allPolys = _allPolys
    allContours = _allContours
    return allPolys, allContours


def imgrcxy(img):

    n = len(img)

    imgxy = np.zeros(shape=(n, n))

    for x in range(n):
        for y in range(n):

            imgxy[x, y] = img[(n - 1) - y, x]

    return imgxy


def imgxyrc(imgxy):

    n = len(imgxy)

    img = np.zeros(shape=(n, n))

    for i in range(n):
        for j in range(n):

            img[(n - 1) - j, i] = imgxy[i, j]

    return img


def imgxAxis(img):

    n = len(img)

    imgx = np.zeros(shape=(n, n))

    for x in range(n):
        for y in range(n):

            imgx[x, y] = img[(n - 1) - x, y]

    return imgx


def vectorBoundary(pts):

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

