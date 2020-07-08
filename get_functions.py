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


def Get_Functions():

    function = []
    function_title = []
    function_name = []
    lim = []
    radians = []

    yes_area = "N"
    yes_perimeter = "N"
    yes_ori = "N"
    yes_trace_S = "N"
    yes_trace_qq = "N"
    yes_trace_QQ = "N"
    yes_cir = "N"
    yes_sf = "Y"
    yes_ell = "N"
    yes_Pol_theta = "N"
    yes_Pol_r = "N"

    if yes_area == "Y":
        function.append(cell.area)
        function_title.append("Area")
        function_name.append(r"$A$")
        lim.append("None")
        radians.append("N")

    if yes_perimeter == "Y":
        function.append(cell.perimeter)
        function_title.append("Perimeter")
        function_name.append(r"$P$")
        lim.append("None")
        radians.append("N")

    if yes_ori == "Y":
        function.append(cell.orientation)
        function_title.append("Orientation")
        function_name.append(r"$\theta$")
        lim.append((0, np.pi))
        radians.append("Y")

    if yes_trace_S == "Y":
        function.append(cell.trace_S)
        function_title.append("Trace(S)")
        function_name.append(r"$Tr(S)$")
        lim.append("None")
        radians.append("N")

    if yes_trace_qq == "Y":
        function.append(cell.trace_qq)
        function_title.append("Trace(q)")
        function_name.append(r"$Tr(q^2)$")
        lim.append("None")
        radians.append("N")

    if yes_trace_QQ == "Y":
        function.append(cell.trace_QQ)
        function_title.append("Trace(QQ)")
        function_name.append(r"$\langle Tr(Q^2) \rangle$")
        lim.append("None")
        radians.append("N")

    if yes_cir == "Y":
        function.append(cell.circularity)
        function_title.append("Circularity")
        function_name.append(r"$Cir$")
        lim.append("None")
        radians.append("N")

    if yes_sf == "Y":
        function.append(cell.shape_factor)
        function_title.append("Shape Factor")
        function_name.append(r"$\langle S_f \rangle$")
        lim.append((0, 1))
        radians.append("N")

    if yes_ell == "Y":
        function.append(cell.ellipticity)
        function_title.append("Ellipticity")
        function_name.append(r"$Ell$")
        lim.append("None")
        radians.append("N")

    if yes_Pol_theta == "Y":
        function.append(cell.polar_ori)
        function_title.append("Polarisation Orientation")
        function_name.append(r"$Pol_(\theta)$")
        lim.append((0, 2 * np.pi))
        radians.append("Y")

    if yes_Pol_r == "Y":
        function.append(cell.polar_mag)
        function_title.append("Polarisation Magnitude")
        function_name.append(r"$Pol_r$")
        lim.append("None")
        radians.append("N")

    return (function, function_title, function_name, lim, radians)
