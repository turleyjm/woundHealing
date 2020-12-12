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

import cellProperties as cell
import findGoodCells as fi

plt.rcParams.update({"font.size": 20})
plt.ioff()
pd.set_option("display.width", 1000)


f = open("pythonText.txt", "r")

filenames = f.read()
filenames = filenames.split(", ")

for filename in filenames:

    df = pd.read_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")
    _dfShape = []

    n = len(df)
    print(f"{filename}")

    for i in range(n):

        _dfShape.append(
            {
                "Time": df["Time"].iloc[i],
                "Polygon": df["Polygon"].iloc[i],
                "Centroid": df["Centroid"].iloc[i],
                "Area": df["Area"].iloc[i],
                "Perimeter": df["Perimeter"].iloc[i],
                "Orientation": df["Orientation"].iloc[i],
                "Shape Factor": df["Shape Factor"].iloc[i],
                "q": -df["Q"].iloc[i],
                "Trace(S)": df["Trace(S)"].iloc[i],
                "Polar": np.array(df["Polar_x"].iloc[i], df["Polar_y"].iloc[i]),
            }
        )

    dfShape = pd.DataFrame(_dfShape)
    dfShape.to_pickle(f"dat/{filename}/boundaryShape{filename}.pkl")
