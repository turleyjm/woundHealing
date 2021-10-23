import os
import shutil
from math import dist, floor, log10

from collections import Counter
import cv2
import matplotlib
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from pandas.core import frame
import scipy as sp
import scipy.linalg as linalg
import scipy.ndimage as nd
import shapely
import skimage as sm
import skimage.io
import skimage.measure
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
import tifffile
from skimage.draw import circle_perimeter
from scipy import optimize
import xml.etree.ElementTree as et

import cellProperties as cell
import findGoodCells as fi
import commonLiberty as cl

plt.rcParams.update({"font.size": 16})


filenames, fileType = cl.getFilesType()

if False:
    for filename in filenames:
        dfDivisions = pd.read_pickle(f"dat/{filename}/divisions{filename}.pkl")
        dfDivisions = dfDivisions.sort_values(["T", "X"], ascending=[True, True])

        focus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)

        (T, X, Y, rgb) = focus.shape

        divisions = np.zeros([T, 552, 552, 3])

        for x in range(X):
            for y in range(Y):
                divisions[:, 20 + x, 20 + y, :] = focus[:, x, y, :]

        for i in range(len(dfDivisions)):

            t = dfDivisions["T"].iloc[i]
            x = int(dfDivisions["X"].iloc[i])
            y = int(dfDivisions["Y"].iloc[i])
            ori = dfDivisions["Orientation"].iloc[i] * np.pi / 180

            rr0, cc0 = sm.draw.disk([551 - (y + 20), x + 20], 17)
            rr1, cc1 = sm.draw.disk([551 - (y + 20), x + 20], 13)
            rr2, cc2, val = sm.draw.line_aa(
                int(551 - (y + 17 * np.sin(ori) + 20)),
                int(x + 17 * np.cos(ori) + 20),
                int(551 - (y - 17 * np.sin(ori) + 20)),
                int(x - 17 * np.cos(ori) + 20),
            )

            divisions[t][rr0, cc0, 2] = 250
            divisions[t][rr1, cc1, 2] = 0
            divisions[t][rr2, cc2, 2] = 250
            divisions[t + 1][rr2, cc2, 2] = 250

        divisions = divisions[:, 20:532, 20:532]

        divisions = np.asarray(divisions, "uint8")
        tifffile.imwrite(f"dat/{filename}/division{filename}.tif", divisions)

        dfDivisions["Y"] = 512 - dfDivisions["Y"]
        dfDivisions.to_excel(f"dat/{filename}/dfDivisions{filename}.xlsx")

filenames = ["Unwound18h17"]
if True:
    for filename in filenames:
        dfDivisions = pd.read_excel(f"dat/{filename}/dfDivisionsEdit{filename}.xlsx")
        dfDivisions["Y"] = 512 - dfDivisions["Y"]
        dfDivisions = dfDivisions.sort_values(["T", "X"], ascending=[True, True])
        dfDivisions.to_pickle(f"dat/{filename}/divisionsEdit{filename}.pkl")

        focus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)

        (T, X, Y, rgb) = focus.shape

        divisions = np.zeros([T, 552, 552, 3])

        for x in range(X):
            for y in range(Y):
                divisions[:, 20 + x, 20 + y, :] = focus[:, x, y, :]

        for i in range(len(dfDivisions)):

            t = int(dfDivisions["T"].iloc[i])
            x = int(dfDivisions["X"].iloc[i])
            y = int(dfDivisions["Y"].iloc[i])
            ori = dfDivisions["Orientation"].iloc[i] * np.pi / 180

            rr0, cc0 = sm.draw.disk([551 - (y + 20), x + 20], 17)
            rr1, cc1 = sm.draw.disk([551 - (y + 20), x + 20], 13)
            rr2, cc2, val = sm.draw.line_aa(
                int(551 - (y + 17 * np.sin(ori) + 20)),
                int(x + 17 * np.cos(ori) + 20),
                int(551 - (y - 17 * np.sin(ori) + 20)),
                int(x - 17 * np.cos(ori) + 20),
            )

            divisions[t][rr0, cc0, 2] = 250
            divisions[t][rr1, cc1, 2] = 0
            divisions[t][rr2, cc2, 2] = 250
            divisions[t + 1][rr2, cc2, 2] = 250

        divisions = divisions[:, 20:532, 20:532]

        divisions = np.asarray(divisions, "uint8")
        tifffile.imwrite(f"dat/{filename}/divisionEdit{filename}.tif", divisions)
