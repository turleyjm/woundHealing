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


folder1 = "dat/datFocus"
folder2 = "dat/datWound"

cwd = os.getcwd()

files = os.listdir(cwd + f"/{folder1}")

for vid_file in files:

    filename = vid_file

    filename = filename.replace("focus", "")
    filename = filename.replace(".tif", "")

    filename1 = filename.replace("H2", "")
    filename1 = filename1.replace("Ecad", "")

    woundMask = sm.io.imread(f"{folder2}/" + f"maskWound{filename1}.tif").astype(int)

    vid_file = f"{folder1}/" + vid_file

    vid = sm.io.imread(vid_file).astype(int)

    vid[woundMask == 255] = 0

    vid = np.asarray(vid, "uint8")
    tifffile.imwrite(f"dat/datFocusRemoveWound/focusRemoveWound{filename}.tif", vid)

