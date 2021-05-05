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
from scipy.stats import pearsonr
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
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters import hessian

import cellProperties as cell
import findGoodCells as fi
import commonLiberty as cl

plt.rcParams.update({"font.size": 20})

filename = "WoundL18h03"
vid = sm.io.imread(f"dat/{filename}/ecadFocus{filename}.tif").astype(int) / 255
vid = vid.astype("float32")

for t in range(len(vid)):
    vid[t] = entropy(vid[t], disk(10))

data = np.asarray(vid, "float32")
tifffile.imwrite(f"results/entropy{filename}.tif", data)

# vid = sm.io.imread(f"dat/{filename}/ecadFocus{filename}.tif").astype(int)

# for t in range(len(vid)):
#     vid[t] = hessian(vid[t], disk(10))

# data = np.asarray(vid, "float32")
# tifffile.imwrite(f"results/hessian{filename}.tif", data)
