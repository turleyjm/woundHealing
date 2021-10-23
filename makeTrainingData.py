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
from random import seed
from random import random
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


# -------------------

filenames, fileType = cl.getFilesType()

seed(100)
training = np.zeros([33, 512, 512])
i = 0
for filename in filenames:

    vid = sm.io.imread(f"dat/{filename}/ecadFocus{filename}.tif").astype(int)

    t = int(92 * random())
    training[3 * i : 3 * i + 3] = vid[t : t + 3]
    i += 1


training = np.asarray(training, "uint8")
tifffile.imwrite(f"dat/wekaBourdary.tif", training)
