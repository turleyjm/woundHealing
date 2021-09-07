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
from PIL import Image
import random
import scipy as sp
import scipy.linalg as linalg
from scipy.stats import mannwhitneyu
import shapely
import skimage as sm
import skimage.feature
import skimage.io
import skimage.measure
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
import tifffile
from skimage.draw import circle_perimeter
from scipy import optimize
import xml.etree.ElementTree as et
import plotly.graph_objects as go
from scipy.optimize import leastsq

import cellProperties as cell
import findGoodCells as fi
import commonLiberty as cl


b = 1
nb = 0
T = 10

generation_b = [b]
generation_nb = [nb]

for i in range(T):
    _b = (1 / (1 - (1 / 4) * b ** 2)) * (b * nb + (1 / 2) * b ** 2)
    _nb = (1 / (1 - (1 / 4) * b ** 2)) * (nb ** 2 + b * nb + (1 / 4) * b ** 2)

    b = _b
    nb = _nb

    generation_b.append(b)
    generation_nb.append(nb)

generation_b = np.array(generation_b)
generation_nb = np.array(generation_nb)

t = range(T + 1)
fig = plt.figure(1, figsize=(9, 8))
plt.plot(t, generation_b, label="balancer")
plt.plot(t, generation_nb, label="non-balancer")
plt.xlabel("Generation")
plt.ylabel("Frequency")
plt.legend()
plt.title(f"balancer proportion")
fig.savefig(
    f"results/balancer proportion",
    dpi=300,
    transparent=True,
)
plt.close("all")