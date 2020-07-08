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

plt.rcParams.update({"font.size": 28})
plt.ioff()
pd.set_option("display.width", 1000)


def best_fit_slope_and_intercept(xs, ys):
    m = ((cell.mean(xs) * cell.mean(ys)) - cell.mean(xs * ys)) / (
        (cell.mean(xs) * cell.mean(xs)) - cell.mean(xs * xs)
    )

    b = cell.mean(ys) - m * cell.mean(xs)

    return (m, b)


folder = "dat_binary"
FIGDIR = "fig"

cwd = os.getcwd()

R_num = 58
T_num = 101

count = 0

Area = []
rho_sd = []

files = os.listdir(cwd + f"/{folder}")
img_list = []

n = len(files) - 1  # because for the .DS_Store file

files = sorted(files)


for i in range(n):
    img_file = f"{folder}/" + files[1 + i]  # because for the .DS_Store file
    img_list.append(img_file)
img_list = sorted(img_list)

for img_file in img_list:

    filename = img_file.replace("dat_binary/", "")
    filename = filename.replace(".tiff", "")
    df = pd.read_pickle(f"databases/df_of_{filename}.pkl")
    Area.append(cell.mean(df["Area"]))
    Pol = []

l = (cell.mean(Area)) ** 0.5

Pol = []
azm = np.linspace(0, 2 * np.pi, T_num)
rad = np.linspace(0, 900 / l, R_num)
R = rad[1]

df = pd.read_pickle(f"databases/df_of_polaristion.pkl")

r = df["correction"][0]
t = np.array(range(len(r)) * R)

df2 = pd.read_pickle(f"databases/df_of_orientation.pkl")
r2 = df2["correction"][0]

fig = plt.figure(1, figsize=(8, 8))
plt.plot(t, r, "bo-", t, r2, "r^-")
plt.ylabel(r"$C_i(R)$")
plt.ylim(-0.2, 1.05)
plt.xlabel(r"$R$")
fig.savefig(
    "results/video_prop/" + FIGDIR + f"_Correlation_function_Polar_Ori",
    dpi=300,
    transparent=True,
)
plt.close("all")


# -------------------------------------

# df2 = pd.read_pickle(f"databases/df_of_orientation.pkl")

# r2 = df["correction"][0]

# fig = plt.figure(1, figsize=(8, 8))
# plt.plot(t, r2, color="#003F72")
# plt.xlabel("Distance by a typical cell length scale")
# plt.ylabel(r"$C(r)$")
# plt.ylim(-0.2, 1.05)
# # plt.title(f"Correlation Function of Orientation")
# fig.savefig(
#     "results/video_prop/" + FIGDIR + f"_Correlation_function_Orientation",
#     dpi=300,
#     transparent=True,
# )
# plt.close("all")
