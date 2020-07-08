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
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing

import cell_properties as cell
import find_good_cells as fi
import standard_deviation_orientation as sdori

plt.rcParams.update({"font.size": 24})

plt.ioff()
pd.set_option("display.width", 1000)


folder = "dat_binary"
FIGDIR = "fig"

cwd = os.getcwd()


# ----------------------------------------------------

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

    for i in range(len(df)):

        x = df["Polar_x"][i]
        y = df["Polar_y"][i]

        if x == 0 and y == 0:
            continue
        else:
            c = (x ** 2 + y ** 2) ** 0.5

            P = np.array([x, y]) / c

            theta = df["Orientation"][i]

            x = P[0] * np.cos(theta) - P[1] * np.sin(theta)
            y = P[0] * np.sin(theta) + P[1] * np.cos(theta)

            Pol.append(np.array([x, y]))

    n = len(Pol)

    P_dash = sum(Pol) / n

    rho = ((P_dash[0]) ** 2 + (P_dash[1]) ** 2) ** 0.5

    sigma_P = sum(((Pol - P_dash) ** 2) / n) ** 0.5

    sigma_P = sum(sigma_P)

    rho_sd.append(rho / sigma_P)

fig = plt.figure(1, figsize=(8, 9))
plt.boxplot(rho_sd, vert=False)
plt.xlabel(r"$\frac{P}{\sigma_p}$")
# plt.title(f"Magnitude of Mean Polarisation over Standard Deviation")
fig.savefig(
    "results/video_prop/" + FIGDIR + f"_Polar_mean_over_sd", dpi=300, transparent=True
)
plt.close("all")


l = (cell.mean(Area)) ** 0.5


# ----------------------------------------------------


R_num = 58
T_num = 101

Pol = []
azm = np.linspace(0, 2 * np.pi, T_num)
rad = np.linspace(0, 900 / l, R_num)

H = []
for t in range(T_num):
    H.append([])
    for r in range(R_num):
        H[t].append([])

R = rad[1]
T = azm[1]

for t in range(T_num):
    H[t][0] = 1

for img_file in img_list:

    filename = img_file.replace("dat_binary/", "")
    filename = filename.replace(".tiff", "")
    df = pd.read_pickle(f"databases/df_of_{filename}.pkl")
    l_img = cell.mean(df["Area"]) ** 0.5

    Q = []

    for i in range(len(df)):
        polygon = df["Polygon"][i]
        Q.append(cell.shape_tensor(polygon))

    S = cell.mean(Q)

    c = (S[0, 0] ** 2 + S[0, 1] ** 2) ** 0.5
    s = S / c
    thetastar = 0.5 * np.arccos(-s[0, 0])

    for i in range(len(df)):

        (cx, cy) = df["Centroid"][i]
        x = df["Polar_x"][i]
        y = df["Polar_y"][i]

        if x == 0 and y == 0:
            continue
        else:
            c = (x ** 2 + y ** 2) ** 0.5
            P = np.array([x, y]) / c
            theta = df["Orientation"][i]

            x = P[0] * np.cos(theta) - P[1] * np.sin(theta)
            y = P[0] * np.sin(theta) + P[1] * np.cos(theta)

            Pol_i = np.array([x, y])

            for j in range(len(df)):

                (cx_j, cy_j) = df["Centroid"][j]
                x = df["Polar_x"][j]
                y = df["Polar_y"][j]

                if x == 0 and y == 0:
                    continue
                elif i == j:
                    continue
                else:
                    c = (x ** 2 + y ** 2) ** 0.5
                    P = np.array([x, y]) / c
                    theta = df["Orientation"][j]

                    x = P[0] * np.cos(theta) - P[1] * np.sin(theta)
                    y = P[0] * np.sin(theta) + P[1] * np.cos(theta)

                    Pol_j = np.array([x, y])
                    cor = np.dot(Pol_i, Pol_j)

                    v = np.array([cx - cx_j, cy - cy_j])
                    phi = (
                        np.arctan(v[1] / v[0]) - thetastar
                    )  # to set the axis of polaristion to the x-axis
                    if v[0] < 0:
                        phi = phi - np.pi

                    while phi < 0:
                        phi = phi + 2 * np.pi

                    while phi > 2 * np.pi:
                        phi = phi - 2 * np.pi

                    t = phi // T
                    r = (((v[0] ** 2 + v[1] ** 2) ** 0.5) / l_img) // R

                    t = int(t)
                    r = int(r + 1)

                    H[t][r].append(cor)

    print(img_file)
    
radius_err = []
radius_err.append(0)
for r in range(R_num - 1):
    error = []
    num = []
    try:
        for t in range(T_num):
            try:
                n = len(H[t][r + 1])
                x = np.array(H[t][r + 1])
                mu = cell.mean(x)
                err = sum((((x - mu) ** 2) / n) ** 2)
                error.append(err)
                num.append(n)
            except:
                continue

        error = np.array(error)
        num = np.array(num)
        radius_err.append((sum(error * num) / sum(num)) / (sum(num)) ** 0.5)
    except:
        continue


radius = []
radius_mean = []
radius_mean.append(1)

for r in range(R_num - 1):
    for t in range(T_num):
        if H[t][r + 1] == []:
            H[t][r + 1] = 0
        else:
            H[t][r + 1] = sum(H[t][r + 1]) / len(H[t][r + 1])

        radius.append(H[t][r + 1])
    radius_mean.append(cell.mean(radius))
    radius = []

ra, th = np.meshgrid(rad, azm)

fig = plt.figure()
ax = Axes3D(fig)

plt.subplot(projection="polar")

pc = plt.pcolormesh(th, ra, H, vmin=-1, vmax=1)

plt.plot(azm, ra, color="k", ls="none")
plt.colorbar(pc)
plt.grid()
fig.savefig(
    "results/video_prop/" + FIGDIR + f"_Cor_polarheatmap", dpi=300, transparent=True
)
plt.close("all")


# ----------------------------------------------------

azm2 = np.linspace(0, 2 * np.pi, T_num)
rad2 = rad[0:8]

s = []
for t in range(T_num):
    s.append([])
    for r in range(8):
        s[t].append([])

for t in range(T_num):
    for r in range(8):
        s[t][r] = H[t][r]


ra2, th2 = np.meshgrid(rad2, azm2)

fig = plt.figure()
ax = Axes3D(fig)

plt.subplot(projection="polar")

pc = plt.pcolormesh(th2, ra2, s, vmin=-0.4, vmax=1)

plt.plot(azm2, ra2, color="k", ls="none")
plt.colorbar(pc)
plt.grid()
fig.savefig(
    "results/video_prop/" + FIGDIR + f"_Cor_polarheatmap_centre",
    dpi=300,
    transparent=True,
)
plt.close("all")


# ----------------------------------------------------
t = np.array(range(len(radius_mean[0:31])) * R)
t = t[0:31]
r = np.linspace(0.95, 29.3, 290)

_df = []

_df.append({"correction": radius_mean[0:31], "error": radius_err[0:31], "radius": t})

df = pd.DataFrame(_df)

df.to_pickle(f"databases/df_of_polaristion.pkl")

fig = plt.figure(1, figsize=(8, 8))
plt.scatter(t, radius_mean[0:31], color="#003F72")
plt.plot(
    r, -0.17 * np.exp(-1.3 * r),
)
plt.ylim(-0.1, 1)
plt.xlabel("Distance by average cell length")
plt.ylabel(f"Correlation function")
# plt.title(f"Correlation Function of Polarisation")
fig.savefig(
    "results/video_prop/" + FIGDIR + f"_Correlation_function", dpi=300, transparent=True
)
plt.close("all")

