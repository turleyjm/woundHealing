import os
from math import floor, log10
from scipy.special import gamma

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

import cellProperties as cell
import findGoodCells as fi

plt.rcParams.update({"font.size": 28})

plt.ioff()
pd.set_option("display.width", 1000)

folder = "dat_binary"
FIGDIR = "fig"

cwd = os.getcwd()


files = os.listdir(cwd + f"/{folder}")
img_list = []

n = len(files)  # because for the .DS_Store file

files = sorted(files)

for i in range(n):
    img_file = f"{folder}/" + files[i]  # because for the .DS_Store file
    img_list.append(img_file)


def rotation_matrix(theta):

    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    return R


a = []
b = []
num_frames = 14
num_videos = int(n / 14)
q0 = np.zeros((num_frames, num_videos))
dQr12 = np.zeros((num_frames, num_videos))
dQr22 = np.zeros((num_frames, num_videos))
count = np.zeros((num_frames, num_videos))

for video in range(num_videos):

    img_video = img_list[video * num_frames : video * num_frames + num_frames]
    img_file = img_video[0]
    videoname = img_file.replace("dat_binary/", "")
    videoname = videoname.replace("01.tiff", "")

    for frame in range(num_frames):

        img_file = img_video[frame]
        filename = img_file.replace("dat_binary/", "")
        filename = filename.replace(".tiff", "")
        df = pd.read_pickle(f"databases/df_of_{filename}.pkl")

        m = len(df)

        Q = []

        for i in range(m):
            polygon = df["Polygon"][i]
            Q.append(cell.qTensor(polygon))

        S = np.mean(Q, axis=0)

        c = (S[0, 0] ** 2 + S[0, 1] ** 2) ** 0.5
        s = S / c
        thetastar = 0.5 * np.arccos(-s[0, 0])

        if s[0, 1] > 0:
            thetastar = np.pi - thetastar

        R = rotation_matrix(2 * thetastar)

        Qr = np.matmul(R.transpose(), Q)
        Sr = np.matmul(R.transpose(), S)

        dQr = Qr - Sr

        dQr1 = []
        dQr2 = []
        for i in range(m):
            dQr1.append(dQr[i][0, 0])
            dQr2.append(dQr[i][0, 1])
        dQr1 = np.array(dQr1)
        dQr2 = np.array(dQr2)

        q0[frame, video] = c * 2 ** 0.5
        dQr12[frame, video] = np.mean(dQr1 ** 2)
        dQr22[frame, video] = np.mean(dQr2 ** 2)
        count[frame, video] = len(dQr1)

        # fig = plt.figure(1, figsize=(8, 8))

        # heatmap, xedges, yedges = np.histogram2d(
        #    dQr1, dQr2, range=[[-0.07, 0.07], [-0.07, 0.07]], bins=25
        # )
        # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        # plt.clf()
        # pos = plt.imshow(heatmap.T, extent=extent, origin="lower", vmin=0, vmax=20)
        # fig.colorbar(pos)
        # plt.axis((-0.07, 0.07, -0.07, 0.07))
        # fig.savefig(
        #    "results/free_energy/" + FIGDIR + f"_Video_dist_frame_{filename}",
        #    dpi=300,
        #    transparent=True,
        # )
        # plt.close("all")

x = range(14)


c = 0
dQ1 = 0
dQ2 = 0

for frame in range(num_frames):  # finds means and ver over the videos

    c += sum(count[frame, 0 : num_videos + 1])
    dQ1 += sum(dQr12[frame, 0 : num_videos + 1] * count[frame, 0 : num_videos + 1])
    dQ2 += sum(dQr22[frame, 0 : num_videos + 1] * count[frame, 0 : num_videos + 1])

dQ1 = dQ1 / c
dQ2 = dQ2 / c

err_q0 = []
err_dQr12 = []
err_dQr22 = []
err_dQ1 = 0
err_dQ2 = 0

for frame in range(num_frames):  # finds means and var over the videos

    err_dQ1 += sum(
        (dQr12[frame, 0 : num_videos + 1] - dQ1) ** 2 * count[frame, 0 : num_videos + 1]
    )
    err_dQ2 += sum(
        (dQr22[frame, 0 : num_videos + 1] - dQ2) ** 2 * count[frame, 0 : num_videos + 1]
    )

    sigma2 = sum(
        q0[frame, 0 : num_videos + 1] ** 2 * count[frame, 0 : num_videos + 1]
    ) / sum(count[frame, 0 : num_videos + 1])
    err_q0.append((sigma2 / sum(count[frame, 0 : num_videos + 1])) ** 0.5)

    q0[frame] = sum(
        q0[frame, 0 : num_videos + 1] * count[frame, 0 : num_videos + 1]
    ) / sum(count[frame, 0 : num_videos + 1])

    sigma2 = sum(
        dQr12[frame, 0 : num_videos + 1] ** 2 * count[frame, 0 : num_videos + 1]
    ) / sum(count[frame, 0 : num_videos + 1])
    err_dQr12.append((sigma2 / sum(count[frame, 0 : num_videos + 1])) ** 0.5)

    dQr12[frame] = sum(
        dQr12[frame, 0 : num_videos + 1] * count[frame, 0 : num_videos + 1]
    ) / sum(count[frame, 0 : num_videos + 1])

    sigma2 = sum(
        dQr22[frame, 0 : num_videos + 1] ** 2 * count[frame, 0 : num_videos + 1]
    ) / sum(count[frame, 0 : num_videos + 1])
    err_dQr22.append((sigma2 / sum(count[frame, 0 : num_videos + 1])) ** 0.5)

    dQr22[frame] = sum(
        dQr22[frame, 0 : num_videos + 1] * count[frame, 0 : num_videos + 1]
    ) / sum(count[frame, 0 : num_videos + 1])


err_dQ1 = err_dQ1 ** 0.5 / c
err_dQ2 = err_dQ2 ** 0.5 / c
q0 = q0[0 : num_frames + 1, 0]
dQr12 = dQr12[0 : num_frames + 1, 0]
dQr22 = dQr22[0 : num_frames + 1, 0]


def best_fit_slope_and_intercept(xs, ys):
    m = ((np.mean(xs) * np.mean(ys)) - np.mean(xs * ys)) / (
        (np.mean(xs) * np.mean(xs)) - np.mean(xs * xs)
    )

    b = np.mean(ys) - m * np.mean(xs)

    return (m, b)


# fig = plt.figure(1, figsize=(9, 8))
# plt.errorbar(x, dQr12, yerr=err_dQr12, fmt=".")
# # plt.plot([0, 13], [b, 13 * m + b], "k-", lw=2)
# plt.ylim(0.0002, 0.0008)
# plt.xlabel("Time")
# # plt.ylabel(r"$\langle (\delta Q_1)^2 \rangle$")
# fig.savefig(
#     "results/free_energy/" + FIGDIR + f"_dQr12", dpi=300, transparent=True,
# )
# plt.close("all")

fig = plt.figure(1, figsize=(9, 8))
plt.gcf().subplots_adjust(left=0.2)
plt.errorbar(x, dQr22, yerr=err_dQr22, fmt=".")
plt.errorbar(x, dQr12, yerr=err_dQr12, fmt=".")
# plt.plot([0, 13], [b, 13 * m + b], "k-", lw=2)
plt.ylim(0.0003, 0.0007)
plt.xlabel("Time")
# plt.ylabel(r"$\langle (\delta Q_2)^2 \rangle$")
fig.savefig(
    "results/free_energy/" + FIGDIR + f"_dQr12_and_dQr22",
    dpi=300,
    transparent=True,
)
plt.close("all")

xs = np.array(list(x), dtype=np.float64)
ys = np.array(q0, dtype=np.float64)
(m, d) = best_fit_slope_and_intercept(xs, ys)

fig = plt.figure(1, figsize=(9, 8))
plt.gcf().subplots_adjust(left=0.2)
plt.errorbar(x, q0, yerr=err_q0, fmt=".")
plt.plot([0, 13], [d, 13 * m + d], "k-", lw=2)
# plt.title(f"m = {m}, d = {d}")
plt.xlabel("Time")
plt.ylabel(r"$\aver{\norm{\bmQ}^2}$")
fig.savefig(
    "results/free_energy/" + FIGDIR + f"_q0",
    dpi=300,
    transparent=True,
)
plt.close("all")

theta = ((2 * gamma(5 / 4) * dQ2) / gamma(3 / 4)) ** 2
theta_up = ((2 * gamma(5 / 4) * (dQ2 + err_dQ2)) / gamma(3 / 4)) ** 2
theta_down = ((2 * gamma(5 / 4) * (dQ2 - err_dQ2)) / gamma(3 / 4)) ** 2
ab = theta / (2 * dQ1)
ab_up = theta_up / (2 * (dQ1 - err_dQ1))
ab_down = theta_down / (2 * (dQ1 + err_dQ1))
C = theta / (d - ab)
C_up = theta_up / (d - ab_up)
C_down = theta_down / (d - ab_down)
kBb = m / ab
kBb_up = m / ab_down
kBb_down = m / ab_up
kBa = m / (ab ** 2)
kBa_up = m / (ab_down ** 2)
kBa_down = m / (ab_up ** 2)


if abs(theta - theta_up) > abs(theta - theta_down):
    print(r"$\theta =$" + f" {theta}" + r" $\pm$ " + f"{abs(theta - theta_up)}")
else:
    print(r"$\theta =$" + f" {theta}" + r" $\pm$ " + f"{abs(theta - theta_down)}")

if abs(ab - ab_up) > abs(ab - ab_down):
    print(r"$\frac{a_0}{b_0} =$" + f" {ab}" + r" $\pm$ " + f"{abs(ab - ab_up)}")
else:
    print(r"$\frac{a_0}{b_0} =$" + f" {ab}" + r" $\pm$ " + f"{abs(ab - ab_down)}")

if abs(C - C_up) > abs(C - C_down):
    print(r"$C =$" + f" {C}" + r" $\pm$ " + f"{abs(C - C_up)}")
else:
    print(r"$C =$" + f" {C}" + r" $\pm$ " + f"{abs(C - C_down)}")

if abs(kBb - kBb_up) > abs(kBb - kBb_down):
    print(r"$\frac{k_B}{b_0} =$" + f" {kBb}" + r" $\pm$ " + f"{abs(kBb - kBb_up)}")
else:
    print(r"$\frac{k_B}{b_0} =$" + f" {kBb}" + r" $\pm$ " + f"{abs(kBb - kBb_down)}")

if abs(kBa - kBa_up) > abs(kBa - kBa_down):
    print(r"$\frac{k_B}{a_0} =$" + f" {kBa}" + r" $\pm$ " + f"{abs(kBa - kBa_up)}")
else:
    print(r"$\frac{k_B}{a_0} =$" + f" {kBa}" + r" $\pm$ " + f"{abs(kBa - kBa_down)}")
