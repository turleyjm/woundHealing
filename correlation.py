import os
from os.path import exists
import shutil
from math import floor, log10, factorial

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
import scipy.special as sc
import scipy.linalg as linalg
import shapely
import skimage as sm
import skimage.io
import skimage.measure
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
from mpl_toolkits.mplot3d import Axes3D
import tifffile
from skimage.draw import circle_perimeter
from scipy import optimize
import xml.etree.ElementTree as et
from scipy.optimize import leastsq
from datetime import datetime
import cellProperties as cell
import utils as util

pd.options.mode.chained_assignment = None
plt.rcParams.update({"font.size": 10})

# -------------------


def weighted_avg_and_std(values, weight, axis=0):
    average = np.average(values, weights=weight, axis=axis)
    variance = np.average((values - average) ** 2, weights=weight, axis=axis)
    return average, np.sqrt(variance)


filenames, fileType = util.getFilesType()
T = 90
scale = 123.26 / 512

# -------------------

# collect all correlations

if False:
    _df = []
    for filename in filenames:
        dfCorMid_12 = pd.read_pickle(f"databases/dfCorMidway{filename}_1-2.pkl")
        dfCorMid_34 = pd.read_pickle(f"databases/dfCorMidway{filename}_3-4.pkl")
        dfCorMid_56 = pd.read_pickle(f"databases/dfCorMidway{filename}_5-6.pkl")
        dfCorMid_78 = pd.read_pickle(f"databases/dfCorMidway{filename}_7-8.pkl")
        dfCorRho = pd.read_pickle(f"databases/dfCorRho{filename}.pkl")
        dfCorRhoQ = pd.read_pickle(f"databases/dfCorRhoQ{filename}.pkl")

        dRhodRho = np.nan_to_num(dfCorRho["dRhodRhoCorrelation"].iloc[0])
        dRhodRho_std = np.nan_to_num(dfCorRho["dRhodRhoCorrelation_std"].iloc[0])
        count_Rho = np.nan_to_num(dfCorRho["Count"].iloc[0])

        dQ1dRho = np.nan_to_num(dfCorRhoQ["dRhodQ1Correlation"].iloc[0])
        dQ1dRho_std = np.nan_to_num(dfCorRhoQ["dRhodQ1Correlation_std"].iloc[0])
        dQ2dRho = np.nan_to_num(dfCorRhoQ["dRhodQ2Correlation"].iloc[0])
        dQ2dRho_std = np.nan_to_num(dfCorRhoQ["dRhodQ2Correlation_std"].iloc[0])
        count_RhoQ = np.nan_to_num(dfCorRhoQ["Count"].iloc[0])

        dQ1dQ1 = np.nan_to_num(dfCorMid_34["dQ1dQ1Correlation"].iloc[0])
        dQ1dQ1_std = np.nan_to_num(dfCorMid_34["dQ1dQ1Correlation_std"].iloc[0])
        dQ1dP1 = np.nan_to_num(dfCorMid_56["dP1dQ1Correlation"].iloc[0])
        dQ1dP1_std = np.nan_to_num(dfCorMid_56["dP1dQ1Correlation_std"].iloc[0])

        dQ2dQ1 = np.nan_to_num(dfCorMid_56["dQ1dQ2Correlation"].iloc[0])
        dQ2dQ1_std = np.nan_to_num(dfCorMid_56["dQ1dQ2Correlation_std"].iloc[0])
        dQ2dQ2 = np.nan_to_num(dfCorMid_34["dQ2dQ2Correlation"].iloc[0])
        dQ2dQ2_std = np.nan_to_num(dfCorMid_34["dQ2dQ2Correlation_std"].iloc[0])
        dQ2dP1 = np.nan_to_num(dfCorMid_78["dP1dQ2Correlation"].iloc[0])
        dQ2dP1_std = np.nan_to_num(dfCorMid_78["dP1dQ2Correlation_std"].iloc[0])
        dQ2dP2 = np.nan_to_num(dfCorMid_78["dP2dQ2Correlation"].iloc[0])
        dQ2dP2_std = np.nan_to_num(dfCorMid_78["dP2dQ2Correlation_std"].iloc[0])

        dP1dP1 = np.nan_to_num(dfCorMid_12["dP1dP1Correlation"].iloc[0])
        dP1dP1_std = np.nan_to_num(dfCorMid_12["dP1dP1Correlation_std"].iloc[0])
        dP2dP2 = np.nan_to_num(dfCorMid_12["dP2dP2Correlation"].iloc[0])
        dP2dP2_std = np.nan_to_num(dfCorMid_12["dP2dP2Correlation_std"].iloc[0])

        count = np.nan_to_num(dfCorMid_56["Count"].iloc[0])

        _df.append(
            {
                "Filename": filename,
                "dRhodRho": dRhodRho,
                "dRhodRho_std": dRhodRho_std,
                "Count Rho": count_Rho,
                "dQ1dRho": dQ1dRho,
                "dQ1dRho_std": dQ1dRho_std,
                "dQ2dRho": dQ2dRho,
                "dQ2dRho_std": dQ2dRho_std,
                "Count Rho Q": count_RhoQ,
                "dQ1dQ1": dQ1dQ1,
                "dQ1dQ1_std": dQ1dQ1_std,
                "dQ1dP1": dQ1dP1,
                "dQ1dP1_std": dQ1dP1_std,
                "dQ2dQ1": dQ2dQ1,
                "dQ2dQ1_std": dQ2dQ1_std,
                "dQ2dQ2": dQ2dQ2,
                "dQ2dQ2_std": dQ2dQ2_std,
                "dQ2dP1": dQ2dP1,
                "dQ2dP1_std": dQ2dP1_std,
                "dQ2dP2": dQ2dP2,
                "dQ2dP2_std": dQ2dP2_std,
                "dP1dP1": dP1dP1,
                "dP1dP1_std": dP1dP1_std,
                "dP2dP2": dP2dP2,
                "dP2dP2_std": dP2dP2_std,
                "Count": count,
            }
        )

    df = pd.DataFrame(_df)
    df.to_pickle(f"databases/dfCorrelations{fileType}.pkl")

# total comparisions
if False:
    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    total = 0
    for i in range(len(dfCor)):
        total += np.sum(dfCor["Count Rho"].iloc[i])
        total += np.sum(dfCor["Count Rho Q"].iloc[i]) * 2
        total += np.sum(dfCor["Count"].iloc[i]) * 8

    numbers = "{:,}".format(int(total))
    print(numbers)


# display all correlations

if True:
    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    fig, ax = plt.subplots(3, 4, figsize=(30, 18))

    T, R, Theta = dfCor["dRhodRho"].iloc[0].shape

    dRhodRho = np.zeros([len(filenames), T, R])
    RhoCount = np.zeros([len(filenames), T, R])
    dQ1dRho = np.zeros([len(filenames), T, R - 1])
    dQ2dRho = np.zeros([len(filenames), T, R - 1])
    RhoQCount = np.zeros([len(filenames), T, R - 1])
    for i in range(len(filenames)):
        dRhodRho[i] = np.mean(dfCor["dRhodRho"].iloc[i][:, :, :-1], axis=2)
        RhoCount[i] = np.mean(dfCor["Count Rho"].iloc[i][:, :, :-1], axis=2)
        dQ1dRho[i] = np.mean(dfCor["dQ1dRho"].iloc[i][:, :, :-1], axis=2)
        dQ2dRho[i] = np.mean(dfCor["dQ2dRho"].iloc[i][:, :, :-1], axis=2)
        RhoQCount[i] = np.mean(dfCor["Count Rho Q"].iloc[i][:, :, :-1], axis=2)

    dRhodRho = weighted_avg_and_std(dRhodRho, RhoCount, axis=0)[0]
    dQ1dRho = weighted_avg_and_std(dQ1dRho, RhoQCount, axis=0)[0]
    dQ2dRho = weighted_avg_and_std(dQ2dRho, RhoQCount, axis=0)[0]

    maxCorr = np.max([dRhodRho, -dRhodRho])
    t, r = np.mgrid[0:85:5, 0:90:10]
    c = ax[0, 0].pcolor(
        t,
        r,
        dRhodRho,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 0])
    ax[0, 0].set_xlabel("Time (mins)")
    ax[0, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 0].title.set_text(r"$\langle \delta \rho \delta \rho \rangle$")

    maxCorr = np.max([dQ1dRho, -dQ1dRho])
    t, r = np.mgrid[0:85:5, 0:120:20]
    c = ax[0, 1].pcolor(
        t,
        r,
        dQ1dRho,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 1])
    ax[0, 1].set_xlabel("Time (mins)")
    ax[0, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 1].title.set_text(r"$\langle \delta Q^1 \delta \rho \rangle$")

    maxCorr = np.max([dQ2dRho, -dQ2dRho])
    c = ax[0, 2].pcolor(
        t,
        r,
        dQ2dRho,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 2])
    ax[0, 2].set_xlabel("Time (mins)")
    ax[0, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 2].title.set_text(r"$\langle \delta Q^2 \delta \rho \rangle$")

    T, R, Theta = dfCor["dQ1dQ1"].iloc[0].shape

    dQ1dQ1 = np.zeros([len(filenames), T, R])
    dQ1dP1 = np.zeros([len(filenames), T, R])
    dQ2dQ1 = np.zeros([len(filenames), T, R])
    dQ2dQ2 = np.zeros([len(filenames), T, R])
    dQ2dP1 = np.zeros([len(filenames), T, R])
    dQ2dP2 = np.zeros([len(filenames), T, R])
    dP1dP1 = np.zeros([len(filenames), T, R])
    dP2dP2 = np.zeros([len(filenames), T, R])
    count = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):
        dQ1dQ1[i] = np.mean(dfCor["dQ1dQ1"].iloc[i][:, :, :-1], axis=2)
        dQ1dP1[i] = np.mean(dfCor["dQ1dP1"].iloc[i][:, :, :-1], axis=2)
        dQ2dQ1[i] = np.mean(dfCor["dQ2dQ1"].iloc[i][:, :, :-1], axis=2)
        dQ2dQ2[i] = np.mean(dfCor["dQ2dQ2"].iloc[i][:, :, :-1], axis=2)
        dQ2dP1[i] = np.mean(dfCor["dQ2dP1"].iloc[i][:, :, :-1], axis=2)
        dQ2dP2[i] = np.mean(dfCor["dQ2dP2"].iloc[i][:, :, :-1], axis=2)
        dP1dP1[i] = np.mean(dfCor["dP1dP1"].iloc[i][:, :, :-1], axis=2)
        dP2dP2[i] = np.mean(dfCor["dP2dP2"].iloc[i][:, :, :-1], axis=2)
        count[i] = np.mean(dfCor["Count"].iloc[i][:, :, :-1], axis=2)

    dQ1dQ1 = weighted_avg_and_std(dQ1dQ1, count, axis=0)[0]
    dQ1dP1 = weighted_avg_and_std(dQ1dP1, count, axis=0)[0]
    dQ2dQ1 = weighted_avg_and_std(dQ2dQ1, count, axis=0)[0]
    dQ2dQ2 = weighted_avg_and_std(dQ2dQ2, count, axis=0)[0]
    dQ2dP1 = weighted_avg_and_std(dQ2dP1, count, axis=0)[0]
    dQ2dP2 = weighted_avg_and_std(dQ2dP2, count, axis=0)[0]
    dP1dP1 = weighted_avg_and_std(dP1dP1, count, axis=0)[0]
    dP2dP2 = weighted_avg_and_std(dP2dP2, count, axis=0)[0]

    t, r = np.mgrid[0:102:2, 0:82:2]
    maxCorr = np.max([dQ1dQ1, -dQ1dQ1])
    c = ax[1, 0].pcolor(
        t,
        r,
        dQ1dQ1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set_xlabel("Time (mins)")
    ax[1, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 0].title.set_text(r"$\langle \delta Q^1 \delta Q^1 \rangle$")

    maxCorr = np.max([dQ1dP1, -dQ1dP1])
    c = ax[1, 1].pcolor(
        t,
        r,
        dQ1dP1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 1])
    ax[1, 1].set_xlabel("Time (mins)")
    ax[1, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 1].title.set_text(r"$\langle \delta Q^1 \delta P_1 \rangle$")

    maxCorr = np.max([dQ2dQ1, -dQ2dQ1])
    c = ax[1, 2].pcolor(
        t,
        r,
        dQ2dQ1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 2])
    ax[1, 2].set_xlabel("Time (mins)")
    ax[1, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 2].title.set_text(r"$\langle \delta Q^2 \delta Q^1 \rangle$")

    maxCorr = np.max([dQ2dQ2, -dQ2dQ2])
    c = ax[1, 3].pcolor(
        t,
        r,
        dQ2dQ2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[1, 3])
    ax[1, 3].set_xlabel("Time (mins)")
    ax[1, 3].set_ylabel(r"$R (\mu m)$ ")
    ax[1, 3].title.set_text(r"$\langle \delta Q^2 \delta Q^2 \rangle$")

    maxCorr = np.max([dQ2dP1, -dQ2dP1])
    c = ax[2, 0].pcolor(
        t,
        r,
        dQ2dP1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 0])
    ax[2, 0].set_xlabel("Time (mins)")
    ax[2, 0].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 0].title.set_text(r"$\langle \delta Q^2 \delta P_1 \rangle$")

    maxCorr = np.max([dQ2dP2, -dQ2dP2])
    c = ax[2, 1].pcolor(
        t,
        r,
        dQ2dP2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 1])
    ax[2, 1].set_xlabel("Time (mins)")
    ax[2, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 1].title.set_text(r"$\langle \delta Q^2 \delta P_2 \rangle$")

    maxCorr = np.max([dP1dP1, -dP1dP1])
    c = ax[2, 2].pcolor(
        t,
        r,
        dP1dP1,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 2])
    ax[2, 2].set_xlabel("Time (mins)")
    ax[2, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 2].title.set_text(r"$\langle \delta P_1 \delta P_1 \rangle$")

    maxCorr = np.max([dP2dP2, -dP2dP2])
    c = ax[2, 3].pcolor(
        t,
        r,
        dP2dP2,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[2, 3])
    ax[2, 3].set_xlabel("Time (mins)")
    ax[2, 3].set_ylabel(r"$R (\mu m)$ ")
    ax[2, 3].title.set_text(r"$\langle \delta P_2 \delta P_2 \rangle$")

    fig.savefig(
        f"results/Correlations {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")
