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
plt.rcParams.update({"font.size": 14})

# -------------------

filenames, fileType = util.getFilesType()
T = 90
scale = 123.26 / 512

# -------------------

# collect all correlations

if False:
    _df = []
    for filename in filenames:
        # dfCorMid_12 = pd.read_pickle(f"databases/dfCorMidway{filename}_1-2.pkl")
        # dfCorMid_34 = pd.read_pickle(f"databases/dfCorMidway{filename}_3-4.pkl")
        # dfCorMid_56 = pd.read_pickle(f"databases/dfCorMidway{filename}_5-6.pkl")
        # dfCorMid_78 = pd.read_pickle(f"databases/dfCorMidway{filename}_7-8.pkl")
        dfCorMid = pd.read_pickle(f"databases/dfCorMidway{filename}.pkl")
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

        # dQ1dQ1 = np.nan_to_num(dfCorMid_34["dQ1dQ1Correlation"].iloc[0])
        # dQ1dQ1_std = np.nan_to_num(dfCorMid_34["dQ1dQ1Correlation_std"].iloc[0])
        # dQ1dP1 = np.nan_to_num(dfCorMid_56["dP1dQ1Correlation"].iloc[0])
        # dQ1dP1_std = np.nan_to_num(dfCorMid_56["dP1dQ1Correlation_std"].iloc[0])

        # dQ2dQ1 = np.nan_to_num(dfCorMid_56["dQ1dQ2Correlation"].iloc[0])
        # dQ2dQ1_std = np.nan_to_num(dfCorMid_56["dQ1dQ2Correlation_std"].iloc[0])
        # dQ2dQ2 = np.nan_to_num(dfCorMid_34["dQ2dQ2Correlation"].iloc[0])
        # dQ2dQ2_std = np.nan_to_num(dfCorMid_34["dQ2dQ2Correlation_std"].iloc[0])
        # dQ2dP1 = np.nan_to_num(dfCorMid_78["dP1dQ2Correlation"].iloc[0])
        # dQ2dP1_std = np.nan_to_num(dfCorMid_78["dP1dQ2Correlation_std"].iloc[0])
        # dQ2dP2 = np.nan_to_num(dfCorMid_78["dP2dQ2Correlation"].iloc[0])
        # dQ2dP2_std = np.nan_to_num(dfCorMid_78["dP2dQ2Correlation_std"].iloc[0])

        # dP1dP1 = np.nan_to_num(dfCorMid_12["dP1dP1Correlation"].iloc[0])
        # dP1dP1_std = np.nan_to_num(dfCorMid_12["dP1dP1Correlation_std"].iloc[0])
        # dP2dP2 = np.nan_to_num(dfCorMid_12["dP2dP2Correlation"].iloc[0])
        # dP2dP2_std = np.nan_to_num(dfCorMid_12["dP2dP2Correlation_std"].iloc[0])

        # count = np.nan_to_num(dfCorMid_56["Count"].iloc[0])

        dQ1dQ1 = np.nan_to_num(dfCorMid["dQ1dQ1Correlation"].iloc[0])
        dQ1dQ1_std = np.nan_to_num(dfCorMid["dQ1dQ1Correlation_std"].iloc[0])
        dQ1dP1 = np.nan_to_num(dfCorMid["dP1dQ1Correlation"].iloc[0])
        dQ1dP1_std = np.nan_to_num(dfCorMid["dP1dQ1Correlation_std"].iloc[0])

        dQ2dQ1 = np.nan_to_num(dfCorMid["dQ1dQ2Correlation"].iloc[0])
        dQ2dQ1_std = np.nan_to_num(dfCorMid["dQ1dQ2Correlation_std"].iloc[0])
        dQ2dQ2 = np.nan_to_num(dfCorMid["dQ2dQ2Correlation"].iloc[0])
        dQ2dQ2_std = np.nan_to_num(dfCorMid["dQ2dQ2Correlation_std"].iloc[0])
        dQ2dP1 = np.nan_to_num(dfCorMid["dP1dQ2Correlation"].iloc[0])
        dQ2dP1_std = np.nan_to_num(dfCorMid["dP1dQ2Correlation_std"].iloc[0])
        dQ2dP2 = np.nan_to_num(dfCorMid["dP2dQ2Correlation"].iloc[0])
        dQ2dP2_std = np.nan_to_num(dfCorMid["dP2dQ2Correlation_std"].iloc[0])

        dP1dP1 = np.nan_to_num(dfCorMid["dP1dP1Correlation"].iloc[0])
        dP1dP1_std = np.nan_to_num(dfCorMid["dP1dP1Correlation_std"].iloc[0])
        dP2dP2 = np.nan_to_num(dfCorMid["dP2dP2Correlation"].iloc[0])
        dP2dP2_std = np.nan_to_num(dfCorMid["dP2dP2Correlation_std"].iloc[0])

        count = np.nan_to_num(dfCorMid["Count"].iloc[0])

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

if False:
    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    fig, ax = plt.subplots(3, 4, figsize=(30, 18))

    T, R, Theta = dfCor["dRhodRho"].iloc[0].shape

    dRhodRho = np.zeros([len(filenames), T, R])
    dQ1dRho = np.zeros([len(filenames), T, R])
    dQ2dRho = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):
        RhoCount = dfCor["Count Rho"].iloc[i][:, :, :-1]
        dRhodRho[i] = np.sum(
            dfCor["dRhodRho"].iloc[i][:, :, :-1] * RhoCount, axis=2
        ) / np.sum(RhoCount, axis=2)
        RhoQCount = dfCor["Count Rho Q"].iloc[i][:, :, :-1]
        dQ1dRho[i] = np.sum(
            dfCor["dQ1dRho"].iloc[i][:, :, :-1] * RhoQCount, axis=2
        ) / np.sum(RhoQCount, axis=2)
        dQ2dRho[i] = np.sum(
            dfCor["dQ2dRho"].iloc[i][:, :, :-1] * RhoQCount, axis=2
        ) / np.sum(RhoQCount, axis=2)

    dRhodRho = np.mean(dRhodRho, axis=0)
    dQ1dRho = np.mean(dQ1dRho, axis=0)
    dQ2dRho = np.mean(dQ2dRho, axis=0)

    maxCorr = np.max([dRhodRho, -dRhodRho])
    t, r = np.mgrid[0:180:10, 0:90:10]
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

    c = ax[0, 1].pcolor(
        t,
        r,
        dRhodRho - dRhodRho[-1],
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 1])
    ax[0, 1].set_xlabel("Time (mins)")
    ax[0, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 1].title.set_text(r"$\langle \delta \rho \delta \rho \rangle$ nosie")

    maxCorr = np.max([dQ1dRho, -dQ1dRho])
    c = ax[0, 2].pcolor(
        t,
        r,
        dQ1dRho,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 2])
    ax[0, 2].set_xlabel("Time (mins)")
    ax[0, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 2].title.set_text(r"$\langle \delta Q^1 \delta \rho \rangle$")

    maxCorr = np.max([dQ2dRho, -dQ2dRho])
    c = ax[0, 3].pcolor(
        t,
        r,
        dQ2dRho,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 3])
    ax[0, 3].set_xlabel("Time (mins)")
    ax[0, 3].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 3].title.set_text(r"$\langle \delta Q^2 \delta \rho \rangle$")

    T, R, Theta = dfCor["dQ1dQ1"].iloc[0].shape

    dQ1dQ1 = np.zeros([len(filenames), T, R - 1])
    dQ1dP1 = np.zeros([len(filenames), T, R - 1])
    dQ2dQ1 = np.zeros([len(filenames), T, R - 1])
    dQ2dQ2 = np.zeros([len(filenames), T, R - 1])
    dQ2dP1 = np.zeros([len(filenames), T, R - 1])
    dQ2dP2 = np.zeros([len(filenames), T, R - 1])
    dP1dP1 = np.zeros([len(filenames), T, R - 1])
    dP2dP2 = np.zeros([len(filenames), T, R - 1])
    for i in range(len(filenames)):
        count = dfCor["Count"].iloc[i][:, :-1, :-1]
        dQ1dQ1[i] = np.sum(
            dfCor["dQ1dQ1"].iloc[i][:, :-1, :-1] * count, axis=2
        ) / np.sum(count, axis=2)
        dQ1dP1[i] = np.sum(
            dfCor["dQ1dP1"].iloc[i][:, :-1, :-1] * count, axis=2
        ) / np.sum(count, axis=2)
        dQ2dQ1[i] = np.sum(
            dfCor["dQ2dQ1"].iloc[i][:, :-1, :-1] * count, axis=2
        ) / np.sum(count, axis=2)
        dQ2dQ2[i] = np.sum(
            dfCor["dQ2dQ2"].iloc[i][:, :-1, :-1] * count, axis=2
        ) / np.sum(count, axis=2)
        dQ2dP1[i] = np.sum(
            dfCor["dQ2dP1"].iloc[i][:, :-1, :-1] * count, axis=2
        ) / np.sum(count, axis=2)
        dQ2dP2[i] = np.sum(
            dfCor["dQ2dP2"].iloc[i][:, :-1, :-1] * count, axis=2
        ) / np.sum(count, axis=2)
        dP1dP1[i] = np.sum(
            dfCor["dP1dP1"].iloc[i][:, :-1, :-1] * count, axis=2
        ) / np.sum(count, axis=2)
        dP2dP2[i] = np.sum(
            dfCor["dP2dP2"].iloc[i][:, :-1, :-1] * count, axis=2
        ) / np.sum(count, axis=2)

    dQ1dQ1 = np.mean(dQ1dQ1, axis=0)
    dQ1dP1 = np.mean(dQ1dP1, axis=0)
    dQ2dQ1 = np.mean(dQ2dQ1, axis=0)
    dQ2dQ2 = np.mean(dQ2dQ2, axis=0)
    dQ2dP1 = np.mean(dQ2dP1, axis=0)
    dQ2dP2 = np.mean(dQ2dP2, axis=0)
    dP1dP1 = np.mean(dP1dP1, axis=0)
    dP2dP2 = np.mean(dP2dP2, axis=0)

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


# display all norm correlations
if False:
    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")
    df = pd.read_pickle(f"databases/dfShape{fileType}.pkl")

    fig, ax = plt.subplots(3, 4, figsize=(30, 18))

    T, R, Theta = dfCor["dRhodRho"].iloc[0].shape

    dRhodRho = np.zeros([len(filenames), T, R])
    dQ1dRho = np.zeros([len(filenames), T, R])
    dQ2dRho = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):
        RhoCount = dfCor["Count Rho"].iloc[i][:, :, :-1]
        dRhodRho[i] = np.sum(
            dfCor["dRhodRho"].iloc[i][:, :, :-1] * RhoCount, axis=2
        ) / np.sum(RhoCount, axis=2)
        RhoQCount = dfCor["Count Rho Q"].iloc[i][:, :, :-1]
        dQ1dRho[i] = np.sum(
            dfCor["dQ1dRho"].iloc[i][:, :, :-1] * RhoQCount, axis=2
        ) / np.sum(RhoQCount, axis=2)
        dQ2dRho[i] = np.sum(
            dfCor["dQ2dRho"].iloc[i][:, :, :-1] * RhoQCount, axis=2
        ) / np.sum(RhoQCount, axis=2)

    std_dq = np.std(np.stack(np.array(df.loc[:, "dq"]), axis=0), axis=0)

    dRhodRho = np.mean(dRhodRho, axis=0)
    std_rho = dRhodRho[0, 0] ** 0.5
    dRhodRho = dRhodRho / dRhodRho[0, 0]
    dQ1dRho = np.mean(dQ1dRho, axis=0)
    dQ1dRho = dQ1dRho / (std_dq[0, 0] * std_rho)
    dQ2dRho = np.mean(dQ2dRho, axis=0)
    dQ2dRho = dQ2dRho / (std_dq[0, 1] * std_rho)

    maxCorr = np.max([1, -1])
    t, r = np.mgrid[0:180:10, 0:90:10]
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

    c = ax[0, 1].pcolor(
        t,
        r,
        dRhodRho - dRhodRho[-1],
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 1])
    ax[0, 1].set_xlabel("Time (mins)")
    ax[0, 1].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 1].title.set_text(r"$\langle \delta \rho \delta \rho \rangle$ nosie")

    c = ax[0, 2].pcolor(
        t,
        r,
        dQ1dRho,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 2])
    ax[0, 2].set_xlabel("Time (mins)")
    ax[0, 2].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 2].title.set_text(r"$\langle \delta Q^1 \delta \rho \rangle$")

    c = ax[0, 3].pcolor(
        t,
        r,
        dQ2dRho,
        cmap="RdBu_r",
        vmin=-maxCorr,
        vmax=maxCorr,
        shading="auto",
    )
    fig.colorbar(c, ax=ax[0, 3])
    ax[0, 3].set_xlabel("Time (mins)")
    ax[0, 3].set_ylabel(r"$R (\mu m)$ ")
    ax[0, 3].title.set_text(r"$\langle \delta Q^2 \delta \rho \rangle$")

    T, R, Theta = dfCor["dQ1dQ1"].iloc[0].shape

    dQ1dQ1 = np.zeros([len(filenames), T, R - 1])
    dQ1dP1 = np.zeros([len(filenames), T, R - 1])
    dQ2dQ1 = np.zeros([len(filenames), T, R - 1])
    dQ2dQ2 = np.zeros([len(filenames), T, R - 1])
    dQ2dP1 = np.zeros([len(filenames), T, R - 1])
    dQ2dP2 = np.zeros([len(filenames), T, R - 1])
    dP1dP1 = np.zeros([len(filenames), T, R - 1])
    dP2dP2 = np.zeros([len(filenames), T, R - 1])
    for i in range(len(filenames)):
        count = dfCor["Count"].iloc[i][:, :-1, :-1]
        dQ1dQ1[i] = np.sum(
            dfCor["dQ1dQ1"].iloc[i][:, :-1, :-1] * count, axis=2
        ) / np.sum(count, axis=2)
        dQ1dP1[i] = np.sum(
            dfCor["dQ1dP1"].iloc[i][:, :-1, :-1] * count, axis=2
        ) / np.sum(count, axis=2)
        dQ2dQ1[i] = np.sum(
            dfCor["dQ2dQ1"].iloc[i][:, :-1, :-1] * count, axis=2
        ) / np.sum(count, axis=2)
        dQ2dQ2[i] = np.sum(
            dfCor["dQ2dQ2"].iloc[i][:, :-1, :-1] * count, axis=2
        ) / np.sum(count, axis=2)
        dQ2dP1[i] = np.sum(
            dfCor["dQ2dP1"].iloc[i][:, :-1, :-1] * count, axis=2
        ) / np.sum(count, axis=2)
        dQ2dP2[i] = np.sum(
            dfCor["dQ2dP2"].iloc[i][:, :-1, :-1] * count, axis=2
        ) / np.sum(count, axis=2)
        dP1dP1[i] = np.sum(
            dfCor["dP1dP1"].iloc[i][:, :-1, :-1] * count, axis=2
        ) / np.sum(count, axis=2)
        dP2dP2[i] = np.sum(
            dfCor["dP2dP2"].iloc[i][:, :-1, :-1] * count, axis=2
        ) / np.sum(count, axis=2)

    dQ1dQ1 = np.mean(dQ1dQ1, axis=0)
    dQ1dP1 = np.mean(dQ1dP1, axis=0)
    dQ2dQ1 = np.mean(dQ2dQ1, axis=0)
    dQ2dQ2 = np.mean(dQ2dQ2, axis=0)
    dQ2dP1 = np.mean(dQ2dP1, axis=0)
    dQ2dP2 = np.mean(dQ2dP2, axis=0)
    dP1dP1 = np.mean(dP1dP1, axis=0)
    dP2dP2 = np.mean(dP2dP2, axis=0)

    std_dp = np.std(np.stack(np.array(df.loc[:, "dp"]), axis=0), axis=0)
    dQ1dQ1 = dQ1dQ1 / (std_dq[0, 0] * std_dq[0, 0])
    dQ1dP1 = dQ1dP1 / (std_dq[0, 0] * std_dp[0])
    dQ2dQ1 = dQ2dQ1 / (std_dq[0, 1] * std_dq[0, 1])
    dQ2dQ2 = dQ2dQ2 / (std_dq[0, 1] * std_dq[0, 1])
    dQ2dP1 = dQ2dP1 / (std_dq[0, 1] * std_dp[0])
    dQ2dP2 = dQ2dP2 / (std_dq[0, 1] * std_dp[1])
    dP1dP1 = dP1dP1 / (std_dp[0] * std_dp[0])
    dP2dP2 = dP2dP2 / (std_dp[1] * std_dp[1])

    t, r = np.mgrid[0:102:2, 0:82:2]
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
        f"results/Correlations Norm {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


def corRho_T(T, C):
    return C / T


def corRho_R(R, C, D):
    T = 2.5
    return C / T * np.exp(-(R ** 2) / (4 * D * T))


def corRhoS_R(R, C, D):
    return C * np.exp(-D * R)


# fit carves dRhodRho
if True:
    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(1, 3, figsize=(26, 6))

    T, R, Theta = dfCor["dRhodRho"].iloc[0].shape

    dRhodRho = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):
        RhoCount = dfCor["Count Rho"].iloc[i][:, :, :-1]
        dRhodRho[i] = np.sum(
            dfCor["dRhodRho"].iloc[i][:, :, :-1] * RhoCount, axis=2
        ) / np.sum(RhoCount, axis=2)

    dRhodRho = np.mean(dRhodRho, axis=0)

    dRhodRhoS = np.mean(dRhodRho, axis=0)

    m = sp.optimize.curve_fit(
        f=corRhoS_R,
        xdata=np.linspace(0, 80, 9),
        ydata=dRhodRhoS,
        p0=(0.0003, 0.04),
    )[0]

    limMax = np.max(dRhodRhoS)
    limMin = np.min(dRhodRhoS)

    ax[0].plot(np.linspace(0, 80, 9), dRhodRhoS)
    ax[0].plot(np.linspace(0, 80, 9), corRhoS_R(np.linspace(0, 80, 9), m[0], m[1]))
    ax[0].set_xlabel(r"$R (\mu m)$ ")
    ax[0].set_ylabel(r"$\delta\rho$ Correlation")
    ax[0].set_ylim([limMin * 1.1, limMax * 1.05])
    ax[0].title.set_text(r"$\langle \delta \rho \delta \rho \rangle$ structure")

    dRhodRhoN = dRhodRho - dRhodRho[-1]
    limMax = np.max(dRhodRhoN[1:, 0])
    limMin = np.min(dRhodRhoN[0, 1:])

    m = sp.optimize.curve_fit(
        f=corRho_R,
        xdata=np.linspace(10, 80, 8),
        ydata=dRhodRhoN[0, 1:],
        p0=(0.003, 10),
    )[0]

    ax[1].plot(np.linspace(10, 80, 8), dRhodRhoN[0, 1:])
    ax[1].plot(np.linspace(10, 80, 8), corRho_R(np.linspace(10, 80, 8), m[0], m[1]))
    ax[1].set_xlabel(r"$R (\mu m)$ ")
    ax[1].set_ylabel(r"$\delta\rho$ Correlation")
    ax[1].set_ylim([limMin * 1.1, limMax * 1.05])
    ax[1].title.set_text(r"$\langle \delta \rho \delta \rho \rangle$ noise $R$")

    m = sp.optimize.curve_fit(
        f=corRho_T,
        xdata=np.linspace(10, 170, 17),
        ydata=dRhodRhoN[1:, 0],
        p0=0.003,
    )[0]

    ax[2].plot(np.linspace(10, 170, 17), dRhodRhoN[1:, 0])
    ax[2].plot(np.linspace(10, 170, 17), corRho_T(np.linspace(10, 170, 17), m))
    ax[2].set_xlabel(r"Time (mins)")
    ax[2].set_ylabel(r"$\delta\rho$ Correlation")
    ax[2].set_ylim([limMin * 1.1, limMax * 1.05])
    ax[2].title.set_text(r"$\langle \delta \rho \delta \rho \rangle$ noise $T$")

    fig.savefig(
        f"results/fit dRhodRho",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")