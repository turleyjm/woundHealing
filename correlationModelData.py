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


def cor_R0(t, c, D):
    return D * -sc.expi(-c * t)


def forIntegral(y, R, T, c, D, L):
    y, R, T = np.meshgrid(y, R, T, indexing="ij")
    return D * np.exp(-y) * sc.jv(0, R * ((y - c * T) / (L * T)) ** 0.5) / y


def corQ2_Integral_T2(R, L):
    c = 0.014231800277153952
    D = 8.06377854e-06

    T = 2
    y = np.linspace(c * T, c * T * 200, 200000)
    h = y[1] - y[0]
    return np.sum(forIntegral(y, R, T, c, D, L) * h, axis=0)[:, 0]


def Integral_Q2(R, L):
    c = 0.005710229096088735
    D = 0.0001453973391906898

    T = 2
    y = np.linspace(c * T, c * T * 200, 200000)
    h = y[1] - y[0]
    return np.sum(forIntegral(y, R, T, c, D, L) * h, axis=0)[:, 0]


def CorrdP1(R, T):
    a = 0.014231800277153952
    b = 0.02502418
    C = 8.06377854e-06
    y = np.linspace(a, a * 100, 100000)
    h = y[1] - y[0]
    return np.sum(forIntegral(y, b, R, a, T, C) * h, axis=0)[:, 0]


def CorrdP2(R, T):
    a = 0.014231800277153952
    b = 0.02502418
    C = 2.89978933e-06
    y = np.linspace(a, a * 100, 100000)
    h = y[1] - y[0]
    return np.sum(forIntegral(y, b, R, a, T, C) * h, axis=0)[:, 0]


# -------------------

grid = 26
timeGrid = 51


# deltaQ2 (model)
if True:

    dfCor = pd.read_pickle(f"databases/dfCorrelations{fileType}.pkl")

    plt.rcParams.update({"font.size": 12})

    T, R, Theta = dfCor["dQ1dQ1Correlation"].iloc[0][:, :-1, :-1].shape

    dQ1dQ1 = np.zeros([len(filenames), T, R])
    dQ2dQ2 = np.zeros([len(filenames), T, R])
    for i in range(len(filenames)):
        dQ1dQ1total = dfCor["dQ1dQ1Count"].iloc[i][:, :-1, :-1]
        dQ1dQ1[i] = np.sum(
            dfCor["dQ1dQ1Correlation"].iloc[i][:, :-1, :-1] * dQ1dQ1total, axis=2
        ) / np.sum(dQ1dQ1total, axis=2)

        dQ2dQ2total = dfCor["dQ2dQ2Count"].iloc[i][:, :-1, :-1]
        dQ2dQ2[i] = np.sum(
            dfCor["dQ2dQ2Correlation"].iloc[i][:, :-1, :-1] * dQ2dQ2total, axis=2
        ) / np.sum(dQ2dQ2total, axis=2)

    dfCor = 0

    dQ1dQ1 = np.mean(dQ1dQ1, axis=0)
    dQ2dQ2 = np.mean(dQ2dQ2, axis=0)

    T = np.linspace(0, 2 * (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)

    m = sp.optimize.curve_fit(
        f=cor_R0,
        xdata=T[1:],
        ydata=dQ2dQ2[:, 0][1:],
        p0=(0.000006, 0.01),
    )[0]

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    ax[0].plot(T[1:], dQ2dQ2[:, 0][1:], label="Data")
    ax[0].plot(T[1:], cor_R0(T[1:], m[0], m[1]), label="Model")
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel(r"$Q^{(2)}$ Correlation")
    # ax[0].set_ylim([-0.000005, 0.000025])
    ax[0].set_xlim([0, 2 * timeGrid])
    ax[0].title.set_text(r"Correlation of $\delta Q^{(2)}$, $R=0$")
    ax[0].legend()

    m = sp.optimize.curve_fit(
        f=Integral_Q2,
        xdata=R[:25],
        ydata=dQ2dQ2[0][5:25],
        p0=0.025,
        method="lm",
    )[0]

    ax[1].plot(R, dQ2dQ2[0], label="Data")
    ax[1].plot(R, Integral_Q2(R, m), label="Model")
    ax[1].set_xlabel(r"$R (\mu m)$")
    ax[1].set_ylabel(r"$Q^{(2)}$ Correlation")
    # ax[1].set_ylim([-0.000005, 0.000025])
    ax[1].title.set_text(r"Correlation of $\delta Q^{(2)}$, $T=0$")
    ax[1].legend()
    fig.savefig(
        f"results/Correlation Q2 in T and R {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# deltaP2 (model)
if False:
    grid = 9
    timeGrid = 51

    dfCorrelation = pd.read_pickle(f"databases/dfCorrelation{fileType}.pkl")

    deltaP1Correlation = dfCorrelation["deltaP1Correlation"].iloc[0]
    deltaP1Correlation = np.mean(deltaP1Correlation[:, :, :-1], axis=2)
    deltaP2Correlation = dfCorrelation["deltaP2Correlation"].iloc[0]
    deltaP2Correlation = np.mean(deltaP2Correlation[:, :, :-1], axis=2)

    T = np.linspace(0, 2 * (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)

    m_P1 = sp.optimize.curve_fit(
        f=CorR0,
        xdata=T[1:],
        ydata=deltaP1Correlation[:, 0][1:],
        p0=(0.000006, 0.01),
    )[0]
    m = sp.optimize.curve_fit(
        f=CorR0,
        xdata=T[1:],
        ydata=deltaP2Correlation[:, 0][1:],
        p0=(0.000006, 0.01),
    )[0]

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    plt.subplots_adjust(wspace=0.3)
    plt.gcf().subplots_adjust(bottom=0.15)

    ax[0].plot(T[1:], deltaP2Correlation[:, 0][1:], label="Data")
    ax[0].plot(T[1:], CorR0(T[1:], m[0], m_P1[1]), label="Model")
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel(r"$P_2$ Correlation")
    ax[0].set_ylim([-0.000002, 0.00001])
    ax[0].set_xlim([0, 2 * timeGrid])
    ax[0].title.set_text(r"Correlation of $\delta P_2$, $R=0$")
    ax[0].legend()

    m = sp.optimize.curve_fit(
        f=Integral_P2,
        xdata=R,
        ydata=deltaP2Correlation[1],
        p0=0.025,
        method="lm",
    )[0]

    ax[1].plot(R, deltaP2Correlation[1], label="Data")
    ax[1].plot(R, Integral_P2(R, m), label="Model")
    ax[1].set_xlabel(r"$R (\mu m)$")
    ax[1].set_ylabel(r"$P_2$ Correlation")
    ax[1].set_ylim([-0.000002, 0.00001])
    ax[1].title.set_text(r"Correlation of $\delta P_2$, $T=2$")
    ax[1].legend()
    fig.savefig(
        f"results/Correlation P2 in T and R {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")
