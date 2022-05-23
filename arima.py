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
import skimage.feature
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
import statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as statsm

pd.options.mode.chained_assignment = None
plt.rcParams.update({"font.size": 10})

# -------------------


def weighted_avg_and_std(values, weight, axis=0):
    average = np.average(values, weights=weight, axis=axis)
    variance = np.average((values - average) ** 2, weights=weight, axis=axis)
    return average, np.sqrt(variance)


def divisionDensity(fileType, timeStep, T):
    filenames = util.getFilesOfType(fileType)
    count = np.zeros([len(filenames), int(T / timeStep)])
    area = np.zeros([len(filenames), int(T / timeStep)])
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    for k in range(len(filenames)):
        filename = filenames[k]
        t0 = util.findStartTime(filename)
        dfFile = dfDivisions[dfDivisions["Filename"] == filename]

        for t in range(count.shape[1]):
            df1 = dfFile[dfFile["T"] > timeStep * t]
            df = df1[df1["T"] <= timeStep * (t + 1)]
            count[k, t] = len(df)

        inPlane = 1 - (
            sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype(int) / 255
        )
        for t in range(area.shape[1]):
            t1 = int(timeStep / 2 * t - t0 / 2)
            t2 = int(timeStep / 2 * (t + 1) - t0 / 2)
            if t1 < 0:
                t1 = 0
            if t2 < 0:
                t2 = 0
            area[k, t] = np.sum(inPlane[t1:t2]) * scale ** 2

    time = []
    dd = []
    std = []
    for t in range(area.shape[1]):
        _area = area[:, t][area[:, t] > 0]
        _count = count[:, t][area[:, t] > 0]
        if len(_area) > 0:
            _dd, _std = weighted_avg_and_std(_count / _area, _area)
            dd.append(_dd * 10000 * timeStep / 2)
            std.append(_std)
            time.append(t * timeStep + timeStep / 2)

    return dd, time


# -------------------

scale = 123.26 / 512
T = 180
R = 110
rStep = 10

# Make arima for unwounded

if True:
    _df = []
    timeStep = 4
    fileType = "Unwound"
    dd = divisionDensity(fileType, timeStep, T)[0]
    for density in dd:
        _df.append(
            {
                "FileType": fileType,
                "dd": density,
            }
        )
    df = pd.DataFrame(_df)

    # Original Series
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    ax[0, 0].plot(df.dd)
    ax[0, 0].set_title("Original Series")
    plot_acf(df.dd, ax=ax[0, 1])
    ax[0, 1].set_ylim([-1.1, 1.1])

    # 1st Differencing
    ax[1, 0].plot(df.dd.diff())
    ax[1, 0].set_title("1st Order Differencing")
    plot_acf(df.dd.diff().dropna(), ax=ax[1, 1])
    ax[1, 1].set_ylim([-1.1, 1.1])

    # 2nd Differencing
    ax[2, 0].plot(df.dd.diff().diff())
    ax[2, 0].set_title("2nd Order Differencing")
    plot_acf(df.dd.diff().diff().dropna(), ax=ax[2, 1])
    ax[2, 1].set_ylim([-1.1, 1.1])

    fig.savefig(
        f"results/autocorrelation of division density {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")
    d = 1

    # find p
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(df.dd.diff())
    ax[0].set_title("1st Differencing")
    ax[1].set(ylim=(0, 5))
    plot_pacf(df.dd.diff().dropna(), ax=ax[1])
    ax[1].set_ylim([-1.1, 1.1])

    fig.savefig(
        f"results/division density AR term {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")
    p = 0

    # find q
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(df.dd.diff())
    ax[0].set_title("1st Differencing")
    ax[1].set(ylim=(0, 1.2))
    plot_acf(df.dd.diff().dropna(), ax=ax[1])
    ax[1].set_ylim([-1.1, 1.1])

    fig.savefig(
        f"results/division density MA term {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")
    q = 0

    model = statsm.tsa.arima.ARIMA(df.dd, order=(p, d, q))
    model_fit = model.fit()
    # print(model_fit.summary())

    residuals = pd.DataFrame(model_fit.resid)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind="kde", title="Density", ax=ax[1])
    fig.savefig(
        f"results/Residuals Density {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

    mu = np.mean(list(df.dd.diff())[1:])
    conf_int = model_fit.get_prediction(start=1).conf_int() - mu
    time = timeStep / 2 + np.array(range(int(T / timeStep))) * timeStep
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    lower_dd = (conf_int["upper dd"] - conf_int["lower dd"]) / 2 - mu
    upper_dd = (conf_int["upper dd"] - conf_int["lower dd"]) / 2 + mu
    ax.errorbar(
        time[1:],
        model_fit.predict(start=1, dynamic=False),
        yerr=[lower_dd, upper_dd],
        label="confidence intervals",
    )
    ax.set_ylim([0, 11])

    fileType = "WoundL"
    dd, time = divisionDensity(fileType, timeStep, T)
    _df = []
    for density in dd:
        _df.append(
            {
                "FileType": fileType,
                "dd": density,
            }
        )
    df = pd.DataFrame(_df)
    ax.plot(time, df.dd, label="Small wound")

    fileType = "WoundS"
    dd, time = divisionDensity(fileType, timeStep, T)
    _df = []
    for density in dd:
        _df.append(
            {
                "FileType": fileType,
                "dd": density,
            }
        )
    df = pd.DataFrame(_df)
    ax.plot(time, df.dd, label="large wound")
    ax.legend()

    fig.savefig(
        f"results/wound confidence intervals",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")


# Compare divison density with time

if False:
    timeStep = 4
    fileTypes = ["WoundS", "WoundL", "Unwound"]
    for fileType in fileTypes:
        dd = divisionDensity(fileType, timeStep, T)