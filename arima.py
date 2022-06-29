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
from scipy.stats import shapiro

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


def divisionDensitySpace(fileType, timeStep, T, rStep, R):
    filenames = util.getFilesOfType(fileType)
    count = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
    area = np.zeros([len(filenames), int(T / timeStep), int(R / rStep)])
    dfDivisions = pd.read_pickle(f"databases/dfDivisions{fileType}.pkl")
    for k in range(len(filenames)):
        filename = filenames[k]
        dfFile = dfDivisions[dfDivisions["Filename"] == filename]
        if "Wound" in filename:
            t0 = util.findStartTime(filename)
        else:
            t0 = 0
        t2 = int(timeStep / 2 * (int(T / timeStep) + 1) - t0 / 2)

        for r in range(count.shape[2]):
            for t in range(count.shape[1]):
                df1 = dfFile[dfFile["T"] > timeStep * t]
                df2 = df1[df1["T"] <= timeStep * (t + 1)]
                df3 = df2[df2["R"] > rStep * r]
                df = df3[df3["R"] <= rStep * (r + 1)]
                count[k, t, r] = len(df)

        inPlane = 1 - (
            sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype(int)[:t2]
            / 255
        )
        dist = (
            sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)[:t2]
            * scale
        )

        for r in range(area.shape[2]):
            for t in range(area.shape[1]):
                t1 = int(timeStep / 2 * t - t0 / 2)
                t2 = int(timeStep / 2 * (t + 1) - t0 / 2)
                if t1 < 0:
                    t1 = 0
                if t2 < 0:
                    t2 = 0
                area[k, t, r] = (
                    np.sum(
                        inPlane[t1:t2][
                            (dist[t1:t2] > rStep * r) & (dist[t1:t2] <= rStep * (r + 1))
                        ]
                    )
                    * scale ** 2
                )

    dd = np.zeros([int(T / timeStep), int(R / rStep)])
    std = np.zeros([int(T / timeStep), int(R / rStep)])
    sumArea = np.zeros([int(T / timeStep), int(R / rStep)])

    for r in range(area.shape[2]):
        for t in range(area.shape[1]):
            _area = area[:, t, r][area[:, t, r] > 800]
            _count = count[:, t, r][area[:, t, r] > 800]
            if len(_area) > 0:
                _dd, _std = weighted_avg_and_std(_count / _area, _area)
                dd[t, r] = _dd
                std[t, r] = _std
                sumArea[t, r] = np.sum(_area)
            else:
                dd[t, r] = np.nan
                std[t, r] = np.nan

    dd[sumArea < 8000] = np.nan
    dd = dd * 10000 * timeStep / 2

    return dd


# -------------------

scale = 123.26 / 512
T = 180
R = 110

# Make arima for unwounded

if False:
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


# Make arima for unwounded

if True:
    _df = []
    timeStep = 4
    rStep = 10
    fileType = "Unwound"
    dd = divisionDensitySpace(fileType, timeStep, T, rStep, R)
    dd_pred = np.zeros([dd.shape[0], dd.shape[1]])
    dd_pred[1:] = np.nan
    dd_pred[:2] = np.nan

    for r in range(dd.shape[1]):
        for t in range(dd.shape[0]):
            if np.isnan(dd[t, r]) == False:
                _df.append(
                    {
                        "FileType": fileType,
                        f"dd_{r}": dd[t, r],
                    }
                )
    df = pd.DataFrame(_df)

    # for r in range(dd.shape[1] - 1):
    #     df0 = df[f"dd_{r+1}"].dropna()
    #     if len(df0) > 34:

    #         # Original Series
    #         fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    #         ax[0, 0].plot(df0.dropna())
    #         ax[0, 0].set_title("Original Series")
    #         plot_acf(df0.dropna(), ax=ax[0, 1])
    #         ax[0, 1].set_ylim([-1.1, 1.1])

    #         # 1st Differencing
    #         ax[1, 0].plot(df0.diff().dropna())
    #         ax[1, 0].set_title("1st Order Differencing")
    #         plot_acf(df0.diff().dropna(), ax=ax[1, 1])
    #         ax[1, 1].set_ylim([-1.1, 1.1])

    #         # 2nd Differencing
    #         ax[2, 0].plot(df0.diff().dropna().diff().dropna())
    #         ax[2, 0].set_title("2nd Order Differencing")
    #         plot_acf(df0.diff().dropna().diff().dropna(), ax=ax[2, 1])
    #         ax[2, 1].set_ylim([-1.1, 1.1])

    #         fig.savefig(
    #             f"results/autocorrelation of division density r={r+1} {fileType}",
    #             transparent=True,
    #             bbox_inches="tight",
    #             dpi=300,
    #         )
    #         plt.close("all")
    d = 1

    # # find p
    # for r in range(dd.shape[1] - 1):
    #     df0 = df[f"dd_{r+1}"].dropna()
    #     if len(df0) > 34:
    #         fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    #         ax[0].plot(df0.diff().dropna())
    #         ax[0].set_title("1st Differencing")
    #         ax[1].set(ylim=(0, 5))
    #         plot_pacf(df0.diff().dropna(), ax=ax[1])
    #         ax[1].set_ylim([-1.1, 1.1])

    #         fig.savefig(
    #             f"results/division density AR term r={r+1} {fileType}",
    #             transparent=True,
    #             bbox_inches="tight",
    #             dpi=300,
    #         )
    #         plt.close("all")
    p = 1

    # find q
    # for r in range(dd.shape[1] - 1):
    #     df0 = df[f"dd_{r+1}"].dropna()
    #     if len(df0) > 34:
    #         fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    #         ax[0].plot(df0.diff().dropna())
    #         ax[0].set_title("1st Differencing")
    #         ax[1].set(ylim=(0, 1.2))
    #         plot_acf(df0.diff().dropna(), ax=ax[1])
    #         ax[1].set_ylim([-1.1, 1.1])

    #         fig.savefig(
    #             f"results/division density MA term r={r+1} {fileType}",
    #             transparent=True,
    #             bbox_inches="tight",
    #             dpi=300,
    #         )
    #         plt.close("all")
    q = 1

    sigma = []
    ar1 = []
    ma1 = []
    dist = []
    for r in range(dd.shape[1] - 1):
        df0 = df[f"dd_{r+1}"].dropna()
        if len(df0) > 34:
            dist.append((r + 1) * 10)
            model = statsm.tsa.arima.ARIMA(df0.diff().dropna(), order=(p, d, q))
            model_fit = model.fit()
            sigma.append(model_fit.params["sigma2"] ** 0.5)
            ar1.append(model_fit.params["ma.L1"])
            ma1.append(model_fit.params["ar.L1"])

    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    ax[0].plot(dist, sigma)
    ax[0].set_title("Sigma")
    ax[1].plot(dist, ar1)
    ax[1].set_title("ar1")
    ax[2].plot(dist, ma1)
    ax[2].set_title("ma1")
    fig.savefig(
        f"results/model parameters with distance {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

    # fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    # ax[0].hist(sigma)
    # ax[0].set_title("Sigma")
    # ax[1].hist(ar1)
    # ax[1].set_title("ar1")
    # ax[2].hist(ma1)
    # ax[2].set_title("ma1")
    # fig.savefig(
    #     f"results/model parameters {fileType}",
    #     transparent=True,
    #     bbox_inches="tight",
    #     dpi=300,
    # )
    # plt.close("all")

    sigma = np.mean(sigma)
    ar1 = np.mean(ar1)
    ma1 = np.mean(ma1)

    residuals = []
    for r in range(dd.shape[1] - 1):
        df0 = df[f"dd_{r+1}"].dropna()
        if len(df0) > 34:
            model = statsm.tsa.arima.ARIMA(df0.diff().dropna(), order=(p, d, q))
            with model.fix_params({"ar.L1": ar1, "ma.L1": ma1, "sigma2": sigma}):
                model_fit = model.fit()

            mu = np.mean(df0.diff().dropna())
            conf_int = model_fit.get_prediction(start=1).conf_int() - mu
            time = timeStep / 2 + np.array(range(int(T / timeStep))) * timeStep
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            lower_dd = (
                conf_int[f"upper dd_{r+1}"] - conf_int[f"lower dd_{r+1}"]
            ) / 2 - mu
            upper_dd = (
                conf_int[f"upper dd_{r+1}"] - conf_int[f"lower dd_{r+1}"]
            ) / 2 + mu
            predict = []
            for t in range(dd.shape[0] - 2):
                pred = (
                    df0.iloc[t + 2] + model_fit.predict(start=1, dynamic=False).iloc[t]
                )
                predict.append(pred)
                dd_pred[t + 2, r + 1] = pred

            ax.errorbar(
                time[3:],
                predict[1:],
                yerr=[lower_dd[1:], upper_dd[1:]],
                label="confidence intervals",
            )
            ax.plot(time, df0)
            # ax.set_ylim([0, 20])

            # fig.savefig(
            #     f"results/unwound confidence intervals r={r+1}",
            #     transparent=True,
            #     bbox_inches="tight",
            #     dpi=300,
            # )
            plt.close("all")

            for i in range(len(pd.DataFrame(model_fit.resid))):
                residuals.append(pd.DataFrame(model_fit.resid).iloc[i][0])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(residuals, bins=10)
    stats, p = shapiro(residuals)
    if p > 0.05:
        ax.title.set_text("Shapiro-Wilk Test Pass Normality")
    else:
        ax.title.set_text("Shapiro-Wilk Test Fail Normality")
    fig.savefig(
        f"results/Residuals Density starima {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

    t, r = np.mgrid[0:T:timeStep, 0:R:rStep]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    c = ax[0].pcolor(
        t,
        r,
        dd,
        vmin=0,
        vmax=18,
    )
    fig.colorbar(c, ax=ax[0])
    ax[0].set(xlabel="Time (mins)", ylabel=r"$R (\mu m)$")
    ax[0].title.set_text(f"Division density {fileType}")

    c = ax[1].pcolor(
        t,
        r,
        dd_pred,
        vmin=0,
        vmax=18,
    )
    fig.colorbar(c, ax=ax[1])
    ax[1].set(xlabel="Time (mins)", ylabel=r"$R (\mu m)$")
    ax[1].title.set_text(f"Division density predict {fileType}")

    Maxdd = np.max(
        [np.max(np.nan_to_num(dd_pred - dd)), -np.min(np.nan_to_num(dd_pred - dd))]
    )
    c = ax[2].pcolor(
        t,
        r,
        dd_pred - dd,
        vmin=-Maxdd,
        vmax=Maxdd,
        cmap="RdBu_r",
    )
    fig.colorbar(c, ax=ax[2])
    ax[2].set(xlabel="Time (mins)", ylabel=r"$R (\mu m)$")
    ax[2].title.set_text(f"Division density difference {fileType}")

    fig.savefig(
        f"results/Division density heatmap predict {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")