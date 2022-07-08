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
            dd.append(_dd * 10000)
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
    dd = dd * 10000

    return dd


def starSig(z):

    for t in range(z.shape[0]):
        for r in range(z.shape[1]):
            if z[t, r] >= 0.05:
                z[t, r] = 0
            else:
                z[t, r] = int(-np.log10(z[t, r]))

    return z


# -------------------

scale = 123.26 / 512
T = 180
R = 100

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
    # fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    # ax[0, 0].plot(df.dd)
    # ax[0, 0].set_title("Original Series")
    # plot_acf(df.dd, ax=ax[0, 1])
    # ax[0, 1].set_ylim([-1.1, 1.1])

    # # 1st Differencing
    # ax[1, 0].plot(df.dd.diff())
    # ax[1, 0].set_title("1st Order Differencing")
    # plot_acf(df.dd.diff().dropna(), ax=ax[1, 1])
    # ax[1, 1].set_ylim([-1.1, 1.1])

    # # 2nd Differencing
    # ax[2, 0].plot(df.dd.diff().diff())
    # ax[2, 0].set_title("2nd Order Differencing")
    # plot_acf(df.dd.diff().diff().dropna(), ax=ax[2, 1])
    # ax[2, 1].set_ylim([-1.1, 1.1])

    # fig.savefig(
    #     f"results/autocorrelation of division density {fileType}",
    #     transparent=True,
    #     bbox_inches="tight",
    #     dpi=300,
    # )
    # plt.close("all")
    d = 1

    # find p
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].plot(df.dd.diff())
    # ax[0].set_title("1st Differencing")
    # ax[1].set(ylim=(0, 5))
    # plot_pacf(df.dd.diff().dropna(), ax=ax[1])
    # ax[1].set_ylim([-1.1, 1.1])

    # fig.savefig(
    #     f"results/division density AR term {fileType}",
    #     transparent=True,
    #     bbox_inches="tight",
    #     dpi=300,
    # )
    # plt.close("all")
    p = 0

    # find q
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].plot(df.dd.diff())
    # ax[0].set_title("1st Differencing")
    # ax[1].set(ylim=(0, 1.2))
    # plot_acf(df.dd.diff().dropna(), ax=ax[1])
    # ax[1].set_ylim([-1.1, 1.1])

    # fig.savefig(
    #     f"results/division density MA term {fileType}",
    #     transparent=True,
    #     bbox_inches="tight",
    #     dpi=300,
    # )
    # plt.close("all")
    q = 0

    model = statsm.tsa.arima.ARIMA(df.dd, order=(p, d, q))
    model_fit = model.fit()
    # print(model_fit.summary())

    residuals = pd.DataFrame(model_fit.resid)
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # residuals.plot(title="Residuals", ax=ax[0])
    # residuals.plot(kind="kde", title="Density", ax=ax[1])
    # fig.savefig(
    #     f"results/Residuals Density {fileType}",
    #     transparent=True,
    #     bbox_inches="tight",
    #     dpi=300,
    # )
    # plt.close("all")

    mu = np.mean(list(df.dd.diff())[1:])
    # model_fit.params["sigma2"]
    sigma = model_fit.params["sigma2"] ** 0.5
    lower_1star = 1.96 * sigma - mu
    upper_1star = 1.96 * sigma + mu
    lower_2star = 2.576 * sigma - mu
    upper_2star = 2.576 * sigma + mu
    lower_3star = 3.291 * sigma - mu
    upper_3star = 3.291 * sigma + mu
    lower_4star = 3.91 * sigma - mu
    upper_4star = 3.91 * sigma + mu

    time = timeStep / 2 + np.array(range(int(T / timeStep))) * timeStep
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(
        time[1:],
        model_fit.predict(start=1, dynamic=False) + mu,
        label="Model prediction",
    )
    y = model_fit.predict(start=1, dynamic=False)

    ax.fill_between(
        time[1:],
        y - lower_1star,
        y + upper_1star,
        facecolor=f"Magenta",
        edgecolor="none",
        alpha=0.2,
    )
    # ax.fill_between(
    #     time[1:],
    #     y - lower_2star,
    #     y + upper_2star,
    #     facecolor=f"Red",
    #     edgecolor="none",
    #     alpha=0.1,
    # )
    # ax.fill_between(
    #     time[1:],
    #     y - lower_3star,
    #     y + upper_3star,
    #     facecolor=f"Red",
    #     edgecolor="none",
    #     alpha=0.1,
    # )
    ax.fill_between(
        time[1:],
        y - lower_4star,
        y + upper_4star,
        facecolor=f"Magenta",
        edgecolor="none",
        alpha=0.2,
    )
    ax.set_ylim([0, 6])

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
    ax.plot(time, df.dd, label="Small wound")

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
    ax.plot(time, df.dd, label="Large wound")
    ax.legend()

    fig.savefig(
        f"results/wound confidence intervals",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")


# Make starima for unwounded

if True:
    allFig = False
    _df = []
    timeStep = 4
    rStep = 20
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

    if allFig:
        for r in range(dd.shape[1] - 1):
            df0 = df[f"dd_{r}"].dropna()
            if len(df0) > 34:
                # Original Series
                fig, ax = plt.subplots(3, 2, figsize=(10, 10))
                ax[0, 0].plot(df0.dropna())
                ax[0, 0].set_title("Original Series")
                plot_acf(df0.dropna(), ax=ax[0, 1])
                ax[0, 1].set_ylim([-1.1, 1.1])

                # 1st Differencing
                ax[1, 0].plot(df0.diff().dropna())
                ax[1, 0].set_title("1st Order Differencing")
                plot_acf(df0.diff().dropna(), ax=ax[1, 1])
                ax[1, 1].set_ylim([-1.1, 1.1])

                # 2nd Differencing
                ax[2, 0].plot(df0.diff().dropna().diff().dropna())
                ax[2, 0].set_title("2nd Order Differencing")
                plot_acf(df0.diff().dropna().diff().dropna(), ax=ax[2, 1])
                ax[2, 1].set_ylim([-1.1, 1.1])

                fig.savefig(
                    f"results/autocorrelation of division density r={r} {fileType}",
                    transparent=True,
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.close("all")
    d = 1

    # find p
    if allFig:
        for r in range(dd.shape[1] - 1):
            df0 = df[f"dd_{r}"].dropna()
            if len(df0) > 34:
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].plot(df0.diff().dropna())
                ax[0].set_title("1st Differencing")
                ax[1].set(ylim=(0, 5))
                plot_pacf(df0.diff().dropna(), ax=ax[1])
                ax[1].set_ylim([-1.1, 1.1])

                fig.savefig(
                    f"results/division density AR term r={r} {fileType}",
                    transparent=True,
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.close("all")
    p = 1

    # find q
    if allFig:
        for r in range(dd.shape[1] - 1):
            df0 = df[f"dd_{r}"].dropna()
            if len(df0) > 34:
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].plot(df0.diff().dropna())
                ax[0].set_title("1st Differencing")
                ax[1].set(ylim=(0, 1.2))
                plot_acf(df0.diff().dropna(), ax=ax[1])
                ax[1].set_ylim([-1.1, 1.1])

                fig.savefig(
                    f"results/division density MA term r={r} {fileType}",
                    transparent=True,
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.close("all")
    q = 1

    sigma = []
    ar1 = []
    ma1 = []
    dist = []
    for r in range(dd.shape[1] - 1):
        df0 = df[f"dd_{r}"].dropna()
        if len(df0) > 34:
            dist.append((r) * 10)
            model = statsm.tsa.arima.ARIMA(df0.diff().dropna(), order=(p, d, q))
            model_fit = model.fit()
            sigma.append(model_fit.params["sigma2"] ** 0.5)
            ar1.append(model_fit.params["ma.L1"])
            ma1.append(model_fit.params["ar.L1"])

    if allFig:
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

    if allFig:
        fig, ax = plt.subplots(1, 3, figsize=(10, 4))
        ax[0].hist(sigma)
        ax[0].set_title("Sigma")
        ax[1].hist(ar1)
        ax[1].set_title("ar1")
        ax[2].hist(ma1)
        ax[2].set_title("ma1")
        fig.savefig(
            f"results/model parameters {fileType}",
            transparent=True,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")

    sigma = np.mean(sigma)
    ar1 = np.mean(ar1)
    ma1 = np.mean(ma1)

    residuals = []
    for r in range(dd.shape[1] - 1):
        df0 = df[f"dd_{r}"].dropna()
        if len(df0) > 34:
            model = statsm.tsa.arima.ARIMA(df0.diff().dropna(), order=(p, d, q))
            with model.fix_params({"ar.L1": ar1, "ma.L1": ma1, "sigma2": sigma}):
                model_fit = model.fit()

            mu = np.mean(df0.diff().dropna())
            lower_1star = 1.96 * sigma - mu
            upper_1star = 1.96 * sigma + mu
            lower_2star = 2.576 * sigma - mu
            upper_2star = 2.576 * sigma + mu
            lower_3star = 3.291 * sigma - mu
            upper_3star = 3.291 * sigma + mu
            lower_4star = 3.91 * sigma - mu
            upper_4star = 3.91 * sigma + mu

            predict = []
            for t in range(dd.shape[0] - 2):
                pred = (
                    df0.iloc[t + 2] + model_fit.predict(start=1, dynamic=False).iloc[t]
                )
                predict.append(pred)
                dd_pred[t + 2, r] = pred

            for i in range(len(pd.DataFrame(model_fit.resid))):
                residuals.append(pd.DataFrame(model_fit.resid).iloc[i][0])

    if allFig:
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
    fig, ax = plt.subplots(2, 2, figsize=(12, 4))

    c = ax[0, 0].pcolor(
        t,
        r,
        dd,
        vmin=0,
        vmax=8,
    )
    fig.colorbar(c, ax=ax[0, 0])
    ax[0, 0].set(xlabel="Time (mins)", ylabel=r"$R (\mu m)$")
    ax[0, 0].title.set_text(f"Division density {fileType}")

    c = ax[0, 1].pcolor(
        t,
        r,
        dd_pred,
        vmin=0,
        vmax=8,
    )
    fig.colorbar(c, ax=ax[0, 1])
    ax[0, 1].set(xlabel="Time (mins)", ylabel=r"$R (\mu m)$")
    ax[0, 1].title.set_text(f"Division density predict {fileType}")

    Maxdd = np.max(
        [np.max(np.nan_to_num(dd_pred - dd)), -np.min(np.nan_to_num(dd_pred - dd))]
    )
    c = ax[1, 0].pcolor(
        t,
        r,
        dd_pred - dd,
        vmin=-Maxdd,
        vmax=Maxdd,
        cmap="RdBu_r",
    )
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set(xlabel="Time (mins)", ylabel=r"$R (\mu m)$")
    ax[1, 0].title.set_text(f"Division density difference {fileType}")

    dd_z = (dd_pred - dd) / sigma

    Maxdd = np.max([np.max(np.nan_to_num(dd_z)), -np.min(np.nan_to_num(dd_z))])
    c = ax[1, 1].pcolor(
        t,
        r,
        dd_z,
        vmin=-Maxdd,
        vmax=Maxdd,
        cmap="RdBu_r",
    )
    fig.colorbar(c, ax=ax[1, 1])
    ax[1, 1].set(xlabel="Time (mins)", ylabel=r"$R (\mu m)$")
    ax[1, 1].title.set_text(f"Division density difference z score {fileType}")

    # plt.subplot_tool()

    plt.subplots_adjust(
        left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.1, hspace=0.65
    )

    fig.savefig(
        f"results/Division density heatmap predict {fileType}",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")

    fileType = "WoundL"
    dd = divisionDensitySpace(fileType, timeStep, T, rStep, R)

    t, r = np.mgrid[0:T:timeStep, 0:R:rStep]
    fig, ax = plt.subplots(3, 2, figsize=(12, 6))

    c = ax[0, 0].pcolor(
        t,
        r,
        dd,
        vmin=0,
        vmax=8,
    )
    fig.colorbar(c, ax=ax[0, 0])
    ax[0, 0].set(xlabel="Time (mins)", ylabel=r"$R (\mu m)$")
    ax[0, 0].title.set_text(f"Division density {fileType}")

    c = ax[1, 0].pcolor(
        t,
        r,
        dd - dd_pred,
        vmin=-6,
        vmax=6,
        cmap="RdBu_r",
    )
    fig.colorbar(c, ax=ax[1, 0])
    ax[1, 0].set(xlabel="Time (mins)", ylabel=r"$R (\mu m)$")
    ax[1, 0].title.set_text(f"delta prediction and {fileType} data")

    z = (dd - dd_pred) / sigma
    sig = np.sign(np.nan_to_num(z)) * starSig(sp.stats.norm.sf(abs(np.nan_to_num(z))))
    Maxdd = np.max([np.max(sig), -np.min(sig)])

    c = ax[2, 0].pcolor(
        t,
        r,
        sig,
        vmin=-6,
        vmax=6,
        cmap="RdBu_r",
    )
    fig.colorbar(c, ax=ax[2, 0])
    ax[2, 0].set(xlabel="Time (mins)", ylabel=r"$R (\mu m)$")
    ax[2, 0].title.set_text(f"significance {fileType}")

    fileType = "WoundS"
    dd = divisionDensitySpace(fileType, timeStep, T, rStep, R)

    c = ax[0, 1].pcolor(
        t,
        r,
        dd,
        vmin=0,
        vmax=8,
    )
    fig.colorbar(c, ax=ax[0, 1])
    ax[0, 1].set(xlabel="Time (mins)", ylabel=r"$R (\mu m)$")
    ax[0, 1].title.set_text(f"Division density {fileType}")

    c = ax[1, 1].pcolor(
        t,
        r,
        dd - dd_pred,
        vmin=-6,
        vmax=6,
        cmap="RdBu_r",
    )
    fig.colorbar(c, ax=ax[1, 1])
    ax[1, 1].set(xlabel="Time (mins)", ylabel=r"$R (\mu m)$")
    ax[1, 1].title.set_text(f"delta prediction and {fileType} data")

    z = (dd - dd_pred) / sigma
    sig = np.sign(np.nan_to_num(z)) * starSig(sp.stats.norm.sf(abs(np.nan_to_num(z))))

    c = ax[2, 1].pcolor(
        t,
        r,
        sig,
        vmin=-6,
        vmax=6,
        cmap="RdBu_r",
    )
    fig.colorbar(c, ax=ax[2, 1])
    ax[2, 1].set(xlabel="Time (mins)", ylabel=r"$R (\mu m)$")
    ax[2, 1].title.set_text(f"significance {fileType}")

    # plt.subplot_tool()

    plt.subplots_adjust(
        left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.1, hspace=0.65
    )

    fig.savefig(
        f"results/delta unwounded prediction and wounded data",
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")
