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
from scipy.special import j0
from scipy.integrate import trapezoid
import shapely
import skimage as sm
import skimage.measure
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
from mpl_toolkits.mplot3d import Axes3D
import tifffile
from skimage.draw import circle_perimeter
import xml.etree.ElementTree as et
from scipy.optimize import leastsq, curve_fit
from datetime import datetime
import cellProperties as cell
import utils as util

plt.rcParams.update({"font.size": 8})

filenames, fileType = util.getFilesType()
T = 90
scale = 123.26 / 512
grid = 26
timeGrid = 51
T = np.linspace(0, (timeGrid - 1), timeGrid)
R = np.linspace(0, 2 * (grid - 1), grid)

def thetaAverage(cor, count):
    """Count-weighted mean over the theta axis of (T, R, theta) arrays.

    Returns (cor, count) shaped (T, R); bins with no pairs are NaN.
    """
    total = np.sum(count, axis=2)
    weighted = np.sum(np.where(count > 0, cor, 0) * count, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return weighted / total, total

def forIntegral(R, T, B, L, mQ, kMax=20, nK=10000):
    """Model correlation C(R, T) = mQ * int k exp(-(B + L k^2) T) J0(R k) / (B + L k^2) dk.

    Evaluated as a matrix product over k (midpoint rule), returning shape (n_R, n_T).
    Midpoints avoid k = 0, where B + L k^2 can be 0.
    """
    dk = kMax / nK
    k = (np.arange(nK) + 0.5) * dk
    decay = np.exp(-(B + L * k[:, None] ** 2) * T[None, :]) / (B + L * k[:, None] ** 2)
    kernel = k[None, :] * sc.jv(0, R[:, None] * k[None, :])
    return mQ * dk * (kernel @ decay)


B = []
L = []
mQ_1 = []
mQ_2 = []
# fit B and L from Hankel transform data C_q(k, t)
for filename in filenames:
    print(datetime.now().strftime("%H:%M:%S ") + filename)
    dfCor = pd.read_pickle(f"databases/correlations/dfCordQ{filename}.pkl")

    cor_q1, count_q1 = thetaAverage(
        dfCor["dQ1dQ1Correlation"][0], dfCor["dQ1dQ1Count"][0]
    )
    cor_q1, count_q1 = cor_q1.transpose(), count_q1.transpose()
    cor_q2, count_q2 = thetaAverage(
        dfCor["dQ2dQ2Correlation"][0], dfCor["dQ2dQ2Count"][0]
    )
    cor_q2, count_q2 = cor_q2.transpose(), count_q2.transpose()

    cor_q1 = cor_q1[:-1]
    cor_q2 = cor_q2[:-1]

    # plot C_q(R, t) Heatmap
    if True:  # Change to True to enable plotting
        T_grid, R_grid = np.meshgrid(T, R)  # shape (n_R, n_T), same as cor_q
        fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
        for a, cor, label in zip(
            ax, [cor_q1, cor_q2], [r"$C^{11}_{qq}(R,T)$", r"$C^{22}_{qq}(R,T)$"]
        ):
            maxCorr = np.nanmax(np.abs(cor))
            c = a.pcolor(
                T_grid,
                R_grid,
                cor,
                cmap="RdBu_r",
                vmin=-maxCorr,
                vmax=maxCorr,
                shading="auto",
            )
            fig.colorbar(c, ax=a)
            a.set_xlabel("Time apart $T$ (min)")
            a.set_ylabel(r"Distance apart $R$ $(\mu m)$")
            a.set_title(r"Experiment " + label)
        fig.tight_layout()
        fig.savefig(
            f"results/mathPostWoundPaper/Heatmap C_q(R,T) {filename}",
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )
        plt.close("all")

    # convert to Hankel transform space C_q(R, t) => C_q(k, t)

    # Choose wavenumbers (units inverse to R)
    k_values = np.linspace(0.1, 0.6, 100)

    # Fourier-space correlation
    C_kT_q1 = np.zeros((len(k_values), len(T)))
    C_kT_q2 = np.zeros((len(k_values), len(T)))

    for i, k in enumerate(k_values):
        integrand_q1 = (
            2 * np.pi
            * R[:, None]
            * j0(k * R[:, None])
            * cor_q1
        )
        C_kT_q1[i, :] = trapezoid(integrand_q1, x=R, axis=0)

        integrand_q2 = (
            2 * np.pi
            * R[:, None]
            * j0(k * R[:, None])
            * cor_q2
        )
        C_kT_q2[i, :] = trapezoid(integrand_q2, x=R, axis=0)

    # Normalize using T = 0 
    f_kT_q1 = C_kT_q1 / C_kT_q1[:, 0, None]
    f_kT_q2 = C_kT_q2 / C_kT_q2[:, 0, None]

    Gamma_k_q1 = []
    residuals_k_q1 = []
    Gamma_k_q2 = []
    residuals_k_q2 = []
    for i, k in enumerate(k_values):

        y = f_kT_q1[i, :]
        # Fit to exponential decay model
        def model(t, Gamma):
            return np.exp(-Gamma * t)

        def residuals(Gamma, t, y):
            return y - model(t, Gamma)

        # Initial guess for Gamma
        Gamma_initial = 0.01

        # Perform least squares fitting
        result = leastsq(residuals, Gamma_initial, args=(T, y))
        Gamma_k_q1.append(result[0][0])
        residuals_k_q1.append(residuals(result[0][0], T, y))

        y = f_kT_q2[i, :]
        result = leastsq(residuals, Gamma_initial, args=(T, y))
        Gamma_k_q2.append(result[0][0])
        residuals_k_q2.append(residuals(result[0][0], T, y))

        #plot f_kT vs T for each k
        if False:  # Change to True to enable plotting
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot(T, f_kT_q1[i, :], "o", label="dQ1dQ1")
            ax.plot(T, f_kT_q2[i, :], "s", label="dQ2dQ2")
            ax.set_xlabel(r"Time lag $T$ $(mins)$")
            ax.set_ylabel(r"$f(k, T)$")
            ax.set_title(r"$f(k, T)$ vs $T$ for $k =$" + str(round(k, 2)))
            ax.legend()
            fig.savefig(
                f"results/mathPostWoundPaper/f_kT vs T for i={i} {filename}",
                dpi=300,
                transparent=True,
                bbox_inches="tight",
            )
            plt.close("all")

    # filter out large residuals
    threshold = 1
    Gamma_k_q1_filtered = [Gamma_k_q1[i] for i in range(len(Gamma_k_q1)) if np.max(np.abs(residuals_k_q1[i])) < threshold]
    k_values_filtered_q1 = [k_values[i] for i in range(len(Gamma_k_q1)) if np.max(np.abs(residuals_k_q1[i])) < threshold]
    Gamma_k_q2_filtered = [Gamma_k_q2[i] for i in range(len(Gamma_k_q2)) if np.max(np.abs(residuals_k_q2[i])) < threshold]
    k_values_filtered_q2 = [k_values[i] for i in range(len(Gamma_k_q2)) if np.max(np.abs(residuals_k_q2[i])) < threshold]
    k_values_filtered_q1 = np.array(k_values_filtered_q1)
    k_values_filtered_q2 = np.array(k_values_filtered_q2)

    # plot Gamma_k vs k_values^2
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(k_values_filtered_q1**2, Gamma_k_q1_filtered, "o", label="dQ1dQ1")
    ax.plot(k_values_filtered_q2**2, Gamma_k_q2_filtered, "s", label="dQ2dQ2")
    ax.set_xlabel(r"$k^2$")
    ax.set_ylabel(r"$\Gamma(k)$")
    ax.set_title(r"$\Gamma(k)$ vs $k^2$")
    # fit Gamma(k) = B + L k^2 with B, L constrained to be positive
    Gamma_k = np.concatenate([np.array(Gamma_k_q1_filtered), np.array(Gamma_k_q2_filtered)])
    k_val = np.concatenate([np.array(k_values_filtered_q1**2), np.array(k_values_filtered_q2**2)])

    def linear_fit(k2, B, L):
        return B + L * k2

    (filename_B, filename_L), _ = curve_fit(
        linear_fit, k_val, Gamma_k, p0=[0.01, 0.01], bounds=(0, np.inf)
    )
    B.append(filename_B)
    L.append(filename_L)
    ax.plot(
        k_values**2,
        linear_fit(k_values**2, filename_B, filename_L),
        "r--",
        label="Linear fit",
    )
    ax.legend()
    # ax.set_ylim([-0.1, 0.3])
    fig.savefig(
        f"results/mathPostWoundPaper/Gamma_k vs k_values^2 {filename}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


    # fit the amplitude mQ for q1 and q2 separately, using this file's B and L
    # and only distances R > 10
    fitR = R > 10
    model = forIntegral(R[fitR], T, filename_B, filename_L, 1)  # shape (n_R, n_T)
    mQ_fit = []
    for cor in [cor_q1, cor_q2]:
        data = cor[fitR]
        valid = np.isfinite(data)
        # the model is linear in mQ, so least squares has a closed form
        mQ_fit.append(np.sum(model[valid] * data[valid]) / np.sum(model[valid] ** 2))
    mQ_1.append(mQ_fit[0])
    mQ_2.append(mQ_fit[1])
    print(f"mQ_1 = {mQ_fit[0]:.3e}  mQ_2 = {mQ_fit[1]:.3e}")

    # heatmaps of experiment, fitted model and residual over the fitted region (R > 10)
    T_grid, R_grid = np.meshgrid(T, R[fitR])
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    for row, (cor, mQ, label) in enumerate(
        zip([cor_q1, cor_q2], mQ_fit, ["11", "22"])
    ):
        data = cor[fitR]
        fitted = mQ * model
        residual = data - fitted
        # experiment and model share a colour scale so they can be compared directly
        maxCorr = np.nanmax(np.abs(data))
        maxRes = np.nanmax(np.abs(residual))
        panels = [
            (data, maxCorr, rf"Experiment $C^{{{label}}}_{{qq}}(R,T)$"),
            (fitted, maxCorr, rf"Model $C^{{{label}}}_{{qq}}(R,T)$"),
            (residual, maxRes, "Experiment - Model"),
        ]
        for a, (values, vmax, title) in zip(ax[row], panels):
            c = a.pcolor(
                T_grid,
                R_grid,
                values,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                shading="auto",
            )
            fig.colorbar(c, ax=a)
            a.set_xlabel("Time apart $T$ (min)")
            a.set_ylabel(r"Distance apart $R$ $(\mu m)$")
            a.set_title(title)
    fig.tight_layout()
    fig.savefig(
        f"results/mathPostWoundPaper/mQ fit {filename}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")



# plot B, L and mQ distributions
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].hist(B, bins=10, alpha=0.7)
ax[0].set_xlabel(r"$B$")
ax[0].set_ylabel("Count")
ax[0].set_title(r"Distribution of $B$")
ax[1].hist(L, bins=10, alpha=0.7)
ax[1].set_xlabel(r"$L$")
ax[1].set_ylabel("Count")
ax[1].set_title(r"Distribution of $L$")
ax[2].hist(mQ_1, bins=10, alpha=0.7, label=r"$mQ_1$")
ax[2].hist(mQ_2, bins=10, alpha=0.7, label=r"$mQ_2$")
ax[2].set_xlabel(r"$mQ$")
ax[2].set_ylabel("Count")
ax[2].set_title(r"Distribution of $mQ$")
ax[2].legend()
fig.tight_layout()
fig.savefig(
    f"results/mathPostWoundPaper/B and L distributions",
    dpi=300,
    transparent=True,
    bbox_inches="tight",
)   
plt.close("all")

mean_B = np.mean(B)
std_B = np.std(B)
mean_L = np.mean(L)
std_L = np.std(L)


# ---- full fit of B, L, mQ_1 and mQ_2 to C(R, T) for R > 10 ----
# B and L are shared by q1 and q2; each has its own amplitude mQ. The Gamma(k) fit
# above gives the starting point.
B_full = []
L_full = []
mQ_1_full = []
mQ_2_full = []
rmse_full = []
fitR = R > 10
for f, filename in enumerate(filenames):
    print(datetime.now().strftime("%H:%M:%S ") + filename + " full fit")
    dfCor = pd.read_pickle(f"databases/correlations/dfCordQ{filename}.pkl")
    cor_q1 = thetaAverage(dfCor["dQ1dQ1Correlation"][0], dfCor["dQ1dQ1Count"][0])[0]
    cor_q2 = thetaAverage(dfCor["dQ2dQ2Correlation"][0], dfCor["dQ2dQ2Count"][0])[0]
    cor_q1 = cor_q1.transpose()[:-1][fitR]  # (n_R, n_T) after dropping the empty bin
    cor_q2 = cor_q2.transpose()[:-1][fitR]

    # fit in units of the data scale so all four parameters are of order 1
    dataScale = np.nanmax(np.abs(np.concatenate([cor_q1.ravel(), cor_q2.ravel()])))
    data = np.concatenate([cor_q1.ravel(), cor_q2.ravel()]) / dataScale
    valid = np.isfinite(data)

    def fullModel(_, B_, L_, m1, m2):
        model = forIntegral(R[fitR], T, B_, L_, 1)
        return np.concatenate([m1 * model.ravel(), m2 * model.ravel()])[valid]

    # starting amplitudes from the closed form at the Gamma(k) values of B and L
    model0 = forIntegral(R[fitR], T, B[f], L[f], 1).ravel()
    m1_0 = np.nansum(model0 * cor_q1.ravel() / dataScale) / np.sum(model0**2)
    m2_0 = np.nansum(model0 * cor_q2.ravel() / dataScale) / np.sum(model0**2)

    popt, pcov = curve_fit(
        fullModel,
        None,
        data[valid],
        p0=[B[f], L[f], max(m1_0, 1e-3), max(m2_0, 1e-3)],
        bounds=([0, 0, 0, 0], np.inf),
        max_nfev=2000,
    )
    B_f, L_f = popt[0], popt[1]
    m1_f, m2_f = popt[2] * dataScale, popt[3] * dataScale
    B_full.append(B_f)
    L_full.append(L_f)
    mQ_1_full.append(m1_f)
    mQ_2_full.append(m2_f)
    rmse_full.append(np.sqrt(np.mean((fullModel(None, *popt) - data[valid]) ** 2)))
    print(
        f"B = {B_f:.4g} (Gamma fit {B[f]:.4g})  L = {L_f:.4g} (Gamma fit {L[f]:.4g})  "
        f"mQ_1 = {m1_f:.3e}  mQ_2 = {m2_f:.3e}"
    )

    # heatmaps of experiment, fitted model and residual (R > 10)
    model = forIntegral(R[fitR], T, B_f, L_f, 1)
    T_grid, R_grid = np.meshgrid(T, R[fitR])
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    for row, (cor, mQ, label) in enumerate(
        zip([cor_q1, cor_q2], [m1_f, m2_f], ["11", "22"])
    ):
        fitted = mQ * model
        residual = cor - fitted
        maxCorr = np.nanmax(np.abs(cor))
        maxRes = np.nanmax(np.abs(residual))
        panels = [
            (cor, maxCorr, rf"Experiment $C^{{{label}}}_{{qq}}(R,T)$"),
            (fitted, maxCorr, rf"Model $C^{{{label}}}_{{qq}}(R,T)$"),
            (residual, maxRes, "Experiment - Model"),
        ]
        for a, (values, vmax, title) in zip(ax[row], panels):
            c = a.pcolor(
                T_grid,
                R_grid,
                values,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                shading="auto",
            )
            fig.colorbar(c, ax=a)
            a.set_xlabel("Time apart $T$ (min)")
            a.set_ylabel(r"Distance apart $R$ $(\mu m)$")
            a.set_title(title)
    fig.suptitle(rf"Full fit: $B$={B_f:.3g}, $L$={L_f:.3g}", y=1.0)
    fig.tight_layout()
    fig.savefig(
        f"results/mathPostWoundPaper/Full fit {filename}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# compare the full-fit B and L with those from the Gamma(k) fit
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(B, B_full, "o")
ax[0].set_xlabel(r"$B$ from $\Gamma(k)$ fit")
ax[0].set_ylabel(r"$B$ from full fit")
ax[1].plot(L, L_full, "o")
ax[1].set_xlabel(r"$L$ from $\Gamma(k)$ fit")
ax[1].set_ylabel(r"$L$ from full fit")
for a, (x, y) in zip(ax, [(B, B_full), (L, L_full)]):
    lim = [0, max(np.max(x), np.max(y))]
    a.plot(lim, lim, "k--", linewidth=0.8)
fig.tight_layout()
fig.savefig(
    "results/mathPostWoundPaper/B and L full fit vs Gamma fit",
    dpi=300,
    transparent=True,
    bbox_inches="tight",
)
plt.close("all")

print(f"Full fit B = {np.mean(B_full):.4g} +- {np.std(B_full):.4g}")
print(f"Full fit L = {np.mean(L_full):.4g} +- {np.std(L_full):.4g}")
print(f"Full fit mQ_1 = {np.mean(mQ_1_full):.3e} +- {np.std(mQ_1_full):.3e}")
print(f"Full fit mQ_2 = {np.mean(mQ_2_full):.3e} +- {np.std(mQ_2_full):.3e}")




