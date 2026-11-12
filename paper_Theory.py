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
import warnings
import scipy as sp
import scipy.special as sc    
import seaborn as sns
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

plt.rcParams.update({"font.size": 10.4})

T = 90
scale = 123.26 / 512


def healthyName(name):
    """Use healthy terminology for presentation while preserving source IDs."""
    return (name.replace("Unwounded", "Healthy").replace("Unwound", "Healthy")
            .replace("unwounded", "healthy").replace("unwound", "healthy"))


def enlargeHeatmapText(fig):
    fig.canvas.draw()
    for text in fig.findobj(match=matplotlib.text.Text):
        text.set_fontsize(text.get_fontsize() * 1.125)


def spaceTimeDensityCorrelation(density, observed, grid=7, windowSize=5):
    """Correlate centered density maps over forward non-overlapping windows."""
    nFrames, xGrid, yGrid = density.shape
    windowStarts = np.arange(0, nFrames - windowSize + 1, windowSize)
    timeGrid = len(windowStarts)
    windowDensity = np.zeros((timeGrid, xGrid, yGrid))
    windowObserved = np.zeros((timeGrid, xGrid, yGrid), dtype=bool)

    for i, start in enumerate(windowStarts):
        frameObserved = observed[start : start + windowSize]
        validFrames = frameObserved.sum(axis=0)
        windowObserved[i] = validFrames > 0
        windowDensity[i] = np.divide(
            np.where(frameObserved, density[start : start + windowSize], 0).sum(axis=0),
            validFrames,
            out=np.zeros((xGrid, yGrid)),
            where=validFrames > 0,
        )

    shape = (timeGrid, grid, 17)
    count = np.zeros(shape)
    total = np.zeros(shape)
    totalSq = np.zeros(shape)

    for dx in range(-(grid - 1), grid):
        for dy in range(-(grid - 1), grid):
            dR = int(np.sqrt(dx**2 + dy**2))
            if dR >= grid:
                continue

            # Preserve the legacy angle convention: anchor minus partner.
            dtheta = np.arctan2(-dy, -dx)
            if dtheta < 0:
                dtheta += 2 * np.pi
            iTheta = int(8 * dtheta / np.pi)

            x0, x1 = max(0, -dx), min(xGrid, xGrid - dx)
            y0, y1 = max(0, -dy), min(yGrid, yGrid - dy)
            if x1 <= x0 or y1 <= y0:
                continue
            anchorDensity = windowDensity[:, x0:x1, y0:y1]
            partnerDensity = windowDensity[
                :, x0 + dx : x1 + dx, y0 + dy : y1 + dy
            ]
            anchorObserved = windowObserved[:, x0:x1, y0:y1]
            partnerObserved = windowObserved[
                :, x0 + dx : x1 + dx, y0 + dy : y1 + dy
            ]

            for deltaT in range(timeGrid):
                nStarts = timeGrid - deltaT
                valid = anchorObserved[:nStarts] & partnerObserved[deltaT:]
                values = (anchorDensity[:nStarts] * partnerDensity[deltaT:])[valid]
                count[deltaT, dR, iTheta] += len(values)
                total[deltaT, dR, iTheta] += np.sum(values)
                totalSq[deltaT, dR, iTheta] += np.sum(values**2)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        variance = np.maximum(totalSq / count - mean**2, 0)
    return mean, np.sqrt(variance), count

def spaceTimeCorrelation(
    x,
    y,
    t,
    props,
    anchors,
    timeGrid=51,
    grid=27,
    nTheta=17,
    rStep=2,
    tStep=1,
    maxPairs=2_000_000,
):
    """
    Binned forward-in-time pair statistics between anchor cells and all other cells.

    x, y, t : arrays of cell positions and integer frames
    props   : {name: (a, b)} arrays; the pair value is a[anchor] * b[other]
    anchors : indices (into x, y, t) of the cells used as the first cell of a pair
    rStep   : distance bin width (same units as x, y)
    tStep   : time bin width in frames

    Pairs are binned by dT = t_other - t_anchor (0 <= dT < timeGrid * tStep),
    distance (width rStep, dR < rStep * (grid - 1)) and angle (width pi / 8). Only
    count, sum and sum of squares are accumulated, so memory does not grow with
    the number of pairs.
    Returns {name: (mean, std, count)}, each shaped (timeGrid, grid, nTheta).
    """
    rMax = rStep * (grid - 1)
    shape = (timeGrid, grid, nTheta)
    nBins = timeGrid * grid * nTheta

    # Sort cells by frame so the cells in any time window are one contiguous slice
    order = np.argsort(t, kind="stable")
    xs, ys, ts = x[order], y[order], t[order]
    props = {k: (a[order], b[order]) for k, (a, b) in props.items()}
    # position[i] = where original row i ended up after sorting
    position = np.empty(len(t), dtype=int)
    position[order] = np.arange(len(t))

    # Translate anchors to sorted positions and group them by frame
    anchors = position[np.asarray(anchors, dtype=int)]
    anchors = anchors[np.argsort(ts[anchors], kind="stable")]
    anchorTimes, starts = np.unique(ts[anchors], return_index=True)
    ends = np.append(starts[1:], len(anchors))

    # Running totals per flattened (dT, dR, dtheta) bin; individual pairs are not stored
    count = np.zeros(nBins)
    total = {k: np.zeros(nBins) for k in props}
    totalSq = {k: np.zeros(nBins) for k in props}

    for tg, s0, s1 in zip(anchorTimes, starts, ends):
        # Candidate cells: frames tg up to (not including) tg + timeGrid * tStep,
        # so only same-time and forward-in-time pairs are considered
        lo = np.searchsorted(ts, tg, side="left")
        hi = np.searchsorted(ts, tg + timeGrid * tStep, side="left")
        # Limit anchors per batch so the (anchors x candidates) matrices stay small
        chunk = max(1, maxPairs // (hi - lo))
        for c0 in range(s0, s1, chunk):
            a = anchors[c0 : min(c0 + chunk, s1)]
            # Offsets from each anchor to every candidate, shape (len(a), hi - lo)
            dx = xs[lo:hi][None, :] - xs[a][:, None]
            dy = ys[lo:hi][None, :] - ys[a][:, None]
            dR = np.sqrt(dx**2 + dy**2)
            # Keep only pairs within the maximum distance; ai indexes the batch,
            # ci indexes the candidate window
            ai, ci = np.nonzero(dR < rMax)
            if len(ai) == 0:
                continue
            dx, dy, dR = dx[ai, ci], dy[ai, ci], dR[ai, ci]
            # Convert batch/window indices to positions in the full sorted arrays
            ci = ci + lo
            ai = a[ai]

            # Drop pairs of a cell with itself
            notSelf = ai != ci
            ai, ci = ai[notSelf], ci[notSelf]
            dx, dy, dR = dx[notSelf], dy[notSelf], dR[notSelf]
            if len(ai) == 0:
                continue

            # Angle of the other cell relative to the anchor, mapped to [0, 2 pi)
            dtheta = np.arctan2(dy, dx)
            dtheta = np.where(dtheta < 0, 2 * np.pi + dtheta, dtheta)
            # Bin indices: time lag, distance, angle (pi / 8 wide)
            iT = ((ts[ci] - tg) // tStep).astype(int)
            iR = (dR / rStep).astype(int)
            iTheta = (8 * dtheta / np.pi).astype(int)
            # Single index into the flattened (timeGrid, grid, nTheta) array
            flat = (iT * grid + iR) * nTheta + iTheta

            # bincount adds each pair's value into its bin in one vectorised step
            count += np.bincount(flat, minlength=nBins)
            for k, (pa, pb) in props.items():
                w = pa[ai] * pb[ci]
                total[k] += np.bincount(flat, weights=w, minlength=nBins)
                totalSq[k] += np.bincount(flat, weights=w**2, minlength=nBins)

    # mean = sum / n and population std from sum of squares; empty bins become NaN
    results = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in props:
            mean = total[k] / count
            # max(..., 0) guards against tiny negative values from rounding
            var = np.maximum(totalSq[k] / count - mean**2, 0)
            results[k] = (
                mean.reshape(shape),
                np.sqrt(var).reshape(shape),
                count.reshape(shape),
            )
    return results

# --------- density ----------

# space time cell density correlation
if False:
    filenames, fileType = util.getFilesType("Unwound18h")
    grid = 7
    gridSize = 10
    gridSizeT = 5
    nFrames = 90

    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    xMin, xMax = dfShape["X"].min(), dfShape["X"].max()
    yMin, yMax = dfShape["Y"].min(), dfShape["Y"].max()
    xGrid = int(1 + (xMax - xMin) // gridSize)
    yGrid = int(1 + (yMax - yMin) // gridSize)
    timeGrid = (nFrames - gridSizeT) // gridSizeT + 1
    T = np.arange(timeGrid) * gridSizeT
    R = np.arange(grid) * gridSize

    for filename in filenames:
        print(filename + datetime.now().strftime(" %H:%M:%S"))
        df = dfShape[dfShape["Filename"] == filename]
        df = df[df["T"].between(0, nFrames - 1)]
        if df.empty:
            raise ValueError(f"No shape data in frames 0-{nFrames - 1} for {filename}")

        frame = df["T"].to_numpy(dtype=int)
        xBin = np.floor((df["X"].to_numpy() - xMin) / gridSize).astype(int)
        yBin = np.floor((df["Y"].to_numpy() - yMin) / gridSize).astype(int)
        xBin = np.clip(xBin, 0, xGrid - 1)
        yBin = np.clip(yBin, 0, yGrid - 1)
        flat = (frame * xGrid + xBin) * yGrid + yBin
        nCells = np.bincount(flat, minlength=nFrames * xGrid * yGrid).reshape(
            nFrames, xGrid, yGrid
        )
        area = np.bincount(
            flat,
            weights=df["Area"].to_numpy(),
            minlength=nFrames * xGrid * yGrid,
        ).reshape(nFrames, xGrid, yGrid)
        observed = nCells > 0
        density = np.divide(nCells, area, out=np.zeros_like(area), where=observed)
        frameCount = observed.sum(axis=(1, 2))
        frameMean = np.divide(
            density.sum(axis=(1, 2)),
            frameCount,
            out=np.zeros(nFrames),
            where=frameCount > 0,
        )
        density[observed] -= np.broadcast_to(
            frameMean[:, None, None], density.shape
        )[observed]

        dRhodRhoCorrelation, dRhodRhoCorrelation_std, total = (
            spaceTimeDensityCorrelation(
                density,
                observed,
                grid=grid,
                windowSize=gridSizeT,
            )
        )
        pd.DataFrame(
            [
                {
                    "Filename": filename,
                    "dRhodRhoCorrelation": dRhodRhoCorrelation,
                    "dRhodRhoCorrelation_std": dRhodRhoCorrelation_std,
                    "Count": total,
                }
            ]
        ).to_pickle(f"databases/correlations/dfCorRho{filename}.pkl")

# --------- cell shape (q-tensor) ----------

# space time cell-cell q-tensor correlation full video
if False:
    filenames, fileType = util.getFilesType("Unwound18h")
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    grid = 27
    timeGrid = 51
    rStep = 2
    tStep = 1

    for k in range(len(filenames)):
        filename = filenames[k]
        path_to_file = f"databases/paperTheory/dfCordQ{filename}.pkl"
        if False == exists(path_to_file):
            print(datetime.now().strftime("%H:%M:%S ") + filename)
            dfShapeF = dfShape[dfShape["Filename"] == filename]
            Is = np.arange(len(dfShapeF))

            dq = np.stack(dfShapeF["dq"].to_list(), axis=0)
            dq1 = dq[:, 0, 0]
            dq2 = dq[:, 0, 1]

            results = spaceTimeCorrelation(
                dfShapeF["X"].to_numpy(),
                dfShapeF["Y"].to_numpy(),
                dfShapeF["T"].to_numpy(),
                {"dQ1dQ1": (dq1, dq1), "dQ2dQ2": (dq2, dq2)},
                Is,
                timeGrid=timeGrid,
                grid=grid,
                rStep=rStep,
                tStep=tStep,
            )

            _df = [
                {
                    "Filename": filename,
                    "dQ1dQ1Correlation": results["dQ1dQ1"][0],
                    "dQ2dQ2Correlation": results["dQ2dQ2"][0],
                    "dQ1dQ1Correlation_std": results["dQ1dQ1"][1],
                    "dQ2dQ2Correlation_std": results["dQ2dQ2"][1],
                    "dQ1dQ1Count": results["dQ1dQ1"][2],
                    "dQ2dQ2Count": results["dQ2dQ2"][2],
                }
            ]
            dfCorrelation = pd.DataFrame(_df)
            dfCorrelation.to_pickle(f"databases/paperTheory/dfCordQ{filename}.pkl")

# space time cell-cell q-tensor correlation change over time 0-1hr, 1-2hr, 2-3hr
if False:
    filenames, fileType = util.getFilesType("Unwound18h")
    timepoints = [[0,29],[30,59],[60,89]]
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    grid = 27
    timeGrid = 26
    rStep = 2
    tStep = 1

    for filename in filenames:
        dfShapeF = dfShape[dfShape["Filename"] == filename]
        for startTime, endTime in timepoints:
            path_to_file = (
                f"databases/paperTheory/dfCordQ{filename}_{startTime}-{endTime}.pkl"
            )
            if not exists(path_to_file):
                print(
                    datetime.now().strftime("%H:%M:%S ")
                    + f"{filename} frames {startTime}-{endTime}"
                )
                dfShapeWindow = dfShapeF[
                    (dfShapeF["T"] >= startTime) & (dfShapeF["T"] <= endTime)
                ]
                Is = np.arange(len(dfShapeWindow))

                dq = np.stack(dfShapeWindow["dq"].to_list(), axis=0)
                dq1 = dq[:, 0, 0]
                dq2 = dq[:, 0, 1]

                results = spaceTimeCorrelation(
                    dfShapeWindow["X"].to_numpy(),
                    dfShapeWindow["Y"].to_numpy(),
                    dfShapeWindow["T"].to_numpy(),
                    {"dQ1dQ1": (dq1, dq1), "dQ2dQ2": (dq2, dq2)},
                    Is,
                    timeGrid=timeGrid,
                    grid=grid,
                    rStep=rStep,
                    tStep=tStep,
                )

                _df = [
                    {
                        "Filename": filename,
                        "StartFrame": startTime,
                        "EndFrame": endTime,
                        "dQ1dQ1Correlation": results["dQ1dQ1"][0],
                        "dQ2dQ2Correlation": results["dQ2dQ2"][0],
                        "dQ1dQ1Correlation_std": results["dQ1dQ1"][1],
                        "dQ2dQ2Correlation_std": results["dQ2dQ2"][1],
                        "dQ1dQ1Count": results["dQ1dQ1"][2],
                        "dQ2dQ2Count": results["dQ2dQ2"][2],
                    }
                ]
                pd.DataFrame(_df).to_pickle(path_to_file)

# Shape correlations for 1.5–2.5 hours, with wound-distance groups separated.
if False:

    figure6StartTime, figure6EndTime = 90, 150
    figure6RadialStep, figure6Grid = 2, 21
    figure6WoundDistance = 30
    figure6Conditions = [
        ("Wild type", "Close", "WoundL18h", "#8b005b"),
        ("Wild type", "Far", "WoundL18h", "#cf369a"),
        ("Wild type", "Unwounded", "Unwound18h", "#f0a0d4"),
        ("JNK", "Close", "WoundLJNK", "#006b30"),
        ("JNK", "Far", "WoundLJNK", "#36a65c"),
        ("JNK", "Unwounded", "UnwoundJNK", "#a0d6aa"),
    ]

    def figure6CorrelationPath(filename, region):
        return (
            f"databases/correlationsWound/dfCorFigure6{region}{filename}"
            f"_minutes{figure6StartTime}-{figure6EndTime}"
            f"_rmax{figure6RadialStep * (figure6Grid - 1)}.pkl"
        )
    figure6ShapeData = {}
    for _, region, shapeFileType, _ in figure6Conditions:
        shapeFilenames, _ = util.getFilesType(shapeFileType)
        if not shapeFilenames:
            raise ValueError(f"No recordings found for {shapeFileType}")
        for filename in shapeFilenames:
            path = figure6CorrelationPath(filename, region)
            if exists(path):
                continue
            if shapeFileType not in figure6ShapeData:
                figure6ShapeData[shapeFileType] = pd.read_pickle(
                    f"databases/dfShapeWound{shapeFileType}.pkl"
                )
            df = figure6ShapeData[shapeFileType]
            df = df[
                (df["Filename"] == filename)
                & df["T"].between(figure6StartTime, figure6EndTime)
            ]
            if region == "Close":
                df = df[df["R"] < figure6WoundDistance]
            elif region == "Far":
                df = df[df["R"] >= figure6WoundDistance]
            if df.empty:
                raise ValueError(f"{filename} {region}: no cells in selected time window")
            print(datetime.now().strftime("%H:%M:%S ") + f"{filename} {region} correlations")
            dq = np.stack(df["dq"].to_list(), axis=0)
            results = spaceTimeCorrelation(
                df["X"].to_numpy(), df["Y"].to_numpy(), df["T"].to_numpy(),
                {"dQ1dQ1": (dq[:, 0, 0], dq[:, 0, 0]),
                 "dQ2dQ2": (dq[:, 0, 1], dq[:, 0, 1])},
                np.arange(len(df)), timeGrid=figure6EndTime - figure6StartTime + 1,
                grid=figure6Grid, rStep=figure6RadialStep, tStep=1,
            )
            pd.DataFrame([{
                "Filename": filename,
                "Region": region,
                "StartTime": figure6StartTime, "EndTime": figure6EndTime,
                "Grid": figure6Grid, "RadialStep": figure6RadialStep,
                "dQ1dQ1Correlation": results["dQ1dQ1"][0],
                "dQ2dQ2Correlation": results["dQ2dQ2"][0],
                "dQ1dQ1Correlation_std": results["dQ1dQ1"][1],
                "dQ2dQ2Correlation_std": results["dQ2dQ2"][1],
                "dQ1dQ1Count": results["dQ1dQ1"][2],
                "dQ2dQ2Count": results["dQ2dQ2"][2],
            }]).to_pickle(path)
    del figure6ShapeData

# ------------------------- Figure 3 -------------------------

# Mean density over time
if True:
    T = np.linspace(0, 89, 90)
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    filenames, fileType = util.getFilesType("Unwound18h")
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    rho = np.zeros([len(filenames), len(T)])
    for i in range(len(filenames)):
        filename = filenames[i]
        df = dfShape[dfShape["Filename"] == filename]
        for j in range(len(T)):
            t = T[j]
            rho[i, j] = 1 / np.mean(df["Area"][df["T"] == t])

    time = 2 * T

    std = np.std(rho, axis=0)
    rho = np.mean(rho, axis=0)
    colour, mark = util.getColorLineMarker(fileType, "18h")
    fileTitle = healthyName(util.getFileTitle(fileType))
    ax.plot(time, rho, color=colour)
    ax.fill_between(time, rho - std, rho + std, alpha=0.15, color=colour)

    ax.set_ylim([0.055, 0.09])
    ax.set(xlabel="Time (mins)", ylabel=r"$\bar{\rho}$ $(\mu m^{-2})$")
    ax.set_title(r"$\bar{\rho}$" + " with time \n healthy")
    fig.savefig(
        f"results/paperTheory/mean rho {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

    rho_star = np.mean(rho)

    print(f"Mean density over time for {fileType}: {rho_star:.4f} um^-2")

# Mean Q1 tensor over time
if True:
    T = np.linspace(0, 89, 90)
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    filenames, fileType = util.getFilesType("Unwound18h")
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    q1 = np.zeros([len(filenames), len(T)])
    for i in range(len(filenames)):
        filename = filenames[i]
        df = dfShape[dfShape["Filename"] == filename]
        for j in range(len(T)):
            t = T[j]
            q1[i, j] = np.mean(df["q"][df["T"] == t])[0, 0]

    time = 2 * T

    std = np.std(q1, axis=0)
    Q1 = np.mean(q1, axis=0)
    if not np.all(np.isfinite(Q1)):
        raise ValueError(f"{fileType}: mean Q1 contains non-finite values")
    m, Q_bar_0 = np.polyfit(time, Q1, 1)
    b = m/(2*Q_bar_0*rho_star)
    print(f"Mean Q1 linear fit for {fileType}: Initial average Q^(1) = {Q_bar_0:.6g}, q-tensor bulk (b) = {b:.6g}")

    colour, mark = util.getColorLineMarker(fileType, "18h")
    fileTitle = healthyName(util.getFileTitle(fileType))
    ax.plot(time, Q1, label=fileTitle, color=colour)
    ax.fill_between(time, Q1 - std, Q1 + std, alpha=0.15, color=colour)
    ax.plot(time, m * time + Q_bar_0, "--", color="black", label="Linear fit")

    ax.set_ylim([0.002, 0.035])
    ax.set(xlabel="Time (mins)", ylabel=r"$\bar{Q}^{(1)}$")
    ax.set_title(r"$\bar{Q}^{(1)}$" + " with time \n healthy")
    ax.legend()
    fig.savefig(
        f"results/paperTheory/mean Q1 {fileType}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

# ------------------------- Figure 4 -------------------------

def densityAnnealedModel(coordinates, m_rho, D):
    R, T = coordinates
    return m_rho / (32 * np.pi**3 * T * D**2) * np.exp(-(R**2) / (4 * D * T))

def densityQuenchedModel(R, m_s, decayLength):
    return m_s * np.exp(-R / decayLength)

def plotParameterBoxplots(parameters, fileType, outputName):
    fig, axes = plt.subplots(1, len(parameters), figsize=(2 * len(parameters), 3))
    for ax, (values, ylabel) in zip(np.atleast_1d(axes), parameters):
        ax.boxplot(values, positions=[0], widths=0.5, showfliers=False)
        sns.swarmplot(y=values, ax=ax, color="tab:blue", size=6, zorder=3)
        ax.set_ylabel(ylabel)
        ax.set_xticks([])
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(f"results/paperTheory/{outputName} {fileType}",
                dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)

# Fit annealed and quenched density correlations independently for each video.
if True:
    filenames, fileType = util.getFilesType("Unwound18h")
    gridSize = 10
    gridSizeT = 5
    minutesPerFrame = 2
    lateStart = 10
    expectedShape = (18, 7, 17)
    sampleFits = []
    sampleAnnealed = []
    sampleQuenched = []
    sampleTotal = []

    for filename in filenames:
        path = f"databases/correlations/dfCorRho{filename}.pkl"
        dfCor = pd.read_pickle(path)
        if len(dfCor) != 1 or dfCor["Filename"].iloc[0] != filename:
            raise ValueError(f"{path} must contain exactly one row for {filename}")
        correlation = np.asarray(dfCor["dRhodRhoCorrelation"].iloc[0])
        count = np.asarray(dfCor["Count"].iloc[0])
        if correlation.shape != expectedShape or count.shape != expectedShape:
            raise ValueError(f"{path}: expected correlation and count shape {expectedShape}")
        if np.any(~np.isfinite(count)) or np.any(count < 0):
            raise ValueError(f"{path}: invalid pair counts")
        if np.any((count > 0) & ~np.isfinite(correlation)):
            raise ValueError(f"{path}: non-finite correlation in a populated bin")

        thetaCount = np.sum(count, axis=2)
        correlationRT = np.divide(
            np.sum(np.where(count > 0, correlation, 0) * count, axis=2),
            thetaCount,
            out=np.full(thetaCount.shape, np.nan),
            where=thetaCount > 0,
        )
        lateCorrelation = correlationRT[lateStart:-1]
        lateCount = np.sum(np.isfinite(lateCorrelation), axis=0)
        quenched = np.divide(
            np.nansum(lateCorrelation, axis=0),
            lateCount,
            out=np.full(correlationRT.shape[1], np.nan),
            where=lateCount > 0,
        )
        annealed = correlationRT - quenched[None, :]
        radii = np.arange(expectedShape[1]) * gridSize
        times = np.arange(expectedShape[0]) * gridSizeT * minutesPerFrame
        rGrid, tGrid = np.meshgrid(radii[1:], times[1:])
        annealedData = annealed[1:, 1:]
        validAnnealed = np.isfinite(annealedData)
        validQuenched = np.isfinite(quenched)
        if np.count_nonzero(validAnnealed) < 3 or np.count_nonzero(validQuenched) < 3:
            raise ValueError(f"{filename}: insufficient finite density correlations to fit")
        annealedScale = np.max(np.abs(annealedData[validAnnealed]))
        quenchedScale = np.max(np.abs(quenched[validQuenched]))
        if annealedScale == 0 or quenchedScale == 0:
            raise ValueError(f"{filename}: density correlation has zero fit scale")

        try:
            annealedFit, _ = curve_fit(
                densityAnnealedModel,
                (rGrid[validAnnealed], tGrid[validAnnealed]),
                annealedData[validAnnealed] / annealedScale,
                p0=(0.003 / annealedScale, 4),
                bounds=([-np.inf, 1e-8], [np.inf, np.inf]),
                x_scale="jac",
                maxfev=20000,
            )
            quenchedFit, _ = curve_fit(
                densityQuenchedModel,
                radii[validQuenched],
                quenched[validQuenched] / quenchedScale,
                p0=(0.0003 / quenchedScale, 20),
                bounds=([-np.inf, 1e-8], [np.inf, np.inf]),
                x_scale="jac",
                maxfev=20000,
            )
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(f"{filename}: density model fit failed") from error
        m_rho, D = annealedFit[0] * annealedScale, annealedFit[1]
        m_s, decayLength = quenchedFit[0] * quenchedScale, quenchedFit[1]
        if decayLength < gridSize:
            warnings.warn(
                f"{filename}: fitted decay length {decayLength:.4g} um is below "
                f"the {gridSize} um radial bin size and may not be resolved",
                RuntimeWarning,
            )
        sampleFits.append(
            {"Filename": filename, "m_rho": m_rho, "D": D,
             "m_s": m_s, "lambda": decayLength}
        )
        sampleAnnealed.append(annealed)
        sampleQuenched.append(quenched)
        sampleTotal.append(correlationRT)
        print(
            f"{filename}: m_rho={m_rho:.4g}, D={D:.4g}, "
            f"m_s={m_s:.4g}, lambda={decayLength:.4g}"
        )

    if not sampleFits:
        raise ValueError(f"No recordings found for {fileType}")
    dfDensityFits = pd.DataFrame(sampleFits)
    dfDensityFits.to_csv(f"results/paperTheory/density fits {fileType}.csv", index=False)
    print(f"Density model parameters (n={len(sampleFits)}, population SD):")
    for parameter in ("m_rho", "D", "m_s", "lambda"):
        values = dfDensityFits[parameter].to_numpy()
        print(f"{parameter} = {np.mean(values):.4g} +- {np.std(values):.4g}")

    rSmooth = np.linspace(0, radii[-1], 61)
    for fit, annealed, quenched, total in zip(
        sampleFits, sampleAnnealed, sampleQuenched, sampleTotal
    ):
        filename = fit["Filename"]
        norm = total[0, 0]
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(
                f"{filename}: cannot normalize density plots; "
                "zero-lag correlation is not positive"
            )
        annealedData = annealed[1:, 1:]
        annealedModel = densityAnnealedModel((rGrid, tGrid), fit["m_rho"], fit["D"])
        maxCorr = np.nanmax(np.abs(np.stack([annealedData, annealedModel]))) / norm
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        for panel, values, title in zip(
            ax,
            [annealedData, annealedModel],
            [r"Experiment $C^{nn}_{\rho\rho}(R,T)$",
             r"Fitted model $C^{nn}_{\rho\rho}(R,T)$"],
        ):
            c = panel.pcolor(tGrid, rGrid, values / norm, cmap="RdBu_r",
                             vmin=-maxCorr, vmax=maxCorr, shading="auto")
            fig.colorbar(c, ax=panel).formatter.set_powerlimits((0, 0))
            panel.set_xlabel("Time apart $T$ (min)")
            panel.set_ylabel(r"Distance apart $R$ $(\mu m)$")
            panel.set_title(title, pad=25)
        enlargeHeatmapText(fig)
        fig.tight_layout()
        fig.savefig(f"results/paperTheory/Correlation dRho_n {filename}",
                    dpi=300, transparent=True, bbox_inches="tight")
        plt.close(fig)

        quenchedModel = densityQuenchedModel(rSmooth, fit["m_s"], fit["lambda"])
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.plot(radii, quenched / norm, label="Data")
        ax.plot(rSmooth, quenchedModel / norm, label="Fitted model")
        ax.set_xlabel(r"Distance apart $R$ $(\mu m)$")
        ax.set_ylabel(r"$\delta \rho_s$ Correlation")
        ax.set_title(r"$C^{ss}_{\rho\rho}(R)$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"results/paperTheory/Correlation rho_s {filename}",
                    dpi=300, transparent=True, bbox_inches="tight")
        plt.close(fig)

    annealedPrefactor = (
        dfDensityFits["m_rho"] / (32 * np.pi**3 * dfDensityFits["D"]**2)
    ).to_numpy()
    plotParameterBoxplots(
        [
            (dfDensityFits["m_rho"].to_numpy(), r"$m_\rho$"),
            (dfDensityFits["D"].to_numpy(), r"$D$ $(\mu m^2/\mathrm{min})$"),
            (annealedPrefactor, r"$m_\rho/(32\pi^3 D^2)$"),
            (dfDensityFits["m_s"].to_numpy(), r"$m_s$"),
            (dfDensityFits["lambda"].to_numpy(), r"$\lambda$ $(\mu m)$"),
        ],
        fileType,
        "density parameter boxplots",
    )

# ------------------------- Figure 5 -------------------------

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

# ---- full fit of B, L, mQ_1 and mQ_2 to C_Q(R, T) for R > 10 ----
# B and L are shared by q1 and q2; each has its own amplitude mQ. The Gamma(k) fit
# above gives the starting point.
if True:
    grid = 26
    timeGrid = 51
    T = np.linspace(0, (timeGrid - 1), timeGrid)
    R = np.linspace(0, 2 * (grid - 1), grid)
    B_full = []
    L_full = []
    mQ_1_full = []
    mQ_2_full = []
    rmse_full = []
    fitR = R > 10
    filenames, fileType = util.getFilesType("Unwound18h")
    for f, filename in enumerate(filenames):
        print(datetime.now().strftime("%H:%M:%S ") + filename + " full fit")
        dfCor = pd.read_pickle(f"databases/paperTheory/dfCordQ{filename}.pkl")
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
        model0 = forIntegral(R[fitR], T, 0.01, 2, 1).ravel()
        m1_0 = np.nansum(model0 * cor_q1.ravel() / dataScale) / np.sum(model0**2)
        m2_0 = np.nansum(model0 * cor_q2.ravel() / dataScale) / np.sum(model0**2)

        popt, pcov = curve_fit(
            fullModel,
            None,
            data[valid],
            p0=[0.01, 2, max(m1_0, 1e-3), max(m2_0, 1e-3)],
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
            f"B = {B_f:.4g},  L = {L_f:.4g},  "
            f"mQ_1 = {m1_f:.3e},  mQ_2 = {m2_f:.3e}"
        )

        # heatmaps of experiment, fitted model and residual (R > 10)
        model = forIntegral(R[fitR], T, B_f, L_f, 1)
        T_grid, R_grid = np.meshgrid(T, R[fitR])
        fitted_q1, fitted_q2 = m1_f * model, m2_f * model
        residual_q1, residual_q2 = cor_q1 - fitted_q1, cor_q2 - fitted_q2
        vmax = np.nanmax(
            np.abs(
                np.concatenate(
                    [
                        values.ravel()
                        for values in (
                            cor_q1,
                            fitted_q1,
                            residual_q1,
                            cor_q2,
                            fitted_q2,
                            residual_q2,
                        )
                    ]
                )
            )
        )
        fig, ax = plt.subplots(2, 3, figsize=(12, 6))
        for row, (cor, fitted, residual, label) in enumerate(
            zip(
                [cor_q1, cor_q2],
                [fitted_q1, fitted_q2],
                [residual_q1, residual_q2],
                ["11", "22"],
            )
        ):
            panels = [
                (cor, rf"Experiment $C^{{{label}}}_{{qq}}(R,T)$"),
                (fitted, rf"Model $C^{{{label}}}_{{qq}}(R,T)$"),
                (residual, "Experiment - Model"),
            ]
            for a, (values, title) in zip(ax[row], panels):
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
                a.set_title(title, pad=25)
        fig.suptitle(rf"Full fit: $B$={B_f:.3g}, $L$={L_f:.3g}", y=1.0)
        enlargeHeatmapText(fig)
        fig.tight_layout()
        fig.savefig(
            f"results/paperTheory/Full fit {filename}",
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )
        plt.close("all")

    print(f"Full fit B = {np.mean(B_full):.4g} +- {np.std(B_full):.4g}")
    print(f"Full fit L = {np.mean(L_full):.4g} +- {np.std(L_full):.4g}")
    print(f"Full fit mQ_1 = {np.mean(mQ_1_full):.3e} +- {np.std(mQ_1_full):.3e}")
    print(f"Full fit mQ_2 = {np.mean(mQ_2_full):.3e} +- {np.std(mQ_2_full):.3e}")
    plotParameterBoxplots(
        [
            (B_full, r"$B$"),
            (L_full, r"$L$"),
            (mQ_1_full, r"$m_{Q_1}$"),
            (mQ_2_full, r"$m_{Q_2}$"),
        ],
        fileType,
        "Q fit parameter boxplots",
    )

# ------------------------- Figure 6 -------------------------

if True:
    figure6StartTime, figure6EndTime = 90, 150
    figure6RadialStep, figure6Grid = 2, 21
    figure6WoundDistance = 30
    figure6Conditions = [
        ("Wild type", "Close", "WoundL18h", "#8b005b"),
        ("Wild type", "Far", "WoundL18h", "#cf369a"),
        ("Wild type", "Unwounded", "Unwound18h", "#f0a0d4"),
        ("JNK", "Close", "WoundLJNK", "#006b30"),
        ("JNK", "Far", "WoundLJNK", "#36a65c"),
        ("JNK", "Unwounded", "UnwoundJNK", "#a0d6aa"),
    ]    
    def figure6CorrelationPath(filename, region):
        return (
            f"databases/correlationsWound/dfCorFigure6{region}{filename}"
            f"_minutes{figure6StartTime}-{figure6EndTime}"
            f"_rmax{figure6RadialStep * (figure6Grid - 1)}.pkl"
        )
    
    startTime, endTime = figure6StartTime, figure6EndTime
    rStep, grid = figure6RadialStep, figure6Grid
    lagStep = 2
    times = np.arange(0, endTime - startTime + 1, lagStep)
    radii = np.arange(grid - 1) * rStep
    fitR = radii >= 10
    conditions = figure6Conditions
    figure6Fits = []
    figure6Correlations = {}

    for genotype, region, shapeFileType, colour in conditions:
        filenames, _ = util.getFilesType(shapeFileType)
        if not filenames:
            raise ValueError(f"No recordings found for {shapeFileType}")
        for filename in filenames:
            path = figure6CorrelationPath(filename, region)
            if not exists(path):
                raise FileNotFoundError(
                    f"Missing Figure 6 correlation {path}; enable the correlation block above"
                )
            dfCor = pd.read_pickle(path)
            if (len(dfCor) != 1 or dfCor["Filename"].iloc[0] != filename
                    or dfCor["Region"].iloc[0] != region
                    or dfCor["StartTime"].iloc[0] != startTime
                    or dfCor["EndTime"].iloc[0] != endTime
                    or dfCor["Grid"].iloc[0] != grid
                    or dfCor["RadialStep"].iloc[0] != rStep):
                raise ValueError(f"Incorrect Figure 6 correlation metadata: {path}")
            correlations = []
            for component in ("dQ1dQ1", "dQ2dQ2"):
                cor = np.asarray(dfCor[component + "Correlation"].iloc[0])
                count = np.asarray(dfCor[component + "Count"].iloc[0])
                if cor.shape != (61, grid, 17) or count.shape != cor.shape:
                    raise ValueError(f"{path}: unexpected {component} array shape")
                if (np.any(~np.isfinite(count)) or np.any(count < 0)
                        or np.any((count > 0) & ~np.isfinite(cor))):
                    raise ValueError(f"{path}: invalid {component} correlation or count")
                correlations.append(thetaAverage(cor, count)[0][::lagStep, :-1].T)
            cor_q1, cor_q2 = correlations
            dataScale = np.nanmax(np.abs(np.stack([cor_q1[fitR], cor_q2[fitR]])))
            if not np.isfinite(dataScale) or dataScale <= 0:
                raise ValueError(f"{filename} {region}: invalid correlation scale")
            data = np.concatenate([cor_q1[fitR].ravel(), cor_q2[fitR].ravel()]) / dataScale
            valid = np.isfinite(data)
            if valid.sum() < 5:
                raise ValueError(f"{filename} {region}: insufficient data for fit")

            def figure6Model(_, B, L, m1, m2):
                model = forIntegral(radii[fitR], times, B, L, 1)
                return np.concatenate([m1 * model.ravel(), m2 * model.ravel()])[valid]

            model0 = forIntegral(radii[fitR], times, 0.01, 2, 1).ravel()
            m1_0 = np.nansum(model0 * cor_q1[fitR].ravel() / dataScale) / np.sum(model0**2)
            m2_0 = np.nansum(model0 * cor_q2[fitR].ravel() / dataScale) / np.sum(model0**2)
            try:
                popt, _ = curve_fit(
                    figure6Model, None, data[valid],
                    p0=[0.01, 2, max(m1_0, 1e-3), max(m2_0, 1e-3)],
                    bounds=([0, 0, 0, 0], np.inf), max_nfev=2000,
                )
            except (RuntimeError, ValueError) as error:
                raise RuntimeError(f"{filename} {region}: Q correlation fit failed") from error
            B, L = popt[:2]
            m1, m2 = popt[2:] * dataScale
            if not np.all(np.isfinite([B, L, m1, m2])):
                raise ValueError(f"{filename} {region}: non-finite fit parameters")
            figure6Fits.append({
                "Genotype": genotype, "Region": region, "Filename": filename,
                "B": B, "L": L, "mQ_1": m1, "mQ_2": m2,
            })
            fullModel = forIntegral(radii, times, B, L, 1)
            figure6Correlations.setdefault((genotype, region), []).append(
                (cor_q1, cor_q2, m1 * fullModel, m2 * fullModel)
            )
            print(f"{filename} {region}: B={B:.4g}, L={L:.4g}, "
                  f"mQ_1={m1:.3e}, mQ_2={m2:.3e}")

            fitted_q1, fitted_q2 = m1 * fullModel[fitR], m2 * fullModel[fitR]
            residual_q1 = cor_q1[fitR] - fitted_q1
            residual_q2 = cor_q2[fitR] - fitted_q2
            vmax = np.nanmax(np.abs(np.stack([
                cor_q1[fitR], fitted_q1, residual_q1,
                cor_q2[fitR], fitted_q2, residual_q2,
            ])))
            tGrid, rGrid = np.meshgrid(times, radii[fitR])
            fig, axes = plt.subplots(2, 3, figsize=(12, 6))
            for row, (cor, model, residual, label) in enumerate((
                (cor_q1[fitR], fitted_q1, residual_q1, "11"),
                (cor_q2[fitR], fitted_q2, residual_q2, "22"),
            )):
                for ax, values, title in zip(
                    axes[row], (cor, model, residual),
                    (rf"Experiment $C^{{{label}}}_{{qq}}(R,T)$",
                     rf"Model $C^{{{label}}}_{{qq}}(R,T)$", "Experiment - Model"),
                ):
                    image = ax.pcolor(tGrid, rGrid, values, cmap="RdBu_r",
                                      vmin=-vmax, vmax=vmax, shading="auto")
                    fig.colorbar(image, ax=ax)
                    ax.set_xlabel("Time apart $T$ (min)")
                    ax.set_ylabel(r"Distance apart $R$ $(\mu m)$")
                    ax.set_title(title, pad=25)
            fig.suptitle(f"{healthyName(filename)} {healthyName(region)}: " + rf"$B$={B:.3g}, $L$={L:.3g}")
            enlargeHeatmapText(fig)
            fig.tight_layout()
            fig.savefig(f"results/paperTheory/Figure 6 fit {region} {filename}",
                        dpi=300, transparent=True, bbox_inches="tight")
            plt.close(fig)

    dfFigure6 = pd.DataFrame(figure6Fits)
    dfFigure6.to_csv("results/paperTheory/Figure 6 Q fit parameters.csv", index=False)
    figure6SliceFigures = []
    for component, label in enumerate(("11", "22")):
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        for genotype, region, _, colour in conditions:
            samples = figure6Correlations[(genotype, region)]
            raw = np.stack([sample[component] for sample in samples])
            model = np.stack([sample[component + 2] for sample in samples])
            counts = np.isfinite(raw).sum(axis=0)
            meanRaw = np.divide(np.nansum(raw, axis=0), counts,
                                out=np.full(counts.shape, np.nan), where=counts > 0)
            meanModel = np.mean(model, axis=0)
            row = 0 if genotype == "Wild type" else 1
            axes[row, 0].plot(radii[fitR], meanRaw[fitR, 0], "o",
                              color=colour, label=healthyName(region))
            axes[row, 0].plot(radii[fitR], meanModel[fitR, 0], "-", color=colour)
            axes[row, 1].plot(times, meanRaw[radii == 10][0], "o",
                              color=colour, label=healthyName(region))
            axes[row, 1].plot(times, meanModel[radii == 10][0], "-", color=colour)
        for row, genotype in enumerate(("Wild type", "JNK")):
            axes[row, 0].set_title(f"{genotype}: T = 0 min")
            axes[row, 1].set_title(rf"{genotype}: R = 10 $\mu$m")
            axes[row, 0].set_xlabel(r"Distance apart $R$ $(\mu m)$")
            axes[row, 0].set_xlim(radii[fitR][0], radii[fitR][-1])
            axes[row, 1].set_xlabel("Time apart $T$ (min)")
            axes[row, 1].set_xlim(times[0], times[-1])
            for ax in axes[row]:
                ax.set_ylabel(rf"$C^{{{label}}}_{{qq}}(R,T)$")
                ax.legend(title="Points: mean data; lines: mean fits", fontsize=9.1)
        figure6SliceFigures.append((fig, axes, component))

    sliceAxes = [ax for _, axes, _ in figure6SliceFigures for ax in axes.flat]
    sliceYLimits = (min(ax.get_ylim()[0] for ax in sliceAxes),
                    max(ax.get_ylim()[1] for ax in sliceAxes))
    for fig, axes, component in figure6SliceFigures:
        for ax in axes.flat:
            ax.set_ylim(sliceYLimits)
        fig.tight_layout()
        fig.savefig(f"results/paperTheory/Figure 6 Q{component + 1} correlation slices",
                    dpi=300, transparent=True, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    conditionNames = [f"{genotype}\n{healthyName(region)}" for genotype, region, _, _ in conditions]
    for ax, parameter, ylabel in zip(
        axes.flat, ("B", "L", "mQ_1", "mQ_2"),
        (r"$B$", r"$L$", r"$m_{Q_1}$", r"$m_{Q_2}$"),
    ):
        groups = [
            dfFigure6.loc[(dfFigure6["Genotype"] == genotype)
                          & (dfFigure6["Region"] == region), parameter].to_numpy()
            for genotype, region, _, _ in conditions
        ]
        values = np.concatenate(groups)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"{parameter}: expected finite, non-negative fit values")
        positive = values[values > 0]
        if positive.size == 0:
            raise ValueError(f"{parameter}: no positive values for logarithmic axis")
        if np.any(values == 0):
            linearThreshold = np.median(positive) / 1000
            ax.set_yscale("symlog", linthresh=linearThreshold)
            ax.set_title(f"Linear below {linearThreshold:.1g}")
        else:
            ax.set_yscale("log")
        ax.boxplot(groups, positions=np.arange(6), widths=0.5, showfliers=False)
        sns.swarmplot(data=groups, palette=[entry[3] for entry in conditions],
                      size=6, ax=ax, zorder=3)
        ax.set_xticks(np.arange(6), conditionNames, rotation=45, ha="right", fontsize=7.8)
        ax.set_ylabel(ylabel)
        if np.any(values == 0):
            ax.set_ylim(bottom=-linearThreshold / 4)
    fig.tight_layout()
    fig.savefig("results/paperTheory/Figure 6 Q fit parameter boxplots",
                dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)
