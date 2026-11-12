import os
from os.path import exists
import shutil
from math import floor, log10, factorial

from collections import Counter
from functools import lru_cache
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
    ax.set_title(r"$\bar{\rho}$" + " with time in healthy tissue")
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
    ax.set_title(r"$\bar{Q}^{(1)}$" + " with time in healthy tissue")
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

def finerHeatmapGrid(coordinates, factor=4):
    """Subdivide each original heatmap cell while preserving its boundaries."""
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 1 or len(coordinates) < 2:
        raise ValueError("Heatmap coordinates must contain at least two points")
    steps = np.diff(coordinates)
    if factor < 1 or not np.all(steps > 0) or not np.allclose(steps, steps[0]):
        raise ValueError("Heatmap coordinates must be regularly spaced and increasing")
    spacing = steps[0] / factor
    return coordinates[0] - steps[0] / 2 + spacing * (np.arange(len(coordinates) * factor) + 0.5)


def poolCorrelations(correlations, counts):
    """Pool correlation bins using their contributing pair counts."""
    correlations = np.asarray(correlations, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if correlations.shape != counts.shape or correlations.ndim < 2:
        raise ValueError("Correlation and count arrays must have matching shapes")
    if (np.any(~np.isfinite(counts)) or np.any(counts < 0)
            or np.any((counts > 0) & ~np.isfinite(correlations))):
        raise ValueError("Invalid pair counts or populated correlation bins")
    total = counts.sum(axis=0)
    weighted = (np.where(counts > 0, correlations, 0) * counts).sum(axis=0)
    return np.divide(weighted, total, out=np.full(total.shape, np.nan),
                     where=total > 0), total

def plotParameterBoxplots(parameters, fileType, outputName):
    fig, axes = plt.subplots(1, len(parameters), figsize=(2 * len(parameters), 3))
    for ax, (values, pooledValue, ylabel) in zip(np.atleast_1d(axes), parameters):
        if not np.isfinite(pooledValue):
            raise ValueError(f"{ylabel}: pooled parameter must be finite")
        ax.boxplot(values, positions=[0], widths=0.5, showfliers=False)
        sns.swarmplot(y=values, ax=ax, color="tab:blue", size=6, zorder=3)
        ax.scatter(0, pooledValue, marker="*", s=120, color="gold",
                   edgecolor="black", linewidth=0.7, zorder=5, label="Pooled fit")
        ax.set_ylabel(ylabel)
        ax.set_xticks([])
        if pooledValue >= 0 and np.all(np.asarray(values) >= 0):
            ax.set_ylim(bottom=0)
    np.atleast_1d(axes)[-1].legend(loc="upper left", bbox_to_anchor=(1.03, 1),
                                  fontsize=8)
    fig.tight_layout()
    fig.savefig(f"results/paperTheory/{outputName} {fileType}",
                dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)

# Fit each video and a pair-count-weighted pooled correlation.
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
    densityCorrelations = []
    densityCounts = []

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

        densityCorrelations.append(correlation)
        densityCounts.append(count)

    if not densityCorrelations:
        raise ValueError(f"No recordings found for {fileType}")
    pooledDensity, pooledDensityCount = poolCorrelations(densityCorrelations, densityCounts)
    densityInputs = list(zip(filenames, densityCorrelations, densityCounts))
    densityInputs.append((f"pooled {fileType}", pooledDensity, pooledDensityCount))

    for filename, correlation, count in densityInputs:
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
        fit = {"Filename": filename, "m_rho": m_rho, "D": D,
               "m_s": m_s, "lambda": decayLength}
        sampleFits.append(fit)
        sampleAnnealed.append(annealed)
        sampleQuenched.append(quenched)
        sampleTotal.append(correlationRT)
        print(
            f"{filename}: m_rho={m_rho:.4g}, D={D:.4g}, "
            f"m_s={m_s:.4g}, lambda={decayLength:.4g}"
        )

    if not sampleFits:
        raise ValueError(f"No recordings found for {fileType}")
    dfDensityFits = pd.DataFrame(sampleFits[:-1])
    dfDensityFits.to_csv(f"results/paperTheory/density fits {fileType}.csv", index=False)
    pd.DataFrame([sampleFits[-1]]).to_csv(
        f"results/paperTheory/density pooled fit {fileType}.csv", index=False
    )
    print(f"Density model parameters (n={len(dfDensityFits)}, population SD):")
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
        totalTGrid, totalRGrid = np.meshgrid(times, radii, indexing="ij")
        maxTotal = np.nanmax(np.abs(total))
        if not np.isfinite(maxTotal) or maxTotal <= 0:
            raise ValueError(f"{filename}: invalid total density correlation")
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        c = ax.pcolor(totalTGrid, totalRGrid, total, cmap="RdBu_r",
                      vmin=-maxTotal, vmax=maxTotal, shading="auto")
        fig.colorbar(c, ax=ax).formatter.set_powerlimits((0, 0))
        ax.set_xlabel("Time apart $T$ (min)")
        ax.set_ylabel(r"Distance apart $R$ $(\mu m)$")
        ax.set_title(r"Total correlation $C_{\rho\rho}(R,T)$", pad=25)
        if filename == f"pooled {fileType}":
            fig.suptitle("Pooled recordings")
        enlargeHeatmapText(fig)
        fig.tight_layout()
        fig.savefig(f"results/paperTheory/Correlation rho total {filename}",
                    dpi=300, transparent=True, bbox_inches="tight")
        plt.close(fig)

        annealedData = annealed[1:, 1:]
        annealedModel = densityAnnealedModel((rGrid, tGrid), fit["m_rho"], fit["D"])
        maxCorr = np.nanmax(np.abs(np.stack([annealedData, annealedModel]))) / norm
        fineR, fineT = finerHeatmapGrid(radii[1:]), finerHeatmapGrid(times[1:])
        fineRGrid, fineTGrid = np.meshgrid(fineR, fineT)
        fineModel = densityAnnealedModel((fineRGrid, fineTGrid), fit["m_rho"], fit["D"])
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        for panel, values, xGrid, yGrid, title in zip(
            ax,
            [annealedData, fineModel],
            [tGrid, fineTGrid],
            [rGrid, fineRGrid],
            [r"Experiment $C^{nn}_{\rho\rho}(R,T)$",
             r"Fitted model $C^{nn}_{\rho\rho}(R,T)$"],
        ):
            c = panel.pcolor(xGrid, yGrid, values / norm, cmap="RdBu_r",
                             vmin=-maxCorr, vmax=maxCorr, shading="auto")
            fig.colorbar(c, ax=panel).formatter.set_powerlimits((0, 0))
            panel.set_xlabel("Time apart $T$ (min)")
            panel.set_ylabel(r"Distance apart $R$ $(\mu m)$")
            panel.set_title(title, pad=25)
        if filename == f"pooled {fileType}":
            fig.suptitle("Pooled recordings")
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
        if filename == f"pooled {fileType}":
            fig.suptitle("Pooled recordings")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"results/paperTheory/Correlation rho_s {filename}",
                    dpi=300, transparent=True, bbox_inches="tight")
        plt.close(fig)

    annealedPrefactor = (
        dfDensityFits["m_rho"] / (32 * np.pi**3 * dfDensityFits["D"]**2)
    ).to_numpy()
    pooledDensityFit = sampleFits[-1]
    pooledPrefactor = pooledDensityFit["m_rho"] / (32 * np.pi**3 * pooledDensityFit["D"]**2)
    plotParameterBoxplots(
        [
            (dfDensityFits["m_rho"].to_numpy(), pooledDensityFit["m_rho"], r"$m_\rho$"),
            (dfDensityFits["D"].to_numpy(), pooledDensityFit["D"], r"$D$ $(\mu m^2/\mathrm{min})$"),
            (annealedPrefactor, pooledPrefactor, r"$m_\rho/(32\pi^3 D^2)$"),
            (dfDensityFits["m_s"].to_numpy(), pooledDensityFit["m_s"], r"$m_s$"),
            (dfDensityFits["lambda"].to_numpy(), pooledDensityFit["lambda"], r"$\lambda$ $(\mu m)$"),
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

# ---- full fit of A, L, mQ_1 and mQ_2 to C_Q(R, T) for R > 10 ----
# A and the model coefficient L* are shared by q1 and q2; L = L* / rho_star.
# The pooled fit weights each correlation bin by its contributing pair count.
if True:
    grid = 26
    timeGrid = 51
    minutesPerFrame = 2
    T = np.arange(timeGrid) * minutesPerFrame
    R = np.linspace(0, 2 * (grid - 1), grid)
    A_full = []
    L_full = []
    mQ_1_full = []
    mQ_2_full = []
    rmse_full = []
    fitR = R > 10
    filenames, fileType = util.getFilesType("Unwound18h")
    qInputs = []
    qCorrelations = [[], []]
    qCounts = [[], []]
    for filename in filenames:
        path = f"databases/paperTheory/dfCordQ{filename}.pkl"
        dfCor = pd.read_pickle(path)
        if len(dfCor) != 1 or dfCor["Filename"].iloc[0] != filename:
            raise ValueError(f"{path} must contain exactly one row for {filename}")
        components = []
        for i, component in enumerate(("dQ1dQ1", "dQ2dQ2")):
            correlation = np.asarray(dfCor[component + "Correlation"].iloc[0])
            count = np.asarray(dfCor[component + "Count"].iloc[0])
            if correlation.shape != (timeGrid, grid + 1, 17) or count.shape != correlation.shape:
                raise ValueError(f"{path}: unexpected {component} array shape")
            if (np.any(~np.isfinite(count)) or np.any(count < 0)
                    or np.any((count > 0) & ~np.isfinite(correlation))):
                raise ValueError(f"{path}: invalid {component} correlation or count")
            qCorrelations[i].append(correlation)
            qCounts[i].append(count)
            components.append(thetaAverage(correlation, count)[0].T[:-1][fitR])
        qInputs.append((filename, *components))

    if not qInputs:
        raise ValueError(f"No recordings found for {fileType}")
    pooledComponents = []
    for correlations, counts in zip(qCorrelations, qCounts):
        pooled, pooledCount = poolCorrelations(correlations, counts)
        pooledComponents.append(thetaAverage(pooled, pooledCount)[0].T[:-1][fitR])
    qInputs.append((f"pooled {fileType}", *pooledComponents))
    rho_star = 0.07125876326265877
    b = 0.09169152171935838
    bulkOffset = 2 * rho_star * b
    for filename, cor_q1, cor_q2 in qInputs:
        isPooled = filename == f"pooled {fileType}"
        print(datetime.now().strftime("%H:%M:%S ") + filename + " full fit")

        # fit in units of the data scale so all four parameters are of order 1
        dataScale = np.nanmax(np.abs(np.concatenate([cor_q1.ravel(), cor_q2.ravel()])))
        if not np.isfinite(dataScale) or dataScale <= 0:
            raise ValueError(f"{filename}: invalid Q correlation scale")
        data = np.concatenate([cor_q1.ravel(), cor_q2.ravel()]) / dataScale
        valid = np.isfinite(data)
        if valid.sum() < 5:
            raise ValueError(f"{filename}: insufficient finite Q correlations to fit")

        def fullModel(_, A_, L_, m1, m2):
            B_ = A_ - bulkOffset
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
            p0=[bulkOffset + 0.01, 2, max(m1_0, 1e-3), max(m2_0, 1e-3)],
            bounds=([bulkOffset + 1e-8, 0, 0, 0], np.inf),
            max_nfev=2000,
        )
        A_f, L_star_f = popt[0], popt[1]
        L_f = L_star_f / rho_star
        m1_f, m2_f = popt[2] * dataScale, popt[3] * dataScale
        rmse = np.sqrt(np.mean((fullModel(None, *popt) - data[valid]) ** 2))
        if isPooled:
            pooledQFit = (A_f, L_f, m1_f, m2_f)
            pd.DataFrame([{
                "A": A_f, "L": L_f, "L_star": L_star_f,
                "mQ_1": m1_f, "mQ_2": m2_f,
                "B": A_f - bulkOffset, "scaled_RMSE": rmse,
            }]).to_csv(f"results/paperTheory/Q pooled fit {fileType}.csv", index=False)
        else:
            A_full.append(A_f)
            L_full.append(L_f)
            mQ_1_full.append(m1_f)
            mQ_2_full.append(m2_f)
            rmse_full.append(rmse)
        print(
            f"A = {A_f:.4g},  L = {L_f:.4g},  "
            f"mQ_1 = {m1_f:.3e},  mQ_2 = {m2_f:.3e}"
        )

        # heatmaps of experiment, fitted model and residual (R > 10)
        model = forIntegral(R[fitR], T, A_f - bulkOffset, L_star_f, 1)
        T_grid, R_grid = np.meshgrid(T, R[fitR])
        fineR, fineT = finerHeatmapGrid(R[fitR]), finerHeatmapGrid(T)
        fineTGrid, fineRGrid = np.meshgrid(fineT, fineR)
        # The zero-lag cell extends below zero; hold its model at T=0 there.
        fineModel = forIntegral(fineR, np.maximum(fineT, 0), A_f - bulkOffset, L_star_f, 1)
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
                [m1_f * fineModel, m2_f * fineModel],
                [residual_q1, residual_q2],
                ["11", "22"],
            )
        ):
            panels = [
                (cor, rf"Experiment $C^{{{label}}}_{{qq}}(R,T)$"),
                (fitted, rf"Model $C^{{{label}}}_{{qq}}(R,T)$"),
                (residual, "Experiment - Model"),
            ]
            for column, (a, (values, title)) in enumerate(zip(ax[row], panels)):
                xGrid, yGrid = (fineTGrid, fineRGrid) if column == 1 else (T_grid, R_grid)
                c = a.pcolor(
                    xGrid,
                    yGrid,
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
        fitTitle = "Pooled fit" if isPooled else "Full fit"
        fig.suptitle(fitTitle + rf": $A$={A_f:.3g}, $L$={L_f:.3g}", y=1.0)
        enlargeHeatmapText(fig)
        fig.tight_layout()
        fig.savefig(
            f"results/paperTheory/Full fit {filename}",
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )
        plt.close("all")

    print(f"Full fit A = {np.mean(A_full):.4g} +- {np.std(A_full):.4g}")
    print(f"Full fit L = {np.mean(L_full):.4g} +- {np.std(L_full):.4g}")
    print(f"Full fit mQ_1 = {np.mean(mQ_1_full):.3e} +- {np.std(mQ_1_full):.3e}")
    print(f"Full fit mQ_2 = {np.mean(mQ_2_full):.3e} +- {np.std(mQ_2_full):.3e}")
    plotParameterBoxplots(
        [
            (A_full, pooledQFit[0], r"$A$"),
            (L_full, pooledQFit[1], r"$L$"),
            (mQ_1_full, pooledQFit[2], r"$m_{Q_1}$"),
            (mQ_2_full, pooledQFit[3], r"$m_{Q_2}$"),
        ],
        fileType,
        "Q fit parameter boxplots",
    )



# ------------------------- Figure 6 -------------------------

# The shared constants are treated as fixed; bootstrap uncertainty is conditional on them.
figure6RhoStar = 0.07125876326265877
figure6Bulk = 0.09169152171935838
figure6BootstrapReplicates = 1000
figure6BootstrapSeed = 2026


def figure6PoolSamples(samples, indices):
    """Pool theta and recording contributions using pair counts."""
    weighted, counts = samples
    weightedSum = weighted[indices].sum(axis=0)
    countSum = counts[indices].sum(axis=0)
    return np.divide(weightedSum, countSum,
                     out=np.full(countSum.shape, np.nan), where=countSum > 0)


@lru_cache(maxsize=4)
def figure6Kernel(radii):
    dk = 20 / 10000
    k = (np.arange(10000) + 0.5) * dk
    return k, dk * k[None, :] * sc.jv(0, np.asarray(radii)[:, None] * k)


def figure6FitPooled(correlations, radii, times, rhoStar=figure6RhoStar):
    """Fit the Figure 5 Q model to Q1 and Q2 correlations."""
    if not np.isfinite(rhoStar) or rhoStar <= 0:
        raise ValueError("Q fit density must be finite and positive")
    fitR = radii >= 10
    values = correlations[:, fitR, :]
    scale = np.nanmax(np.abs(values))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Q correlation has an invalid scale")
    data = values.ravel() / scale
    valid = np.isfinite(data)
    if valid.sum() < 5:
        raise ValueError("Q correlation has insufficient finite bins")
    offset = 2 * rhoStar * figure6Bulk
    k, kernel = figure6Kernel(tuple(radii[fitR]))
    k2 = k**2
    cachedParameters = None
    cachedModel = None
    cachedJacobian = None

    def evaluate(A, Lstar, m1, m2):
        nonlocal cachedParameters, cachedModel, cachedJacobian
        current = (A, Lstar, m1, m2)
        if current != cachedParameters:
            rate = A - offset + Lstar * k2
            decay = np.exp(-rate[:, None] * times[None, :])
            base = decay / rate[:, None]
            derivative = -base * (times[None, :] + 1 / rate[:, None])
            integral = (kernel @ base).ravel()
            dA = (kernel @ derivative).ravel()
            dL = (kernel @ (derivative * k2[:, None])).ravel()
            zeros = np.zeros_like(integral)
            cachedModel = np.concatenate((m1 * integral, m2 * integral))[valid]
            cachedJacobian = np.stack((
                np.concatenate((m1 * dA, m2 * dA)),
                np.concatenate((m1 * dL, m2 * dL)),
                np.concatenate((integral, zeros)),
                np.concatenate((zeros, integral)),
            ), axis=1)[valid]
            cachedParameters = current
        return cachedModel, cachedJacobian

    def model(_, A, Lstar, m1, m2):
        return evaluate(A, Lstar, m1, m2)[0]

    def jacobian(_, A, Lstar, m1, m2):
        return evaluate(A, Lstar, m1, m2)[1]

    initial = forIntegral(radii[fitR], times, 0.01, 2, 1).ravel()
    initialNorm = np.sum(initial**2)
    m1 = np.nansum(initial * values[0].ravel() / scale) / initialNorm
    m2 = np.nansum(initial * values[1].ravel() / scale) / initialNorm
    popt, _ = curve_fit(
        model, None, data[valid],
        p0=[offset + 0.01, 2, max(m1, 1e-3), max(m2, 1e-3)],
        bounds=([offset + 1e-8, 0, 0, 0], np.inf), jac=jacobian, max_nfev=2000,
    )
    parameters = np.array([popt[0], popt[1] / rhoStar,
                           popt[2] * scale, popt[3] * scale])
    if not np.all(np.isfinite(parameters)):
        raise ValueError("Q fit produced non-finite parameters")
    # Amplitude thresholds refer to the optimizer's data-scaled parameters.
    nearBounds = (popt[0] - offset <= 1e-6, popt[1] <= 1e-6,
                  popt[2] <= 1e-6, popt[3] <= 1e-6)
    return parameters, nearBounds


def figure6ModelValues(parameters, radii, times):
    A, L, m1, m2 = parameters
    integral = forIntegral(radii, times, A - 2 * figure6RhoStar * figure6Bulk,
                           L * figure6RhoStar, 1)
    return np.stack((m1 * integral, m2 * integral))


if False:
    figure6StartTime, figure6EndTime = 90, 150
    figure6RadialStep, figure6Grid = 2, 21
    conditions = [
        ("Wild type", "Close", "WoundL18h", "#8b005b"),
        ("Wild type", "Far", "WoundL18h", "#cf369a"),
        ("Wild type", "Unwounded", "Unwound18h", "#f0a0d4"),
        ("JNK", "Close", "WoundLJNK", "#006b30"),
        ("JNK", "Far", "WoundLJNK", "#36a65c"),
        ("JNK", "Unwounded", "UnwoundJNK", "#a0d6aa"),
    ]
    times = np.arange(0, figure6EndTime - figure6StartTime + 1, 2)
    radii = np.arange(figure6Grid - 1) * figure6RadialStep
    fitR = radii >= 10
    parameterNames = ("A", "L", "mQ_1", "mQ_2")
    samplesByCondition = {}
    recordingsByType = {}

    for genotype, region, shapeFileType, colour in conditions:
        filenames = sorted(util.getFilesType(shapeFileType)[0])
        if not filenames:
            raise ValueError(f"No recordings found for {shapeFileType}")
        if shapeFileType in recordingsByType and filenames != recordingsByType[shapeFileType]:
            raise ValueError(f"Inconsistent paired recordings for {shapeFileType}")
        recordingsByType[shapeFileType] = filenames
        sampleSums, sampleCounts = [], []
        for filename in filenames:
            path = (
                f"databases/correlationsWound/dfCorFigure6{region}{filename}"
                f"_minutes{figure6StartTime}-{figure6EndTime}"
                f"_rmax{figure6RadialStep * (figure6Grid - 1)}.pkl"
            )
            if not exists(path):
                raise FileNotFoundError(f"Missing Figure 6 correlation {path}")
            dfCor = pd.read_pickle(path)
            if (len(dfCor) != 1 or dfCor["Filename"].iloc[0] != filename
                    or dfCor["Region"].iloc[0] != region
                    or dfCor["StartTime"].iloc[0] != figure6StartTime
                    or dfCor["EndTime"].iloc[0] != figure6EndTime
                    or dfCor["Grid"].iloc[0] != figure6Grid
                    or dfCor["RadialStep"].iloc[0] != figure6RadialStep):
                raise ValueError(f"Incorrect Figure 6 correlation metadata: {path}")
            sums, counts = [], []
            for component in ("dQ1dQ1", "dQ2dQ2"):
                cor = np.asarray(dfCor[component + "Correlation"].iloc[0])
                count = np.asarray(dfCor[component + "Count"].iloc[0])
                if cor.shape != (61, figure6Grid, 17) or count.shape != cor.shape:
                    raise ValueError(f"{path}: unexpected {component} array shape")
                if (np.any(~np.isfinite(count)) or np.any(count < 0)
                        or np.any((count > 0) & ~np.isfinite(cor))):
                    raise ValueError(f"{path}: invalid {component} correlation or count")
                cor, count = cor[::2, :-1], count[::2, :-1]
                sums.append(np.sum(np.where(count > 0, cor, 0) * count, axis=2).T)
                counts.append(count.sum(axis=2).T)
            sampleSums.append(sums)
            sampleCounts.append(counts)
        samplesByCondition[(genotype, region)] = (
            np.asarray(sampleSums), np.asarray(sampleCounts)
        )

    pooledResults = {}
    for genotype, region, shapeFileType, colour in conditions:
        key = (genotype, region)
        correlation = figure6PoolSamples(
            samplesByCondition[key], np.arange(len(recordingsByType[shapeFileType]))
        )
        try:
            parameters, nearBounds = figure6FitPooled(correlation, radii, times)
        except (ValueError, RuntimeError) as error:
            raise RuntimeError(f"{genotype} {region}: pooled fit failed") from error
        pooledResults[key] = (correlation, parameters, nearBounds)
        print(f"{genotype} {region}: " + ", ".join(
            f"{name}={value:.5g}" for name, value in zip(parameterNames, parameters)
        ))

    rng = np.random.default_rng(figure6BootstrapSeed)
    bootstrapRows, failureRows = [], []
    attemptsByType = {}
    if figure6BootstrapReplicates < 2:
        raise ValueError("At least two bootstrap replicates are needed for a standard deviation")
    maxAttempts = int(np.ceil(figure6BootstrapReplicates * 1.2))
    for shapeFileType, filenames in recordingsByType.items():
        groupConditions = [entry for entry in conditions if entry[2] == shapeFileType]
        successes = 0
        for attempt in range(1, maxAttempts + 1):
            if successes == figure6BootstrapReplicates:
                break
            indices = rng.integers(0, len(filenames), size=len(filenames))
            draw = "|".join(filenames[i] for i in indices)
            attemptResults = []
            for genotype, region, _, _ in groupConditions:
                correlation = figure6PoolSamples(samplesByCondition[(genotype, region)], indices)
                try:
                    parameters, nearBounds = figure6FitPooled(correlation, radii, times)
                except (ValueError, RuntimeError) as error:
                    failureRows.append({
                        "FileType": shapeFileType, "Attempt": attempt,
                        "Genotype": genotype, "Region": region,
                        "Recordings": draw, "Error": str(error),
                    })
                    break
                attemptResults.append((genotype, region, parameters, nearBounds))
            if len(attemptResults) != len(groupConditions):
                continue
            successes += 1
            for genotype, region, parameters, nearBounds in attemptResults:
                row = {
                    "Genotype": genotype, "Region": region, "FileType": shapeFileType,
                    "Replicate": successes, "Attempt": attempt, "Recordings": draw,
                }
                row.update(dict(zip(parameterNames, parameters)))
                row.update({name + "_near_bound": flag
                            for name, flag in zip(parameterNames, nearBounds)})
                bootstrapRows.append(row)
            if successes % 100 == 0:
                print(f"{shapeFileType}: {successes}/{figure6BootstrapReplicates} bootstrap fits")
        attemptsByType[shapeFileType] = successes + sum(
            failure["FileType"] == shapeFileType for failure in failureRows
        )
        if successes < figure6BootstrapReplicates:
            pd.DataFrame(failureRows).to_csv(
                "results/paperTheory/Figure 6 bootstrap failures.csv", index=False
            )
            raise RuntimeError(
                f"{shapeFileType}: only {successes} successful bootstrap draws "
                f"in {maxAttempts} attempts; see Figure 6 bootstrap failures.csv"
            )

    dfBootstrap = pd.DataFrame(bootstrapRows)
    dfBootstrap.to_csv("results/paperTheory/Figure 6 bootstrap samples.csv", index=False)
    pd.DataFrame(failureRows, columns=[
        "FileType", "Attempt", "Genotype", "Region", "Recordings", "Error"
    ]).to_csv("results/paperTheory/Figure 6 bootstrap failures.csv", index=False)
    if failureRows:
        warnings.warn(f"{len(failureRows)} bootstrap draws failed; see Figure 6 bootstrap failures.csv",
                      RuntimeWarning)

    summaryRows = []
    for genotype, region, shapeFileType, colour in conditions:
        correlation, parameters, nearBounds = pooledResults[(genotype, region)]
        draws = dfBootstrap[(dfBootstrap["Genotype"] == genotype)
                            & (dfBootstrap["Region"] == region)]
        row = {
            "Genotype": genotype, "Region": region, "FileType": shapeFileType,
            "Recordings": len(recordingsByType[shapeFileType]),
            "BootstrapSuccesses": len(draws),
            "BootstrapAttempts": attemptsByType[shapeFileType],
            "BootstrapSeed": figure6BootstrapSeed,
            "rho_star_fixed": figure6RhoStar, "b_fixed": figure6Bulk,
        }
        for name, estimate, bound in zip(parameterNames, parameters, nearBounds):
            values = draws[name].to_numpy()
            row[name] = estimate
            row[name + "_bootstrap_sd"] = np.std(values, ddof=1)
            row[name + "_ci_low"], row[name + "_ci_high"] = np.percentile(values, [2.5, 97.5])
            row[name + "_near_bound"] = bound
            row[name + "_bootstrap_near_bound"] = int(draws[name + "_near_bound"].sum())
        summaryRows.append(row)
    dfFigure6Summary = pd.DataFrame(summaryRows)
    dfFigure6Summary.to_csv("results/paperTheory/Figure 6 pooled bootstrap parameters.csv", index=False)

    for genotype, region, _, colour in conditions:
        correlation, parameters, _ = pooledResults[(genotype, region)]
        fitted = figure6ModelValues(parameters, radii, times)
        residual = correlation - fitted
        vmax = np.nanmax(np.abs(np.stack((correlation[:, fitR], fitted[:, fitR],
                                         residual[:, fitR]))))
        tGrid, rGrid = np.meshgrid(times, radii[fitR])
        fineR, fineT = finerHeatmapGrid(radii[fitR]), finerHeatmapGrid(times)
        fineTGrid, fineRGrid = np.meshgrid(fineT, fineR)
        fineFitted = figure6ModelValues(parameters, fineR, np.maximum(fineT, 0))
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        for row, label in enumerate(("11", "22")):
            for column, (ax, values, title) in enumerate(zip(
                axes[row], (correlation[row, fitR], fineFitted[row], residual[row, fitR]),
                (rf"Pooled $C^{{{label}}}_{{qq}}(R,T)$",
                 rf"Model $C^{{{label}}}_{{qq}}(R,T)$", "Pooled - Model"),
            )):
                xGrid, yGrid = (fineTGrid, fineRGrid) if column == 1 else (tGrid, rGrid)
                image = ax.pcolor(xGrid, yGrid, values, cmap="RdBu_r",
                                  vmin=-vmax, vmax=vmax, shading="auto")
                fig.colorbar(image, ax=ax)
                ax.set_xlabel("Time apart $T$ (min)")
                ax.set_ylabel(r"Distance apart $R$ $(\mu m)$")
                ax.set_title(title, pad=25)
        fig.suptitle(f"{genotype} {healthyName(region)} pooled: "
                     + rf"$A$={parameters[0]:.3g}, $L$={parameters[1]:.3g}")
        enlargeHeatmapText(fig)
        fig.tight_layout()
        fig.savefig(f"results/paperTheory/Figure 6 pooled fit {genotype} {region}",
                    dpi=300, transparent=True, bbox_inches="tight")
        plt.close(fig)

    sliceFigures = []
    for component, label in enumerate(("11", "22")):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for genotype, region, _, colour in conditions:
            correlation, parameters, _ = pooledResults[(genotype, region)]
            fitted = figure6ModelValues(parameters, radii, times)
            row = 0 if genotype == "Wild type" else 1
            axes[row, 0].plot(radii[fitR], correlation[component, fitR, 0], "o",
                              color=colour, label=healthyName(region))
            axes[row, 0].plot(radii[fitR], fitted[component, fitR, 0], "-", color=colour)
            axes[row, 1].plot(times, correlation[component, radii == 10][0], "o",
                              color=colour, label=healthyName(region))
            axes[row, 1].plot(times, fitted[component, radii == 10][0], "-", color=colour)
        for row, genotype in enumerate(("Wild type", "JNK")):
            axes[row, 0].set_title(f"{genotype}: T = 0 min")
            axes[row, 1].set_title(rf"{genotype}: R = 10 $\mu$m")
            axes[row, 0].set_xlabel(r"Distance apart $R$ $(\mu m)$")
            axes[row, 0].set_xlim(radii[fitR][0], radii[fitR][-1])
            axes[row, 1].set_xlabel("Time apart $T$ (min)")
            axes[row, 1].set_xlim(times[0], times[-1])
            for ax in axes[row]:
                ax.set_ylabel(rf"$C^{{{label}}}_{{qq}}(R,T)$")
                ax.legend(title="Points: pooled data; lines: pooled fit", fontsize=9.1)
        sliceFigures.append((fig, axes, component))
    sliceAxes = [ax for _, axes, _ in sliceFigures for ax in axes.flat]
    limits = (min(ax.get_ylim()[0] for ax in sliceAxes),
              max(ax.get_ylim()[1] for ax in sliceAxes))
    for fig, axes, component in sliceFigures:
        for ax in axes.flat:
            ax.set_ylim(limits)
        fig.tight_layout()
        fig.savefig(f"results/paperTheory/Figure 6 pooled Q{component + 1} correlation slices",
                    dpi=300, transparent=True, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for ax, parameter, ylabel in zip(
        axes.flat, parameterNames, (r"$A$", r"$L$", r"$m_{Q_1}$", r"$m_{Q_2}$")
    ):
        values = dfFigure6Summary[[parameter, parameter + "_ci_low",
                                   parameter + "_ci_high"]].to_numpy()
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"{parameter}: invalid bootstrap interval or estimate")
        positive = values[values > 0]
        if positive.size == 0:
            raise ValueError(f"{parameter}: no positive values for logarithmic axis")
        for i, (genotype, region, _, colour) in enumerate(conditions):
            row = dfFigure6Summary[(dfFigure6Summary["Genotype"] == genotype)
                                   & (dfFigure6Summary["Region"] == region)].iloc[0]
            ax.vlines(i, row[parameter + "_ci_low"], row[parameter + "_ci_high"],
                      color=colour, linewidth=2)
            ax.scatter(i, row[parameter], color=colour, zorder=3)
        if np.any(values == 0):
            linearThreshold = np.median(positive) / 1000
            ax.set_yscale("symlog", linthresh=linearThreshold)
            ax.set_ylim(bottom=-linearThreshold / 4)
            ax.set_title(f"Linear below {linearThreshold:.1g}")
        else:
            ax.set_yscale("log")
        ax.set_xticks(np.arange(len(conditions)), [
            f"{genotype}\n{healthyName(region)}" for genotype, region, _, _ in conditions
        ], rotation=45, ha="right", fontsize=7.8)
        ax.set_ylabel(ylabel)
    fig.suptitle("Pooled estimates and 95% recording-bootstrap intervals")
    fig.tight_layout()
    fig.savefig("results/paperTheory/Figure 6 pooled parameter intervals",
                dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)




# ------------------------- Supplementary Figure 4 -------------------------


def supplementary4Holm(pvalues):
    """Holm-adjust one family of p-values, returning the original order."""
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.ndim != 1 or np.any(~np.isfinite(pvalues)) or np.any(
        (pvalues < 0) | (pvalues > 1)
    ):
        raise ValueError("Holm correction requires finite p-values in [0, 1]")
    order = np.argsort(pvalues)
    adjusted = np.minimum(1, np.maximum.accumulate(
        pvalues[order] * np.arange(len(pvalues), 0, -1)
    ))
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def supplementary4Tests(values, parameterNames, windowLabels):
    """Complete-case Friedman and paired Wilcoxon tests for each parameter."""
    values = np.asarray(values, dtype=float)
    if (values.ndim != 3 or values.shape[1:] != (3, len(parameterNames))
            or len(windowLabels) != 3 or len(values) < 3
            or not np.all(np.isfinite(values))):
        raise ValueError("Tests require at least three complete, finite paired recordings")
    rows = []
    pairs = ((0, 1), (0, 2), (1, 2))
    for k, name in enumerate(parameterNames):
        sample = values[:, :, k]
        if np.all(sample == sample[:, :1]):
            statistic, pvalue = 0.0, 1.0
        else:
            statistic, pvalue = sp.stats.friedmanchisquare(*sample.T)
        if not np.isfinite(statistic) or not np.isfinite(pvalue):
            raise ValueError(f"{name}: Friedman test returned non-finite results")
        rows.append({
            "Parameter": name, "Test": "Friedman", "Window1": "all",
            "Window2": "all", "N": len(sample), "Statistic": statistic,
            "MedianDifference_Window2_minus_Window1": np.nan,
            "P_raw": pvalue, "P_Holm": np.nan,
        })
        pairRows = []
        for i, j in pairs:
            difference = sample[:, j] - sample[:, i]
            if np.all(difference == 0):
                statistic, pvalue = 0.0, 1.0
            else:
                # Pass precomputed differences to avoid subtraction inside SciPy.
                statistic, pvalue = sp.stats.wilcoxon(
                    difference, alternative="two-sided"
                )
            if not np.isfinite(statistic) or not np.isfinite(pvalue):
                raise ValueError(f"{name}: Wilcoxon test returned non-finite results")
            pairRows.append({
                "Parameter": name, "Test": "Wilcoxon", "Window1": windowLabels[i],
                "Window2": windowLabels[j], "N": len(sample), "Statistic": statistic,
                "MedianDifference_Window2_minus_Window1": np.median(difference),
                "P_raw": pvalue,
            })
        for row, adjusted in zip(
            pairRows, supplementary4Holm([row["P_raw"] for row in pairRows])
        ):
            row["P_Holm"] = adjusted
        rows.extend(pairRows)
    # A separate family covers the four omnibus parameter tests.
    omnibus = [row for row in rows if row["Test"] == "Friedman"]
    for row, adjusted in zip(omnibus, supplementary4Holm([r["P_raw"] for r in omnibus])):
        row["P_Holm"] = adjusted
    return pd.DataFrame(rows)


def supplementary4Plot(values, pooled, tests, parameterNames, windowLabels, path):
    """Plot paired fits and the pair-count-weighted pooled fit in each window."""
    labels = (r"$A$", r"$L$", r"$m_{Q_1}$", r"$m_{Q_2}$")
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for k, (ax, name, ylabel) in enumerate(zip(axes, parameterNames, labels)):
        sample = values[:, :, k]
        ax.boxplot(sample, positions=[0, 1, 2], widths=0.5, showfliers=False)
        for recording in sample:
            ax.plot([0, 1, 2], recording, color="tab:blue", alpha=0.25, linewidth=0.8)
        ax.scatter(np.tile(np.arange(3), len(sample)), sample.ravel(),
                   color="tab:blue", s=28, alpha=0.75, zorder=3)
        ax.scatter(np.arange(3), pooled[:, k], marker="*", s=150, color="gold",
                   edgecolor="black", linewidth=0.7, zorder=5, label="Pooled fit")
        ax.set_xticks([0, 1, 2], windowLabels)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time window (hours)")
        highest = max(np.max(sample), np.max(pooled[:, k]))
        span = highest if highest > 0 else 1.0
        ax.set_ylim(0, highest + 0.63 * span)
        rows = tests[(tests["Parameter"] == name) & (tests["Test"] == "Wilcoxon")]
        for level, ((i, j), (_, row)) in enumerate(zip(
            ((0, 1), (1, 2), (0, 2)),
            rows.set_index(["Window1", "Window2"]).loc[
                [(windowLabels[0], windowLabels[1]),
                 (windowLabels[1], windowLabels[2]),
                 (windowLabels[0], windowLabels[2])]
            ].iterrows(),
        )):
            y = highest + (0.08 + 0.14 * level) * span
            ax.plot([i, i, j, j], [y, y + 0.02 * span, y + 0.02 * span, y],
                    color="black", linewidth=0.8)
            pvalue = row["P_Holm"]
            significance = "***" if pvalue < 0.001 else "**" if pvalue < 0.01 else "*" if pvalue < 0.05 else "ns"
            ax.text((i + j) / 2, y + 0.025 * span, significance,
                    ha="center", va="bottom", fontsize=9)
        omnibus = tests[(tests["Parameter"] == name) & (tests["Test"] == "Friedman")]
        ax.set_title(f"Friedman\nHolm p={omnibus['P_Holm'].iloc[0]:.3g}", fontsize=10)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.03, 1), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)


def supplementary4Heatmaps(correlations, radii, times, windowLabels, path):
    """Plot pooled data with one shared colour scale per Q component."""
    correlations = np.asarray(correlations)
    if correlations.shape != (3, 2, len(radii), len(times)) or len(windowLabels) != 3:
        raise ValueError("Supplementary Figure 4: unexpected pooled correlation shape")
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7))
    tGrid, rGrid = np.meshgrid(times, radii)
    for component, label in enumerate(("11", "22")):
        values = correlations[:, component]
        if not np.any(np.isfinite(values)):
            raise ValueError(f"Supplementary Figure 4: no finite Q{component + 1} data")
        vmax = np.nanmax(np.abs(values))
        if vmax <= 0:
            raise ValueError(f"Supplementary Figure 4: zero Q{component + 1} colour scale")
        for window, ax in enumerate(axes[component]):
            image = ax.pcolor(tGrid, rGrid, values[window], cmap="RdBu_r",
                              vmin=-vmax, vmax=vmax, shading="auto")
            fig.colorbar(image, ax=ax)
            ax.set_xlabel("Time apart $T$ (min)")
            ax.set_ylabel(r"Distance apart $R$ $(\mu m)$")
            ax.set_title(rf"Pooled $C^{{{label}}}_{{qq}}$: {windowLabels[window]} h")
    enlargeHeatmapText(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)


def supplementary4WindowDensities(dfShape, filenames, windows):
    """Average 1 / mean cell area equally across frames and recordings."""
    densities = np.empty((len(filenames), len(windows)))
    for i, filename in enumerate(filenames):
        sample = dfShape[dfShape["Filename"] == filename]
        for j, (startFrame, endFrame) in enumerate(windows):
            window = sample[sample["T"].between(startFrame, endFrame)]
            frameAreas = window.groupby("T")["Area"].mean().reindex(
                np.arange(startFrame, endFrame + 1)
            ).to_numpy()
            if np.any(~np.isfinite(frameAreas)) or np.any(frameAreas <= 0):
                raise ValueError(f"{filename} frames {startFrame}-{endFrame}: missing or invalid mean cell area")
            densities[i, j] = np.mean(1 / frameAreas)
    return densities


# Fit recording-specific and pooled Q correlations in three one-hour windows.
if True:
    filenames, fileType = util.getFilesType("Unwound18h")
    filenames = sorted(filenames)
    if not filenames or len(filenames) != len(set(filenames)):
        raise ValueError("Supplementary Figure 4 requires unique recordings")
    windows = ((0, 29), (30, 59), (60, 89))
    windowLabels = ("0-1", "1-2", "2-3")
    parameterNames = ("A", "L", "mQ_1", "mQ_2")
    radii = np.arange(26) * 2
    times = np.arange(26) * 2
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    recordingDensities = supplementary4WindowDensities(dfShape, filenames, windows)
    windowDensities = recordingDensities.mean(axis=0)
    fitRows = []
    individualFits = np.full((len(filenames), len(windows), 4), np.nan)
    pooledFits = np.full((len(windows), 4), np.nan)
    pooledCorrelations = []
    for windowIndex, (startFrame, endFrame) in enumerate(windows):
        rhoStar = windowDensities[windowIndex]
        print(f"Supplementary Figure 4 {windowLabels[windowIndex]} h: rho_star={rhoStar:.6g} um^-2")
        sampleSums, sampleCounts = [], []
        for recordingIndex, filename in enumerate(filenames):
            path = f"databases/paperTheory/dfCordQ{filename}_{startFrame}-{endFrame}.pkl"
            dfCor = pd.read_pickle(path)
            if (len(dfCor) != 1 or dfCor["Filename"].iloc[0] != filename
                    or dfCor["StartFrame"].iloc[0] != startFrame
                    or dfCor["EndFrame"].iloc[0] != endFrame):
                raise ValueError(f"Incorrect supplementary correlation metadata: {path}")
            sums, counts = [], []
            for component in ("dQ1dQ1", "dQ2dQ2"):
                cor = np.asarray(dfCor[component + "Correlation"].iloc[0])
                count = np.asarray(dfCor[component + "Count"].iloc[0])
                if cor.shape != (26, 27, 17) or count.shape != cor.shape:
                    raise ValueError(f"{path}: unexpected {component} array shape")
                if (np.any(~np.isfinite(count)) or np.any(count < 0)
                        or np.any((count > 0) & ~np.isfinite(cor))):
                    raise ValueError(f"{path}: invalid {component} correlation or count")
                cor, count = cor[:, :-1], count[:, :-1]
                sums.append(np.sum(np.where(count > 0, cor, 0) * count, axis=2).T)
                counts.append(count.sum(axis=2).T)
            sampleSums.append(sums)
            sampleCounts.append(counts)
            correlation = figure6PoolSamples((np.asarray([sums]), np.asarray([counts])),
                                             np.array([0]))
            try:
                parameters, nearBounds = figure6FitPooled(correlation, radii, times, rhoStar)
                individualFits[recordingIndex, windowIndex] = parameters
                status = "ok"
            except (ValueError, RuntimeError) as error:
                parameters = np.full(4, np.nan)
                nearBounds = (False,) * 4
                status = f"failed: {error}"
                warnings.warn(f"Supplementary Figure 4 {filename} {windowLabels[windowIndex]}: {status}")
            if any(nearBounds):
                warnings.warn(f"Supplementary Figure 4 {filename} {windowLabels[windowIndex]}: fit near bounds {nearBounds}")
            fitRows.append({
                "Filename": filename, "Fit": "individual", "Window": windowLabels[windowIndex],
                "StartFrame": startFrame, "EndFrame": endFrame, "Status": status,
                "rho_star": rhoStar, "b_fixed": figure6Bulk,
                **dict(zip(parameterNames, parameters)),
                **{f"{name}_near_bound": near for name, near in zip(parameterNames, nearBounds)},
            })
        correlation = figure6PoolSamples(
            (np.asarray(sampleSums), np.asarray(sampleCounts)), np.arange(len(filenames))
        )
        pooledCorrelations.append(correlation)
        try:
            parameters, nearBounds = figure6FitPooled(correlation, radii, times, rhoStar)
        except (ValueError, RuntimeError) as error:
            raise RuntimeError(f"Supplementary Figure 4 pooled {windowLabels[windowIndex]} fit failed") from error
        pooledFits[windowIndex] = parameters
        if any(nearBounds):
            warnings.warn(f"Supplementary Figure 4 pooled {windowLabels[windowIndex]}: fit near bounds {nearBounds}")
        fitRows.append({
            "Filename": fileType, "Fit": "pooled", "Window": windowLabels[windowIndex],
            "StartFrame": startFrame, "EndFrame": endFrame, "Status": "ok",
            "rho_star": rhoStar, "b_fixed": figure6Bulk,
            **dict(zip(parameterNames, parameters)),
            **{f"{name}_near_bound": near for name, near in zip(parameterNames, nearBounds)},
        })
    output = "results/paperTheory"
    densityRows = []
    for windowIndex, (startFrame, endFrame) in enumerate(windows):
        for recordingIndex, filename in enumerate(filenames):
            densityRows.append({
                "Filename": filename, "Window": windowLabels[windowIndex],
                "StartFrame": startFrame, "EndFrame": endFrame,
                "recording_rho_star": recordingDensities[recordingIndex, windowIndex],
                "window_rho_star": windowDensities[windowIndex],
            })
    pd.DataFrame(densityRows).to_csv(
        f"{output}/Sup Figure 4 window densities {fileType}.csv", index=False
    )
    fits = pd.DataFrame(fitRows)
    fits.to_csv(f"{output}/Sup Figure 4 Q window fits {fileType}.csv", index=False)
    complete = np.all(np.isfinite(individualFits), axis=(1, 2))
    if not np.all(complete):
        warnings.warn(f"Supplementary Figure 4 complete-case tests exclude: {', '.join(np.asarray(filenames)[~complete])}")
    tests = supplementary4Tests(individualFits[complete], parameterNames, windowLabels)
    tests.to_csv(f"{output}/Sup Figure 4 Q window statistics {fileType}.csv", index=False)
    supplementary4Plot(individualFits[complete], pooledFits, tests, parameterNames,
                       windowLabels, f"{output}/Sup Figure 4 Q window boxplots {fileType}.png")
    supplementary4Heatmaps(pooledCorrelations, radii, times, windowLabels,
                           f"{output}/Sup Figure 4 pooled Q window heatmaps {fileType}.png")
    print(f"Supplementary Figure 4: {complete.sum()} of {len(filenames)} complete recordings")
