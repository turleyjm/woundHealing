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

T = 90
scale = 123.26 / 512


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

    done = 0
    nextPrint = 0
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

        # Print progress roughly every 10% of anchors
        done += s1 - s0
        if done >= nextPrint:
            print(
                datetime.now().strftime("%H:%M:%S")
                + f" {int(100 * done / len(anchors))}%"
            )
            nextPrint = done + max(1, len(anchors) // 10)

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
if True:
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
if True:
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
if True:
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

# Shape correlations for frames 1.5-2.5hr, with each wound-distance group kept separate.
if True:
    for shapeFileType in ["WoundL18h", "WoundLJNK"]:
        shapeFilenames, _ = util.getFilesType(shapeFileType)
        dfShape = pd.read_pickle(f"databases/dfShapeWound{shapeFileType}.pkl")
        startFrame, endFrame = 45, 75
        grid = 20
        timeGrid = endFrame - startFrame + 1
        rStep = 2
        tStep = 1
        woundDistanceThreshold = 30

        for filename in shapeFilenames:
            dfShapeF = dfShape[
                (dfShape["Filename"] == filename)
                & (dfShape["T"] >= startFrame)
                & (dfShape["T"] <= endFrame)
            ]
            if dfShapeF.empty:
                raise ValueError(
                    f"No shape data for {filename} in frames {startFrame}-{endFrame}"
                )

            for region, inRegion in (
                ("Close", dfShapeF["R"] < woundDistanceThreshold),
                ("Far", dfShapeF["R"] >= woundDistanceThreshold),
            ):
                path_to_file = (
                    f"databases/correlationsWound/dfCor{region}{filename}"
                    f"_frames{startFrame}-{endFrame}.pkl"
                )
                if exists(path_to_file):
                    continue

                dfRegion = dfShapeF[inRegion]
                print(
                    datetime.now().strftime("%H:%M:%S ")
                    + f"{filename} {region.lower()} "
                    + f"({len(dfRegion)} cells in frames {startFrame}-{endFrame})"
                )
                dq = (
                    np.stack(dfRegion["dq"].to_list(), axis=0)
                    if not dfRegion.empty
                    else np.empty((0, 2, 2))
                )
                results = spaceTimeCorrelation(
                    dfRegion["X"].to_numpy(),
                    dfRegion["Y"].to_numpy(),
                    dfRegion["T"].to_numpy(),
                    {
                        "dQ1dQ1": (dq[:, 0, 0], dq[:, 0, 0]),
                        "dQ2dQ2": (dq[:, 0, 1], dq[:, 0, 1]),
                    },
                    np.arange(len(dfRegion)),
                    timeGrid=timeGrid,
                    grid=grid,
                    rStep=rStep,
                    tStep=tStep,
                )
                pd.DataFrame(
                    [
                        {
                            "Filename": filename,
                            "StartFrame": startFrame,
                            "EndFrame": endFrame,
                            "Region": region,
                            "WoundDistanceThreshold": woundDistanceThreshold,
                            "dQ1dQ1Correlation": results["dQ1dQ1"][0],
                            "dQ2dQ2Correlation": results["dQ2dQ2"][0],
                            "dQ1dQ1Correlation_std": results["dQ1dQ1"][1],
                            "dQ2dQ2Correlation_std": results["dQ2dQ2"][1],
                            "dQ1dQ1Count": results["dQ1dQ1"][2],
                            "dQ2dQ2Count": results["dQ2dQ2"][2],
                        }
                    ]
                ).to_pickle(path_to_file)

# Shape correlations for frames 1.5-2.5hr in unwounded control wings.
if True:
    for shapeFileType in ["Unwound18h", "UnwoundJNK"]:
        shapeFilenames, _ = util.getFilesType(shapeFileType)
        dfShape = pd.read_pickle(f"databases/dfShapeWound{shapeFileType}.pkl")
        startFrame, endFrame = 45, 75
        grid = 20
        timeGrid = endFrame - startFrame + 1
        rStep = 2
        tStep = 1

        for filename in shapeFilenames:
            path_to_file = (
                f"databases/correlationsWound/dfCor{filename}"
                f"_frames{startFrame}-{endFrame}.pkl"
            )
            if exists(path_to_file):
                continue

            dfShapeWindow = dfShape[
                (dfShape["Filename"] == filename)
                & (dfShape["T"] >= startFrame)
                & (dfShape["T"] <= endFrame)
            ]
            if dfShapeWindow.empty:
                raise ValueError(
                    f"No shape data for {filename} in frames {startFrame}-{endFrame}"
                )

            print(
                datetime.now().strftime("%H:%M:%S ")
                + f"{filename} frames {startFrame}-{endFrame}"
            )
            dq = np.stack(dfShapeWindow["dq"].to_list(), axis=0)
            dq1 = dq[:, 0, 0]
            dq2 = dq[:, 0, 1]
            results = spaceTimeCorrelation(
                dfShapeWindow["X"].to_numpy(),
                dfShapeWindow["Y"].to_numpy(),
                dfShapeWindow["T"].to_numpy(),
                {"dQ1dQ1": (dq1, dq1), "dQ2dQ2": (dq2, dq2)},
                np.arange(len(dfShapeWindow)),
                timeGrid=timeGrid,
                grid=grid,
                rStep=rStep,
                tStep=tStep,
            )
            pd.DataFrame(
                [
                    {
                        "Filename": filename,
                        "StartFrame": startFrame,
                        "EndFrame": endFrame,
                        "dQ1dQ1Correlation": results["dQ1dQ1"][0],
                        "dQ2dQ2Correlation": results["dQ2dQ2"][0],
                        "dQ1dQ1Correlation_std": results["dQ1dQ1"][1],
                        "dQ2dQ2Correlation_std": results["dQ2dQ2"][1],
                        "dQ1dQ1Count": results["dQ1dQ1"][2],
                        "dQ2dQ2Count": results["dQ2dQ2"][2],
                    }
                ]
            ).to_pickle(path_to_file)




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


# ---- full fit of B, L, mQ_1 and mQ_2 to C(R, T) for R > 10 ----
# B and L are shared by q1 and q2; each has its own amplitude mQ. The Gamma(k) fit
# above gives the starting point.
grid = 27
timeGrid = 51
T = np.linspace(0, (timeGrid - 1), timeGrid)
R = np.linspace(0, 2 * (grid - 1), grid)
B_full = []
L_full = []
mQ_1_full = []
mQ_2_full = []
rmse_full = []
fitR = R > 10
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
