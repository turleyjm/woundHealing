import numpy as np
import os
import scipy
from scipy import ndimage
import skimage as sm
import skimage.io
import pandas as pd
import tifffile

import commonLiberty as cl


def surfaceFind(p):

    n = len(p) - 4

    localMax = []
    for i in range(n):
        q = p[i : i + 5]
        localMax.append(max(q))

    Max = localMax[0]
    for i in range(n):
        if Max < localMax[i]:
            Max = localMax[i]
        elif Max < 250:
            continue
        else:
            return Max

    return Max


def heightScale(z0, z):

    # e where scaling starts from the surface and d is the cut off
    d = 10
    e = 9

    if z0 + e > z:
        scale = 1
    elif z > z0 + d:
        scale = 0
    else:
        scale = 1 - abs(z - z0 - e) / (d - e)

    return scale


# Returns the full macro code with the filepath and focus range inserted as
# hard-coded values.


def focusStack(image, focusRange):

    image = image.astype("uint16")
    (T, Z, Y, X) = image.shape
    variance = np.zeros([T, Z, Y, X])
    varianceMax = np.zeros([T, Y, X])
    surface = np.zeros([T, Y, X])
    focus = np.zeros([T, Y, X])

    for t in range(T):
        for z in range(Z):
            winMean = ndimage.uniform_filter(image[t, z], (focusRange, focusRange))
            winSqrMean = ndimage.uniform_filter(
                image[t, z] ** 2, (focusRange, focusRange)
            )
            variance[t, z] = winSqrMean - winMean ** 2

    for t in range(T):
        varianceMax[t] = np.max(variance[t], axis=0)

    for t in range(T):
        for z in range(Z):
            surface[t][variance[t, z] == varianceMax[t]] = z

    for t in range(T):
        for z in range(Z):
            focus[t][surface[t] == z] = image[t, z][surface[t] == z]

    surface = surface.astype("uint8")
    focus = focus.astype("uint8")

    return surface, focus


def normalise(vid, calc, mu0):
    vid = vid.astype("float")
    (T, X, Y) = vid.shape

    for t in range(T):
        mu = vid[t, 50:450, 50:450][vid[t, 50:450, 50:450] > 0]

        if calc == "MEDIAN":
            mu = np.quantile(mu, 0.5)
        elif calc == "UPPER_Q":
            mu = np.quantile(mu, 0.75)

        ratio = mu0 / mu

        vid[t] = vid[t] * ratio
        vid[t][vid[t] > 255] = 255

    return vid.astype("uint8")


filenames, fileType = cl.getFilesType()

if True:
    for filename in filenames:

        stackFile = f"dat/{filename}/{filename}.tif"
        stack = sm.io.imread(stackFile).astype(int)

        (T, Z, C, Y, X) = stack.shape

        print("Median Filter")

        for t in range(T):
            stack[t, :, 1] = ndimage.median_filter(stack[t, :, 1], size=(3, 3, 3))

        print("Finding Surface")
        surfaceEcad = focusStack(stack[:, :, 0], 21)[0]

        print("Focussing the image stack")

        ecadFocus = focusStack(stack[:, :, 0], 9)[1]
        h2Focus = focusStack(stack[:, :, 1], 9)[1]
        surfaceH2 = focusStack(stack[:, :, 1], 3)[0]

        depth = surfaceH2.astype("float") - surfaceEcad.astype("float")
        depth[depth < 0] = 0
        depth = 100 - depth * 5
        depth[depth < 0] = 0

        surfaceH2 = np.asarray(surfaceH2, "uint8")
        tifffile.imwrite(f"dat/{filename}/surfaceH2{filename}.tif", surfaceH2)
        depth = np.asarray(depth, "uint8")
        tifffile.imwrite(f"dat/{filename}/depth{filename}.tif", depth)

        print("Normalising images")

        ecadNormalise = normalise(ecadFocus, "MEDIAN", 25)
        h2Normalise = normalise(h2Focus, "UPPER_Q", 60)

        vid = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(float)

        vid[:, :, :, 2] = depth

        vid = np.asarray(vid, "uint8")
        tifffile.imwrite(f"dat/{filename}/prepDeep{filename}.tif", vid)


def sphere(shape, radius, position):
    # https://stackoverflow.com/questions/46626267/how-to-generate-a-sphere-in-3d-numpy-array
    # assume shape and position are both a 3-tuple of int or float
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    semisizes = (radius,) * 3

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        arr += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below 1
    return arr <= 1.0


if False:
    for filename in filenames:
        dfDivisions = pd.read_pickle(f"dat/{filename}/mitosisTracks{filename}.pkl")
        df = dfDivisions[dfDivisions["Chain"] == "parent"]
        _dfSpaceTime = []

        for i in range(len(df)):

            label = df["Label"].iloc[i]
            t = df["Time"].iloc[i][-1]
            [x, y] = df["Position"].iloc[i][-1]
            ori = df["Division Orientation"].iloc[i]

            _dfSpaceTime.append(
                {
                    "Filename": filename,
                    "Label": label,
                    "Orientation": ori,
                    "T": t,
                    "X": x,
                    "Y": y,
                }
            )

        dfSpaceTime = pd.DataFrame(_dfSpaceTime)

        segmentation = np.zeros([181, 512, 512])

        T = np.array(dfSpaceTime["T"])
        X = np.array(dfSpaceTime["X"])
        Y = np.array(dfSpaceTime["Y"])

        for i in range(len(T)):

            division = sphere(
                [181, 512, 512], 10, (round(T[i]), round(512 - Y[i]), round(X[i]))
            ).astype(int)
            segmentation[division == 1] = 255

        segmentation = np.asarray(segmentation, "uint8")
        tifffile.imwrite(
            f"dat/{filename}/segmentationLabel{filename}.tif", segmentation
        )

        vid = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(float)

        vid[:, :, :, 2] = segmentation

        vid = np.asarray(vid, "uint8")
        tifffile.imwrite(f"dat/{filename}/segmentationLabelFocus{filename}.tif", vid)