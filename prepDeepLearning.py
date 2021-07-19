from re import I
import numpy as np
import os
import scipy
from scipy import ndimage
import skimage as sm
import skimage.io
import pandas as pd
import tifffile
from PIL import Image
import cv2

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

if False:
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
                [181, 512, 512], 5, (round(T[i]), round(512 - Y[i]), round(X[i]))
            ).astype(int)
            segmentation[division == 1] = 1

        segmentation = np.asarray(segmentation, "uint8")
        tifffile.imwrite(
            f"dat/{filename}/segmentationLabel{filename}.tif", segmentation
        )

        vid = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(float)

        vid[:, :, :, 2] = segmentation * 150

        vid = np.asarray(vid, "uint8")
        tifffile.imwrite(f"dat/{filename}/segmentationLabelFocus{filename}.tif", vid)


def reorder16(stack):

    out = []

    if len(stack.shape) == 3:
        T, X, Y = stack.shape
        steps = T // 8 - 1

        for i in range(4):
            for j in range(4):
                for t in range(steps):
                    img = np.zeros([515, 515])
                    substack = stack[
                        t * 8 : t * 8 + 16,
                        i * 128 : i * 128 + 128,
                        j * 128 : j * 128 + 128,
                    ]

                    for n in range(4):
                        for m in range(4):
                            img[
                                n * 129 : n * 129 + 128, m * 129 : m * 129 + 128
                            ] = substack[n * 4 + m]

                    out.append(img)
    else:
        T, X, Y, C = stack.shape
        steps = T // 8 - 1

        for i in range(4):
            for j in range(4):
                for t in range(steps):
                    img = np.zeros([515, 515, 3])
                    substack = stack[
                        t * 8 : t * 8 + 16,
                        i * 128 : i * 128 + 128,
                        j * 128 : j * 128 + 128,
                    ]

                    for n in range(4):
                        for m in range(4):
                            img[
                                n * 129 : n * 129 + 128, m * 129 : m * 129 + 128
                            ] = substack[n * 4 + m]

                    out.append(img)

    return np.asarray(out, "uint8")


# time depentant data
if False:
    for filename in filenames:
        segmentation = sm.io.imread(
            f"dat/{filename}/segmentationLabel{filename}.tif"
        ).astype(int)
        prepDeep = sm.io.imread(f"dat/{filename}/prepDeep{filename}.tif").astype(int)

        segmentation16 = reorder16(segmentation)
        prepDeep16 = reorder16(prepDeep)

        tifffile.imwrite(f"dat/{filename}/segmentation16{filename}.tif", segmentation16)
        tifffile.imwrite(f"dat/{filename}/prepDeep16{filename}.tif", prepDeep16)

        print("d")


if True:
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

        dfSpaceTime = dfSpaceTime[dfSpaceTime["T"] < 180]

        segmentation = np.zeros([180, 552, 552])

        T = np.array(dfSpaceTime["T"])
        X = np.array(dfSpaceTime["X"])
        Y = np.array(dfSpaceTime["Y"])

        for i in range(len(T)):

            rr0, cc0 = sm.draw.disk([551 - round(Y[i] + 20), round(X[i]) + 20], 10)
            segmentation[round(T[i])][rr0, cc0] = 1

        segmentation = segmentation[:, 20:532, 20:532]
        segmentation = np.asarray(segmentation, "uint8")
        tifffile.imwrite(f"train/segmentationLabel3{filename}.tif", segmentation)

        prepDeep3 = np.zeros([180, 512, 512, 3])

        h2Focus = sm.io.imread(f"dat/{filename}/h2Focus{filename}.tif").astype(int)
        ecadFocus = sm.io.imread(f"dat/{filename}/ecadFocus{filename}.tif").astype(int)

        for i in range(180):
            prepDeep3[i, :, :, 0] = h2Focus[i]
            prepDeep3[i, :, :, 2] = h2Focus[i + 1]
            prepDeep3[i, :, :, 1] = ecadFocus[i]
            # im = Image.fromarray(prepDeep3[i])
            # im.save(f"train/{filename}_{i}.jpeg")

        prepDeep3 = np.asarray(prepDeep3, "uint8")
        tifffile.imwrite(f"train/prepDeep3{filename}.tif", prepDeep3)

if True:
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

        dfSpaceTime = dfSpaceTime[dfSpaceTime["T"] < 179]

        segmentation = np.zeros([179, 552, 552])

        T = np.array(dfSpaceTime["T"])
        X = np.array(dfSpaceTime["X"])
        Y = np.array(dfSpaceTime["Y"])

        for i in range(len(T)):

            rr0, cc0 = sm.draw.disk([551 - round(Y[i] + 20), round(X[i]) + 20], 10)
            segmentation[round(T[i])][rr0, cc0] = 1

        segmentation = segmentation[:, 20:532, 20:532]
        segmentation = np.asarray(segmentation, "uint8")
        tifffile.imwrite(f"train/segmentationLabel3h2{filename}.tif", segmentation)

        prepDeep3 = np.zeros([179, 512, 512, 3])

        h2Focus = sm.io.imread(f"dat/{filename}/h2Focus{filename}.tif").astype(int)

        for i in range(179):
            prepDeep3[i, :, :, 0] = h2Focus[i]
            prepDeep3[i, :, :, 1] = h2Focus[i + 1]
            prepDeep3[i, :, :, 2] = h2Focus[i + 2]
            # im = Image.fromarray(prepDeep3[i])
            # im.save(f"train/{filename}_{i}.jpeg")

        prepDeep3 = np.asarray(prepDeep3, "uint8")
        tifffile.imwrite(f"train/prepDeep3h2{filename}.tif", prepDeep3)

if True:
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

        dfSpaceTime = dfSpaceTime[dfSpaceTime["T"] < 180]

        segmentation = np.zeros([90, 552, 552])

        T = np.array(dfSpaceTime["T"])
        X = np.array(dfSpaceTime["X"])
        Y = np.array(dfSpaceTime["Y"])

        for i in range(len(T)):

            rr0, cc0 = sm.draw.disk([551 - round(Y[i] + 20), round(X[i]) + 20], 10)
            segmentation[int(T[i] / 2)][rr0, cc0] = 1

        segmentation = segmentation[:, 20:532, 20:532]
        segmentation = np.asarray(segmentation, "uint8")
        tifffile.imwrite(f"train/segmentationLabel3-2{filename}.tif", segmentation)

        prepDeep3 = np.zeros([90, 512, 512, 3])

        h2Focus = sm.io.imread(f"dat/{filename}/h2Focus{filename}.tif").astype(int)
        ecadFocus = sm.io.imread(f"dat/{filename}/ecadFocus{filename}.tif").astype(int)
        h2Focus = h2Focus[::2]
        ecadFocus = ecadFocus[::2]

        for i in range(90):
            prepDeep3[i, :, :, 0] = h2Focus[i]
            prepDeep3[i, :, :, 2] = h2Focus[i + 1]
            prepDeep3[i, :, :, 1] = ecadFocus[i]

        prepDeep3 = np.asarray(prepDeep3, "uint8")
        tifffile.imwrite(f"train/prepDeep3-2{filename}.tif", prepDeep3)


if True:
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

        dfSpaceTime = dfSpaceTime[dfSpaceTime["T"] < 178]

        segmentation = np.zeros([89, 552, 552])

        T = np.array(dfSpaceTime["T"])
        X = np.array(dfSpaceTime["X"])
        Y = np.array(dfSpaceTime["Y"])

        for i in range(len(T)):

            rr0, cc0 = sm.draw.disk([551 - round(Y[i] + 20), round(X[i]) + 20], 10)
            segmentation[int(T[i] / 2)][rr0, cc0] = 1

        segmentation = segmentation[:, 20:532, 20:532]
        segmentation = np.asarray(segmentation, "uint8")
        tifffile.imwrite(f"train/segmentationLabel3-2h2{filename}.tif", segmentation)

        prepDeep3 = np.zeros([89, 512, 512, 3])

        h2Focus = sm.io.imread(f"dat/{filename}/h2Focus{filename}.tif").astype(int)
        h2Focus = h2Focus[::2]

        for i in range(89):
            prepDeep3[i, :, :, 0] = h2Focus[i]
            prepDeep3[i, :, :, 1] = h2Focus[i + 1]
            prepDeep3[i, :, :, 2] = h2Focus[i + 2]

        prepDeep3 = np.asarray(prepDeep3, "uint8")
        tifffile.imwrite(f"train/prepDeep3-2h2{filename}.tif", prepDeep3)

if True:
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

        dfSpaceTime = dfSpaceTime[dfSpaceTime["T"] < 180]

        segmentation = np.zeros([60, 552, 552])

        T = np.array(dfSpaceTime["T"])
        X = np.array(dfSpaceTime["X"])
        Y = np.array(dfSpaceTime["Y"])

        for i in range(len(T)):

            rr0, cc0 = sm.draw.disk([551 - round(Y[i] + 20), round(X[i]) + 20], 10)
            segmentation[int(T[i] / 3)][rr0, cc0] = 1

        segmentation = segmentation[:, 20:532, 20:532]
        segmentation = np.asarray(segmentation, "uint8")
        tifffile.imwrite(f"train/segmentationLabel3-3{filename}.tif", segmentation)

        prepDeep3 = np.zeros([60, 512, 512, 3])

        h2Focus = sm.io.imread(f"dat/{filename}/h2Focus{filename}.tif").astype(int)
        ecadFocus = sm.io.imread(f"dat/{filename}/ecadFocus{filename}.tif").astype(int)
        h2Focus = h2Focus[::3]
        ecadFocus = ecadFocus[::3]

        for i in range(60):
            prepDeep3[i, :, :, 0] = h2Focus[i]
            prepDeep3[i, :, :, 2] = h2Focus[i + 1]
            prepDeep3[i, :, :, 1] = ecadFocus[i]

        prepDeep3 = np.asarray(prepDeep3, "uint8")
        tifffile.imwrite(f"train/prepDeep3-3{filename}.tif", prepDeep3)


if True:
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

        dfSpaceTime = dfSpaceTime[dfSpaceTime["T"] < 177]

        segmentation = np.zeros([59, 552, 552])

        T = np.array(dfSpaceTime["T"])
        X = np.array(dfSpaceTime["X"])
        Y = np.array(dfSpaceTime["Y"])

        for i in range(len(T)):

            rr0, cc0 = sm.draw.disk([551 - round(Y[i] + 20), round(X[i]) + 20], 10)
            segmentation[int(T[i] / 3)][rr0, cc0] = 1

        segmentation = segmentation[:, 20:532, 20:532]
        segmentation = np.asarray(segmentation, "uint8")
        tifffile.imwrite(f"train/segmentationLabel3-3h2{filename}.tif", segmentation)

        prepDeep3 = np.zeros([59, 512, 512, 3])

        h2Focus = sm.io.imread(f"dat/{filename}/h2Focus{filename}.tif").astype(int)
        h2Focus = h2Focus[::3]

        for i in range(59):
            prepDeep3[i, :, :, 0] = h2Focus[i]
            prepDeep3[i, :, :, 1] = h2Focus[i + 1]
            prepDeep3[i, :, :, 2] = h2Focus[i + 2]

        prepDeep3 = np.asarray(prepDeep3, "uint8")
        tifffile.imwrite(f"train/prepDeep3-3h2{filename}.tif", prepDeep3)


# boundary segmentation
if False:
    overlay = sm.io.imread(f"train/deepOverlayBinaryEdits.tif").astype(int)

    binary = np.zeros([26, 512, 512])
    ecad3 = np.zeros([26, 512, 512, 3])

    for i in range(26):
        binary[i] = overlay[3 * i + 1, :, :, 0]

    filenames = [
        "WoundL18h01",
        "WoundL18h02",
        "WoundL18h03",
        "WoundL18h04",
        "WoundL18h05",
        "WoundS18h01",
        "WoundS18h02",
        "WoundS18h03",
        "WoundS18h04",
        "WoundS18h05",
        "WoundS18h06",
        "WoundS18h07",
        "WoundS18h08",
    ]

    frame = [
        [2, 14],
        [8, 129],
        [4, 83],
        [6, 123],
        [22, 85],
        [14, 136],
        [8, 156],
        [10, 120],
        [23, 107],
        [16, 4],
        [5, 26],
        [8, 136],
        [4, 35],
    ]
    t = 0
    for i in range(len(filenames)):
        filename = filenames[i]
        t1, t2 = frame[i]

        ecadFocus = sm.io.imread(f"dat/{filename}/ecadFocus{filename}.tif").astype(int)

        ecad3[t, :, :, 0] = ecadFocus[t1 - 1]
        ecad3[t, :, :, 1] = ecadFocus[t1]
        ecad3[t, :, :, 2] = ecadFocus[t1 + 1]
        t += 1

        ecad3[t, :, :, 0] = ecadFocus[t2 - 1]
        ecad3[t, :, :, 1] = ecadFocus[t2]
        ecad3[t, :, :, 2] = ecadFocus[t2 + 1]
        t += 1

    binary[binary > 0] = 1
    kernel = np.ones((2, 2), np.uint8)

    for i in range(26):
        binary[i] = cv2.dilate(binary[i], kernel, iterations=1)

    binary = np.asarray(binary, "uint8")
    tifffile.imwrite(f"train/binaryLabel.tif", binary)
    ecad3 = np.asarray(ecad3, "uint8")
    tifffile.imwrite(f"train/prepDeepEcad3.tif", ecad3)