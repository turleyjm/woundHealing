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


filenames, fileType = cl.getFilesType()

# filenames = [
#     "Unwound18h11",
#     "Unwound18h12",
#     "WoundL18h07",
#     "WoundL18h08",
#     "WoundL18h09",
#     "WoundS18h10",
#     "WoundS18h11",
#     "WoundS18h12",
#     "WoundS18h13",
# ]
# fileType = "validation"

fileType = "training"
filenames = [
    "Unwound18h01",
    "Unwound18h02",
    "Unwound18h03",
    "Unwound18h04",
    "Unwound18h05",
    "Unwound18h06",
    "Unwound18h07",
    "Unwound18h08",
    "Unwound18h09",
    "Unwound18h10",
    "WoundL18h01",
    "WoundL18h02",
    "WoundL18h03",
    "WoundL18h04",
    "WoundL18h05",
    "WoundL18h06",
    "WoundS18h01",
    "WoundS18h02",
    "WoundS18h03",
    "WoundS18h04",
    "WoundS18h05",
    "WoundS18h06",
    "WoundS18h07",
    "WoundS18h08",
    "WoundS18h09",
]


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
        # tifffile.imwrite(f"train/label1e2h1f{filename}.tif", segmentation)

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
        tifffile.imwrite(f"train/1e2h1f{filename}.tif", prepDeep3)

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
        # tifffile.imwrite(f"train/label3h1f{filename}.tif", segmentation)

        prepDeep3 = np.zeros([179, 512, 512, 3])

        h2Focus = sm.io.imread(f"dat/{filename}/h2Focus{filename}.tif").astype(int)

        for i in range(179):
            prepDeep3[i, :, :, 0] = h2Focus[i]
            prepDeep3[i, :, :, 1] = h2Focus[i + 1]
            prepDeep3[i, :, :, 2] = h2Focus[i + 2]
            # im = Image.fromarray(prepDeep3[i])
            # im.save(f"train/{filename}_{i}.jpeg")

        prepDeep3 = np.asarray(prepDeep3, "uint8")
        tifffile.imwrite(f"train/3h1f{filename}.tif", prepDeep3)


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
        # tifffile.imwrite(f"train/label1e2h2f{filename}.tif", segmentation)

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
        tifffile.imwrite(f"train/1e2h2f{filename}.tif", prepDeep3)


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
        # tifffile.imwrite(f"train/label3h2f{filename}.tif", segmentation)

        prepDeep3 = np.zeros([89, 512, 512, 3])

        h2Focus = sm.io.imread(f"dat/{filename}/h2Focus{filename}.tif").astype(int)
        h2Focus = h2Focus[::2]

        for i in range(89):
            prepDeep3[i, :, :, 0] = h2Focus[i]
            prepDeep3[i, :, :, 1] = h2Focus[i + 1]
            prepDeep3[i, :, :, 2] = h2Focus[i + 2]

        prepDeep3 = np.asarray(prepDeep3, "uint8")
        tifffile.imwrite(f"train/3h2f{filename}.tif", prepDeep3)

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
        # tifffile.imwrite(f"train/label1e2h3f{filename}.tif", segmentation)

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
        tifffile.imwrite(f"train/1e2h3f{filename}.tif", prepDeep3)


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
        # tifffile.imwrite(f"train/label3h3f{filename}.tif", segmentation)

        prepDeep3 = np.zeros([59, 512, 512, 3])

        h2Focus = sm.io.imread(f"dat/{filename}/h2Focus{filename}.tif").astype(int)
        h2Focus = h2Focus[::3]

        for i in range(59):
            prepDeep3[i, :, :, 0] = h2Focus[i]
            prepDeep3[i, :, :, 1] = h2Focus[i + 1]
            prepDeep3[i, :, :, 2] = h2Focus[i + 2]

        prepDeep3 = np.asarray(prepDeep3, "uint8")
        tifffile.imwrite(f"train/3h3f{filename}.tif", prepDeep3)


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