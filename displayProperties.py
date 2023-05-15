import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage as sm
from skimage import color
import tifffile
import skimage.io
import skimage.draw
import utils as util
import cv2
import shutil
import cellProperties as cell
from shapely.geometry import Polygon
from PIL import ImageColor
from PIL import Image
from skimage.morphology import square, erosion

plt.rcParams.update({"font.size": 20})

# -------------------


def angleDiff(theta, phi):

    diff = theta - phi

    if abs(diff) > 90:
        if diff > 0:
            diff = 180 - diff
        else:
            diff = 180 + diff

    return abs(diff)


def rgb2gray(rgb):

    r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
    gray = 0.4 * r + 0.5870 * g + 0.1140 * b

    return gray


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros(
            (image.shape[0] + padding * 2, image.shape[1] + padding * 2)
        )
        imagePadded[
            int(padding) : int(-1 * padding), int(padding) : int(-1 * padding)
        ] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (
                            kernel * imagePadded[x : x + xKernShape, y : y + yKernShape]
                        ).sum()
                except:
                    break

    return output


# -------------------

filenames, fileType = util.getFilesType()
# filename = "prettyWound"
scale = 123.26 / 512
T = 93

# Display divisons
if False:
    for filename in filenames:
        focus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)
        dfDivisions = pd.read_pickle(f"dat/{filename}/dfDivision{filename}.pkl")

        (T, X, Y, rgb) = focus.shape

        divisions = np.zeros([T, 552, 552, 3])

        for x in range(X):
            for y in range(Y):
                divisions[:, 20 + x, 20 + y, :] = focus[:, x, y, :]

        for i in range(len(dfDivisions)):

            t0 = dfDivisions["T"].iloc[i]
            ori = np.pi * dfDivisions["Orientation"].iloc[i] / 180
            (x, y) = (dfDivisions["X"].iloc[i], dfDivisions["Y"].iloc[i])
            x = int(x)
            y = int(y)

            rr0, cc0 = sm.draw.disk([551 - (y + 20), x + 20], 16)
            rr1, cc1 = sm.draw.disk([551 - (y + 20), x + 20], 11)

            rr2, cc2, val = sm.draw.line_aa(
                int(551 - (y + 17 * np.sin(ori) + 20)),
                int(x + 17 * np.cos(ori) + 20),
                int(551 - (y - 17 * np.sin(ori) + 20)),
                int(x - 17 * np.cos(ori) + 20),
            )

            times = [t0, int(t0 + 1)]

            timeVid = []
            for t in times:
                if t >= 0 and t <= T - 1:
                    timeVid.append(t)

            for t in timeVid:
                divisions[t][rr0, cc0, 2] = 200
                divisions[t][rr1, cc1, 2] = 0
                divisions[t][rr2, cc2, 2] = 200

        divisions = divisions[:, 20:532, 20:532]

        divisions = np.asarray(divisions, "uint8")
        tifffile.imwrite(
            f"results/displayProperties/divisionsDisplay{filename}.tif", divisions
        )

# orientation to wound
if False:
    for filename in filenames:
        focus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)
        dfDivisions = pd.read_pickle(f"dat/{filename}/dfDivision{filename}.pkl")

        (T, X, Y, rgb) = focus.shape
        gray = rgb2gray(focus)
        gray = gray * (255 / np.max(gray))
        # gray = np.asarray(gray, "uint8")
        # tifffile.imwrite(f"results/gray{filename}.tif", gray)

        divisions = np.zeros([T, 552, 552, 3])

        dfShape = pd.read_pickle(f"dat/{filename}/shape{filename}.pkl")
        Q = np.mean(dfShape["q"])
        theta0 = 0.5 * np.arctan2(Q[1, 0], Q[0, 0])

        t0 = util.findStartTime(filename)
        if "Wound" in filename:
            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
            dist = sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)
            for t in range(T):
                (xc, yc) = dfWound["Position"].iloc[t]
                rr0, cc0 = sm.draw.disk([551 - (yc + 20), xc + 20], 5)
                divisions[t][rr0, cc0, 0] = 255
                divisions[t][rr0, cc0, 1] = 255
                divisions[t][rr0, cc0, 2] = 255

            for i in range(len(dfDivisions)):
                t = dfDivisions["T"].iloc[i]
                (x_w, y_w) = dfWound["Position"].iloc[t]
                x = dfDivisions["X"].iloc[i]
                y = dfDivisions["Y"].iloc[i]
                ori = (dfDivisions["Orientation"].iloc[i] - theta0 * 180 / np.pi) % 180
                theta = (np.arctan2(y - y_w, x - x_w) - theta0) * 180 / np.pi
                ori_w = (ori - theta) % 180
                if ori_w > 90:
                    ori_w = 180 - ori_w

                t0 = dfDivisions["T"].iloc[i]
                ori = np.pi * dfDivisions["Orientation"].iloc[i] / 180
                (x, y) = (dfDivisions["X"].iloc[i], dfDivisions["Y"].iloc[i])
                x = int(x)
                y = int(y)

                rr0, cc0 = sm.draw.disk([551 - (y + 20), x + 20], 16)
                rr1, cc1 = sm.draw.disk([551 - (y + 20), x + 20], 11)
                rr2, cc2, val = sm.draw.line_aa(
                    int(551 - (y + 17 * np.sin(ori) + 20)),
                    int(x + 17 * np.cos(ori) + 20),
                    int(551 - (y - 17 * np.sin(ori) + 20)),
                    int(x - 17 * np.cos(ori) + 20),
                )

                times = [t0, int(t0 + 1), int(t0 + 2), int(t0 + 3)]

                timeVid = []
                for t in times:
                    if t >= 0 and t <= T - 1:
                        timeVid.append(t)

                blue = 255 * gaussian(ori_w, 0, 45)
                red = 255 * gaussian(ori_w, 90, 45)
                green = 155 * gaussian(ori_w, 45, 10)
                blue, red, green = (
                    np.array([blue, red, green]) * 255 / np.max([blue, red, green])
                )
                for t in timeVid:
                    divisions[t][rr0, cc0, 0] = red
                    divisions[t][rr1, cc1, 0] = 0
                    divisions[t][rr2, cc2, 0] = red
                    divisions[t][rr0, cc0, 1] = green
                    divisions[t][rr1, cc1, 1] = 0
                    divisions[t][rr2, cc2, 1] = green
                    divisions[t][rr0, cc0, 2] = blue
                    divisions[t][rr1, cc1, 2] = 0
                    divisions[t][rr2, cc2, 2] = blue
        else:
            dfVelocityMean = pd.read_pickle(f"databases/dfVelocityMean{fileType}.pkl")
            dfFilename = dfVelocityMean[
                dfVelocityMean["Filename"] == filename
            ].reset_index()
            for t in range(T):
                mig = np.sum(
                    np.stack(np.array(dfFilename.loc[:t, "V"]), axis=0), axis=0
                )
                xc = int(255 + mig[0] / scale)
                yc = int(255 + mig[1] / scale)
                rr0, cc0 = sm.draw.disk([551 - (yc + 20), xc + 20], 5)
                divisions[t][rr0, cc0, 0] = 255
                divisions[t][rr0, cc0, 1] = 255
                divisions[t][rr0, cc0, 2] = 255

            for i in range(len(dfDivisions)):
                t = dfDivisions["T"].iloc[i]
                mig = np.sum(
                    np.stack(np.array(dfFilename.loc[:t, "V"]), axis=0), axis=0
                )
                xc = 255 + mig[0] / scale
                yc = 255 + mig[1] / scale
                x = dfDivisions["X"].iloc[i]
                y = dfDivisions["Y"].iloc[i]
                ori = (dfDivisions["Orientation"].iloc[i] - theta0 * 180 / np.pi) % 180
                theta = (np.arctan2(y - yc, x - xc) - theta0) * 180 / np.pi
                ori_w = (ori - theta) % 180
                if ori_w > 90:
                    ori_w = 180 - ori_w

                t0 = dfDivisions["T"].iloc[i]
                ori = np.pi * dfDivisions["Orientation"].iloc[i] / 180
                (x, y) = (dfDivisions["X"].iloc[i], dfDivisions["Y"].iloc[i])
                x = int(x)
                y = int(y)

                rr0, cc0 = sm.draw.disk([551 - (y + 20), x + 20], 16)
                rr1, cc1 = sm.draw.disk([551 - (y + 20), x + 20], 11)
                rr2, cc2, val = sm.draw.line_aa(
                    int(551 - (y + 17 * np.sin(ori) + 20)),
                    int(x + 17 * np.cos(ori) + 20),
                    int(551 - (y - 17 * np.sin(ori) + 20)),
                    int(x - 17 * np.cos(ori) + 20),
                )

                times = [t0, int(t0 + 1), int(t0 + 2), int(t0 + 3)]

                timeVid = []
                for t in times:
                    if t >= 0 and t <= T - 1:
                        timeVid.append(t)

                blue = 255 * gaussian(ori_w, 0, 45)
                red = 255 * gaussian(ori_w, 90, 45)
                green = 155 * gaussian(ori_w, 45, 10)
                blue, red, green = (
                    np.array([blue, red, green]) * 255 / np.max([blue, red, green])
                )
                for t in timeVid:
                    divisions[t][rr0, cc0, 0] = red
                    divisions[t][rr1, cc1, 0] = 0
                    divisions[t][rr2, cc2, 0] = red
                    divisions[t][rr0, cc0, 1] = green
                    divisions[t][rr1, cc1, 1] = 0
                    divisions[t][rr2, cc2, 1] = green
                    divisions[t][rr0, cc0, 2] = blue
                    divisions[t][rr1, cc1, 2] = 0
                    divisions[t][rr2, cc2, 2] = blue

        divisions = divisions[:, 20:532, 20:532]

        mask = np.all((divisions - np.zeros(3)) == 0, axis=3)

        divisions[:, :, :, 0][mask] = gray[mask]
        divisions[:, :, :, 1][mask] = gray[mask]
        divisions[:, :, :, 2][mask] = gray[mask]

        divisions = np.asarray(divisions, "uint8")
        tifffile.imwrite(
            f"results/displayProperties/orientationWound{filename}.tif", divisions
        )

# orientation to of division shape and TCJs
if True:
    dfDivisionShape = pd.read_pickle(f"databases/dfDivisionShape{fileType}.pkl")
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    dfDivisionTrack = dfDivisionTrack[dfDivisionTrack["Type"] == "parent"]
    dfShape = pd.read_pickle(f"databases/dfShape{fileType}.pkl")
    for filename in filenames:
        focus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)
        tracks = sm.io.imread(f"dat/{filename}/binaryTracks{filename}.tif").astype(int)

        (T, X, Y, rgb) = focus.shape
        gray = rgb2gray(focus)
        gray = gray * (255 / np.max(gray))
        # gray = np.asarray(gray, "uint8")
        # tifffile.imwrite(f"results/gray{filename}.tif", gray)

        divisions = np.zeros([T, 552, 552, 3])
        divisions_tcj = np.zeros([T, 552, 552, 3])

        dfShape = pd.read_pickle(f"dat/{filename}/shape{filename}.pkl")
        Q = np.mean(dfShape["q"])
        theta0 = 0.5 * np.arctan2(Q[1, 0], Q[0, 0])

        dfFileShape = dfDivisionShape[dfDivisionShape["Filename"] == filename]
        dfFileShape = dfFileShape[dfFileShape["Track length"] > 18]
        df = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        labels = list(dfFileShape["Label"])
        for label in labels:
            dfDiv = df[df["Label"] == label]
            tcjs = dfDiv["TCJ"][dfDiv["Division Time"] == -15].iloc[0]
            if tcjs != False:
                t15 = dfDiv["Time"][dfDiv["Division Time"] == -15].iloc[0]

                ori = dfFileShape["Orientation"][dfFileShape["Label"] == label].iloc[0]
                oriPre = dfDiv["Orientation"][dfDiv["Division Time"] == -15].iloc[0]
                SfPre = dfDiv["Shape Factor"][dfDiv["Division Time"] == -15].iloc[0]
                oriPre_tcj = dfDiv["Orientation tcj"][
                    dfDiv["Division Time"] == -15
                ].iloc[0]
                SfPre_tcj = dfDiv["Shape Factor tcj"][
                    dfDiv["Division Time"] == -15
                ].iloc[0]
                x = int(dfFileShape["X"][dfFileShape["Label"] == label].iloc[0])
                y = int(dfFileShape["Y"][dfFileShape["Label"] == label].iloc[0])
                t0 = int(
                    dfFileShape["Anaphase Time"][dfFileShape["Label"] == label].iloc[0]
                )

                times = [t0, int(t0 + 1), int(t0 + 2), int(t0 + 3)]

                timeVid = []
                for t in times:
                    if t >= 0 and t <= T - 1:
                        timeVid.append(t)

                rr0, cc0, val = sm.draw.line_aa(
                    int(551 - (y + 17 * np.sin(ori * np.pi / 180) + 20)),
                    int(x + 17 * np.cos(ori * np.pi / 180) + 20),
                    int(551 - (y - 17 * np.sin(ori * np.pi / 180) + 20)),
                    int(x - 17 * np.cos(ori * np.pi / 180) + 20),
                )
                rr1, cc1, val = sm.draw.line_aa(
                    int(551 - (y + 17 * np.sin(oriPre * np.pi / 180) + 20)),
                    int(x + 17 * np.cos(oriPre * np.pi / 180) + 20),
                    int(551 - (y - 17 * np.sin(oriPre * np.pi / 180) + 20)),
                    int(x - 17 * np.cos(oriPre * np.pi / 180) + 20),
                )

                andDiff = angleDiff(ori, oriPre)
                blue = 255 * gaussian(andDiff, 0, 45)
                red = 255 * gaussian(andDiff, 90, 45)
                green = 155 * gaussian(andDiff, 45, 10)
                blue, red, green = (
                    np.array([blue, red, green]) * 255 / np.max([blue, red, green])
                )
                for t in timeVid:
                    divisions[t][rr0, cc0, 0] = red
                    divisions[t][rr1, cc1, 0] = 255
                    divisions[t][rr0, cc0, 1] = green
                    divisions[t][rr1, cc1, 1] = 255
                    divisions[t][rr0, cc0, 2] = blue
                    divisions[t][rr1, cc1, 2] = 255

                rr0, cc0, val = sm.draw.line_aa(
                    int(551 - (y + 17 * np.sin(ori * np.pi / 180) + 20)),
                    int(x + 17 * np.cos(ori * np.pi / 180) + 20),
                    int(551 - (y - 17 * np.sin(ori * np.pi / 180) + 20)),
                    int(x - 17 * np.cos(ori * np.pi / 180) + 20),
                )
                rr1, cc1, val = sm.draw.line_aa(
                    int(551 - (y + 14 * np.sin(oriPre_tcj * np.pi / 180) + 20)),
                    int(x + 14 * np.cos(oriPre_tcj * np.pi / 180) + 20),
                    int(551 - (y - 14 * np.sin(oriPre_tcj * np.pi / 180) + 20)),
                    int(x - 14 * np.cos(oriPre_tcj * np.pi / 180) + 20),
                )

                andDiff = angleDiff(ori, oriPre_tcj)
                blue = 255 * gaussian(andDiff, 0, 45)
                red = 255 * gaussian(andDiff, 90, 45)
                green = 155 * gaussian(andDiff, 45, 10)
                blue, red, green = (
                    np.array([blue, red, green]) * 255 / np.max([blue, red, green])
                )
                for t in timeVid:
                    divisions_tcj[t][rr0, cc0, 0] = red
                    divisions_tcj[t][rr1, cc1, 0] = 255
                    divisions_tcj[t][rr0, cc0, 1] = green
                    divisions_tcj[t][rr1, cc1, 1] = 255
                    divisions_tcj[t][rr0, cc0, 2] = blue
                    divisions_tcj[t][rr1, cc1, 2] = 255

                x_tcj = int(np.mean(np.array(tcjs)[:, 0]))
                y_tcj = int(np.mean(np.array(tcjs)[:, 1]))
                rr1, cc1, val = sm.draw.line_aa(
                    int(551 - (y_tcj + 14 * np.sin(oriPre_tcj * np.pi / 180) + 20)),
                    int(x_tcj + 14 * np.cos(oriPre_tcj * np.pi / 180) + 20),
                    int(551 - (y_tcj - 14 * np.sin(oriPre_tcj * np.pi / 180) + 20)),
                    int(x_tcj - 14 * np.cos(oriPre_tcj * np.pi / 180) + 20),
                )
                divisions_tcj[t15][rr1, cc1, 0] = 255
                divisions_tcj[t15][rr1, cc1, 1] = 255
                divisions_tcj[t15][rr1, cc1, 2] = 255
                for tcj in tcjs:
                    rr2, cc2 = sm.draw.disk(
                        [551 - (int(tcj[1]) + 20), int(tcj[0]) + 20], 2
                    )
                    divisions_tcj[t15][rr2, cc2, 1] = 255

                rr1, cc1, val = sm.draw.line_aa(
                    int(551 - (y_tcj + 14 * np.sin(oriPre * np.pi / 180) + 20)),
                    int(x_tcj + 14 * np.cos(oriPre * np.pi / 180) + 20),
                    int(551 - (y_tcj - 14 * np.sin(oriPre * np.pi / 180) + 20)),
                    int(x_tcj - 14 * np.cos(oriPre * np.pi / 180) + 20),
                )
                divisions[t15][rr1, cc1, 0] = 255
                divisions[t15][rr1, cc1, 1] = 255
                divisions[t15][rr1, cc1, 2] = 255
                colour = tracks[t, 511 - y, x]
                if np.all(colour != np.array([255, 255, 255])):
                    divisions[t15, 20:532, 20:532, 1][
                        np.all((tracks[t15] - colour) == 0, axis=2)
                    ] = 255

        divisions = divisions[:, 20:532, 20:532]

        mask = np.all((divisions - np.zeros(3)) == 0, axis=3)

        divisions[:, :, :, 0][mask] = gray[mask]
        divisions[:, :, :, 1][mask] = gray[mask]
        divisions[:, :, :, 2][mask] = gray[mask]

        divisions = np.asarray(divisions, "uint8")
        tifffile.imwrite(
            f"results/displayProperties/orientationShape{filename}.tif", divisions
        )

        divisions_tcj = divisions_tcj[:, 20:532, 20:532]

        mask = np.all((divisions_tcj - np.zeros(3)) == 0, axis=3)

        divisions_tcj[:, :, :, 0][mask] = gray[mask]
        divisions_tcj[:, :, :, 1][mask] = gray[mask]
        divisions_tcj[:, :, :, 2][mask] = gray[mask]

        divisions_tcj = np.asarray(divisions_tcj, "uint8")
        tifffile.imwrite(
            f"results/displayProperties/orientationShape_tcj{filename}.tif",
            divisions_tcj,
        )

# course grain velocity feild
if False:
    grid = 12

    df = pd.read_pickle(f"dat/{filename}/nucleusVelocity{filename}.pkl")

    util.createFolder("results/video/")
    for t in range(T - 4):
        _dft = df[df["T"] >= t]
        dft = _dft[_dft["T"] < t + 4]

        a = util.ThreeD(grid)

        for i in range(grid):
            for j in range(grid):
                x = [(512 / grid) * j, (512 / grid) * (j + 1)]
                y = [(512 / grid) * i, (512 / grid) * (i + 1)]
                dfxy = util.sortGrid(dft, x, y)
                if len(dfxy) == 0:
                    a[i][j] = np.array([0, 0])
                else:
                    a[i][j] = np.mean(dfxy["Velocity"], axis=0)

        x, y = np.meshgrid(
            np.linspace(0, 512 * scale, grid),
            np.linspace(0, 512 * scale, grid),
        )

        u = np.zeros([grid, grid])
        v = np.zeros([grid, grid])

        for i in range(grid):
            for j in range(grid):
                u[i, j] = a[i][j][0]
                v[i, j] = a[i][j][1]

        fig, ax = plt.subplots(figsize=(5, 5))
        plt.quiver(x, y, u, v, scale=50)
        plt.title(f"time = {t}")
        fig.savefig(
            f"results/video/Velocity field {t}",
            dpi=300,
            transparent=True,
        )
        plt.close("all")

        # make video
    img_array = []

    for t in range(T - 4):
        img = cv2.imread(f"results/video/Velocity field {t}.png")
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(
        f"results/displayProperties/Velocity field {filename}.mp4",
        cv2.VideoWriter_fourcc(*"DIVX"),
        3,
        size,
    )
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
    cv2.destroyAllWindows()

    shutil.rmtree("results/video")

# heatmap shape factor
if False:

    binary = sm.io.imread(f"dat/{filename}/binary1{filename}.tif").astype(int)

    img_xy = fi.imgrcxy(binary)
    img_xy = 255 - img_xy

    img_label = sm.measure.label(img_xy, background=0, connectivity=1)
    # img_label = np.asarray(img_label, "uint16")
    # tifffile.imwrite(f"results/img_label{filename}.tif", img_label)
    img_labels = np.unique(img_label)[1:]
    all_polys = []
    all_contours = []
    heatmap = np.zeros([512, 512])

    for label in img_labels:
        contour = sm.measure.find_contours(img_label == label, level=0)[0]
        poly = sm.measure.approximate_polygon(contour, tolerance=1)

        if label == 364:
            heatmap[img_label == label] = 0
        else:
            if poly != []:
                polygon = Polygon(poly)
                if polygon.area != 0:
                    heatmap[img_label == label] = cell.shapeFactor(polygon)

    heatmap[heatmap == 0] = np.nan
    t = 0

    fig, ax = plt.subplots(figsize=(20, 20))
    pos = ax.imshow(heatmap)
    fig.colorbar(pos)
    fig.suptitle(f"")
    # plt.title(f"Heat map of {function_title}")
    fig.savefig(
        f"results/displayProperties/heatmap {filename} {t}.png",
        dpi=300,
        transparent=True,
    )
    plt.close()

# Deep learning 3 frame
if False:

    focus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)
    h2Focus = focus[:, :, :, 0]
    ecadFocus = focus[:, :, :, 1]

    T = h2Focus.shape[0]

    prepDeep3 = np.zeros([T - 1, 512, 512, 3])

    for i in range(T - 1):
        prepDeep3[i, :, :, 0] = h2Focus[i]
        prepDeep3[i, :, :, 2] = h2Focus[i + 1]
        prepDeep3[i, :, :, 1] = ecadFocus[i]

    prepDeep3 = np.asarray(prepDeep3, "uint8")
    tifffile.imwrite(f"results/displayProperties/deepLearning{filename}.tif", prepDeep3)

# Director Q field
if False:
    for filename in filenames:
        focus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)
        binary = sm.io.imread(f"dat/{filename}/binary{filename}.tif").astype(int)

        (T, X, Y, rgb) = focus.shape
        gray = rgb2gray(focus)
        gray = gray * (255 / np.max(gray))
        # gray = np.asarray(gray, "uint8")
        # tifffile.imwrite(f"results/gray{filename}.tif", gray)

        directorNorm = np.zeros([T, 572, 572, 3])
        director = np.zeros([T, 572, 572, 3])

        dfShape = pd.read_pickle(f"dat/{filename}/shape{filename}.pkl")

        for i in range(len(dfShape)):
            x = dfShape["Centroid"].iloc[i][0]
            y = dfShape["Centroid"].iloc[i][1]
            t = dfShape["Time"].iloc[i]
            q = dfShape["q"].iloc[i]
            ori = np.arctan2(q[0, 1], q[0, 0]) / 2
            q_0 = (q[0, 0] ** 2 + q[0, 1] ** 2) ** 0.5
            if q_0 > 0.4:
                q_0 = 0.4

            rr0, cc0, val = sm.draw.line_aa(
                int(571 - (y + 100 * q_0 * np.sin(ori) + 30)),
                int(x + 100 * q_0 * np.cos(ori) + 30),
                int(571 - (y - 100 * q_0 * np.sin(ori) + 30)),
                int(x - 100 * q_0 * np.cos(ori) + 30),
            )
            director[t][rr0, cc0, 0] = 255

            rr0, cc0, val = sm.draw.line_aa(
                int(571 - (y + 6 * np.sin(ori) + 30)),
                int(x + 6 * np.cos(ori) + 30),
                int(571 - (y - 6 * np.sin(ori) + 30)),
                int(x - 6 * np.cos(ori) + 30),
            )
            directorNorm[t][rr0, cc0, 0] = 255

        director = director[:, 30:542, 30:542]

        director[:, :, :, 1][binary == 255] = 200

        mask = np.all((director - np.zeros(3)) == 0, axis=3)

        director[:, :, :, 0][mask] = gray[mask]
        director[:, :, :, 1][mask] = gray[mask]
        director[:, :, :, 2][mask] = gray[mask]

        director = np.asarray(director, "uint8")
        tifffile.imwrite(f"results/displayProperties/director{filename}.tif", director)

        directorNorm = directorNorm[:, 30:542, 30:542]
        directorNorm[:, :, :, 1][binary == 255] = 200

        mask = np.all((directorNorm - np.zeros(3)) == 0, axis=3)

        directorNorm[:, :, :, 0][mask] = gray[mask]
        directorNorm[:, :, :, 1][mask] = gray[mask]
        directorNorm[:, :, :, 2][mask] = gray[mask]

        directorNorm = np.asarray(directorNorm, "uint8")
        tifffile.imwrite(
            f"results/displayProperties/directorNorm{filename}.tif", directorNorm
        )

# Polar field
if False:
    for filename in filenames:
        focus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)
        binary = sm.io.imread(f"dat/{filename}/binary{filename}.tif").astype(int)

        (T, X, Y, rgb) = focus.shape
        gray = rgb2gray(focus)
        gray = gray * (255 / np.max(gray))
        # gray = np.asarray(gray, "uint8")
        # tifffile.imwrite(f"results/gray{filename}.tif", gray)

        polar = np.zeros([T, 552, 552, 3])

        df = pd.read_pickle(f"dat/{filename}/shape{filename}.pkl")

        for i in range(len(df)):
            [x, y] = [
                df["Centroid"].iloc[i][0],
                df["Centroid"].iloc[i][1],
            ]
            t = int(df["Time"].iloc[i])
            p = df["Polar"].iloc[i]
            ori = np.arctan2(p[1], p[0])
            p_0 = (p[0] ** 2 + p[1] ** 2) ** 0.5
            if p_0 > 0.015:
                p_0 = 0.015

            rr0, cc0, val = sm.draw.line_aa(
                int(531 - (y + 400 * p_0 * np.sin(ori))),
                int(x + 400 * p_0 * np.cos(ori) + 20),
                int(531 - y),
                int(x + 20),
            )
            rr1, cc1 = sm.draw.disk([551 - (y + 20), x + 20], 2)
            polar[t][rr0, cc0, 0] = 255
            polar[t][rr1, cc1, 2] = 255
            polar[t][rr0, cc0, 2] = 0

        polar = polar[:, 20:532, 20:532]
        polar[:, :, :, 1][binary == 255] = 200

        mask = np.all((polar - np.zeros(3)) == 0, axis=3)

        polar[:, :, :, 0][mask] = gray[mask]
        polar[:, :, :, 1][mask] = gray[mask]
        polar[:, :, :, 2][mask] = gray[mask]

        polar = np.asarray(polar, "uint8")
        tifffile.imwrite(f"results/displayProperties/polar{filename}.tif", polar)

# Velocity field
if False:
    for filename in filenames:
        focus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)

        (T, X, Y, rgb) = focus.shape
        gray = rgb2gray(focus)
        gray = gray * (255 / np.max(gray))
        # gray = np.asarray(gray, "uint8")
        # tifffile.imwrite(f"results/gray{filename}.tif", gray)

        velocity = np.zeros([T, 552, 552, 3])
        velocityNorm = np.zeros([T, 552, 552, 3])

        dfVelocity = pd.read_pickle(f"dat/{filename}/nucleusVelocity{filename}.pkl")

        for i in range(len(dfVelocity)):
            x = dfVelocity["X"].iloc[i]
            y = dfVelocity["Y"].iloc[i]
            t = int(dfVelocity["T"].iloc[i])
            v = dfVelocity["Velocity"].iloc[i]
            ori = np.arctan2(v[1], v[0])
            v_0 = (v[0] ** 2 + v[1] ** 2) ** 0.5

            rr0, cc0, val = sm.draw.line_aa(
                int(531 - (y + 2 * v_0 * np.sin(ori))),
                int(x + 2 * v_0 * np.cos(ori) + 20),
                int(531 - y),
                int(x + 20),
            )
            rr1, cc1 = sm.draw.disk([551 - (y + 20), x + 20], 2)
            velocity[t][rr0, cc0, 0] = 255
            velocity[t][rr1, cc1, 2] = 255

            rr0, cc0, val = sm.draw.line_aa(
                int(531 - (y + 10 * np.sin(ori))),
                int(x + 10 * np.cos(ori) + 20),
                int(531 - y),
                int(x + 20),
            )
            velocityNorm[t][rr0, cc0, 0] = 255
            velocityNorm[t][rr1, cc1, 2] = 255

        velocity = velocity[:, 20:532, 20:532]

        mask = np.all((velocity - np.zeros(3)) == 0, axis=3)

        velocity[:, :, :, 0][mask] = gray[mask]
        velocity[:, :, :, 1][mask] = gray[mask]
        velocity[:, :, :, 2][mask] = gray[mask]

        velocity = np.asarray(velocity, "uint8")
        tifffile.imwrite(f"results/displayProperties/velocity{filename}.tif", velocity)

        velocityNorm = velocityNorm[:, 20:532, 20:532]

        mask = np.all((velocityNorm - np.zeros(3)) == 0, axis=3)

        velocityNorm[:, :, :, 0][mask] = gray[mask]
        velocityNorm[:, :, :, 1][mask] = gray[mask]
        velocityNorm[:, :, :, 2][mask] = gray[mask]

        velocityNorm = np.asarray(velocityNorm, "uint8")
        tifffile.imwrite(
            f"results/displayProperties/velocityNorm{filename}.tif", velocityNorm
        )

# dv field
if False:
    for filename in filenames:
        focus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)

        (T, X, Y, rgb) = focus.shape
        gray = rgb2gray(focus)
        gray = gray * (255 / np.max(gray))
        # gray = np.asarray(gray, "uint8")
        # tifffile.imwrite(f"results/gray{filename}.tif", gray)

        velocity = np.zeros([T, 552, 552, 3])

        dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
        dfVelocity = pd.read_pickle(f"dat/{filename}/nucleusVelocity{filename}.pkl")

        if "Wound" in filename:
            for t in range(T):
                dft = dfVelocity[dfVelocity["T"] == t]

                xw, yw = dfWound["Position"].iloc[t]
                V = np.mean(dft["Velocity"])

                for i in range(len(dft)):
                    x = dft["X"].iloc[i]
                    y = dft["Y"].iloc[i]
                    dv = dft["Velocity"].iloc[i] - V

                    ori = np.arctan2(dv[1], dv[0])
                    v_0 = (dv[0] ** 2 + dv[1] ** 2) ** 0.5

                    rr0, cc0, val = sm.draw.line_aa(
                        int(531 - (y + 2 * v_0 * np.sin(ori))),
                        int(x + 2 * v_0 * np.cos(ori) + 20),
                        int(531 - y),
                        int(x + 20),
                    )
                    rr1, cc1 = sm.draw.disk([551 - (y + 20), x + 20], 2)
                    velocity[t][rr0, cc0, 0] = 255
                    velocity[t][rr1, cc1, 2] = 255
                    velocity[t][rr0, cc0, 2] = 0

        velocity = velocity[:, 20:532, 20:532]

        mask = np.all((velocity - np.zeros(3)) == 0, axis=3)

        velocity[:, :, :, 0][mask] = gray[mask]
        velocity[:, :, :, 1][mask] = gray[mask]
        velocity[:, :, :, 2][mask] = gray[mask]

        velocity = np.asarray(velocity, "uint8")
        tifffile.imwrite(
            f"results/displayProperties/deltaVelocity{filename}.tif", velocity
        )

# T1s display
if False:
    for filename in filenames:
        focus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)
        T1s = sm.io.imread(f"dat/{filename}/T1s{filename}.tif").astype(int)

        (T, X, Y, rgb) = focus.shape
        # gray = np.asarray(gray, "uint8")
        # tifffile.imwrite(f"results/gray{filename}.tif", gray)

        T1display = np.zeros([T, 5, X, Y])

        T1display[:, 1] = focus[:, :, :, 1]
        redMask = T1s[:, :, :, 0]
        greenMask = T1s[:, :, :, 1]
        for t in range(T):
            redMask[t] = erosion(redMask[t], square(3))
            greenMask[t] = erosion(greenMask[t], square(3))

        T1display[:, 0][redMask == 255] = 255 * 0.6
        T1display[:, 1][greenMask == 255] = 255 * 0.6
        T1display[:, 0][greenMask == 255] = 255 * 0.6

        T1display = np.asarray(T1display, "uint8")
        tifffile.imwrite(
            f"results/displayProperties/T1sdisplay{filename}.tif",
            T1display,
            imagej=True,
            metadata={"axes": "TCYX"},
        )

# display division tracks
if False:
    dfDivisionTrack = pd.read_pickle(f"databases/dfDivisionTrack{fileType}.pkl")
    for filename in filenames:
        focus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)
        tracks = sm.io.imread(f"dat/{filename}/binaryTracks{filename}.tif").astype(int)
        dfFile = dfDivisionTrack[dfDivisionTrack["Filename"] == filename]
        df = dfFile[dfFile["Type"] == "parent"]

        (T, X, Y, rgb) = focus.shape

        for i in range(len(df)):

            colour = df["Colour"].iloc[i]
            t = df["Time"].iloc[i]
            focus[t, :, :, 2][np.all((tracks[t] - colour) == 0, axis=2)] = 255

        df = dfFile[dfFile["Type"] == "daughter1"]

        for i in range(len(df)):

            colour = df["Colour"].iloc[i]
            t = df["Time"].iloc[i]
            focus[t, :, :, 2][np.all((tracks[t] - colour) == 0, axis=2)] = 200
            focus[t, :, :, 0][np.all((tracks[t] - colour) == 0, axis=2)] += 150
            focus[t, :, :, 0][focus[t, :, :, 0] > 255] = 255

        df = dfFile[dfFile["Type"] == "daughter2"]

        for i in range(len(df)):

            colour = df["Colour"].iloc[i]
            t = df["Time"].iloc[i]
            focus[t, :, :, 2][np.all((tracks[t] - colour) == 0, axis=2)] = 200
            focus[t, :, :, 1][np.all((tracks[t] - colour) == 0, axis=2)] += 150
            focus[t, :, :, 1][focus[t, :, :, 1] > 255] = 255

        focus = np.asarray(focus, "uint8")
        tifffile.imwrite(
            f"results/displayProperties/divisionsTracks{filename}.tif", focus
        )

# filter applied to image (from in figure folder)
if False:
    blur3 = sm.io.imread(f"dat/Unwound18h13_3.tif").astype(int)
    image = np.zeros([74, 74, 3])
    filters = np.zeros([15, 15, 3])
    image[7:67, 7:67] = blur3

    kernel = np.zeros([15, 15])
    kernel[:, :5] += 1
    kernel[:, :4] += 1
    kernel[:, :3] += 1
    kernel[:, :2] += 1
    kernel[:, :1] += 1
    kernel[:, 6:] += -1
    filters[:, :, 0] = kernel

    output = convolve2D(image[:, :, 0], kernel, padding=0, strides=1)

    kernel = np.zeros([15, 15])
    kernel[:, 5:10] += 3
    kernel[:, 6:9] += 2
    kernel[:, 7:8] += 1
    kernel[:, 12:] += -1
    kernel[:, 13:] += -1
    kernel[:, 14:] += -1
    kernel[:, :3] += -1
    kernel[:, :2] += -1
    kernel[:, :1] += -1
    filters[:, :, 1] = kernel

    output += convolve2D(image[:, :, 1], kernel, padding=0, strides=1)

    kernel = np.zeros([15, 15])
    kernel[:, 10:] += 1
    kernel[:, 11:] += 1
    kernel[:, 12:] += 1
    kernel[:, 13:] += 1
    kernel[:, 14:] += 1
    kernel[:, :9] += -1
    filters[:, :, 2] = kernel

    output += convolve2D(image[:, :, 2], kernel, padding=0, strides=1)

    output = np.asarray(output, "int32")
    tifffile.imwrite(f"results/myFilteredImage.tif", output)
    filters[filters < 0] = 0
    filters = np.flip(filters, axis=1)
    filters = np.asarray(filters, "uint16")
    tifffile.imwrite(f"results/myFilter.tif", filters)

# Director dQ field heatmap
if False:
    cm = plt.get_cmap("RdBu_r")
    for filename in filenames:
        t0 = util.findStartTime(filename)
        focus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)
        (T, X, Y, rgb) = focus.shape
        binary = sm.io.imread(f"dat/{filename}/binary{filename}.tif").astype(int)
        df = pd.read_pickle(f"dat/{filename}/shape{filename}.pkl")

        Q = np.mean(df["q"])
        theta0 = np.arctan2(Q[0, 1], Q[0, 0]) / 2
        R = util.rotation_matrix(-theta0)

        imgLabel = np.zeros([T, 512, 512])
        for t in range(T):
            img = 255 - binary[t]

            # find and labels cells
            imgLabel[t] = sm.measure.label(img, background=0, connectivity=1)

        gray = rgb2gray(focus)
        gray = gray * (255 / np.max(gray))
        # gray = np.asarray(gray, "uint8")
        # tifffile.imwrite(f"results/gray{filename}.tif", gray)

        dQ1 = np.zeros([T, 512, 512, 3])

        for t in range(T):
            dft = df[df["Time"] == t]
            Q = np.matmul(R, np.matmul(np.mean(dft["q"]), np.matrix.transpose(R)))
            for i in range(len(dft)):
                [x, y] = [
                    dft["Centroid"].iloc[i][0],
                    dft["Centroid"].iloc[i][1],
                ]
                q = np.matmul(R, np.matmul(dft["q"].iloc[i], np.matrix.transpose(R)))
                dq = q - Q
                col = (dq[0, 0] + 0.05) / 0.1
                if col > 0.999:
                    col = 0.999
                elif col < 0:
                    col = 0
                colour = cm(col)

                label = imgLabel[t, int(512 - y), int(x)]
                if label != 0:
                    dQ1[t, :, :, 0][imgLabel[t] == label] = colour[0] * 255
                    dQ1[t, :, :, 1][imgLabel[t] == label] = colour[1] * 255
                    dQ1[t, :, :, 2][imgLabel[t] == label] = colour[2] * 255

        dQ1[:, :, :, 1][binary == 255] = 200

        mask = np.all((dQ1 - np.zeros(3)) == 0, axis=3)

        dQ1[:, :, :, 0][mask] = gray[mask]
        dQ1[:, :, :, 1][mask] = gray[mask]
        dQ1[:, :, :, 2][mask] = gray[mask]

        # imgLabel = np.asarray(imgLabel, "uint16")
        # tifffile.imwrite(f"results/displayProperties/imgLabel{filename}.tif", imgLabel)
        dQ1 = np.asarray(dQ1, "uint8")
        tifffile.imwrite(f"results/displayProperties/tissue_dq1{filename}.tif", dQ1)

# Director dQ field heatmap wound
if False:
    dfShape = pd.read_pickle(f"databases/dfShapeWound{fileType}.pkl")
    cm = plt.get_cmap("RdBu_r")
    for filename in filenames:
        t0 = util.findStartTime(filename)
        focus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)
        (T, X, Y, rgb) = focus.shape
        binary = sm.io.imread(f"dat/{filename}/binary{filename}.tif").astype(int)

        imgLabel = np.zeros([T, 512, 512])
        for t in range(T):
            img = 255 - binary[t]

            # find and labels cells
            imgLabel[t] = sm.measure.label(img, background=0, connectivity=1)

        gray = rgb2gray(focus)
        gray = gray * (255 / np.max(gray))
        # gray = np.asarray(gray, "uint8")
        # tifffile.imwrite(f"results/gray{filename}.tif", gray)

        dQ1 = np.zeros([T, 512, 512, 3])

        df = dfShape[dfShape["Filename"] == filename]

        for i in range(len(df)):
            x = int(df["X"].iloc[i] / scale)
            y = int(df["Y"].iloc[i] / scale)
            t = int((df["T"].iloc[i] - t0) / 2)
            dq = df["dq"].iloc[i]
            col = (dq[0, 0] + 0.05) / 0.1
            if col > 0.999:
                col = 0.999
            elif col < 0:
                col = 0
            colour = cm(col)

            label = imgLabel[t, int(512 - y), int(x)]
            if label != 0:
                dQ1[t, :, :, 0][imgLabel[t] == label] = colour[0] * 255
                dQ1[t, :, :, 1][imgLabel[t] == label] = colour[1] * 255
                dQ1[t, :, :, 2][imgLabel[t] == label] = colour[2] * 255

        if "Wound" in filename:
            dfWound = pd.read_pickle(f"dat/{filename}/woundsite{filename}.pkl")
            for t in range(T):
                (x, y) = dfWound["Position"].iloc[t]
                label = imgLabel[t, int(512 - y), int(x)]
                dQ1[t, :, :, 0][imgLabel[t] == label] = 0
                dQ1[t, :, :, 1][imgLabel[t] == label] = 0
                dQ1[t, :, :, 2][imgLabel[t] == label] = 0
                if x < 5:
                    x = 5
                if x > 507:
                    x = 507
                if y < 5:
                    y = 5
                if y > 507:
                    y = 507
                rr, cc = sm.draw.disk([511 - (y), x], 5)
                dQ1[t][rr, cc, 1] = 255
        else:
            dfVelocityMean = pd.read_pickle(f"databases/dfVelocityMean{fileType}.pkl")
            dfFilename = dfVelocityMean[
                dfVelocityMean["Filename"] == filename
            ].reset_index()
            for t in range(T):
                mig = np.sum(
                    np.stack(np.array(dfFilename.loc[:t, "V"]), axis=0), axis=0
                )
                x = 256 + mig[0] / scale
                y = 256 + mig[1] / scale
                label = imgLabel[t, int(512 - y), int(x)]
                if x < 5:
                    x = 5
                if x > 507:
                    x = 507
                if y < 5:
                    y = 5
                if y > 507:
                    y = 507
                rr, cc = sm.draw.disk([511 - (y), x], 5)
                dQ1[t][rr, cc, 1] = 255

        dQ1[:, :, :, 1][binary == 255] = 200

        mask = np.all((dQ1 - np.zeros(3)) == 0, axis=3)

        dQ1[:, :, :, 0][mask] = gray[mask]
        dQ1[:, :, :, 1][mask] = gray[mask]
        dQ1[:, :, :, 2][mask] = gray[mask]

        # imgLabel = np.asarray(imgLabel, "uint16")
        # tifffile.imwrite(f"results/displayProperties/imgLabel{filename}.tif", imgLabel)
        dQ1 = np.asarray(dQ1, "uint8")
        tifffile.imwrite(f"results/displayProperties/dq1{filename}.tif", dQ1)

# Heatmap area
if False:
    cm = plt.get_cmap("Reds")
    for filename in filenames:
        df = pd.read_pickle(f"dat/{filename}/shape{filename}.pkl")
        t0 = util.findStartTime(filename)
        focus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)
        (T, X, Y, rgb) = focus.shape
        binary = sm.io.imread(f"dat/{filename}/binary{filename}.tif").astype(int)

        imgLabel = np.zeros([T, 512, 512])
        for t in range(T):
            img = 255 - binary[t]

            # find and labels cells
            imgLabel[t] = sm.measure.label(img, background=0, connectivity=1)

        gray = rgb2gray(focus)
        gray = gray * (255 / np.max(gray))
        gray = np.asarray(gray, "uint8")
        tifffile.imwrite(f"results/displayProperties/gray{filename}.tif", gray)

        Area = np.zeros([T, 512, 512, 3])

        for i in range(len(df)):
            [x, y] = [
                df["Centroid"].iloc[i][0],
                df["Centroid"].iloc[i][1],
            ]
            t = df["Time"].iloc[i]
            area = df["Area"].iloc[i] * scale**2
            col = area / 35
            if col > 0.999:
                col = 0.999
            elif col < 0:
                col = 0
            colour = cm(col)

            label = imgLabel[t, int(512 - y), int(x)]
            if label != 0:
                Area[t, :, :, 0][imgLabel[t] == label] = colour[0] * 255
                Area[t, :, :, 1][imgLabel[t] == label] = colour[1] * 255
                Area[t, :, :, 2][imgLabel[t] == label] = colour[2] * 255

        Area[:, :, :, 1][binary == 255] = 200

        mask = np.all((Area - np.zeros(3)) == 0, axis=3)

        Area[:, :, :, 0][mask] = gray[mask]
        Area[:, :, :, 1][mask] = gray[mask]
        Area[:, :, :, 2][mask] = gray[mask]

        Area = np.asarray(Area, "uint8")
        tifffile.imwrite(f"results/displayProperties/Area{filename}.tif", Area)
        imgLabel = np.asarray(imgLabel, "uint16")
        tifffile.imwrite(f"results/displayProperties/imgLabel{filename}.tif", imgLabel)

# Far and close to wound
if False:
    filename = "WoundL18h10"
    focus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)
    (T, X, Y, rgb) = focus.shape
    dist = sm.io.imread(f"dat/{filename}/distance{filename}.tif").astype(int)
    close = np.zeros([X, Y])
    far = np.zeros([X, Y])

    close[(dist[0] < 30 / scale) & (dist[0] > 0)] = 128
    far[dist[60] > 30 / scale] = 128

    close = np.asarray(close, "uint8")
    tifffile.imwrite(f"results/displayProperties/close{filename}.tif", close)
    far = np.asarray(far, "uint8")
    tifffile.imwrite(f"results/displayProperties/far{filename}.tif", far)
