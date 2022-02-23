import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage as sm
import tifffile
import skimage.io
import skimage.draw
import utils as util
import cv2
import shutil
import findGoodCells as fi
import cellProperties as cell
from shapely.geometry import Polygon

plt.rcParams.update({"font.size": 20})

# -------------------

filename = "prettyWound"
scale = 123.26 / 512
T = 8

if False:
    focus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)
    dfDivisions = pd.read_pickle(f"dat/{filename}/dfDivision{filename}.pkl")

    (T, X, Y, rgb) = focus.shape

    divisions = np.zeros([T, 552, 552, 3])

    for x in range(X):
        for y in range(Y):
            divisions[:, 20 + x, 20 + y, :] = focus[:, x, y, :]

    for i in range(len(dfDivisions)):

        t0 = dfDivisions["T"].iloc[i]
        (x, y) = (dfDivisions["X"].iloc[i], dfDivisions["Y"].iloc[i])
        x = int(x)
        y = int(y)

        rr0, cc0 = sm.draw.disk([551 - (y + 20), x + 20], 16)
        rr1, cc1 = sm.draw.disk([551 - (y + 20), x + 20], 11)

        times = [t0, int(t0 + 1)]

        timeVid = []
        for t in times:
            if t >= 0 and t <= T - 1:
                timeVid.append(t)

        for t in timeVid:
            divisions[t][rr0, cc0, 2] = 200
            divisions[t][rr1, cc1, 2] = 0

    divisions = divisions[:, 20:532, 20:532]

    divisions = np.asarray(divisions, "uint8")
    tifffile.imwrite(f"results/divisionsWound{filename}.tif", divisions)


if False:
    _df = []

    df = pd.read_pickle(f"dat/{filename}/shape{filename}.pkl")
    Q = np.mean(df["q"])
    theta0 = np.arccos(Q[0, 0] / (Q[0, 0] ** 2 + Q[0, 1] ** 2) ** 0.5) / 2
    R = util.rotation_matrix(-theta0)

    df = pd.read_pickle(f"dat/{filename}/nucleusVelocity{filename}.pkl")
    mig = np.zeros(2)

    for t in range(T):
        dft = df[df["T"] == t]
        v = np.mean(dft["Velocity"]) * scale
        v = np.matmul(R, v)

        for i in range(len(dft)):
            x = dft["X"].iloc[i] * scale
            y = dft["Y"].iloc[i] * scale
            dv = np.matmul(R, dft["Velocity"].iloc[i] * scale) - v
            [x, y] = np.matmul(R, np.array([x, y]))

            _df.append(
                {
                    "Filename": filename,
                    "T": t,
                    "X": x - mig[0],
                    "Y": y - mig[1],
                    "dv": dv,
                }
            )
        mig += v

    dfVelocity = pd.DataFrame(_df)
    dfVelocity.to_pickle(f"databases/dfVelocity{filename}.pkl")


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
        f"results/Velocity field {filename}.mp4",
        cv2.VideoWriter_fourcc(*"DIVX"),
        3,
        size,
    )
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
    cv2.destroyAllWindows()

    shutil.rmtree("results/video")


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
        f"results/heatmap {filename} {t}.png",
        dpi=300,
        transparent=True,
    )
    plt.close()


if True:

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
    tifffile.imwrite(f"results/deepLearning{filename}.tif", prepDeep3)