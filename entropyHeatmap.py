import os
import shutil
from math import floor, log10

import cv2
import matplotlib.lines as lines
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg as linalg
import shapely
import skimage as sm
import skimage.io
import skimage.measure
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing

import cellProperties as cell
import findGoodCells as fi

plt.rcParams.update({"font.size": 20})


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


# -------- shiftedColorMap

# heatmap = sm.io.imread(file3).astype(int)
# heatmap = heatmap / sum(sum(heatmap))
# heatmap[0, 0] = -0.08

# # make these smaller to increase the resolution
# dx, dy = 0.01, 0.01

# # generate 2 2d grids for the x & y bounds
# y, x = np.mgrid[-0.15:0.15:dy, -0.15:0.15:dx]

# heatmap.max()

# z_min, z_max = -0.08, heatmap.max()
# midpoint = 1 - z_max / (z_max + abs(z_min))

# orig_cmap = matplotlib.cm.seismic
# shifted_cmap = shiftedColorMap(orig_cmap, midpoint=midpoint, name="shifted")
# fig, ax = plt.subplots()
# c = ax.pcolor(x, y, heatmap, cmap=shifted_cmap, vmin=z_min, vmax=z_max)
# fig.colorbar(c, ax=ax)
# fig.savefig(
#     f"results/prob dist q t=11", dpi=300, transparent=True,
# )
# plt.close("all")


# ------------------------------------------------------

createFolder("results/video/")

heatmap = sm.io.imread("dat/entropy histogram sample06_01.tif").astype(int)
heatmap = heatmap / sum(sum(heatmap))

# make these smaller to increase the resolution
dx, dy = 0.01, 0.01

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[-0.15:0.15:dy, -0.15:0.15:dx]

z_min, z_max = 0, 0.05

fig, ax = plt.subplots()
c = ax.pcolor(x, y, heatmap, cmap="Reds", vmin=z_min, vmax=z_max)
fig.colorbar(c, ax=ax)
fig.savefig(
    f"results/prob dist q t=0", dpi=300, transparent=True,
)
fig.savefig(
    f"results/video/prob dist q t=0", dpi=300, transparent=True,
)
plt.close("all")

# -----------

heatmap = sm.io.imread("dat/entropy histogram sample06_02.tif").astype(int)
heatmap = heatmap / sum(sum(heatmap))

# make these smaller to increase the resolution
dx, dy = 0.01, 0.01

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[-0.15:0.15:dy, -0.15:0.15:dx]

z_min, z_max = 0, 0.05

fig, ax = plt.subplots()
c = ax.pcolor(x, y, heatmap, cmap="Reds", vmin=z_min, vmax=z_max)
fig.colorbar(c, ax=ax)
fig.savefig(
    f"results/prob dist q t=1", dpi=300, transparent=True,
)
fig.savefig(
    f"results/video/prob dist q t=1", dpi=300, transparent=True,
)
plt.close("all")

# -----------

heatmap = sm.io.imread("dat/entropy histogram sample06_03.tif").astype(int)
heatmap = heatmap / sum(sum(heatmap))

# make these smaller to increase the resolution
dx, dy = 0.01, 0.01

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[-0.15:0.15:dy, -0.15:0.15:dx]

z_min, z_max = 0, 0.05

fig, ax = plt.subplots()
c = ax.pcolor(x, y, heatmap, cmap="Reds", vmin=z_min, vmax=z_max)
fig.colorbar(c, ax=ax)
fig.savefig(
    f"results/prob dist q t=2", dpi=300, transparent=True,
)
fig.savefig(
    f"results/video/prob dist q t=2", dpi=300, transparent=True,
)
plt.close("all")

# -----------

heatmap = sm.io.imread("dat/entropy histogram sample06_04.tif").astype(int)
heatmap = heatmap / sum(sum(heatmap))

# make these smaller to increase the resolution
dx, dy = 0.01, 0.01

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[-0.15:0.15:dy, -0.15:0.15:dx]

z_min, z_max = 0, 0.05

fig, ax = plt.subplots()
c = ax.pcolor(x, y, heatmap, cmap="Reds", vmin=z_min, vmax=z_max)
fig.colorbar(c, ax=ax)
fig.savefig(
    f"results/prob dist q t=3", dpi=300, transparent=True,
)
fig.savefig(
    f"results/video/prob dist q t=3", dpi=300, transparent=True,
)
plt.close("all")

# -----------

heatmap = sm.io.imread("dat/entropy histogram sample06_05.tif").astype(int)
heatmap = heatmap / sum(sum(heatmap))

# make these smaller to increase the resolution
dx, dy = 0.01, 0.01

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[-0.15:0.15:dy, -0.15:0.15:dx]

z_min, z_max = 0, 0.05

fig, ax = plt.subplots()
c = ax.pcolor(x, y, heatmap, cmap="Reds", vmin=z_min, vmax=z_max)
fig.colorbar(c, ax=ax)
fig.savefig(
    f"results/prob dist q t=4", dpi=300, transparent=True,
)
fig.savefig(
    f"results/video/prob dist q t=4", dpi=300, transparent=True,
)
plt.close("all")

# -----------

heatmap = sm.io.imread("dat/entropy histogram sample06_06.tif").astype(int)
heatmap = heatmap / sum(sum(heatmap))

# make these smaller to increase the resolution
dx, dy = 0.01, 0.01

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[-0.15:0.15:dy, -0.15:0.15:dx]

z_min, z_max = 0, 0.05

fig, ax = plt.subplots()
c = ax.pcolor(x, y, heatmap, cmap="Reds", vmin=z_min, vmax=z_max)
fig.colorbar(c, ax=ax)
fig.savefig(
    f"results/prob dist q t=5", dpi=300, transparent=True,
)
fig.savefig(
    f"results/video/prob dist q t=5", dpi=300, transparent=True,
)
plt.close("all")

# -----------

heatmap = sm.io.imread("dat/entropy histogram sample06_07.tif").astype(int)
heatmap = heatmap / sum(sum(heatmap))

# make these smaller to increase the resolution
dx, dy = 0.01, 0.01

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[-0.15:0.15:dy, -0.15:0.15:dx]

z_min, z_max = 0, 0.05

fig, ax = plt.subplots()
c = ax.pcolor(x, y, heatmap, cmap="Reds", vmin=z_min, vmax=z_max)
fig.colorbar(c, ax=ax)
fig.savefig(
    f"results/prob dist q t=6", dpi=300, transparent=True,
)
fig.savefig(
    f"results/video/prob dist q t=6", dpi=300, transparent=True,
)
plt.close("all")

# -----------

heatmap = sm.io.imread("dat/entropy histogram sample06_08.tif").astype(int)
heatmap = heatmap / sum(sum(heatmap))

# make these smaller to increase the resolution
dx, dy = 0.01, 0.01

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[-0.15:0.15:dy, -0.15:0.15:dx]

z_min, z_max = 0, 0.05

fig, ax = plt.subplots()
c = ax.pcolor(x, y, heatmap, cmap="Reds", vmin=z_min, vmax=z_max)
fig.colorbar(c, ax=ax)
fig.savefig(
    f"results/prob dist q t=7", dpi=300, transparent=True,
)
fig.savefig(
    f"results/video/prob dist q t=7", dpi=300, transparent=True,
)
plt.close("all")


# -----------

heatmap = sm.io.imread("dat/entropy histogram sample06_09.tif").astype(int)
heatmap = heatmap / sum(sum(heatmap))

# make these smaller to increase the resolution
dx, dy = 0.01, 0.01

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[-0.15:0.15:dy, -0.15:0.15:dx]

z_min, z_max = 0, 0.05

fig, ax = plt.subplots()
c = ax.pcolor(x, y, heatmap, cmap="Reds", vmin=z_min, vmax=z_max)
fig.colorbar(c, ax=ax)
fig.savefig(
    f"results/prob dist q t=8", dpi=300, transparent=True,
)
fig.savefig(
    f"results/video/prob dist q t=8", dpi=300, transparent=True,
)
plt.close("all")

# -----------

heatmap = sm.io.imread("dat/entropy histogram sample06_10.tif").astype(int)
heatmap = heatmap / sum(sum(heatmap))

# make these smaller to increase the resolution
dx, dy = 0.01, 0.01

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[-0.15:0.15:dy, -0.15:0.15:dx]

z_min, z_max = 0, 0.05

fig, ax = plt.subplots()
c = ax.pcolor(x, y, heatmap, cmap="Reds", vmin=z_min, vmax=z_max)
fig.colorbar(c, ax=ax)
fig.savefig(
    f"results/prob dist q t=9", dpi=300, transparent=True,
)
fig.savefig(
    f"results/video/prob dist q t=9", dpi=300, transparent=True,
)
plt.close("all")


# -----------

heatmap = sm.io.imread("dat/entropy histogram sample06_11.tif").astype(int)
heatmap = heatmap / sum(sum(heatmap))

# make these smaller to increase the resolution
dx, dy = 0.01, 0.01

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[-0.15:0.15:dy, -0.15:0.15:dx]

z_min, z_max = 0, 0.05

fig, ax = plt.subplots()
c = ax.pcolor(x, y, heatmap, cmap="Reds", vmin=z_min, vmax=z_max)
fig.colorbar(c, ax=ax)
fig.savefig(
    f"results/prob dist q t=10", dpi=300, transparent=True,
)
fig.savefig(
    f"results/video/prob dist q t=10", dpi=300, transparent=True,
)
plt.close("all")

# -----------

heatmap = sm.io.imread("dat/entropy histogram sample06_12.tif").astype(int)
heatmap = heatmap / sum(sum(heatmap))

# make these smaller to increase the resolution
dx, dy = 0.01, 0.01

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[-0.15:0.15:dy, -0.15:0.15:dx]

z_min, z_max = 0, 0.05

fig, ax = plt.subplots()
c = ax.pcolor(x, y, heatmap, cmap="Reds", vmin=z_min, vmax=z_max)
fig.colorbar(c, ax=ax)
fig.savefig(
    f"results/prob dist q t=11", dpi=300, transparent=True,
)
fig.savefig(
    f"results/video/prob dist q t=11", dpi=300, transparent=True,
)
plt.close("all")

# -----------

heatmap = sm.io.imread("dat/entropy histogram sample06_13.tif").astype(int)
heatmap = heatmap / sum(sum(heatmap))

# make these smaller to increase the resolution
dx, dy = 0.01, 0.01

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[-0.15:0.15:dy, -0.15:0.15:dx]

z_min, z_max = 0, 0.05

fig, ax = plt.subplots()
c = ax.pcolor(x, y, heatmap, cmap="Reds", vmin=z_min, vmax=z_max)
fig.colorbar(c, ax=ax)
fig.savefig(
    f"results/prob dist q t=12", dpi=300, transparent=True,
)
fig.savefig(
    f"results/video/prob dist q t=12", dpi=300, transparent=True,
)
plt.close("all")

# -----------

heatmap = sm.io.imread("dat/entropy histogram sample06_14.tif").astype(int)
heatmap = heatmap / sum(sum(heatmap))

# make these smaller to increase the resolution
dx, dy = 0.01, 0.01

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[-0.15:0.15:dy, -0.15:0.15:dx]

z_min, z_max = 0, 0.05

fig, ax = plt.subplots()
c = ax.pcolor(x, y, heatmap, cmap="Reds", vmin=z_min, vmax=z_max)
fig.colorbar(c, ax=ax)
fig.savefig(
    f"results/prob dist q t=13", dpi=300, transparent=True,
)
fig.savefig(
    f"results/video/prob dist q t=13", dpi=300, transparent=True,
)
plt.close("all")

# ------- Video Heatmap

img_array = []

createFolder("results/video/")

for i in range(14):
    img = cv2.imread(f"results/video/prob dist q t={i}.png")
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)


out = cv2.VideoWriter(
    "results/heatmapVideoSample06.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 5, size
)
for i in range(len(img_array)):
    out.write(img_array[i])

out.release()
cv2.destroyAllWindows()

shutil.rmtree("results/video")