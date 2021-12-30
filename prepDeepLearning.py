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

for filename in filenames:
    T = 90

    ecadFocus = sm.io.imread(f"dat/{filename}/ecadFocus{filename}.tif").astype(int)
    h2Focus = sm.io.imread(f"dat/{filename}/h2Focus{filename}.tif").astype(int)

    input3h = np.zeros([91, 512, 512, 3])
    input1e2h = np.zeros([91, 512, 512, 3])
    input2e4h = np.zeros([90, 512, 512, 3])

    for t in range(T):
        input1e2h[t, :, :, 0] = h2Focus[t]
        input1e2h[t, :, :, 1] = ecadFocus[t]
        input1e2h[t, :, :, 2] = h2Focus[t + 1]

        input3h[t, :, :, 0] = h2Focus[t]
        input3h[t, :, :, 1] = h2Focus[t + 1]
        input3h[t, :, :, 2] = h2Focus[t + 2]

        input2e4h[t, :, :, 0] = h2Focus[t] / 2
        input2e4h[t, :, :, 1] = ecadFocus[t] / 2
        input2e4h[t, :, :, 2] = h2Focus[t + 1] / 2

        input2e4h[t, :, :, 0] += h2Focus[t + 2] / 4
        input2e4h[t, :, :, 1] += h2Focus[t + 2] / 4

        input2e4h[t, :, :, 0] += h2Focus[t + 3] / 4
        input2e4h[t, :, :, 2] += h2Focus[t + 3] / 4

        input2e4h[t, :, :, 1] += ecadFocus[t + 3] / 4
        input2e4h[t, :, :, 2] += ecadFocus[t + 3] / 4

    input1e2h = np.asarray(input1e2h, "uint8")
    tifffile.imwrite(f"dat/uploadDL/input1e2h{filename}.tif", input1e2h)
    input3h = np.asarray(input3h, "uint8")
    tifffile.imwrite(f"dat/uploadDL/input3h{filename}.tif", input3h)
    input2e4h = np.asarray(input2e4h, "uint8")
    tifffile.imwrite(f"dat/uploadDL/input2e4h{filename}.tif", input2e4h)

    focus = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)
    focus = np.asarray(focus, "uint8")
    tifffile.imwrite(f"dat/uploadDL/focus{filename}.tif", focus)