import numpy as np
from scipy import ndimage
import skimage as sm
import tifffile
import utils as util
import matplotlib.pyplot as plt


def normaliseMigration(vid, calc, mu0):
    vid = vid.astype("float")
    (T, Z, X, Y) = vid.shape

    for t in range(T):
        mu = vid[t, :, 50:450, 50:450][vid[t, :, 50:450, 50:450] > 0]

        if calc == "MEDIAN":
            mu = np.quantile(mu, 0.5)
        elif calc == "UPPER_Q":
            mu = np.quantile(mu, 0.75)

        ratio = mu0 / mu

        vid[t] = vid[t] * ratio
        vid[t][vid[t] > 255] = 255

    return vid.astype("uint8")


def getSurface(ecad):

    ecad = ecad.astype("float")
    variance = ecad
    (T, Z, Y, X) = ecad.shape

    for t in range(T):
        for z in range(Z):
            win_mean = ndimage.uniform_filter(ecad[t, z], (40, 40))
            win_sqr_mean = ndimage.uniform_filter(ecad[t, z] ** 2, (40, 40))
            variance[t, z] = win_sqr_mean - win_mean**2

    win_sqr_mean = 0
    win_mean = 0

    surface = np.zeros([T, X, Y])

    mu0 = 400
    for t in range(T):
        mu = variance[t, :, 50:450, 50:450][variance[t, :, 50:450, 50:450] > 0]

        mu = np.quantile(mu, 0.5)

        ratio = mu0 / mu

        variance[t] = variance[t] * ratio
        variance[t][variance[t] > 65536] = 65536

    np.argmax(variance[0] > 1000, axis=0)

    surface = np.argmax(variance > 1000, axis=1)

    surface = ndimage.median_filter(surface, size=9)

    return surface


def heightFilter(channel, surface, height):

    (T, Z, Y, X) = channel.shape

    heightFilt = np.zeros([T, Y, X, Z])

    b = np.arange(Z)
    heightFilt += b

    surfaceBelow = np.repeat(surface[:, :, :, np.newaxis], Z, axis=3) + height

    # heightFilt = np.einsum("ijkl->iljk", heightFilt)

    heightFilt = heightFilt < surfaceBelow
    heightFilt = heightFilt.astype(float)

    heightFilt = heightFilt.astype(float)
    heightFilt = np.einsum("ijkl->iljk", heightFilt)
    for t in range(T):
        for z in range(Z):
            heightFilt[t, z] = ndimage.uniform_filter(heightFilt[t, z], (20, 20))

    # heightFilt = np.asarray(heightFilt * 254, "uint8")
    # tifffile.imwrite(f"heightFilt.tif", heightFilt)

    channel = channel * heightFilt

    return channel


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
            variance[t, z] = winSqrMean - winMean**2

    varianceMax = np.max(variance, axis=1)

    for z in range(Z):
        surface[variance[:, z] == varianceMax] = z

    for z in range(Z):
        focus[surface == z] = image[:, z][surface == z]

    surface = surface.astype("uint8")
    focus = focus.astype("uint8")

    return surface, focus


def process_stack(filename):

    print("Finding Surface")
    # print(current_time())

    stack = sm.io.imread(f"datBleach/{filename}.tif").astype(int)

    (T, Z, C, Y, X) = stack.shape

    surface = getSurface(stack[:, :, 0])

    print("Filtering Height")
    # print(current_time())

    ecad = heightFilter(stack[:, :, 0], surface, 10)
    h2 = heightFilter(stack[:, :, 1], surface, 15)

    print("Focussing the image stack")
    # print(current_time())

    ecadBleach = focusStack(ecad, 7)[1]
    h2Bleach = focusStack(h2, 7)[1]

    ecadBleach = np.asarray(ecadBleach, "uint8")
    tifffile.imwrite(f"datBleach/ecadBleach{filename}.tif", ecadBleach)
    h2Bleach = np.asarray(h2Bleach, "uint8")
    tifffile.imwrite(f"datBleach/h2Bleach{filename}.tif", h2Bleach)


plt.rcParams.update({"font.size": 16})

# -------------------

filenames, fileType = util.getFilesType()
scale = 123.26 / 512
T = 93

if False:
    for filename in filenames:
        process_stack(filename)

if False:
    normIntenEcad = np.zeros([len(filenames), 93])
    normIntenH2 = np.zeros([len(filenames), 93])
    for k in range(len(filenames)):
        filename = filenames[k]
        ecadBleach = sm.io.imread(f"datBleach/ecadBleach{filename}.tif").astype(int)
        h2Bleach = sm.io.imread(f"datBleach/h2Bleach{filename}.tif").astype(int)
        muEcad = ecadBleach[0][ecadBleach[t] > 0]
        muH2 = h2Bleach[0][h2Bleach[t] > 0]
        for t in range(T):
            normIntenEcad[k, t] = ecadBleach[t][ecadBleach[t] > 0] / muEcad
            normIntenH2[k, t] = h2Bleach[t][h2Bleach[t] > 0] / muH2

    time = 2 * np.array(range(T))
    stdEcad = np.std(normIntenEcad, axis=0)
    normIntenEcad = np.mean(normIntenEcad, axis=0)
    stdH2 = np.std(normIntenH2, axis=0)
    normIntenH2 = np.mean(normIntenH2, axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(time, normIntenEcad)
    ax[0].fill_between(
        time, normIntenEcad - stdEcad, normIntenEcad + stdEcad, alpha=0.15
    )
    ax[1].plot(time, normIntenH2)
    ax[1].fill_between(time, normIntenH2 - stdH2, normIntenH2 + stdH2, alpha=0.15)

    ax[0].set(xlabel="Time (mins)", ylabel="Normalised intensity")
    ax[0].title.set_text("E-Cadherin-GFP bleaching over time")

    ax[1].set(xlabel="Time (mins)", ylabel="Normalised intensity")
    ax[1].title.set_text("histone2-RFP bleaching over time")

    fig.savefig(
        f"results/Bleaching wild type",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

fileTypes, groupTitle = util.getFilesTypes("18h")

if True:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for fileType in fileTypes:
        filenames = util.getFilesType(fileType)[0]
        surf = np.zeros([len(filenames), 93])
        for k in range(len(filenames)):
            filename = filenames[k]
            surface = sm.io.imread(f"dat/{filename}/surface{filename}.tif").astype(int)
            surf[k] = -(np.mean(np.mean(surface, axis=1), axis=1) - np.mean(surface[0]))

        surf = surf * 0.75
        time = 2 * np.array(range(T))
        std = np.std(surf, axis=0)
        surf = np.mean(surf, axis=0)

        color = util.getColor(fileType)
        fileTitle = util.getFileTitle(fileType)

        ax.plot(time, surf, label=fileTitle, color=color)
        ax.fill_between(time, surf - std, surf + std, alpha=0.15, color=color)

    ax.legend(loc="upper right", fontsize=12)
    ax.set(xlabel="Time (mins)", ylabel=r"Mean surface height ($\mu m$)")
    boldTitle = util.getBoldTitle(groupTitle)
    ax.title.set_text("Surface height over \n time " + boldTitle)

    fig.savefig(
        f"results/Surface height over time {groupTitle}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")
