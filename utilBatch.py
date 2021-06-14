import numpy as np
import os
import scyjava as sj
import scipy
from scipy import ndimage
import skimage as sm
import skimage.io
import tifffile


def process_stack(filename):

    print("Finding Surface")

    stackFile = f"datProcessing/{filename}/{filename}.tif"
    stack = sm.io.imread(stackFile).astype(int)

    (T, Z, C, Y, X) = stack.shape

    surface = getSurface(stack[:, :, 0])
    save = True
    if save:
        surface = np.asarray(surface, "uint8")
        tifffile.imwrite(f"datProcessing/{filename}/surface{filename}.tif", surface)

    print("Filtering Height")

    ecad = heightFilter(stack[:, :, 0], surface)
    for t in range(T):
        stack[t, :, 1] = ndimage.median_filter(stack[t, :, 1], size=(3, 3, 3))

    migration = np.asarray(stack[:, :, 1], "uint8")
    migration = normaliseMigration(migration, "MEDIAN", 10)
    tifffile.imwrite(f"datProcessing/{filename}/migration{filename}.tif", migration)

    h2 = heightFilter(stack[:, :, 1], surface)
    stack = 0

    save = False
    if save:
        ecad = np.asarray(ecad, "uint8")
        tifffile.imwrite(f"datProcessing/{filename}/ecadHeight{filename}.tif", ecad)
        h2 = np.asarray(h2, "uint8")
        tifffile.imwrite(f"datProcessing/{filename}/h2Height{filename}.tif", h2)

    print("Focussing the image stack")

    ecadFocus = focusStack(ecad, 9)[1]
    h2Focus = focusStack(h2, 9)[1]

    save = False
    if save:
        ecadFocus = np.asarray(ecadFocus, "uint8")
        tifffile.imwrite(
            f"datProcessing/{filename}/ecadBleach{filename}.tif", ecadFocus
        )
        h2Focus = np.asarray(h2Focus, "uint8")
        tifffile.imwrite(f"datProcessing/{filename}/h2Bleach{filename}.tif", h2Focus)

    print("Normalising images")

    ecadNormalise = normalise(ecadFocus, "MEDIAN", 25)
    h2Normalise = normalise(h2Focus, "UPPER_Q", 60)

    save = True
    if save:
        ecadNormalise = np.asarray(ecadNormalise, "uint8")
        tifffile.imwrite(
            f"datProcessing/{filename}/ecadFocus{filename}.tif", ecadNormalise
        )
        h2Normalise = np.asarray(h2Normalise, "uint8")
        tifffile.imwrite(f"datProcessing/{filename}/h2Focus{filename}.tif", h2Normalise)


def weka(
    ij,
    filename,
    model_path,
    channel,
    name,
):

    framesMax = 7
    weka = sj.jimport("trainableSegmentation.WekaSegmentation")()
    weka.loadClassifier(model_path)

    ecadFile = f"datProcessing/{filename}/{channel}Focus{filename}.tif"
    ecad = sm.io.imread(ecadFile).astype(int)

    (T, X, Y) = ecad.shape

    stackprob = np.zeros([T, X, Y])
    split = int(T / framesMax - 1)
    stack = ecad[0:framesMax]
    stack_ij2 = ij.py.to_dataset(stack)
    stackprob_ij2 = apply_weka(ij, weka, stack_ij2)
    stackprob[0:framesMax] = ij.py.from_java(stackprob_ij2).values

    j = 1
    print(f" part {j} -----------------------------------------------------------")
    j += 1

    for i in range(split):
        stack = ecad[framesMax * (i + 1) : framesMax + framesMax * (i + 1)]
        stack_ij2 = ij.py.to_dataset(stack)
        stackprob_ij2 = apply_weka(ij, weka, stack_ij2)
        stackprob[
            framesMax * (i + 1) : framesMax + framesMax * (i + 1)
        ] = ij.py.from_java(stackprob_ij2).values

        print(f" part {j} -----------------------------------------------------------")
        j += 1

    stack = ecad[framesMax * (i + 2) :]
    stack_ij2 = ij.py.to_dataset(stack)
    stackprob_ij2 = apply_weka(ij, weka, stack_ij2)
    stackprob[framesMax * (i + 2) :] = ij.py.from_java(stackprob_ij2).values

    print(f" part {j} -----------------------------------------------------------")
    j += 1

    stackprob = np.asarray(stackprob, "uint8")
    tifffile.imwrite(f"datProcessing/{filename}/{name}{filename}.tif", stackprob)


# -----------------


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


def getSurface(ecad):

    ecad = ecad.astype("float")
    variance = ecad
    (T, Z, Y, X) = ecad.shape

    for t in range(T):
        for z in range(Z):
            win_mean = ndimage.uniform_filter(ecad[t, z], (20, 20))
            win_sqr_mean = ndimage.uniform_filter(ecad[t, z] ** 2, (20, 20))
            variance[t, z] = win_sqr_mean - win_mean ** 2

    win_sqr_mean = 0
    win_mean = 0

    surface = np.zeros([T, X, Y])

    for t in range(T):
        for x in range(X):
            for y in range(Y):
                p = variance[t, :, x, y]

                # p = list(p)
                # p.reverse()
                # p = np.array(p)

                m = surfaceFind(p)
                h = [i for i, j in enumerate(p) if j == m][0]

                surface[t, x, y] = h

    surface = ndimage.median_filter(surface, size=9)
    surface = np.asarray(surface, "uint8")

    return surface


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


def heightFilter(channel, surface):

    (T, Z, Y, X) = channel.shape

    for z in range(Z):
        for z0 in range(Z):
            scale = heightScale(z0, z)
            channel[:, z][surface == z0] = channel[:, z][surface == z0] * scale

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


def apply_weka(ij, classifier, image_ij2):

    ijf = sj.jimport("net.imglib2.img.display.imagej.ImageJFunctions")()

    image_ij1 = ijf.wrap(image_ij2, sj.to_java("IJ1 image"))

    prob_ij1 = classifier.applyClassifier(image_ij1, 6, True)

    n_channels = classifier.getNumOfClasses()
    n_frames = image_ij1.getNChannels()
    prob_ij1.setDimensions(n_channels, 1, n_frames)

    prob_ij2 = ij.py.to_dataset(prob_ij1)
    prob_np = ij.py.from_java(prob_ij2).astype("float16").values * 255
    prob_np = prob_np.astype("uint8")[:, 0]
    prob_ij2 = ij.py.to_dataset(prob_np)

    return prob_ij2


def save_ij2(ij, image_ij2, outname):

    if os.path.exists(outname):
        os.remove(outname)

    ij.io().save(image_ij2, outname)


def get_outPlane_macro(filepath):
    return """
        open("%s");
        rename("outPlane.tif");
        mainWin = "outPlane.tif"

        setOption("BlackBackground", false);
        run("Make Binary", "method=Minimum background=Default calculate");
        run("Median 3D...", "x=4 y=4 z=4");
        run("Invert", "stack");
        run("Make Binary", "method=Minimum background=Default calculate");
        run("Dilate", "stack");
        run("Dilate", "stack");
        run("Median 3D...", "x=2 y=2 z=2");
        run("Invert", "stack");
    """ % (
        filepath
    )


def woundsite(ij, filename):

    filepath = f"/Users/jt15004/Documents/Coding/python/processData/datProcessing/{filename}/woundProb{filename}.tif"

    outPlaneMacro = get_outPlane_macro(filepath)
    ij.script().run("macro.ijm", outPlaneMacro, True).get()

    outPlaneBinary_ij2 = ij.py.active_dataset()
    outPlaneBinary = ij.py.from_java(outPlaneBinary_ij2)

    outPlaneBinary = np.asarray(outPlaneBinary, "uint8")
    tifffile.imwrite(f"datProcessing/{filename}/outPlane{filename}.tif", outPlaneBinary)