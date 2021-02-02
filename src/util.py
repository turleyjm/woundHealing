import numpy as np
import os
import scyjava as sj

# This file contains the main processing functions.
#
# Note: ImageJ is currently undergoing a transformation behind the scenes.
# While the ImageJ GUI looks the same as it always has done, ImageJ2 uses a
# completely different method for storing images.  This format scales better
# with massive datasets (e.g. it can break them down into chunks) and works with
# more than 5 dimensions (the limit for ImageJ1).  As such, when working with
# PyImageJ, depending on what you're currently doing,  you may need to switch
# between ImageJ1, ImageJ2 and Numpy image formats.  The way this project worked
# out, I’ve had to use all three formats, so you'll see examples of each
# approach.

# This is the main image processing function.  It takes all the parameters we
# defined in the GUI and uses them to run the steps of the analysis - focusing,
# normalisation and WEKA.
def process_stack(
    ij,
    input_path,
    norm_mode,
    focus_range,
    mu0,
    background,
    model_path,
    save_focus,
    save_norm,
    save_prob,
):
    # Splitting the main filename from the file extension.  This is because we
    # will later on append information about the file being saved to the end of
    # the filename (e.g. "_Focused")
    (root_path, file_ext) = os.path.splitext(input_path)

    ### FOCUSING THE IMAGES ###
    # Running focus macro (using "get" on the end so the scipt waits until the
    # macro is complete)
    print("Focussing the image stack")

    # This function gets the macro text, which is stored in the get_focus_macro
    # function.  Since this macro loads an image, we need to update it to
    # hard-code the input image path.
    focus_macro = get_focus_macro(root_path + file_ext, focus_range)

    # Running the macro script via the PyImageJ object.  In this instance, the
    # macro will open the image within our copy of ImageJ (the "ij" variable);
    # however, since we're running in "headless" mode (set during initialisation
    # of PyImageJ in the main script), we don't see the images.  Nonetheless,
    # they're there and the macro runs as if we were running it directly within
    # a normal copy of ImageJ.
    ij.script().run("macro.ijm", focus_macro, True).get()

    # We now need to access the currently-active image within ImageJ.  We can do
    # this with the following command, which returns the image as a Java object.
    # Specifically, it's an ImageJ2 type of object called a DefaultDataset.
    focus_ij2 = ij.py.active_dataset()

    # If selected in the GUI, this saves the focussed image to the same place as
    # the input file, but with the suffix "_Focussed".
    if save_focus:
        outname = "%s_Focussed.tif" % root_path
        save_ij2(ij, focus_ij2, outname)

    ### APPLYING NORMALISATION ###
    # For the normalisation process, we are doing processing in Python, so we
    # need to convert the ImageJ2 DefaultDataset that we got from the focussing
    # step into a Numpy array.  Fortunately, PyImageJ has some handy functions
    # which do this for us.
    print("Normalising images")
    focus_np = ij.py.from_java(focus_ij2)

    # Running the normalise function defined later on in this file.  This will
    # update the input focus_np image.
    normalise(focus_np.values, background, mu0, norm_mode)

    # If selected in the GUI, this saves the normalised image to the same place
    # as the input file, but with the suffix "_Norm".  While we did the
    # processing for this step in Python, with the image as a Numpy array, the
    # ImageJ2 DefaultDataset and the Numpy array still correspond to the same
    # image in memory, so we can save using the ImageJ2 DefaultDataset.
    if save_norm:
        outname = "%s_Norm.tif" % root_path
        save_ij2(ij, focus_ij2, outname)

    ### APPLYING PIXEL CLASSIFICATION (WEKA) ###
    # For WEKA pixel classification we go back to processing in ImageJ; however,
    # this time we can't run it as a simple macro.  This is a limitation of the
    # WEKA plugin, that it can't be run in headless mode as a macro.  Instead,
    # we use ScyJava's jimport to create an instance of the
    # WekaSegmentation Java class.  We're then able to use this class with all
    # it's associated functions.  By accessing the WekaSegmentation class
    # directly we can load in the .model classifier file and run classification
    # on a specific image.
    print("Running pixel classification (WEKA)")
    weka = sj.jimport("trainableSegmentation.WekaSegmentation")()
    weka.loadClassifier(model_path)

    # The apply_weka function takes our current ImageJ2 DefaultDataset object as
    # an input; however, it will convert it to ImageJ1 format when passing it to
    # WEKA - this is simply because the WekaSegmentation class hasn't been
    # designed to work with the newer ImageJ2 format.  The apply_weka function
    # outputs a new ImageJ2 DefaultDataset object containing the probability
    # maps.
    prob_ij2 = apply_weka(ij, weka, focus_ij2)

    # If selected in the GUI, this saves the probability image to the same place
    # as the input file, but with the suffix "_Prob".
    if save_prob:
        outname = "%s_Prob.tif" % root_path
        save_ij2(ij, prob_ij2, outname)


# Returns the full macro code with the filepath and focus range inserted as
# hard-coded values.
def get_focus_macro(filepath, focus_range):
    return """
        open("%s");
        main_win = getTitle();
        run("Gaussian-based stack focuser", "radius_of_gaussian_blur=%s");
        focus_win = getTitle();
        selectWindow(main_win);
        close();
        selectWindow(focus_win);
        run("8-bit");
    """ % (
        filepath,
        focus_range,
    )


# Applies the normalisation code.  It's been reduced to a single copy (rather
# than having separate ones for Ecad and H2).
def normalise(vid, background, mu0, calc):
    (T, X, Y) = vid.shape

    for t in range(T):
        mu = vid[t][vid[t] > background]
        vid[t][vid[t] <= background] = 0

        # Since we're using the same function for Ecad and H2, we have this
        # conditional statement, which calculates mu appropriately depending
        # on which channel is currently being processed.
        if calc == "MEAN":
            mu = np.quantile(mu, 0.75)
        elif calc == "UPPER_Q":
            mu = np.quantile(mu, 0.75)

        ratio = mu0 / mu

        vid[t] = vid[t] * ratio
        vid[t][vid[t] > 255] = 255


# Applies the WEKA pixel classification step.  This function creates an instance
# of the WekaSegmentation Java class, which allows us to run the class as if we
# were calling it natively inside Java.
def apply_weka(ij, classifier, image_ij2):
    # Using scyjava and jimport to create an instance of the ImageJFunctions
    # class, which will be used to convert ImageJ2 images to ImageJ1
    ijf = sj.jimport("net.imglib2.img.display.imagej.ImageJFunctions")()

    # Converting the ImageJ2 image (DefaultDataset) to an ImageJ1 (ImagePlus)
    # type.  The argument is just the name given to this image.  We can call it
    # anything we like.
    image_ij1 = ijf.wrap(image_ij2, sj.to_java("IJ1 image"))

    # Applying classifier using the WekaSegmentation class' "applyClassifier"
    # function.  This returns a new ImageJ1 image (ImagePlus format).
    prob_ij1 = classifier.applyClassifier(image_ij1, 6, True)

    # At the moment, the probability image is a single stack with alternating
    # predicted classes, so we want to convert it into a multidimensional stack.
    # To do this, we need to know how many frames and channels (classes) there
    # are.  The third dimension is labelled internally within the ImageJ1 image
    # as "channels", so to get the number of frames we actually need to find out
    # how many channels it has.
    n_channels = classifier.getNumOfClasses()
    n_frames = image_ij1.getNChannels()
    prob_ij1.setDimensions(n_channels, 1, n_frames)

    # Converting the probability image to ImageJ2 format, so it can be saved.
    return ij.py.to_dataset(prob_ij1)


# This function saves ImageJ2 format images to the specified output file path.
def save_ij2(ij, image_ij2, outname):
    # The image saving function will throw an error if it finds an image already
    # saved at the target location.  Therefore, we first need to delete such
    # images.
    if os.path.exists(outname):
        os.remove(outname)

    # Saving the ImageJ2 image (DefaultDataset) to file.
    ij.io().save(image_ij2, outname)
