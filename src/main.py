### IMPORTS ###
import configparser
import gui
import imagej
import os
import scyjava as sj
import util

from pathlib import Path


### PARAMETERS ###
# Parameters are stored in a configuration file at [user home/.wbif/autoseg.ini]
# (e.g. on Windows this would be "C:\Users\[username]\.wbif\autoseg.ini").
# These parameters are updated each time the analysis runs, so should always
# display the most recently-input values.

# Getting the path to the configuration file.  "Path.home()" gets the user's
# home folder path.
config_path = os.path.join(Path.home(), ".wbif", "autoseg.ini")

# Creating a new instance of the ConfigParser class, which will handle parameter
# reading and writing
config = configparser.ConfigParser()

# Reading the configuration file at the specified path
config.read(config_path)

# Checking that the config object contains the "defaults" section.  If it
# doesn't we create an empty dict.  The empty dict is there so the following
# steps don't fail.
if "defaults" in config:
    defaults = config["defaults"]
else:
    defaults = {}

# Creating a new GUI class object.  The constructor for this class takes the
# default values as optional arguments (i.e. we don't need to pass them, but if
# we don't we can't get the default values into the GUI).  Each call to "get"
# a value from the default parameters dict also includes a fall-back value.  If
# the dict doesn't contain the key we're looking for (e.g. this should only be
# the case if we've not run this script on a computer before) it will instead
# use that fallback value.
g = gui.GUI(
    ecad_path=defaults.get("ecad_path", ""),
    h2_path=defaults.get("h2_path", ""),
    focus_range=defaults.get("focus_range", 5),
    mu0=defaults.get("mu0", 25),
    background=defaults.get("background", 0),
    ecad_model_path=defaults.get("ecad_model_path", ""),
    h2_model_path=defaults.get("h2_model_path", ""),
    save_focus=defaults.get("save_focus", True),
    save_norm=defaults.get("save_norm", True),
    save_prob=defaults.get("save_prob", True),
)

# Running the GUI.  It's at this point the GUI will be displayed.  The code in
# this file will stop being evaluated until the GUI has been closed by clicking
# on the "Process" button.
g.run()

# Updating the configuration file with the values stored in the GUI.
config["defaults"] = {
    "ecad_path": g.ecad_path,
    "h2_path": g.h2_path,
    "focus_range": g.focus_range,
    "mu0": g.mu0,
    "background": g.background,
    "ecad_model_path": g.ecad_model_path,
    "h2_model_path": g.h2_model_path,
    "save_focus": g.save_focus,
    "save_norm": g.save_norm,
    "save_prob": g.save_prob,
}

# Checking the folder the configuration file will be stored in exists.  If it
# doesn't we create it.
if not os.path.exists(config_path):
    os.makedirs(Path(config_path).parent)

# Writing the updated configuration to file
with open(config_path, "w") as configfile:
    config.write(configfile)


### INITIALISING THE SYSTEM ###
# We now need to start the PyImageJ system running.  This will download a copy
# of ImageJ.  The first time this script runs it can take a couple of minutes;
# however, subsequent runs should get past this step in a matter of seconds.
print("Initialising ImageJ (this may take a couple of minutes first time)")

# Setting the amount of RAM the Java Virtual Machine running ImageJ is allowed
# (e.g. Xmx6g loads 6 GB of RAM)
sj.config.add_option("-Xmx6g")

# Initialising PyImageJ with core ImageJ and the plugins we need.  For this, we
# have the Time_Lapse plugin, which offers an alternative for stack focusing.
# We also import the WEKA plugin.
ij = imagej.init(
    [
        "net.imagej:imagej:2.1.0",
        "net.imagej:imagej-legacy",
        "sc.fiji:Time_Lapse:2.1.1",
        "sc.fiji:Trainable_Segmentation:3.2.34",
    ],
    headless=True,
)

# Displaying information about the running ImageJ
print(ij.getApp().getInfo(True))


### MAIN PROCESSING STEPS ###
# The main processing steps have been bundled into a series of functions within
# the utils.py file.  We call the "process_stack" function, which in turn runs
# the relevant functions for stack focussing, normalisation and WEKA.

# Processing Ecad image stack
print("Processing Ecad image")

# Most of the arguments for this are being read from the GUI object (named "g").
# Furthermore, the numeric values are converted from strings (the GUI output)
# into integers.  The third argument controls the mu calculation during
# intensity normalisation.
util.process_stack(
    ij,
    g.ecad_path,
    "MEAN",
    int(g.focus_range),
    int(g.mu0),
    int(g.background),
    g.ecad_model_path,
    g.save_focus,
    g.save_norm,
    g.save_prob,
)

# # Processing H2 image stack
print("Processing H2 image")
util.process_stack(
    ij,
    g.h2_path,
    "UPPER_Q",
    int(g.focus_range),
    int(g.mu0),
    int(g.background),
    g.h2_model_path,
    g.save_focus,
    g.save_norm,
    g.save_prob,
)

# At this point the analysis is complete
print("Complete")
