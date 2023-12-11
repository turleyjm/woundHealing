import matplotlib.pyplot as plt
import numpy as np
import utils as util


def getColorLineMarker(fileType, groupTitle):

    if (
        groupTitle == "wild type"
        or groupTitle == "JNK DN"
        or groupTitle == "Ca RNAi"
        or groupTitle == "immune ablation"
    ):
        colorDict = {
            "Unwound18h": [3, "--", "o"],
            "WoundL18h": [10, ":", "o"],
            "WoundS18h": [20, "-.", "o"],
            "UnwoundJNK": [3, "--", "^"],
            "WoundLJNK": [10, ":", "^"],
            "WoundSJNK": [20, "-.", "^"],
            "UnwoundCa": [3, "--", "s"],
            "WoundLCa": [10, ":", "s"],
            "WoundSCa": [20, "-.", "s"],
            "Unwoundrpr": [3, "--", "p"],
            "WoundLrpr": [10, ":", "p"],
            "WoundSrpr": [20, "-.", "p"],
        }
    else:
        colorDict = {
            "Unwound18h": [3, "--", "o"],
            "WoundL18h": [3, ":", "o"],
            "WoundS18h": [3, "-.", "o"],
            "UnwoundJNK": [8, "--", "^"],
            "WoundLJNK": [8, ":", "^"],
            "WoundSJNK": [8, "-.", "^"],
            "UnwoundCa": [14, "--", "s"],
            "WoundLCa": [14, ":", "s"],
            "WoundSCa": [14, "-.", "s"],
            "Unwoundrpr": [20, "--", "p"],
            "WoundLrpr": [20, ":", "p"],
            "WoundSrpr": [20, "-.", "p"],
        }
    n = 23
    cm = plt.get_cmap("gist_rainbow")
    i, line, mark = colorDict[fileType]

    return cm(1.0 * i / n), line, mark


fileTypes, groupTitle = util.getFilesTypes()

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
i = 0
for fileType in fileTypes:
    colour, line, mark = getColorLineMarker(fileType, groupTitle)
    ax.plot(np.arange(10) * (i + 1), color=colour, linestyle=line, marker=mark)
    i += 1

fig.savefig("results/moreColors.png")
plt.show()
