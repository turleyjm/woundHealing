import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage as sm
import tifffile
import skimage.io
import skimage.draw

plt.rcParams.update({"font.size": 20})

# -------------------

filename = "Unwound18h13"

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