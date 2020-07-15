import numpy as np
import scipy.linalg as linalg


def area(pts):
    """takes polygon exterior coordinates and finds the area"""

    x = [c[0] for c in pts]
    y = [c[1] for c in pts]
    s = 0
    for i in range(len(pts) - 1):
        s += 0.5 * (x[i + 1] + x[i]) * (y[i + 1] - y[i])
    return s


def centroid(pts):
    """takes polygon exterior coordinates and finds the certre of mass in (x,y)"""

    x = [c[0] for c in pts]
    y = [c[1] for c in pts]
    sx = 0
    sy = 0
    a = area(pts)
    for i in range(len(pts) - 1):
        sx += (x[i] + x[i + 1]) * (x[i] * y[i + 1] - x[i + 1] * y[i])
        sy += (y[i] + y[i + 1]) * (x[i] * y[i + 1] - x[i + 1] * y[i])
    cx = sx / (6 * a)
    cy = sy / (6 * a)
    return (cx, cy)


def inertia(pts):
    """takes the polygon exterior coordinates and finds its inertia tensor matrix"""

    x = [c[0] for c in pts]
    y = [c[1] for c in pts]
    sxx = 0
    sxy = 0
    syy = 0
    a = area(pts)
    cx, cy = centroid(pts)
    for i in range(len(pts) - 1):
        sxx += (y[i] ** 2 + y[i] * y[i + 1] + y[i + 1] ** 2) * (
            x[i] * y[i + 1] - x[i + 1] * y[i]
        )
        syy += (x[i] ** 2 + x[i] * x[i + 1] + x[i + 1] ** 2) * (
            x[i] * y[i + 1] - x[i + 1] * y[i]
        )
        sxy += (
            x[i] * y[i + 1]
            + 2 * x[i] * y[i]
            + 2 * x[i + 1] * y[i + 1]
            + x[i + 1] * y[i]
        ) * (x[i] * y[i + 1] - x[i + 1] * y[i])
    Ixx = sxx / 12 - a * cy ** 2
    Iyy = syy / 12 - a * cx ** 2
    Ixy = sxy / 24 - a * cx * cy
    I = np.zeros(shape=(2, 2))
    I[0, 0] = Ixx
    I[1, 0] = -Ixy
    I[0, 1] = -Ixy
    I[1, 1] = Iyy
    I = I / a ** 2
    return I


def shape_factor(pts):
    """Using the inertia tensor matrix it products the shape factor of the polygon"""

    I = inertia(pts)
    D = linalg.eig(I)[0]
    e1 = D[0]
    e2 = D[1]
    Sf = abs((e1 - e2) / (e1 + e2))
    return Sf


pts = [
    [339.0, 217.0],
    [327.0, 227.0],
    [325.0, 225.0],
    [320.0, 224.0],
    [313.0, 215.0],
    [322.0, 206.0],
    [325.0, 206.0],
    [332.0, 208.0],
    [333.0, 207.0],
    [334.0, 211.0],
    [339.0, 217.0],
]  # example coord

