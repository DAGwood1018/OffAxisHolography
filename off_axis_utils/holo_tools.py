import numpy as np


def reference_wave(XY, tilt, k0, sz, A=1):
    k = k0 * np.sin(tilt)
    return A * np.exp(1j * sz * (k[0] * XY[0] + k[1] * XY[1]))

def object_wave(XY, phi, k0, sz, A=1):
    return A * np.exp(1j * sz * k0 * (XY[0] + XY[1]) + 1j * phi)

def optimal_tilt(k0, sz, err=0.1):
    kmax = np.pi / sz
    k = (1 - err) * (2 * np.sqrt(2) / (3 + np.sqrt(2)) * kmax)
    return np.arcsin(k / k0)

def fringe_dist(f1, dims, pxl_sz=None):
    dims = pxl_sz * np.array(dims) if not pxl_sz is None else np.array(dims)
    M, N = tuple(dims)
    return 1 / np.sqrt((f1[0]/N)**2 + (f1[1]/M)**2)

def resolution_lim(f1, dims, sz):
    M, N = dims
    kmax = 2 * np.pi / sz
    f1 = f1[0] / N * kmax, f1[1] / M * kmax
    k = np.sqrt(np.sum(np.array(f1) ** 2)) / 3
    return 2 * np.pi / k

def phase_diff(h, wl, n, n0):
    return 2 * np.pi * (n-n0) * h / wl

def height_profile(phase, wl, n, n0):
    return phase * wl / (2 * np.pi * (n-n0))


