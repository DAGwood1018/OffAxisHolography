import numpy as np
from scipy.optimize import minimize
from off_axis_holography.utilities import gridspace, binarize
from matplotlib import pyplot as plt
from fluid_driver.system_constants import *


# cost function for the CFS implementation

def parabolic_phase(params, x, y, k0, sz):
    psi = np.exp(-1j * k0 * sz * ((x-params[1]) ** 2 + (y-params[2]) ** 2) / (2 * params[0]))
    return psi


def phase_count(params, field, X, Y, k0, sz):
    psi = parabolic_phase(params, X, Y, k0, sz)
    phase = np.angle(psi * field) + np.pi
    return phase.size - binarize(phase).sum()


def fit_circular_phase(field, k0, sz, method='TNC', tol=1e-6, opts={}):
    print("Starting phase compensation...")
    M, N = field.shape
    X, Y = gridspace(N, M, 1, True)

    params = np.array([1e5, 0, 0])
    res = minimize(phase_count, params, args=(field, X, Y, k0, sz),
                   method=method, tol=tol, options=opts)

    print("Optimization Routine Complete.")
    print(res.x)
    return parabolic_phase(res.x, X, Y, k0, sz)


X, Y = gridspace(800, 800, 1, True)
plt.figure()
plt.imshow(np.angle(parabolic_phase([1e5, 100, 0], X, Y, k0, pxl_sz)), cmap='RdBu')
plt.show()
