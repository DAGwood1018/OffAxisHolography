import numpy as np
from phase_unwrap import PhaseUnwrap
from matplotlib import pyplot as plt
import os

def gaussian_phase(x, y, sigma):
    """
    Generate a Gaussian phase.

    Parameters:
    x (ndarray): X-axis values.
    y (ndarray): Y-axis values.
    sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
    ndarray: Gaussian phase.
    """
    phase = np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    return phase

N = 100
path = os.getcwd()

# Generate coordinates
x = np.linspace(-5, 5, N)
y = np.linspace(-5, 5, N)
X, Y = np.meshgrid(x, y)

# Define standard deviation
sigma = 1.0

# Generate Gaussian phase
phase = gaussian_phase(X, Y, sigma)

# Mask the image to split it in two horizontally
mask = np.zeros_like(phase, dtype=bool)
image_wrapped = np.angle(np.exp(1j * phase))

plt.figure()
plt.imshow(image_wrapped, cmap='rainbow')
plt.colorbar()
plt.savefig(os.path.join(path, "in.png"))

unwrap = PhaseUnwrap(np.array([N, N], dtype=int), 1, 0)
image_unwrapped = unwrap(image_wrapped)

plt.figure()
plt.imshow(image_unwrapped, cmap='rainbow')
plt.colorbar()
plt.savefig(os.path.join(path, "out.png"))

print('finished')