from hologram_filtering import AlignOffAxis
from phase_unwrap import PhaseUnwrap
import numpy as np
import cv2
from holo_utils import format_img
import time
import os

ROOT = os.path.realpath(os.path.dirname(__file__))  # Root director
filepath = os.path.join(ROOT, "still_water.npy")

WAVELENGTH = 589.157315 * 1e-9  # Laser wavelength we are using
PIXEL_SIZE = 3.45 * 1e-6  # Pixel size of camera
fringes = np.load(filepath)

align = AlignOffAxis(fringes, WAVELENGTH, PIXEL_SIZE, optimize=True, visualize=True)
filter = align(threads=1)
print("Threads Used: ", filter.threads_in_use())


t0 = time.time()
phase = filter(fringes.astype('uint8'))
filter.show_input()
filter.show_output()

print(phase.flatten()[0:10])

# Instantiating this right after the filter breaks it for some reason.
# Cannot use filter class after instantiating the unwrapper.
unwrap = PhaseUnwrap(np.array([align.roi[2], align.roi[3]], dtype=int), 1, 0)

# TODO Input not being copied correctly. 
phase = unwrap(phase)
dt = time.time() - t0
print("Runtime: ", dt)

unwrap.show_wrapped()
unwrap.show_unwrapped()
