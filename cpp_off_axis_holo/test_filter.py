from hologram_filtering import OffAxFilter
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

filter = OffAxFilter(fringes, WAVELENGTH, PIXEL_SIZE, optimize=True, visualize=True)

c_filter, unwrap = filter(threads=4)
print("Threads Used: ", c_filter.threads_in_use())


print(np.nonzero(c_filter.mask()))

c_filter.write(fringes.flatten())
c_filter.a()
c_filter.show_input()

'''
c_filter.forwards()
c_filter.crop()
c_filter.backwards()

c_filter.a()
c_filter.b()
c_filter.c()
c_filter.d()

t0 = time.time()
phase = c_filter(np.array(fringes.flatten(), dtype=np.complex128))

# TODO I don't think this phase is right
cv2.imshow('phase_wrap', format_img(phase))
#cv2.waitKey(0)
'''

'''
phase = unwrap.unwrap(phase)
dt = time.time() - t0
print("Runtime: ", dt)
'''

#unwrap.to_string()

#cv2.imshow('phase_unwrap', format_img(phase))
#cv2.waitKey(0)
