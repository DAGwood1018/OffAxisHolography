from hologram_filtering import OffAxFilter
import numpy as np
import os

ROOT = os.path.realpath(os.path.dirname(__file__))  # Root director
filepath = os.path.join(ROOT, "still_water.npy")

WAVELENGTH = 589.157315 * 1e-9  # Laser wavelength we are using
PIXEL_SIZE = 3.45 * 1e-6  # Pixel size of camera
fringes = np.load(filepath)

filter = OffAxFilter(fringes, WAVELENGTH, PIXEL_SIZE, True, True)

c_filter = filter(threads=4)
print(c_filter.size())
c_filter.filter(fringes)
#c_filter.to_string(False)

# TODO Infinite loop upon iteration needs to be fixed