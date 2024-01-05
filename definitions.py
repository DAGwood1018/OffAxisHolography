import os

ROOT = os.path.realpath(os.path.dirname(__file__))  # Root directory

WAVELENGTH = 589.157315 * 1e-9  # Laser wavelength we are using
PIXEL_SIZE = 3.45 * 1e-6  # Pixel size of camera
SENSOR_RESOLUTION = [1440, 1080]  # Camera resolution