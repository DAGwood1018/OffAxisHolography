from abc import ABC
import numpy as np
import logging
import tkinter as tk
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
from .field_propagation import AngularSpectrum
from .holo_utils import format_img
from warnings import warn

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('hologram_filtering.log'),
                              logging.StreamHandler()])


class OffAxisMask(AngularSpectrum):

    def __init__(self, M, N, wl, sz, nb=1, threads=1, dz_max=10, nticks=100, set_radius=False, **kwargs):
        super().__init__(M, N, wl, sz, nb, threads=threads, **kwargs)
        self._root = tk.Tk()
        self._limits = (-dz_max, dz_max)
        self._tick_interval = 2 * dz_max / nticks
        self._dz = 0
        self._set_radius = set_radius
        self._open = False

        self._image = None
        self._canvas = None
        self._fig = None
        self._ax = None
        self._slider = None
        self._circle = None
        self._roi_center = None
        self._roi_radius = None

    '''
    @property
    def masked(self):
        if self._mask is None:
            return False
        else:
            return True

    @property
    def roi(self):
        if self._mask is None:
            return 0, 0, self._dims[0], self._dims[1]
        coords = np.argwhere(self._mask)
        m_min, n_min = coords.min(axis=0)
        m_max, n_max = coords.max(axis=0)
        return m_min, n_min, m_max - m_min + 1, n_max - n_min + 1
    '''

    def _setup_gui(self, fringes):
        self._root.title("Calibrate Hologram Filter")
        self._root.geometry("800x600")

        # Create a matplotlib figure for image display
        self._fringes = fringes
        self._fig, self._ax = plt.subplots()
        self._update_image()

        self._canvas = FigureCanvasTkAgg(self._fig, master=self._root)
        self._canvas_widget = self._canvas.get_tk_widget()
        self._canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add slider with adjustable ticks
        self._slider = tk.Scale(
            self._root, from_=self._limits[0], to=self._limits[1],
            orient="horizontal", resolution=0.1, tickinterval=self._tick_interval,
            length=600, label="Adjust Parameter"
        )
        self._slider.set(self._dz)
        self._slider.pack(side=tk.BOTTOM, fill=tk.X)
        self._slider.bind("<Motion>", self._update_slider)

        # Bind Enter key to finalize selection and return result
        self._root.bind("<Return>", self._finalize)
        self._open = True

    def _update_slider(self, event=None):
        value = self._slider.get()
        self._dz = float(value)
        self._update_image()

    def _update_image(self):
        if self._open:
            Fh = self.forwards(self.propagate(self._fringes, self._dz))
            Fh_scaled = 2 * np.log(np.abs(Fh) + 1e-10)
            updated_image = format_img(Fh_scaled)
            self._ax.clear()
            self._ax.imshow(updated_image, cmap='gray')
            self._ax.axis("off")
            self._canvas.draw()

    def _select_roi(self):
        if self._open:
            # Allow ROI selection
            self._ax.set_title("Click to set center, Drag to adjust radius")

            self._circle = None
            self._roi_center = None
            self._roi_radius = None

            # Connect event handlers for ROI
            self._fig.canvas.mpl_connect('button_press_event', self._on_click)
            self._fig.canvas.mpl_connect('motion_notify_event', self._on_drag)
            return True
        return False

    def _on_click(self, event):
        if self._open:
            if event.inaxes != self._ax:
                return False

            # Set the center of the circle
            if self._circle:
                self._circle.remove()

            self._roi_center = (event.xdata, event.ydata)
            self._circle = Circle(self._roi_center, 0, edgecolor='red', fill=False)
            self._ax.add_patch(self._circle)
            self._canvas.draw()
            return True
        return False

    def _on_drag(self, event):
        if self._open:
            if event.inaxes != self._ax or not self._circle:
                return False

            # Update the radius based on drag
            self._roi_radius = np.sqrt((event.xdata - self._roi_center[0]) ** 2 +
                                       (event.ydata - self._roi_center[1]) ** 2)
            self._circle.set_radius(self._roi_radius)
            self._ax.set_title(f"Radius: {self._roi_radius:.2f}")
            self._canvas.draw()
            return True
        return False

    def _finalize(self, event=None):
        if self._open:
            if self._roi_center is None or self._roi_radius is None:
                self._ax.set_title("ROI not set. Please select a region of interest first.")
                return False

            y, x = np.ogrid[:self.shape[0], :self.shape[1]]
            roi_mask = (x - self._roi_center[0]) ** 2 + (y - self._roi_center[1]) ** 2 <= self._roi_radius ** 2

            try:
                self._roi_center = self._find_max(self.propagate(self._fringes, self._dz) * roi_mask)
            except RuntimeError:
                warn("Failed to automatically find peak. Resorting to chosen center.")

            # Close the GUI
            self._root.quit()
            self._root.destroy()
            return True
        return False

    def _calc_mask(self, f1, radius=None):
        """
        :param f1: Frequency coords of +/-1 order in the form [fx, fy]
        :param radius: Predefined radius. If none the ideal radius is calculated.
        :return:
        """

        r = np.sqrt(np.sum(np.array(f1) ** 2))
        if radius is None:
            r_1 = r / 3
            r_DC = 2 * r_1
        else:
            assert radius < r, "Radius of mask can not exceed distance from image center."
            r_1 = radius
            r_DC = r - r_1

        mask10 = np.sqrt((self._Fx - f1[0]) ** 2 + (self._Fy - f1[1]) ** 2) < r_1
        mask01 = np.sqrt((self._Fx + f1[0]) ** 2 + (self._Fy + f1[1]) ** 2) < r_1
        mask00 = np.sqrt(self._Fx ** 2 + self._Fy ** 2) < r_DC
        return mask10, mask01, mask00

    def _find_max(self, Fh):
        """
        Finds maximum value of a masked Fourier spectrum

        :param Fh: Masked Fourier spectrum
        :type Fh: array
        :return: Indices of maximum in the form [fx, fy]
        :rtype: ndarray
        """

        fy, fx = np.where(Fh == np.amax(Fh))
        if len(fx) == 0:
            raise RuntimeError("No maximum identified.")
        if len(fx) > 1:
            warn("Multiple maximums detected. Using median value.")
            fx = np.median(fx)
            fy = np.median(fy)

        M, N = self.shape
        f = np.array([fx - N / 2, fy - M / 2])
        return f.reshape(f.size, ) / self._nb

    @classmethod
    def crop_to_mask(cls, a, mask):
        assert len(a.shape) == 2 and len(mask.shape) == 2, "Expecting 2D arrays."
        assert a.shape[0] == mask.shape[0] and a.shape[1] == mask.shape[1], "Expecting arrays of the same shape."
        coords = np.argwhere(mask)
        m_min, n_min = coords.min(axis=0)
        m_max, n_max = coords.max(axis=0)
        return a[m_min:m_max + 1, n_min:n_max + 1]

