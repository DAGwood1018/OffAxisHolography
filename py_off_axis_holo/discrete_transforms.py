import pyfftw
import numpy as np

from py_off_axis_holo.holography_helpers import pad_arr, unpad_arr

BUILD_FLAGS = ('FFTW_MEASURE',)

def fft(a):
    """
    Perform Fourier transform.

    :param a: Array to transform.
    :type a: ndarray<complex>
    :return: FFT of a.
    :rtype: ndarray<complex>
    """

    transform = FFT(a.shape)
    return transform(a)


def ifft(b):
    """
    Perform inverse Fourier transform.

    :param b: Array to transform.
    :type b: ndarray<complex>
    :return: IFFT of b.
    :rtype: ndarray<complex>
    """

    transform = IFFT(b.shape)
    return transform(b)


def dct(a):
    """
    Perform discrete cosine transform.

    :param a: Array to transform.
    :type a: ndarray<float>
    :return: DCT of a.
    :rtype: ndarray<float>
    """

    transform = DCT(a.shape)
    return transform(a)


def idct(b):
    """
    Perform inverse discrete cosine transform.

    :param b: Array to transform.
    :type b: ndarray<float>
    :return: IDCT of b.
    :rtype: ndarray<float>
    """

    transform = IDCT(b.shape)
    return transform(b)


def FFT(dims, threads=1, dtype='complex128', flags=BUILD_FLAGS):
    """
    Initialize fast fourier transform.

    :param dims: Dimension of ndarray to operate on.
    :type dims: int
    :param threads: Number of threads to use. Default is 1.
    :type threads: int
    :param dtype: Numpy dtype of arrays to operate on. Default is 'complex128'.
    :type dtype: string
    :param flags: Flags for transform builder.
    :type flags: tuple<str>
    """

    if dtype != 'complex128' and dtype != 'complex64':
        dtype = 'complex128'
    return DiscreteTransform(dims, 'FFTW_FORWARD', dtype=dtype, threads=threads, flags=flags)


def IFFT(dims, threads=1, dtype='complex128', flags=BUILD_FLAGS):
    """
    Initialize inverse fast fourier transform.

    :param dims: Dimensions of ndarrays to operate on.
    :type dims: int
    :param threads: Number of threads to use. Default is 1.
    :type threads: int
    :param dtype: Numpy dtype of arrays to operate on. Default is 'complex128'.
    :type dtype: string
    :param flags: Flags for transform builder.
    :type flags: tuple<str>
    """

    if dtype != 'complex128' and dtype != 'complex64':
        dtype = 'complex128'
    return DiscreteTransform(dims, 'FFTW_BACKWARD', dtype=dtype, threads=threads, flags=flags)


def DCT(dims, threads=1, dtype='float64', flags=BUILD_FLAGS):
    """
    Initialize discrete cosine transform.

    :param dims: Dimensions of ndarrays to operate on.
    :type dims: int
    :param threads: Number of threads to use. Default is 1.
    :type threads: int
    :param dtype: Numpy dtype of arrays to operate on. Default is 'float64'.
    :type dtype: string
    :param flags: Flags for transform builder.
    :type flags: tuple<str>
    """

    if dtype != 'float64' and dtype != 'float32':
        dtype = 'float64'
    return DiscreteTransform(dims, 'FFTW_REDFT10', dtype=dtype, threads=threads, flags=flags)


def IDCT(dims, threads=1, dtype='float64', flags=BUILD_FLAGS):
    """
    Initialize inverse discrete cosine transform.

    :param dims: Dimension of ndarray to operate on.
    :type dims: int
    :param threads: Number of threads to use. Default is 1.
    :type threads: int
    :param dtype: Numpy dtype of arrays to operate on. Default is 'float64'.
    :type dtype: string
    :param flags: Flags for transform builder.
    :type flags: tuple<str>
    """

    if dtype != 'float64' and dtype != 'float32':
        dtype = 'float64'
    return DiscreteTransform(dims, 'FFTW_REDFT01', dtype=dtype, threads=threads, flags=flags)


class DiscreteTransform:

    def __init__(self, dims, direction, dtype, threads=1, nstack=0, flags=BUILD_FLAGS):
        """
        Abstract class for defining discrete transforms.

        :param dims: Dimension of ndarray to operate on.
        :type dims: int
        :param direction: Type of transform to perform.
        :type direction: string
        :param dtype: Numpy dtype of arrays to operate on.
        :type dtype: string
        :param threads: Number of threads to use. Default is 1.
        :type threads: int
        :param nstack: Number of arrays to simultaneously process. Default is 0.
        :type nstack: int
        :param flags: Flags for transform builder.
        :type flags: tuple<str>
        """

        assert isinstance(threads, int) and threads >= 1, "Number of threads must be an integer >= 1."
        assert isinstance(nstack, int) and nstack >= 0, "Number of stacked arrays must be an integer >= 0."
        assert np.ndim(dims) == 1, "Dimensions must given as a 1D array."

        self._dtype = dtype
        self._threads = threads
        self._dir = direction
        self._nstack = nstack
        self._transform = self._build(dims, nstack=nstack, flags=flags)

    def __call__(self, a, **kwargs):
        """
        Performs an N dimensional transform on an array.

        :param a: Array to transform. Can be a stacked ndarray.
        :type a: ndarray<dtype>
        :return: Transform of a.
        :rtype: ndarray<dtype>
        """

        a = np.array(a)
        a = a.astype(self._dtype)
        b = np.zeros(a.shape, dtype=self._dtype)
        self._transform.update_arrays(a, self.output)
        b[:] = self._transform(**kwargs)
        return b

    def _build(self, dims, nstack=1, flags=BUILD_FLAGS):
        """
        Setup FFTW object.

        :param dims: Dimension of ndarray to operate on.
        :type dims: list<int>
        :param nstack: Number of arrays to simultaneously process. Default is 1.
        :type nstack: int
        :param flags: Flags for transform builder.
        :type flags: tuple<str>
        :return: a pyFFTW FFTW object.
        """

        if nstack == 0:
            dims = np.round(np.array(dims)).astype(int)
            axes = tuple(i for i in range(len(dims)))
        else:
            stacked_dims = [nstack]
            stacked_dims.extend(list(dims))
            dims = np.round(np.array(stacked_dims)).astype(int)
            axes = tuple(i for i in range(1, len(dims)))

        flags = () if flags is None else flags
        a = pyfftw.empty_aligned(dims, dtype=self._dtype)
        b = pyfftw.empty_aligned(dims, dtype=self._dtype)
        directions = np.repeat(self._dir, len(axes))
        return pyfftw.FFTW(a, b, threads=self._threads, axes=axes,
                           direction=directions, flags=flags)

    @property
    def shape(self):
        """
        :return: FFTW transform shape.
        :rtype: ndarray
        """

        return np.array(self.input.shape)

    @property
    def dtype(self):
        """
        :return: The dtype of the input and output arrays.
        :rtype: dtype
        """

        return self._dtype


    @property
    def flags(self):
        """
        :return: FFTW flags passed to object.
        :rtype: tuple
        """

        return self._transform.flags

    @property
    def input(self):
        """
        :return: Input array.
        :rtype: ndarray
        """

        return self._transform.input_array

    @property
    def output(self):
        """
        :return: Output array.
        :rtype: ndarray
        """

        return self._transform.output_array

    @property
    def threads(self):
        """
        :return: Number of threads used.
        :rtype: int
        """

        return self._threads

    @property
    def nstack(self):
        """
        :return: Number of stacked ndarrays.
        :rtype: int
        """

        return self._nstack

    @property
    def direction(self):
        """
        :return: Transform direction.
        :rtype: str
        """

        return self._dir

class DFT:

    def __init__(self, dims, nb=0, threads=1, ortho=False, re=False, dtype='complex128', flags=BUILD_FLAGS):
        """
        Class for performing discrete fourier transforms.

        :param dims: New dimension of ndarray to operate on.
        :type dims: int
        :param nb: Number of zeros to pad array dimensions by. Default is '0'.
        :type nb: int
        :param threads: Number of threads to use. Default is '1'.
        :type threads: int
        :param ortho: Whether to use orthonormal normalization. Default is 'False'
        :type ortho: bool
        :param re: Whether a real discrete transform should be used or not.
        :type re: bool
        :param dtype: Numpy dtype (complex) of arrays to operate on. Default is 'complex128'.
        :type dtype: string
        :param flags: Flags for transform builder.
        :type flags: tuple<str>
        """

        assert type(nb) is int, "Padding must be given as an integer."
        padded_shape = 2 * nb + np.array(dims)
        if np.issubdtype(np.dtype(dtype), np.complexfloating) and re:
            dtype = 'float64'

        self._nb = nb
        self._ortho = ortho
        self._stacked = False
        self._fft = FFT(padded_shape, threads=threads, dtype=dtype, flags=flags) if not re else \
            DCT(padded_shape, threads=threads, dtype=dtype, flags=flags)
        self._ifft = IFFT(padded_shape, threads=threads, dtype=dtype, flags=flags) if not re else \
            IDCT(padded_shape, threads=threads, dtype=dtype, flags=flags)

    @property
    def input_shape(self):
        """
        :return: Shape of input array. Accounts for padding.
        :rtype: tuple<int>
        """

        return tuple(self._fft.shape)

    @property
    def output_shape(self):
        """
        :return: Shape of output array. Accounts for padding.
        :rtype: tuple<int>
        """

        return tuple(self._ifft.shape)

    def pad_array(self, a):
        """
        Adds padding to arrays.

        :param a: Input array.
        :type a: ndarray<complex>
        :return: Padding array.
        :rtype: ndarray<complex>
        """

        if self._stacked:
            nb = [0]
            for i in range(1, len(self.input_shape)):
                nb.append(self._nb)
        else:
            nb = self._nb
        return pad_arr(a, nb)

    def unpad_array(self, a):
        """
        Removes array padding.

        :param a: Input array.
        :type a: ndarray<complex>
        :return: Padding array.
        :rtype: ndarray<complex>
        """

        if self._stacked:
            nb = [0]
            for i in range(1, len(self.input_shape)):
                nb.append(self._nb)
        else:
            nb = self._nb
        return unpad_arr(a, nb)

    def forwards(self, a):
        """
        Forward Fourier transform

        :param a: Array to transform
        :type a: ndarray<complex>
        :return: Transformed array.
        :rtype: ndarray<complex>
        """

        if self._nb > 0:
            a = self.pad_array(a)
        assert tuple(a.shape) == self.input_shape, "Shape of array must be " + str(self.input_shape)
        return np.fft.fftshift(self._fft(a.astype(self._fft.dtype), ortho=self._ortho, normalise_idft=False))

    def backwards(self, b):
        """
        Inverse Fourier transform

        :param b: Array to transform
        :type b: ndarray<complex>
        :return: Transformed array.
        :rtype: ndarray<complex>
        """

        assert tuple(b.shape) == self.output_shape, "Shape of array must be " + str(self.output_shape)
        a = self._ifft(np.fft.ifftshift(b.astype(self._ifft.dtype)), ortho=self._ortho, normalise_idft=False)
        if self._nb > 0:
            return self.unpad_array(a)
        return a

    def stack_arrays(self, n):
        """
        Adds another dimension that is not transformed allowing for multiple arrays to be processed at once.
        However, order will not be preserved.

        :param n: How many arrays you want to process.
        :type n: int
        :return: bool
        """

        if n<=0 or not isinstance(n, int):
            return False
        if self._stacked:
            self.unstack_arrays()

        self._stacked = True
        dims_in, dims_out = self.input_shape, self.output_shape
        in_dir, out_dir = self._fft.direction, self._ifft.direction

        self._fft = DiscreteTransform(dims_in, in_dir, self._fft.dtype,
                                      threads=self._fft.threads, nstack=n, flags=self._fft.flags)
        self._ifft = DiscreteTransform(dims_out, out_dir, self._ifft.dtype,
                                       threads=self._ifft.threads, nstack=n, flags=self._ifft.flags)
        return True

    def unstack_arrays(self):
        """
        Return to only processing a single array at a time.

        :return: bool
        """

        if not self._stacked:
            return False

        self._stacked = False
        dims_in, dims_out= self.input_shape[1:], self.output_shape[1:]
        in_dir, out_dir = self._fft.direction, self._ifft.direction

        self._fft = DiscreteTransform(dims_in, in_dir, self._fft.dtype,
                                      threads=self._fft.threads, nstack=0, flags=self._fft.flags)
        self._ifft = DiscreteTransform(dims_out, out_dir, self._ifft.dtype,
                                       threads=self._ifft.threads, nstack=0, flags=self._ifft.flags)
        return True




