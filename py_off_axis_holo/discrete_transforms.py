import pyfftw
import numpy as np
from py_off_axis_holo.holography_helpers import pad_arr, unpad_arr

def fft(a, **kwargs):
    """
    Perform Fourier transform.

    :param a: Array to transform.
    :type a: ndarray<complex>
    :param kwargs: Optional arguments of DiscreteTransform class
    :return: FFT of a.
    :rtype: ndarray<complex>
    """

    transform = FFT(a.shape, **kwargs)
    return transform(a)

def ifft(b, **kwargs):
    """
    Perform inverse Fourier transform.

    :param b: Array to transform.
    :type b: ndarray<complex>
    :param kwargs: Optional arguments of DiscreteTransform class
    :return: IFFT of b.
    :rtype: ndarray<complex>
    """

    transform = IFFT(b.shape, **kwargs)
    return transform(b)

def dct(a, **kwargs):
    """
    Perform discrete cosine transform.

    :param a: Array to transform.
    :type a: ndarray<float>
    :param kwargs: Optional arguments of DiscreteTransform class
    :return: DCT of a.
    :rtype: ndarray<float>
    """

    transform = DCT(a.shape, **kwargs)
    return transform(a)

def idct(b, **kwargs):
    """
    Perform inverse discrete cosine transform.

    :param b: Array to transform.
    :type b: ndarray<float>
    :param kwargs: Optional arguments of DiscreteTransform class
    :return: IDCT of b.
    :rtype: ndarray<float>
    """

    transform = IDCT(b.shape, **kwargs)
    return transform(b)

def FFT(dims, threads=1, dtype='complex128', **kwargs):
    """
    Initialize fast fourier transform.

    :param dims: Dimension of ndarray to operate on.
    :type dims: int
    :param threads: Number of threads to use. Default is 1.
    :type threads: int
    :param dtype: Numpy dtype of arrays to operate on. Default is 'complex128'.
    :type dtype: string
    :param kwargs: Additional parameters for pyfftw.FFTW().
    """

    if dtype != 'complex128' and dtype != 'complex64':
        dtype = 'complex128'
    return DiscreteTransform(dims, 'FFTW_FORWARD', dtype, threads, **kwargs)

def IFFT(dims, threads=1, dtype='complex128', **kwargs):
    """
    Initialize inverse fast fourier transform.

    :param dims: Dimensions of ndarrays to operate on.
    :type dims: int
    :param threads: Number of threads to use. Default is 1.
    :type threads: int
    :param dtype: Numpy dtype of arrays to operate on. Default is 'complex128'.
    :type dtype: string
    :param kwargs: Additional parameters for pyfftw.FFTW().
    """

    if dtype != 'complex128' and dtype != 'complex64':
        dtype = 'complex128'
    return DiscreteTransform(dims, 'FFTW_BACKWARD', dtype, threads, **kwargs)

def DCT(dims, threads=1, dtype='float64', **kwargs):
    """
    Initialize discrete cosine transform.

    :param dims: Dimensions of ndarrays to operate on.
    :type dims: int
    :param threads: Number of threads to use. Default is 1.
    :type threads: int
    :param dtype: Numpy dtype of arrays to operate on. Default is 'float64'.
    :type dtype: string
    :param kwargs: Additional flags for pyfftw.FFTW().
    """

    if dtype != 'float64' and dtype != 'float32':
        dtype = 'float64'
    return DiscreteTransform(dims, 'FFTW_REDFT10', dtype, threads, **kwargs)

def IDCT(dims, threads=1, dtype='float64', **kwargs):
    """
    Initialize inverse discrete cosine transform.

    :param dims: Dimension of ndarray to operate on.
    :type dims: int
    :param threads: Number of threads to use. Default is 1.
    :type threads: int
    :param dtype: Numpy dtype of arrays to operate on. Default is 'float64'.
    :type dtype: string
    :param kwargs: Additional flags for pyfftw.FFTW().
    """

    if dtype != 'float64' and dtype != 'float32':
        dtype = 'float64'
    return DiscreteTransform(dims, 'FFTW_REDFT01', dtype, threads, **kwargs)

class DiscreteTransform:

    def __init__(self, dims, direction, dtype, threads=1, **kwargs):
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
        :param kwargs: Additional parameters for transform builder.
        """

        self._dtype = dtype
        self._threads = threads
        self._dir = direction
        self._transform = self._build(dims, self._parse_flags(**kwargs))

    def __call__(self, a, **kwargs):
        """
        Performs an N dimensional transform on an array.

        :param a: Array to transform.
        :type a: ndarray<dtype>
        :return: Transform of a.
        :rtype: ndarray<dtype>
        """

        a = a.astype(self._dtype)
        b = np.zeros(a.shape, dtype=self._dtype)
        self._transform.update_arrays(a, self.output)
        b[:] = self._transform(**kwargs)
        return b

    def _build(self, dims, flags=()):
        """
        Setup FFTW object.

        :param dims: Dimension of ndarray to operate on.
        :type dims: int
        :param flags: Additional flags for pyfftw.FFTW().
        """

        dims = np.round(np.array(dims)).astype(int)
        a = pyfftw.empty_aligned(dims, dtype=self._dtype)
        b = pyfftw.empty_aligned(dims, dtype=self._dtype)
        axes = tuple(i for i in range(len(dims)))
        directions = np.repeat(self._dir, len(dims))
        return pyfftw.FFTW(a, b, threads=self._threads, axes=axes,
                           direction=directions, flags=flags)

    @classmethod
    def _parse_flags(cls, planner_effort='FFTW_MEASURE', aligned_input=True, overwrite_input=False):
        """
        Parses flags used by FFTW to perform transformations. See FFTW documentation for more info.

        :param planner_effort: Determines amount of planning done beforehand to speed of transforms.
        Options are FFT_{ESTIMATE, MEASURE, PATIENT, EXHAUSTIVE} from least to most effort. Default is 'FFTW_MEASURE'.
        :type planner_effort: string
        :param aligned_input: Whether given arrays have been properly aligned in memory. Default is 'True'.
        :type aligned_input: bool
        :param overwrite_input: Whether the allocated arrays in memory can be overwritten which may provide some speed up.
        Default is 'True'.
        :type overwrite_input: bool
        :return: tuple<string>
        """

        flags = ()
        plan_opts = ['FFTW_ESTIMATE', 'FFTW_MEASURE', 'FFTW_PATIENT', 'FFTW_EXHAUSTIVE']
        if planner_effort in plan_opts:
            flags += (planner_effort,)
        else:
            flags += ('FFTW_MEASURE',)
        if overwrite_input:
            flags += ('FFTW_DESTROY_INPUT',)
        if not aligned_input:
            flags += ('FFTW_UNALIGNED',)
        return flags

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

    def refactor(self, dims):
        """
        Recreate FFTW for a new size.

        :param dims: New dimension of ndarray to operate on.
        :type dims: int
        """

        self._transform = self._build(dims, self.flags)

class DFT:

    def __init__(self, dims, nb=0, threads=1, ortho=False, dtype='complex128', **kwargs):
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
        :param dtype: Numpy dtype (complex) of arrays to operate on. Default is 'complex128'.
        :type dtype: string
        :param kwargs: Optional arguments for DiscreteTransform object.
        """

        self._nb = nb
        self._dims = dims
        self._ortho = ortho
        self._fft = FFT(self.shape, threads=threads, dtype=dtype, **kwargs)
        self._ifft = IFFT(self.shape, threads=threads, dtype=dtype, **kwargs)

    @property
    def shape(self):
        """
        :return: Shape of output array.
        :rtype: tuple<int>
        """

        return tuple(2 * self._nb + np.array(self._dims))

    def pad_arr(self, a):
        """
        Adds padding to arrays.

        :param a: Input array.
        :type a: ndarray<complex>
        :return: Padding array.
        :rtype: ndarray<complex>
        """

        return pad_arr(a, self._nb)

    def unpad_arr(self, a):
        """
        Removes array padding.

        :param a: Input array.
        :type a: ndarray<complex>
        :return: Padding array.
        :rtype: ndarray<complex>
        """

        return unpad_arr(a, self._nb)

    def forwards(self, a):
        """
        Forward Fourier transform

        :param a: Array to transform
        :type a: ndarray<complex>
        :return: Transformed array.
        :rtype: ndarray<complex>
        """

        assert a.shape == self._dims, "Shape of array must be " + str(self._dims)
        if self._nb > 0:
            a = self.pad_arr(a)
        return np.fft.fftshift(self._fft(a, ortho=self._ortho, normalise_idft=False))

    def backwards(self, b):
        """
        Inverse Fourier transform

        :param b: Array to transform
        :type b: ndarray<complex>
        :return: Transformed array.
        :rtype: ndarray<complex>
        """

        assert b.shape == self.shape, "Shape of array must be " + str(self.shape)
        a = self._ifft(np.fft.ifftshift(b), ortho=self._ortho, normalise_idft=False)
        if self._nb > 0:
            return self.unpad_arr(a)
        return a

class REDFT:

    def __init__(self, dims, nb=0, threads=1, ortho=False, dtype='float64', **kwargs):
        """
        Class for performing cosine (real even) discrete Fourier transforms.

        :param dims: New dimension of ndarray to operate on.
        :type dims: int
        :param nb: Number of zeros to pad array dimensions by. Default is '0'.
        :type nb: int
        :param threads: Number of threads to use. Default is '1'.
        :type threads: int
        :param ortho: Whether to use orthonormal normalization. Default is 'False'
        :type ortho: bool
        :param dtype: Numpy dtype (float) of arrays to operate on. Default is 'float64'.
        :type dtype: string
        :param kwargs: Optional arguments for DiscreteTransform object.
        """

        self._nb = nb
        self._dims = dims
        self._ortho = ortho
        self._dct = DCT(self.shape, threads=threads, dtype=dtype, **kwargs)
        self._idct = IDCT(self.shape, threads=threads, dtype=dtype, **kwargs)

    @property
    def shape(self):
        """
        :return: Shape of output array.
        :rtype: tuple<int>
        """

        return tuple(2*self._nb + np.array(self._dims))

    def pad_arr(self, a):
        """
        Adds padding to arrays.

        :param a: Input array.
        :type a: ndarray<float>
        :return: Padding array.
        :rtype: ndarray<float>
        """

        return pad_arr(a, self._nb)

    def unpad_arr(self, a):
        """
        Adds padding to arrays.

        :param a: Input array.
        :type a: ndarray<float>
        :return: Padding array.
        :rtype: ndarray<float>
        """

        return unpad_arr(a, self._nb)

    def forwards(self, a):
        """
        Forward cosine transform

        :param a: Array to transform
        :type a: ndarray<float>
        :return: Transformed array.
        :rtype: ndarray<float>
        """

        assert a.shape == self._dims, "Shape of array must be " + str(self._dims)
        if self._nb > 0:
            a = self.pad_arr(a)
        return self._dct(a, ortho=self._ortho, normalise_idft=False)

    def backwards(self, b):
        """
        Inverse cosine transform

        :param b: Array to transform
        :type b: ndarray<float>
        :return: Transformed array.
        :rtype: ndarray<float>
        """

        assert b.shape == self.shape, "Shape of array must be " + str(self.shape)
        a = self._idct(b, ortho=self._ortho, normalise_idft=False)
        if self._nb > 0:
            a = self.unpad_arr(a)
        return a