import numpy as np


def gridspace(N, M, nb=1, center=False):
    """
    :param N, M: Shape of 2D arrays to operate on.
    :type N, M: int
    :param nb: Determines grid spacing.
    :type nb: float
    :param center: Whether to center meshgrids at 0. Default is 'False'.
    :type center: bools
    :return: Meshgrids of X and Y.
    :rtype: ndarray, ndarray
    """

    nb = nb if nb >= 1 else 1
    dims = np.ceil(nb * np.array([N, M])).astype('int64')
    x = np.linspace(-0.5 * N, 0.5 * N, dims[0]) if center else np.linspace(0, N, dims[0])
    y = np.linspace(-0.5 * M, 0.5 * M, dims[1]) if center else np.linspace(0, M, dims[1])
    return np.meshgrid(x, y, indexing='xy')


def pad_arr(a, nb):
    """
    Adds padding (zero entries) to a ndarray.
    Array shape becomes nb * a.shape.

    :param a: Array to pad.
    :type a: ndarray
    :param nb: Factor to pad size by.
    :type nb: float
    :return: Padded array.
    :rtype: ndarray
    """

    if nb <= 1:
        return a
    shape = np.ceil(nb * np.array(a.shape)).astype('int64')
    diff = shape - a.shape
    rpad = diff // 2
    lpad = rpad + diff % 2
    padding = tuple((r, l) for r, l in zip(rpad, lpad))
    return np.pad(a, padding, mode='constant')


def unpad_arr(a, nb):
    """
    Removes padding (zero entries) from a ndarray.
    Shape becomes a.shape // nb.

    :param a: Array to unpad.
    :type a: ndarray
    :param nb: Factor to unpad size by.
    :type nb: float
    :return: Unpadded array.
    :rtype: ndarray
    """

    if nb <= 1:
        return a
    shape = np.array(a.shape) // nb
    diff = np.array(a.shape) - shape
    rb = (diff // 2).astype('int64')
    lb = (rb + diff % 2).astype('int64')
    ind = tuple(slice(r, -l) for r, l in zip(rb, lb))
    return a[ind]
