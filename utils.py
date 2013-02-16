import numpy as np


def _num_windows(length, window, step):
    return max(0, int((length - window + step) / step))


def window_slice_iterator(length, window, step):
    """Generate slices into a 1-dimensional array of specified *length*
    with the specified *window* size and *step* size.

    Yields slice objects of length *window*. Any remainder at the end is
    unceremoniously truncated.
    """
    num_windows = _num_windows(length, window, step)
    for i in xrange(num_windows):
        start = step * i
        end = start + window
        yield slice(start, end)


def stft(signal, window=1024, step=None, n=None):
    """Compute the short-time Fourier transform on a 1-dimensional array
    *signal*, with the specified *window* size, *step* size, and
    *n*-resolution FFT.

    This function returns a 2-dimensional array of complex floats. The
    0th dimension is time (window steps) and the 1th dimension is
    frequency.
    """
    if step is None:
        step = window
    if n is None:
        n = window
    if signal.ndim != 1:
        raise ValueError("signal must be a 1-dimensional array")
    length = signal.size
    num_windows = _num_windows(length, window, step)
    out = np.zeros((num_windows, n), dtype=np.complex64)
    for (i, s) in enumerate(window_slice_iterator(length, window, step)):
        out[i, :] = np.fft.fft(signal[s], n)
    return out
