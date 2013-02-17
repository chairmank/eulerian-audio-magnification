import numpy as np
from scipy.io import wavfile
from scipy.signal import firwin, lfilter, hamming

default_nyquist = 22050.0


def slurp_wav(path, n=(44100 * 10)):
    """Read first *n* samples from the 0th channel of a WAV file
    specified by *path*."""
    (fs, signal) = wavfile.read(path)
    nyq = fs / 2.0
    # For expediency, just pull one channel
    if signal.ndim > 1:
        signal = signal[:, 0]
    n = min(n, signal.size)
    signal = signal[:n]
    return (nyq, signal)


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
        step = window / 2
    if n is None:
        n = window
    if signal.ndim != 1:
        raise ValueError("signal must be a 1-dimensional array")
    length = signal.size
    num_windows = _num_windows(length, window, step)
    out = np.zeros((num_windows, n), dtype=np.complex64)
    taper = hamming(window)
    for (i, s) in enumerate(window_slice_iterator(length, window, step)):
        out[i, :] = np.fft.fft(signal[s] * taper, n)
    return out

def resynthesize(spectrogram, window=1024, step=None, n=None):
    """Compute the short-time Fourier transform on a 1-dimensional array
    *signal*, with the specified *window* size, *step* size, and
    *n*-resolution FFT.

    This function returns a 2-dimensional array of complex floats. The
    0th dimension is time (window steps) and the 1th dimension is
    frequency.
    """
    if step is None:
        step = window / 2
    if n is None:
        n = window
    if spectrogram.ndim != 2:
        raise ValueError("spectrogram must be a 2-dimensional array")
    (num_windows, num_freqs) = spectrogram.shape
    length = step * (num_windows - 1) + window
    signal = np.zeros((length,))
    for i in xrange(num_windows):
        snippet = np.real(np.fft.ifft(spectrogram[i, :], window))
        signal[(step * i):(step * i + window)] += snippet
    return signal


def stft_svd(signal, **kwargs):
    """Decompose the short-time Fourier transform of a signal using
    singular value decomposition.
    """
    spectrogram = stft(signal, **kwargs)
    # SVD of the spectrogram:
    #   u.shape == (num_windows, n)
    #   s.shape == (k, k)
    #   v.shape == (k, n)
    # where
    #   k == min(num_windows, n)
    k = min(*spectrogram.shape)
    (left, sv, right) = np.linalg.svd(spectrogram, full_matrices=False)
    scaled_right = (sv[:, np.newaxis] * right)
    # spectrogram is approximated by sum(components)
    components = (
        np.dot(left[:, slice(i, i + 1)], scaled_right[slice(i, i + 1), :])
        for i in xrange(k))
    return components


def estimate_power_spectrum(signal, **kwargs):
    return np.power(np.abs(stft(signal, **kwargs)), 2).mean(axis=0)


def whitening_filter(signal, nyq=default_nyquist, band=[20, 20000]):
    pass


def bandpass_filter_signal(signal, low, high, nyq=default_nyquist):
    taps = firwin(1024, [low, high], nyq=nyq, pass_zero=False)
    filtered_signal = lfilter(taps, [1.0], signal).astype(np.int16)
    return filtered_signal





