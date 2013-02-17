from scipy.io import wavfile
import numpy as np
from scipy.signal import firwin, lfilter, hamming



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


class Clip:
    def __init__(self, path, start=0, end=None):
        """Read samples from the 0th channel of a WAV file specified by
        *path*."""
        (fs, self.signal) = wavfile.read(path)
        self.nyq = fs / 2.0

        # For expediency, just pull one channel
        if self.signal.ndim > 1:
            self.signal = self.signal[:, 0]

        if end is None:
            self.signal = self.signal[start:]
        else:
            self.signal = self.signal[start:end]

        self.original = self.signal[:]

    def write(self, path):
        wavfile.write(path, int(2 * self.nyq), self.signal)

class Spectrogram:
    def __init__(self, clip, window=1024, step=None, n=None):
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

        signal = clip.signal
        self.window = window
        self.step = step
        self.n = n

        if signal.ndim != 1:
            raise ValueError("signal must be a 1-dimensional array")
        length = signal.size
        num_windows = _num_windows(length, window, step)
        out = np.zeros((num_windows, n), dtype=np.complex64)
        taper = hamming(window)
        for (i, s) in enumerate(window_slice_iterator(length, window, step)):
            out[i, :] = np.fft.fft(signal[s] * taper, n)
        self.data = out

    def resynthesize(self):
        spectrogram = self.data
        if self.data.ndim != 2:
            raise ValueError("spectrogram must be a 2-dimensional array")
        (num_windows, num_freqs) = spectrogram.shape
        length = self.step * (num_windows - 1) + self.window
        signal = np.zeros((length,))
        for i in xrange(num_windows):
            snippet = np.real(np.fft.ifft(spectrogram[i, :], self.window))
            signal[(self.step * i):(self.step * i + self.window)] += snippet
        signal = signal / np.max(np.abs(signal)) * 0x8000 * 0.9
        signal = signal.astype(np.int16)
        return signal
