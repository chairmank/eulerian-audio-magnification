import optparse
from scipy.io import wavfile
import numpy as np

import utils

if __name__ == '__main__':
    (nyq, x) = utils.slurp_wav("Queen_mono.wav", 44100 * 20)
    s = utils.stft(x)
    r = utils.resynthesize(s).astype(np.int16)
    wavfile.write("resynth.wav", int(2 * nyq), r)
