import optparse
from scipy.io import wavfile
import numpy as np

import utils

if __name__ == '__main__':
    (nyq, signal) = utils.slurp_wav("Queen_mono.wav", 44100 * 13, 44100 * 20)
    print "computing spectrogram"
    spectrogram = utils.stft(signal)

    print "computing truncated spectrogram after singular value decomposition"
    truncated_spectrogram = utils.svd_truncation(spectrogram, k=[0])

    print "resynthesizing from spectrogram"
    # resynthesize without any modifications
    resynth = utils.resynthesize(spectrogram)

    print "resynthesizing from truncated spectrogram"
    # resynthesize after dimensionality reduction
    truncated_resynth = utils.resynthesize(truncated_spectrogram)

    wavfile.write("resynth.wav", int(2 * nyq), resynth)
    wavfile.write("resynth_truncated.wav", int(2 * nyq), truncated_resynth)
