import optparse
from scipy.io import wavfile
import numpy as np

import utils

from optparse import OptionParser



if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-s", type="float", dest="start_time", default=13)
    parser.add_option("-f", type="float", dest="end_time", default=20)
    (options, args) = parser.parse_args()

    if len(args)==0:
	    filename = "Queen_mono.wav"
    else:
	    filename = args[0]
    print options

    (nyq, signal) = utils.slurp_wav(filename, int(options.start_time * 44100), int(44100 * options.end_time))
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
