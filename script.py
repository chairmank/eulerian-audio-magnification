import sys
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
    print "computing power"
    power = utils.estimate_spectral_power(spectrogram)

    print "whitening spectrum"
    whitened = spectrogram / np.sqrt(power)
    whitened = utils.normalize_total_power(whitened, utils.total_power(spectrogram))

    print "unwhitening spectrum"
    unwhitened = whitened * np.sqrt(power)
    unwhitened = utils.normalize_total_power(unwhitened, utils.total_power(spectrogram))

    print "resynthesizing from whitened-unwhitened spectrogram"
    resynth = utils.resynthesize(unwhitened)
    wavfile.write("resynth.wav", int(2 * nyq), resynth)

    print "computing truncated whitened spectrogram after singular value decomposition"
    #k = [0]
    k = range(20)
    truncated_whitened_spectrogram = utils.svd_truncation(whitened, k=k)

    print "unwhitening truncated spectrum"
    truncated_spectrogram = truncated_whitened_spectrogram * np.sqrt(power)
    truncated_spectrogram = utils.normalize_total_power(truncated_spectrogram, utils.total_power(spectrogram))

    print "resynthesizing from unwhitened truncated spectrogram"
    truncated_resynth = utils.resynthesize(truncated_spectrogram)
    wavfile.write("resynth_truncated.wav", int(2 * nyq), truncated_resynth)


