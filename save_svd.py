#!/usr/bin/python

from clip import *
from svd import SVD
from optparse import OptionParser

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-s", type="float", dest="start_time", default=0)
    parser.add_option("-f", type="float", dest="end_time", default=10)
    (options, args) = parser.parse_args()

    if len(args)!=2:
        print "Usage: script input.wav output.pickle"
        exit(0)

    c = Clip(args[0], int(options.start_time * 44100), int(44100 * options.end_time))
    s = Spectrogram(c)
    svd = SVD(spectrogram=s)
    svd.save(args[1])
