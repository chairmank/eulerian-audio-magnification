#!/usr/bin/python

from clip import *
from svd import SVD
from optparse import OptionParser
import sys
if __name__ == '__main__':
    svd = SVD(filename=sys.argv[1])
    k = []

    for arg in sys.argv[2:]:
        if ':' in arg:
            i = arg.index(':')
            s = arg[:i]
            f = arg[i+1:]
            k += range(int(s), int(f))
        else:
            k.append(int(arg))

    svd.mask(k)
    s = svd.reconstruct()
    c2 = s.resynthesize()

    c2.write('reconstruction.wav')
