import numpy as np
import pickle
from clip import Spectrogram

class SVD:
    def __init__(self, spectrogram=None, filename=None):
        if spectrogram is None and filename is None:
            print "Must specify either spectrogram or filename"
            exit(0)
        elif spectrogram is not None and filename is not None:
            print "Can't specify both spectrogram and filename"
            exit(0)
        elif spectrogram is not None:
            (self.left, self.sv, self.right) = np.linalg.svd(spectrogram.data, full_matrices=False)
            self.params = spectrogram.params
        else:
            f = open(filename, "rb")
            self.left = pickle.load(f)
            self.sv = pickle.load(f)
            self.right = pickle.load(f)
            self.params = pickle.load(f)
            f.close()

    def save(self, filename):
        f = open(filename, "wb")
        pickle.dump(self.left, f)
        pickle.dump(self.sv, f)
        pickle.dump(self.right, f)
        pickle.dump(self.params, f)
        f.close()

    def mask(self, k=[0]):
        zero_out = np.array([i for i in xrange(self.sv.size) if i not in k])
        if zero_out.size:
            self.sv[zero_out] = 0.0

    def reconstruct(self):
        truncated = np.dot(self.left, self.sv[:, np.newaxis] * self.right)
        s = Spectrogram(None)
        s.params = self.params
        s.data = truncated
        return s




