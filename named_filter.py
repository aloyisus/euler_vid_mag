import numpy as np
from scipy.signal import convolve2d


def binomial_filter(sz):
    if (sz < 2):
        print('size argument must be larger than 1')
        return
    kernel = np.asarray([[0.5,0.5],[0.5,0.5]])
    for n in range(0,sz-2):
        kernel = convolve2d(kernel, [[0.5,0.5],[0.5,0.5]])
    return kernel


def named_filter(name):
    if (name[:5] == 'binom'):
        kernel = np.sqrt(2) * binomial_filter(int(name[5:]))
    return kernel
