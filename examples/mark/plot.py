#! /usr/bin/env python

import numpy as np
import scipy as sp
import pylab as p
from scipy import fftpack, signal

numchannels = 64*64
fp = open("vis.out", 'r')
while fp:
    a = np.fromfile(fp, np.complex64, numchannels)
    if len(a) == 0:
        break
    p.plot(a, color="red")
    p.show()
    b = np.fft.ifft(a, 2 * numchannels)
    c = np.fft.fftshift(b)
    p.plot(c, color="red")
    p.show()
