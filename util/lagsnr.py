#!/usr/bin/python
import os,sys
import numpy as np

if not len(sys.argv) == 2:
    print "Usage: %s <vis.out>" % sys.argv[0]
    sys.exit()

if not os.path.exists(sys.argv[1]):
    print sys.argv[1] + " doesn't exist"
    sys.exit()

#data = np.fromfile(sys.argv[1])
numcols = len(open(sys.argv[1]).readlines()[0].split())
print numcols
numblpols = numcols - 8
usecols = [1]
for i in range(8,numcols):
    usecols.append(i)
print "Number of bppols", numblpols
data = np.loadtxt(sys.argv[1], usecols=usecols)
print data[0,1]
for i in range(numblpols/4):
    vis = data[:,1+i*4] + 1j*data[:,1+i*4+1]
    lags = np.abs(np.fft.ifft(vis))
    snr = np.max(lags)/np.std(lags)
    argmax = np.argmax(lags)
    print "S/N of blpol", i, "is", snr, "occuring at channel", argmax
    output = open("lag-%d.txt" % (i+1),"w")
    for i in range(len(lags)):
        output.write("%.9f %.9f\n" % (data[i,0], lags[i])) 
    output.close()
