#!/usr/bin/python
import os, sys

def evalpoly(poly, t):
    tval = 1.0
    accum = 0.0
    for i in range(len(poly)):
        accum += poly[i]*tval
        tval *= t
    return accum

if not len(sys.argv) == 2:
    print "Usage: %s <difx prefix>" % sys.argv[0]
    sys.exit()

inputfile = sys.argv[1] + ".input"
imfile = sys.argv[1] + ".im"
conffile = sys.argv[1] + ".conf"

if not os.path.exists(inputfile):
    print "Inputfile ", inputfile, "doesn't exist, aborting"
    sys.exit()

if not os.path.exists(imfile):
    print "IM file ", imfile, "doesn't exist, aborting"
    sys.exit()

if os.path.exists(conffile):
    print "Conffile ", conffile, "already exists, aborting"
    sys.exit()

inputlines = open(inputfile).readlines()
freq = 0
bw = 0
nchan = 0
nsta = 0
startmjd = 0
startsec = 0
telnames = []
datafiles = []
delays = []
for line in inputlines:
    if "FREQ (MHZ) 0" in line:
        freq = float(line.split()[-1])*1e6
    if "BW (MHZ) 0" in line:
        bw = float(line.split()[-1])*1e6
    if "NUM CHANNELS 0" in line:
        nchan = int(line.split()[-1])
    if "TELESCOPE ENTRIES:" in line:
        nsta = int(line.split()[-1])
    if "TELESCOPE NAME" in line:
        telnames.append(line.split()[-1])
    if "FILE" in line and "0:" in line:
        datafiles.append(line.split()[-1])
    if "START MJD" in line:
        startmjd = int(line.split()[-1])
    if "START SECONDS" in line:
        startsec = int(line.split()[-1])

immjd = 0
imsec = 0
imdelays = []
imlines = open(imfile).readlines()
for line in imlines:
    if "POLY" in line and "MJD" in line:
        secoff = startsec - imsec + (startmjd-immjd)*86400
        if secoff >= 0 and secoff < 120:
            break # We've already filled an acceptable value
        immjd = int(line.split()[-1])
    if "POLY" in line and "SEC" in line:
        imsec = int(line.split()[-1])
    if "DELAY (us)" in line:
        delays.append(line.split()[6:])

secoff = startsec - imsec + (startmjd-immjd)*86400
if secoff > 120 or secoff < 0:
    print "Wrong start time or IM file start time!", secoff, startsec, imsec, startmjd, immjd
    sys.exit()

output = open(conffile, "w")
output.write("COMPLEX 0\n")
output.write("NBIT 2\n")
output.write("NPOL 2\n")
output.write("NCHAN %d\n" % nchan)
output.write("LO %f\n" % freq)
output.write("BANDWIDTH %f\n" % bw)
output.write("NUMFFTS 16384\n")
output.write("NANT %d\n" % nsta)
fftdur = nchan/bw
fftspersec = int(1/fftdur + 0.5)
print fftdur, fftspersec
for i in range(len(datafiles)):
    poly = []
    for j in range(len(delays[i])):
        poly.append(float(delays[i][j]))
    t0 = secoff
    t1 = secoff + 0.5
    t2 = secoff + 1.0
    d0 = evalpoly(poly, t0)/1e6
    d1 = evalpoly(poly, t1)/1e6
    d2 = evalpoly(poly, t2)/1e6
    print d0, d1, d2
    a = (2*d0 - 4*d1 + 2*d2)/(fftspersec * fftspersec)
    b = (-3*d0 + 4*d1 - d2)/fftspersec
    c = d0
    output.write("%s %s %.15g %.15g %.15g 0\n" % (telnames[i], datafiles[i], a, b, c))

print "Don't forget to change the data file names to the raw data (headers stripped) file if needed"
