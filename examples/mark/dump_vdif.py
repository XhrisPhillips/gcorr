#! /usr/bin/env python

import os
import struct
import sys
import time

def vex2time(str):
    tupletime = time.strptime(str, "%Yy%jd%Hh%Mm%Ss");
    return time.mktime(tupletime)

def time2vex(secs):
    tupletime = time.gmtime(secs)
    return time.strftime("%Yy%jd%Hh%Mm%Ss", tupletime)

os.environ['TZ'] = 'UTC'
time.tzset()

start = vex2time(sys.argv[3])
stop = vex2time(sys.argv[4])
last_frame = -1
last_t = 0

infp = open(sys.argv[1], 'r')
outfp = open(sys.argv[2], 'w')
while infp:
    header_size = 16
    buf = infp.read(16)
    words = struct.unpack("4I", buf)
    invalid = words[0] >> 31
    legacy = (words[0] >> 30) & 0x1
    seconds = words[0] & 0x3fffffff
    frame = words[1] & 0x00ffffff
    epoch = (words[1] >> 24) & 0x3f
    length = words[2] & 0x00ffffff
    num_channels = 1 << ((words[2] >> 24) & 0x1f)
    bits_per_sample = ((words[3] >> 26) & 0x1f) + 1
    thread_id = ((words[3] >> 16) & 0x3ff)
    station_id = words[3] & 0x0000ffff
    year = 2000 + epoch / 2
    if epoch % 2 == 0:
        month = 1
    else:
        month = 7
        pass
    t = time.mktime((year, month, 1, 0, 0, 0, -1, -1, -1))
    t += seconds

    if not legacy:
        header_size = 32
        infp.read(16)
        pass
    buf = infp.read(length * 8 - header_size)

    if thread_id == 0 and t >= stop:
        break
    if thread_id == 0 and t >= start:
        if last_frame == -1 and frame != 0:
            print "Warning: data doesn't start on integral second boundary"
            pass
        if frame != last_frame + 1 and last_t == t:
            print time.ctime(t)
            print "Warning: missing frame", last_frame + 1
            pass
        last_frame = frame
        last_t = t
        outfp.write(buf)
        pass
    continue
