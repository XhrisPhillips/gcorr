#!/bin/bash

NTHREAD=<testme>
NCHAN=256
NLOOP=10

nant=(4 6 8 10 12 16)

for a in ${nant[@]}; do
    echo  -n "NANT= $a"
    ../src/bench_fxkernel -n $NLOOP -t $NTHREAD test${a}.conf | grep Mbps
done
