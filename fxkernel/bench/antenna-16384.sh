#!/bin/bash

NTHREAD=<setme>
NCHAN=16384
NLOOP=10

nant=(4 6 8 10 12 16)

for a in ${nant[@]}; do
    echo  -n "NANT= $a"
    ../src/bench_fxkernel -n $NLOOP -t $NTHREAD test${a}-16384.conf | grep Mbps
done
