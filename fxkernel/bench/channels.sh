#!/bin/bash

nchan=(64  256 1024 4096 16384)
NTHREAD=<setme>
NLOOP=10

for c in ${nchan[@]}; do
    echo  -n "NCHAN= $c"
    ../src/bench_fxkernel -n $NLOOP -t $NTHREAD test8-${c}.conf | grep Mbps
done
