#!/bin/bash

nproc=$(nproc)
maxThreads=$((nproc+5))
# maxThreads=20
NLOOP=10

for t in $(seq 1 $maxThreads); do
    echo  -n "NTHREAD= $t"
    ../src/bench_fxkernel -n $NLOOP -t $t test6.conf | grep Mbps
done
