#!/bin/bash

nproc=$(nproc)
maxThreads=$((nproc+5))
# maxThread=20

for t in $(seq 1 $maxThreads); do
    echo  -n "NTHREAD= $t"
    ../src/bench_fxkernel -t $t test6.conf | grep Mbps
done
