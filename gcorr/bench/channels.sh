#!/bin/bash

set -e

nchan=(64  256 1024 4096 16384)

NLOOP=100

for c in ${nchan[@]}; do
    echo  -n "NCHAN= $c  "
    ../src/testgpukernel -n $NLOOP test8-${c}.conf | grep Gbps
done
