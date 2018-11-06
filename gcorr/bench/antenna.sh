#!/bin/bash

NLOOP=1

nant=(4 6 8 10 12 16)

for a in ${nant[@]}; do
    echo -n "NANT= $a  "
    ../src/testgpukernel -n $NLOOP test${a}.conf | grep Gbps
done
