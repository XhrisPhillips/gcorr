#!/bin/bash -f

echo "***CHANNELS***"
./channels.sh
echo "***ANTENNA***"
./antenna.sh
echo "***ANTENNA 16384***"
./antenna-16384.sh
echo "***CHANNELS 8bit***"
./channels-8.sh
echo "***ANTENNA 8bit***"
./antenna-8bit.sh
echo "***ANTENNA 16384***"
./antenna-16384-8bit.sh

echo "*****************"
echo "******HALF*******"
echo "*****************"

echo "***CHANNELS***"
./channels-half.sh
echo "***ANTENNA***"
./antenna-half.sh
echo "***ANTENNA 16384***"
./antenna-16384-half.sh
echo "***CHANNELS 8bit***"
./channels-8half.sh
echo "***ANTENNA 8bit***"
./antenna-8bit-half.sh
echo "***ANTENNA 16384***"
./antenna-16384-8bit-half.sh

