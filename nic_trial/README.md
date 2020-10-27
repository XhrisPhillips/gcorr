# branch nic_trial
This branch is created by Xinping Deng and it is a branch to try out network interface performance for BIGCAT project.

# The goal
We want to see if we can apply RoCE and how hard/easy that will be. By doing so, how much improvement we can get.

# Some random pages
[How to Disable RoCE](https://community.mellanox.com/s/article/How-to-Disable-RoCE).

[RDMA/RoCE Solutions](https://community.mellanox.com/s/article/rdma-roce-solutions)

[Recommended Network Configuration Examples for RoCE Deployment](https://community.mellanox.com/s/article/recommended-network-configuration-examples-for-roce-deployment)

[RoCE Configuration on Mellanox Adapters (PCP-Based Lossless Traffic)](https://community.mellanox.com/s/article/roce-configuration-on-mellanox-adapters--pcp-based-lossless-traffic-x)

# [iperf](https://fasterdata.es.net/performance-testing/network-troubleshooting-tools/iperf/)
``sudo apt install iperf`` or ``sudo apt install iperf3``

``pip install iperf3`` for [Python version](https://github.com/thiezn/iperf3-python).

# TCP test
TCP test does not need CPU affinity, it always gives us ~30Gbps (send and receive) per process.

On server side (frigg), ran:

``iperf -i 1 -p 9999 -s``

On client side (odin), run

``iperf -i 1 -p 9999 -c 10.17.4.1 -t 100``

# UDP test
CPU affinity makes difference for UDP test. 

## Without CPU affinity

On server side (frigg), run

``iperf -u -i 1 -p 5201 -l 14700 -s``

On client side (odin), run

``iperf -u -i 1 -p 5201 -l 14700 -c 10.17.4.1 -t 100 -b 100g``

## With CPU affinity

On server side (frigg), run

``taskset -c 0 iperf -u -i 1 -p 5201 -l 14700 -s``

On client side (odin), run

``taskset -c 0 iperf -u -i 1 -p 5201 -l 14700 -c 10.17.4.1 -t 100 -b 100g``

If you run TCP and UDP tests at the same time, it is also important to setup TCP affinity.

The size of datagrams is also importtnat, the current good number is 14700 on both side.

## iperf3

For server

``iperf3 -i 1 -p 5201 -A 0 -s`` this works for both TCP and UDP. 

For UDP client
``iperf3 -i 1 -p 5201 -A 0 -u -l 14700 -c 10.17.4.1 -t 100 -b 100g``

For TCP client
``iperf3 -i 1 -p 5201 -A 0 -c 10.17.4.1 -t 100``

100Gbps NiC is on NUMA 0 for frigg, is on NUMA 1 for odin.

``iperf3 -i 1 -p 5201 -c 10.17.4.1 -u -b 100G -l 8948 -w 50000K -A 0``

# PSRDADA installation configuration
*nvcc must be in PATH to make the configure find CUDA*

./configure --prefix=/home/den15c/.local --with-cuda-dir=/usr/local/cuda-10.2 --with-cuda-include-dir=/usr/local/cuda-10.2/include  --with-cuda-lib-dir=/usr/local/cuda-10.2/lib64 --with-fftw3-dir=/usr --with-fftw3-include-dir=/usr/include --with-fftw3-lib-dir=/usr/lib/x86_64-linux-gnu --with-gsl-dir=/usr --with-gsl-include-dir=/usr/include/gsl --with-gsl-lib-dir=/usr/lib/x86_64-linux-gnu --with-hwloc-dir=/usr --with-hwloc-include-dir=/usr/include/hwloc --with-hwloc-lib-dir=/usr/lib/x86_64-linux-gnu

## Support features

|Supported |Feature                                                             |
|:---------|-------------------------------------------------------------------:|
|YES   	   |CUDA								|
|YES   	   |FFTW 3.3+								|
|YES   	   |Gnu Scientific Library (GSL)					|
|YES   	   |Portable Hardware Locality (hwloc)				  	|
|NO    	   |Intel Performance Primitives (ipp)					|
|NO    	   |Intel Math Kernel Library (mkl)					|
|NO    	   |Remote Direct Memory Access CM libraries (from OFED distribution) 	|

## Create ring buffer at GPU memory
Use ``dada_db`` as usual with additional options ``-g 0 -w``. ``-g`` to give GPU device ID and ``-w`` is required to make it works as expected.

# frigg and odin
|Server |NiC            |IP             |NUMA   |
|:------|:-------------:|:-------------:|------:|
|frigg	|ens17f1	|10.17.4.1	|0	|
|odin 	|ens27f1	|10.17.4.2	|1	|

# To do
1. get a simple UDP capture code with just a counter
2. add in header decoder
3. add in memory copy
4. add in multiple threads feature and sync between multiple ports?