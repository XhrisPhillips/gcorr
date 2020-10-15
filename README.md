# gcorr
Prototype GPU correlator for "small N" radio interferometers

# branch nic_trial
This branch is created by Xinping Deng and it is a branch to try out network interface performance for BIGCAT project.

# The goal
We want to see if we can apply RoCE and how hard/easy that will be. By doing so, how much improvement we can get.

# Some random pages
[How to Disable RoCE](https://community.mellanox.com/s/article/How-to-Disable-RoCE).

[RDMA/RoCE Solutions](https://community.mellanox.com/s/article/rdma-roce-solutions)

[Recommended Network Configuration Examples for RoCE Deployment](https://community.mellanox.com/s/article/recommended-network-configuration-examples-for-roce-deployment)

[RoCE Configuration on Mellanox Adapters (PCP-Based Lossless Traffic)](https://community.mellanox.com/s/article/roce-configuration-on-mellanox-adapters--pcp-based-lossless-traffic-x)

# First test
On the receive side (Odin), I ran:

``iperf -s -i1 -p 9999``

While on the sending side (Frigg) I ran:

``iperf -c 10.17.4.2 -t 100 -i 1 -p 9999``

# UDP with [iperf](https://fasterdata.es.net/performance-testing/network-troubleshooting-tools/iperf/)
On server side (frigg), run
``iperf -s -u``

On client side (odin), run
``iperf -c 10.17.4.1 -u -b 100000m``

``-b`` increases UDP bandwidth from default 1Mbps to 100000 Mbps.

# iperf
``sudo apt install iperf`` or ``sudo apt install iperf3``

``pip install iperf3`` for [Python version](https://github.com/thiezn/iperf3-python).