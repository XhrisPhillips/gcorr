#!/usr/bin/env python

import iperf3
import socket

# Create a client
client = iperf3.Client()

# Setup client interface
client.duration = 100
if socket.gethostname() == "frigg":
    client.server_hostname = "10.17.4.1"
if socket.gethostname() == "odin":
    client.server_hostname = "10.17.4.2"
client.port = 5001

print ("Client is running on address {} and port {}".format(client.server_hostname,
                                                            client.port))

result = client.run()
result.sent_Mbps
