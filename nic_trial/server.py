#!/usr/bin/env python

import iperf3
import socket

# Create a server
server = iperf3.Server()

# Setup server interface
if socket.gethostname() == "frigg":
    server.server_hostname = "10.17.4.1"
if socket.gethostname() == "odin":
    server.server_hostname = "10.17.4.2"
server.port = 5001

print ("Server is running on address {} and port {}".format(server.server_hostname,
                                                            server.port))

result = server.run()
