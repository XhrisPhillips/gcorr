#!/usr/bin/env python

import iperf3
import socket

# Create a server
server = iperf3.Server()

# Setup server interface
if socket.gethostname() == "frigg":
    server.bind_address = "10.17.4.1"
if socket.gethostname() == "odin":
    server.bind_address = "10.17.4.2"
server.port = 5201
server.protocol = 'udp'

print ("Server is running on address {} and port {}".format(server.server_hostname,
                                                            server.port))

result = server.run()
