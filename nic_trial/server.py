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

print('Connecting to {0}:{1}'.format(server.bind_address, server.port))
result = server.run()

if result.error:
    print(result.error)
else:
    print('')
    print('Test completed:')
    print('  started at         {0}'.format(result.time))
    print('  bytes transmitted  {0}'.format(result.bytes))
    print('  jitter (ms)        {0}'.format(result.jitter_ms))
    print('  avg cpu load       {0}%\n'.format(result.local_cpu_total))

    print('Average transmitted data in all sorts of networky formats:')
    print('  bits per second      (bps)   {0}'.format(result.bps))
    print('  Kilobits per second  (kbps)  {0}'.format(result.kbps))
    print('  Megabits per second  (Mbps)  {0}'.format(result.Mbps))
    print('  KiloBytes per second (kB/s)  {0}'.format(result.kB_s))
    print('  MegaBytes per second (MB/s)  {0}'.format(result.MB_s))
