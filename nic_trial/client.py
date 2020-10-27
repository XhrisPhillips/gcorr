#!/usr/bin/env python

import iperf3
import socket

client = iperf3.Client()
client.duration = 10
if socket.gethostname() == "frigg":
    client.server_hostname = "10.17.4.2"
if socket.gethostname() == "odin":
    client.server_hostname = "10.17.4.1"
client.port = 5201
client.protocol = 'udp'
#client.bandwidth = 100000000000

print('Connecting to {0}:{1}'.format(client.server_hostname, client.port))
result = client.run()

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
    print('  Megabits per second  (Gbps)  {0}'.format(result.Mbps/1024.))
    print('  KiloBytes per second (kB/s)  {0}'.format(result.kB_s))
    print('  MegaBytes per second (MB/s)  {0}'.format(result.MB_s))
    print('  MegaBytes per second (GB/s)  {0}'.format(result.MB_s/1024.))
