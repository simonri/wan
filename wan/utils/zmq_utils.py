"""ZMQ helper, mirroring sglang's get_zmq_socket signature.

sglang convention: long-lived process (HTTP) binds, child process (worker) connects.
We use IPC over Unix sockets — single-host, no port allocation, no firewall surface.
"""

import zmq
import zmq.asyncio


def get_zmq_socket(
  context: zmq.Context,
  socket_type: int,
  endpoint: str,
  bind: bool,
) -> zmq.Socket:
  socket = context.socket(socket_type)
  if bind:
    socket.bind(endpoint)
  else:
    socket.connect(endpoint)
  return socket
