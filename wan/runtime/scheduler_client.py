import pickle

import zmq
import zmq.asyncio

from wan.server_args import ServerArgs


class AsyncSchedulerClient:
  def __init__(self):
    self.context = None
    self.server_args = None

  def initialize(self, server_args: ServerArgs):
    if self.context is not None and not self.context.closed:
      print("AsyncSchedulerClient is already initialized")
      self.close()

    self.server_args = server_args
    self.context = zmq.asyncio.Context()
    print("AsyncSchedulerClient initialized")

  async def forward(self, batch: any) -> any:
    if self.context is None:
      raise RuntimeError("AsyncSchedulerClient is not initialized")

    # create temp REQ socket for this request to allow concurrent requests
    socket = self.context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    # 100 min timeout
    socket.setsockopt(zmq.RCVTIMEO, 600000)

    endpoint = self.server_args.scheduler_endpoint
    socket.connect(endpoint)

    try:
      await socket.send(pickle.dumps(batch))
      payload = await socket.recv()
      output_batch = pickle.loads(payload)
      return output_batch
    except zmq.error.Again:
      print("Timeout waiting for response from scheduler")
      raise TimeoutError("Timeout waiting for response from scheduler")
    finally:
      socket.close()

  def close(self):
    if self.context:
      self.context.term()
      self.context = None


# singleton instances for easy access
async_scheduler_client = AsyncSchedulerClient()
