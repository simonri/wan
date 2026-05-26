import zmq

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

  def close(self):
    if self.context:
      self.context.term()
      self.context = None


# singleton instances for easy access
async_scheduler_client = AsyncSchedulerClient()
