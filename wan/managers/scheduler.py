import time

from wan.server_args import ServerArgs, set_global_server_args


class Scheduler:
  def __init__(
    self,
    server_args: ServerArgs,
  ):
    self.server_args = server_args

    set_global_server_args(server_args)

    self._running = True

  def event_loop(self) -> None:
    while self._running:
      time.sleep(1)
