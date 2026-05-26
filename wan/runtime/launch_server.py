import multiprocessing as mp
import os
import signal
import sys
import threading

import psutil
import uvicorn

from wan.entrypoints.http_server import create_app
from wan.managers.gpu_worker import run_scheduler_process
from wan.server_args import ServerArgs, set_global_server_args


def launch_http_server(server_args: ServerArgs):
  set_global_server_args(server_args)
  app = create_app(server_args)
  uvicorn.run(app, use_colors=True, host="0.0.0.0", port=8000, reload=False)


def launch_server(server_args: ServerArgs):
  print("Starting server...")

  scheduler_pipe_reader, scheduler_pipe_writer = mp.Pipe(duplex=False)

  process = mp.Process(target=run_scheduler_process, args=server_args, name="worker", daemon=True)

  process.start()

  scheduler_pipe_writer.close()
  scheduler_pipe_reader.close()

  launch_http_server(server_args)


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
  if threading.current_thread() is threading.main_thread():
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)

  if parent_pid is None:
    parent_pid = os.getpid()
    include_parent = False

  try:
    itself = psutil.Process(parent_pid)
  except psutil.NoSuchProcess:
    return

  children = itself.children(recursive=True)
  for child in children:
    if child.pid == skip_pid:
      continue
    try:
      child.kill()
    except psutil.NoSuchProcess:
      pass

  if include_parent:
    try:
      if parent_pid == os.getpid():
        itself.kill()
        sys.exit(0)

      itself.kill()

      itself.send_signal(signal.SIGQUIT)
    except psutil.NoSuchProcess:
      pass


if __name__ == "__main__":
  server_args = ServerArgs()

  try:
    launch_server(server_args)
  finally:
    kill_process_tree(os.getpid(), include_parent=False)
