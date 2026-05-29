"""Server entrypoint. Forks the GPU worker, waits for it to load the model and
bind sockets, then starts uvicorn.

Mirrors sglang's launch_server: parent reads init info from a one-shot pipe to
gate HTTP startup until the scheduler is actually ready.
"""

import multiprocessing as mp
import os
import signal
import sys
import threading

import psutil
import uvicorn

from wan.entrypoints.http_server import create_app
from wan.managers.scheduler import run_scheduler_process
from wan.server_args import ServerArgs, set_global_server_args


def launch_http_server(server_args: ServerArgs):
  set_global_server_args(server_args)
  app = create_app(server_args)
  uvicorn.run(app, use_colors=True, host=server_args.host, port=server_args.port, reload=False)


def launch_server(server_args: ServerArgs):
  print("Starting server...")
  server_args.init_ipc_names()
  set_global_server_args(server_args)

  # Must use "spawn" start method: the parent already touched CUDA and a fork
  # would inherit a broken context. sglang does the same.
  ctx = mp.get_context("spawn")

  # One-shot pipe: child sends {"status": "ready"} after the model is loaded
  # AND the IPC sockets are connected; parent reads before uvicorn starts.
  pipe_reader, pipe_writer = ctx.Pipe(duplex=False)

  worker = ctx.Process(
    target=run_scheduler_process,
    args=(server_args, pipe_writer),
    name="wan-scheduler",
    daemon=False,
  )
  worker.start()
  # Close our copy of the writer end so the only writer is the child.
  pipe_writer.close()

  # Block here until the scheduler reports ready. If the worker dies before
  # signalling, pipe_reader.recv() raises EOFError.
  try:
    init_info = pipe_reader.recv()
  except EOFError as e:
    worker.join(timeout=5)
    raise RuntimeError(f"scheduler died before sending init info (exit={worker.exitcode})") from e
  finally:
    pipe_reader.close()

  if init_info.get("status") != "ready":
    raise RuntimeError(f"scheduler failed to start: {init_info}")
  print(f"Scheduler ready (pid={worker.pid}); starting HTTP on {server_args.host}:{server_args.port}")

  try:
    launch_http_server(server_args)
  finally:
    print("HTTP server stopped; terminating scheduler...")
    if worker.is_alive():
      worker.terminate()
      worker.join(timeout=10)
      if worker.is_alive():
        worker.kill()


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
  # Use the Wan I2V config explicitly; the base PipelineConfig in ServerArgs's
  # default is generic and lacks VAE z_dim etc.
  from wan.configs.pipeline.wan import WanI2VConfig
  server_args = ServerArgs(pipeline_config=WanI2VConfig())

  try:
    launch_server(server_args)
  finally:
    kill_process_tree(os.getpid(), include_parent=False)
