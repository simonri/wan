"""HTTP-side counterpart to the Scheduler. Mirrors sglang TokenizerManager's IPC.

Persistent PUSH (to scheduler) + persistent PULL (from scheduler). The PULL is
drained by a background asyncio task `handle_loop` which fans outputs back to
per-job asyncio.Events keyed by job_id (sglang's rid_to_state pattern).
"""

import asyncio

import zmq
import zmq.asyncio

from wan.managers.io_struct import BatchGenerateOutput, BatchGenerateReq
from wan.server_args import ServerArgs
from wan.utils.zmq_utils import get_zmq_socket


class _JobState:
  __slots__ = ("event", "output")

  def __init__(self) -> None:
    self.event = asyncio.Event()
    self.output: BatchGenerateOutput | None = None


class AsyncSchedulerClient:
  def __init__(self) -> None:
    self.context: zmq.asyncio.Context | None = None
    self.send_to_scheduler: zmq.asyncio.Socket | None = None
    self.recv_from_scheduler: zmq.asyncio.Socket | None = None
    self.server_args: ServerArgs | None = None
    self.rid_to_state: dict[str, _JobState] = {}
    self._handle_loop_task: asyncio.Task | None = None

  def initialize(self, server_args: ServerArgs) -> None:
    if self.context is not None:
      print("AsyncSchedulerClient already initialized")
      return

    self.server_args = server_args
    self.context = zmq.asyncio.Context(2)
    # HTTP side binds; the worker process connects. Match sglang's convention.
    self.send_to_scheduler = get_zmq_socket(
      self.context, zmq.PUSH, server_args.scheduler_input_ipc_name, bind=True
    )
    self.recv_from_scheduler = get_zmq_socket(
      self.context, zmq.PULL, server_args.tokenizer_ipc_name, bind=True
    )
    print("AsyncSchedulerClient initialized")

  def start_handle_loop(self) -> None:
    """Spawn the background task that drains recv_from_scheduler.

    Must be called from inside a running asyncio loop (e.g. FastAPI lifespan).
    """
    if self._handle_loop_task is not None:
      return
    self._handle_loop_task = asyncio.create_task(self._handle_loop())

  async def _handle_loop(self) -> None:
    assert self.recv_from_scheduler is not None
    while True:
      try:
        recv_obj = await self.recv_from_scheduler.recv_pyobj()
      except asyncio.CancelledError:
        return
      except zmq.ContextTerminated:
        return
      except Exception as e:
        print(f"AsyncSchedulerClient handle_loop recv error: {e}")
        continue

      if not isinstance(recv_obj, BatchGenerateOutput):
        print(f"AsyncSchedulerClient: dropping unknown msg {type(recv_obj).__name__}")
        continue

      state = self.rid_to_state.pop(recv_obj.job_id, None)
      if state is None:
        print(f"AsyncSchedulerClient: no waiter for job_id={recv_obj.job_id}")
        continue
      state.output = recv_obj
      state.event.set()

  async def submit(self, msg: BatchGenerateReq) -> BatchGenerateOutput:
    """Send one job and await its result. msg.job_id is the dispatch key."""
    if self.send_to_scheduler is None:
      raise RuntimeError("AsyncSchedulerClient is not initialized")
    state = _JobState()
    self.rid_to_state[msg.job_id] = state
    await self.send_to_scheduler.send_pyobj(msg)
    await state.event.wait()
    assert state.output is not None
    return state.output

  def close(self) -> None:
    if self._handle_loop_task is not None:
      self._handle_loop_task.cancel()
      self._handle_loop_task = None
    if self.send_to_scheduler is not None:
      self.send_to_scheduler.close(linger=0)
      self.send_to_scheduler = None
    if self.recv_from_scheduler is not None:
      self.recv_from_scheduler.close(linger=0)
      self.recv_from_scheduler = None
    if self.context is not None:
      self.context.term()
      self.context = None


# Singleton used by the HTTP routes.
async_scheduler_client = AsyncSchedulerClient()
