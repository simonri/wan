"""GPU-side scheduler. Owns the model, consumes pickled jobs over ZMQ PULL,
runs forward_batch, saves the MP4s, sends back the result paths via ZMQ PUSH.

Mirrors sglang's Scheduler in shape but without TP/EP/disaggregation: one process,
one GPU, one job at a time. sglang patterns kept:
  - persistent PUSH/PULL sockets (not REQ/REP)
  - worker connects, HTTP side binds
  - send_pyobj / recv_pyobj wire (pickle under the hood)
  - init_info dict sent via pipe back to parent
"""

import ctypes
import gc
import multiprocessing as mp
import os
import os.path
import signal
import time
import traceback

import torch
import zmq

from wan.entrypoints.utils import expand_request_outputs
from wan.managers.io_struct import BatchGenerateOutput, BatchGenerateReq
from wan.pipeline.executor import SyncExecutor
from wan.pipeline.wan_i2v_pipeline import WanImageToVideoPipeline
from wan.postprocess import save_outputs
from wan.server_args import ServerArgs, set_global_server_args
from wan.utils.zmq_utils import get_zmq_socket


class Scheduler:
  def __init__(self, server_args: ServerArgs):
    self.server_args = server_args
    set_global_server_args(server_args)

    # Load the model ONCE. This is the expensive step (~30-60s). LoRAs are
    # NOT loaded here — each request specifies its own set, and the pipeline's
    # LoRA registry caches by nickname so repeated requests hit a fast path.
    print("[scheduler] loading pipeline...")
    t0 = time.perf_counter()
    executor = SyncExecutor(server_args=server_args)
    self.pipeline = WanImageToVideoPipeline(server_args=server_args, executor=executor)
    print(f"[scheduler] pipeline ready in {time.perf_counter() - t0:.2f}s")

    # Connect to HTTP-side bound sockets. sync zmq (not asyncio) since the
    # event loop here is a plain blocking recv -> forward -> send.
    self.zmq_context = zmq.Context(2)
    self.recv_from_tokenizer = get_zmq_socket(
      self.zmq_context, zmq.PULL, server_args.scheduler_input_ipc_name, bind=False
    )
    self.send_to_tokenizer = get_zmq_socket(
      self.zmq_context, zmq.PUSH, server_args.tokenizer_ipc_name, bind=False
    )

    self._running = True
    signal.signal(signal.SIGTERM, self._on_sigterm)
    signal.signal(signal.SIGINT, self._on_sigterm)

  def _on_sigterm(self, *_):
    print("[scheduler] received shutdown signal")
    self._running = False

  def get_init_info(self) -> dict:
    return {"status": "ready"}

  def event_loop(self) -> None:
    print("[scheduler] event loop running")
    while self._running:
      try:
        recv_obj = self.recv_from_tokenizer.recv_pyobj()
      except zmq.ContextTerminated:
        break
      except Exception as e:
        print(f"[scheduler] recv error: {e}")
        continue

      if not isinstance(recv_obj, BatchGenerateReq):
        print(f"[scheduler] dropping unknown message: {type(recv_obj).__name__}")
        continue

      self._handle_request(recv_obj)

    self._shutdown()

  def _handle_request(self, msg: BatchGenerateReq) -> None:
    job_id = msg.job_id
    print(f"[scheduler] job {job_id}: dispatch (loras={[lo.nickname for lo in msg.loras]})")
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    try:
      # Per-request LoRA swap. set_lora's clear_existing semantics replace the
      # current stack with msg.loras; nicknames already seen hit the cache so
      # only the apply/merge step runs, not the full safetensors read.
      t_lora = time.perf_counter()
      self.pipeline.set_lora(
        lora_nicknames=[lo.nickname for lo in msg.loras],
        lora_paths=[lo.path for lo in msg.loras],
        targets=[lo.target for lo in msg.loras],
        strengths=[lo.strength for lo in msg.loras],
      )
      print(f"[scheduler] job {job_id}: set_lora ({len(msg.loras)} adapters) in {time.perf_counter() - t_lora:.2f}s")

      request_group = expand_request_outputs(msg.req, num_prompts=1)
      output_batches = self.pipeline.forward_batch(request_group, self.server_args)
      file_paths: list[str] = []
      t_save = time.perf_counter()
      for ob_idx, ob in enumerate(output_batches):
        paths = save_outputs(
          outputs=ob.output,
          fps=msg.req.fps,
          save_output=True,
          build_output_path=lambda idx, jid=job_id, b=ob_idx: os.path.join(
            self.server_args.output_path, f"{jid}_{b}_{idx}.mp4"
          ),
          crf=msg.crf,
          enable_frame_interpolation=msg.enable_frame_interpolation,
          frame_interpolation_exp=msg.frame_interpolation_exp,
          frame_interpolation_scale=msg.frame_interpolation_scale,
        )
        file_paths.extend(paths)
      print(f"[scheduler] job {job_id}: save_outputs {time.perf_counter() - t_save:.2f}s")

      elapsed = time.perf_counter() - t0
      peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
      print(f"[scheduler] job {job_id}: ok ({elapsed:.2f}s, peak {peak_mb:.0f}MB, {len(file_paths)} files)")
      out = BatchGenerateOutput(
        job_id=job_id,
        file_paths=file_paths,
        inference_time_s=elapsed,
        num_outputs=len(file_paths),
        peak_memory_mb=peak_mb,
      )
    except Exception as e:
      tb = traceback.format_exc()
      print(f"[scheduler] job {job_id}: failed\n{tb}")
      out = BatchGenerateOutput(job_id=job_id, error=f"{type(e).__name__}: {e}")

    self.send_to_tokenizer.send_pyobj(out)
    # Drop request tensors + return reserved-but-unallocated VRAM so the next
    # job sees a clean arena. Peak per job is ~73 GB on 80 GB; without this,
    # back-to-back jobs OOM on the second one.
    gc.collect()
    torch.cuda.empty_cache()

  def _shutdown(self) -> None:
    try:
      self.recv_from_tokenizer.close(linger=0)
      self.send_to_tokenizer.close(linger=0)
      self.zmq_context.term()
    except Exception as e:
      print(f"[scheduler] shutdown socket cleanup error: {e}")
    gc.collect()
    if torch.cuda.is_initialized():
      torch.cuda.empty_cache()
    print("[scheduler] shutdown complete")


_PR_SET_PDEATHSIG = 1


def _die_when_parent_dies() -> None:
  """Linux: ask the kernel to SIGKILL us as soon as our parent exits.

  SIGKILL (not SIGTERM) because pyzmq's blocking recv internally retries on
  EINTR and never surfaces Python-level signals, so a SIGTERM signal handler
  would just queue and the worker would keep holding 70+ GB of VRAM until the
  next message arrives. SIGKILL is uninterruptible — the kernel releases the
  GPU on process death.
  """
  try:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
  except Exception as e:
    print(f"[scheduler] prctl(PR_SET_PDEATHSIG) failed: {e}")


def run_scheduler_process(server_args: ServerArgs, pipe_writer: mp.connection.Connection) -> None:
  """Entrypoint for the worker process.

  Loads the pipeline + binds sockets in Scheduler.__init__, then sends init info
  back to the parent so launch_server can start uvicorn only after we are ready.
  """
  _die_when_parent_dies()
  # If the parent already died between fork and this point, the prctl above will
  # have queued the signal — but guard with an explicit check too.
  if os.getppid() == 1:
    print("[scheduler] parent already gone at startup; aborting")
    return
  try:
    scheduler = Scheduler(server_args)
    pipe_writer.send(scheduler.get_init_info())
    pipe_writer.close()
    scheduler.event_loop()
  except Exception:
    tb = traceback.format_exc()
    print(f"[scheduler] fatal:\n{tb}")
    try:
      pipe_writer.send({"status": "failed", "error": tb})
      pipe_writer.close()
    except Exception:
      pass
    raise
