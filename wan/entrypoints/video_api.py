"""HTTP routes for video generation.

POST /v1/videos              -> enqueue a job, return {id, status:"queued"}
GET  /v1/videos/{video_id}   -> poll job state, returns the same VideoResponse

The dispatch path:
  1. POST handler validates the JSON body, builds a Req, generates a job_id,
     puts a VideoResponse(status="queued") in JOB_STORE, and spawns a background
     asyncio task that awaits AsyncSchedulerClient.submit(job_id, req).
  2. The submit() coroutine PUSHes a BatchGenerateReq onto the scheduler socket
     and waits on an asyncio.Event keyed by job_id.
  3. The Scheduler runs forward_batch, saves MP4s, PUSHes back a
     BatchGenerateOutput. The handle_loop on this side wakes the Event.
  4. The dispatch task updates JOB_STORE[job_id] with completed status + paths.
  5. GET handler just returns JOB_STORE[video_id].
"""

import asyncio
import time
import uuid

from fastapi import HTTPException, Request
from fastapi.routing import APIRouter

from wan.configs.sample.wan import Wan2_2_I2V_SamplingParam
from wan.entrypoints.protocol import VideoGenerationsRequest, VideoResponse
from wan.managers.io_struct import BatchGenerateReq
from wan.runtime.scheduler_client import async_scheduler_client
from wan.server_args import get_global_server_args
from wan.stages.schedule_batch import Req

router = APIRouter(prefix="/v1/videos")

# In-process job store. Cleared on restart; for now a plain dict is fine because
# the GIL serialises HTTP access and only the dispatch task mutates per-key.
JOB_STORE: dict[str, VideoResponse] = {}


def _build_sampling_params(vreq: VideoGenerationsRequest, output_path: str) -> Wan2_2_I2V_SamplingParam:
  """Map the public request fields onto Wan2_2_I2V_SamplingParam defaults."""
  return Wan2_2_I2V_SamplingParam(
    prompt=vreq.prompt,
    height=vreq.height if vreq.height is not None else 1280,
    width=vreq.width if vreq.width is not None else 720,
    num_frames=vreq.num_frames if vreq.num_frames is not None else 81,
    num_inference_steps=vreq.num_inference_steps if vreq.num_inference_steps is not None else 8,
    num_outputs_per_prompt=vreq.num_outputs_per_prompt or vreq.n or 1,
    fps=vreq.fps if vreq.fps is not None else 16,
    seed=vreq.seed if vreq.seed is not None else 42,
    generator_device=vreq.generator_device,
    output_path=output_path,
    output_file_name="output.mp4",
  )


async def _dispatch_job(job_id: str, msg: BatchGenerateReq) -> None:
  try:
    output = await async_scheduler_client.submit(msg)
    response = JOB_STORE.get(job_id)
    if response is None:
      return
    if output.error is not None:
      response.status = "failed"
      response.error = {"message": output.error}
    else:
      response.status = "completed"
      response.completed_at = int(time.time())
      response.file_paths = output.file_paths
      response.file_path = output.file_paths[0] if output.file_paths else None
      response.num_outputs = output.num_outputs
      response.peak_memory_mb = output.peak_memory_mb
      response.inference_time_s = output.inference_time_s
      response.progress = 100
  except Exception as e:
    response = JOB_STORE.get(job_id)
    if response is not None:
      response.status = "failed"
      response.error = {"message": f"{type(e).__name__}: {e}"}


@router.post("", response_model=VideoResponse)
async def create_video(request: Request) -> VideoResponse:
  server_args = get_global_server_args()
  try:
    body = await request.json()
  except Exception:
    body = {}
  vreq = VideoGenerationsRequest(**body)

  job_id = str(uuid.uuid4())
  sampling_params = _build_sampling_params(vreq, server_args.output_path)

  req = Req(sampling_params=sampling_params)
  req.prompt = vreq.prompt
  req.image_path = vreq.input_reference

  msg = BatchGenerateReq(
    job_id=job_id,
    req=req,
    loras=vreq.loras or [],
    enable_frame_interpolation=vreq.enable_frame_interpolation,
    frame_interpolation_exp=vreq.frame_interpolation_exp,
    frame_interpolation_scale=vreq.frame_interpolation_scale,
  )

  response = VideoResponse(id=job_id, status="queued")
  JOB_STORE[job_id] = response
  asyncio.create_task(_dispatch_job(job_id, msg))
  return response


@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(video_id: str) -> VideoResponse:
  response = JOB_STORE.get(video_id)
  if response is None:
    raise HTTPException(status_code=404, detail=f"video_id {video_id} not found")
  return response
