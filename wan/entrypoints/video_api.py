import asyncio
import uuid

from fastapi import Request
from fastapi.params import Form
from fastapi.routing import APIRouter

from wan.entrypoints.protocol import VideoGenerationsRequest, VideoResponse
from wan.runtime.scheduler_client import AsyncSchedulerClient, async_scheduler_client
from wan.server_args import get_global_server_args
from wan.stages.schedule_batch import OutputBatch, Req

router = APIRouter(prefix="/v1/videos")


async def process_generation_batch(
  scheduler_client: AsyncSchedulerClient,
  batch,
) -> tuple[list[str], OutputBatch]:
  result = await scheduler_client.forward([batch])
  print("Done")
  return result


async def _dispatch_job_async(
  job_id: str,
  batch: Req,
  *,
  temp_dirs: list[str] | None = None,
) -> None:
  try:
    await process_generation_batch(async_scheduler_client, batch)
  except Exception as e:
    print(e)


@router.post("", response_model=VideoResponse)
async def create_video(
  request: Request,
) -> VideoResponse:
  server_args = get_global_server_args()

  try:
    body = await request.json()
  except Exception:
    body = {}
  req = VideoGenerationsRequest(**body)

  try:
    sampling_params = Wan2_2_I2V_SamplingParam(
      height=1280,
      width=720,
      num_frames=81,
      num_inference_steps=8,
      num_outputs_per_prompt=1,
      output_path=out_dir,
      output_file_name="output.mp4",
      profile=args.profile,
    )

  request_id = str(uuid.uuid4())

  batch = Req(sampling_params)

  asyncio.create_task(
    _dispatch_job_async(
      request_id,
      batch,
    )
  )

  return VideoResponse(id=job_id)
