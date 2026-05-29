from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.routing import APIRouter

from wan.entrypoints.video_api import router as video_router
from wan.runtime.scheduler_client import async_scheduler_client
from wan.server_args import ServerArgs


@asynccontextmanager
async def lifespan(app: FastAPI):
  # Bind the IPC sockets and spawn the async handle_loop that drains scheduler
  # outputs into per-job waiters. We start the background task here (not in
  # initialize()) because the asyncio loop only exists once the app is running.
  server_args = app.state.server_args
  async_scheduler_client.initialize(server_args)
  async_scheduler_client.start_handle_loop()

  yield

  print("FastAPI app is shutting down...")
  async_scheduler_client.close()


health_router = APIRouter()


@health_router.get("/health")
async def health():
  return {"status": "ok"}


def create_app(server_args: ServerArgs):
  app = FastAPI(lifespan=lifespan)

  app.include_router(health_router)
  app.include_router(video_router)

  app.state.server_args = server_args
  return app
