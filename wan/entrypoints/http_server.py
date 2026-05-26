from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.routing import APIRouter

from wan.runtime.scheduler_client import async_scheduler_client
from wan.server_args import ServerArgs


@asynccontextmanager
async def lifespan(app: FastAPI):
  # 1. init the singleton client
  server_args = app.state.server_args
  async_scheduler_client.initialize(server_args)

  yield

  # on shutdown
  print("FastAPI app is shutting down...")
  async_scheduler_client.close()


health_router = APIRouter()


@health_router.get("/health")
async def health():
  return {"status": "ok"}


def create_app(server_args: ServerArgs):
  app = FastAPI()

  app.state.server_args = server_args
  return app
