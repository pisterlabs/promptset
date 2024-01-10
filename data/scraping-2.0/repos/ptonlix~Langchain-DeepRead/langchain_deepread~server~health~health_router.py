from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field
from langchain_deepread.server.utils.model import RestfulModel
import logging

logger = logging.getLogger(__name__)

# Not authentication or authorization required to get the health status.
health_router = APIRouter(prefix="/v1")


class HealthResponse(BaseModel):
    status: Literal["ok"] = Field(default="ok")


@health_router.get(
    "/health",
    tags=["Health"],
    response_model=RestfulModel[HealthResponse | None],
)
def health() -> RestfulModel:
    logger.error(1234566)
    """Return ok if the system is up."""
    return RestfulModel(data=HealthResponse(status="ok"))
