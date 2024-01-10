import httpx
import logging

from . import router
from .lib.types import Health
from backends.utils._openai import LEAPFROGAI_HEALTH_URL
from backends.utils.exceptions import OPENAI_UNREACHABLE


logger = logging.getLogger("health")


@router.get("/healthz", tags=["health"])
async def healthz() -> Health:
    return Health()


@router.get("/upstream/healthz", tags=["upstream health"])
async def upstream_healthz() -> Health:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(LEAPFROGAI_HEALTH_URL)

            status = response.status_code
            if status == 200:
                return Health()
            else:
                logger.warn(f"Upstream API did not return status 200: {response}")
                return Health(
                    status=f"WARNING: upstream connection is alive but other requests may fail due to upstream status code {status}"
                )
    except Exception as e:
        logger.error(f"{OPENAI_UNREACHABLE.detail}: {e}")
        raise OPENAI_UNREACHABLE
