import logging
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from langchain_deepread.server.utils.auth import authenticated
from langchain_deepread.server.moderation.moderation_server import ModerationService
from langchain_deepread.server.utils.model import RestfulModel
from langchain_deepread.server.moderation.moderation_server import GPTResponse

logger = logging.getLogger(__name__)

moderation_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class ModerationBody(BaseModel):
    context: str
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "context": "This is okay",
                }
            ]
        }
    }


@moderation_router.post(
    "/moderation",
    response_model=RestfulModel[GPTResponse | None],
    tags=["Moderation"],
)
def summary(request: Request, body: ModerationBody) -> RestfulModel:
    """
    Summarize the article based on the given content and requirements
    """
    service = request.state.injector.get(ModerationService)
    return RestfulModel(data=service.moderation(**body.model_dump()))
