import logging
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from langchain_deepread.server.utils.auth import authenticated
from langchain_deepread.server.qa.qa_service import (
    QAService,
    QAResponse,
)
from langchain_deepread.server.utils.model import RestfulModel

logger = logging.getLogger(__name__)

qa_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class QABody(BaseModel):
    question: str
    context: str
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What is Task Decomposition?",
                    "context": "Task decomposition is a technique used to break down complex tasks into smaller and simpler steps.\
                     It can be done through prompting techniques like Chain of Thought or Tree of Thoughts",
                }
            ]
        }
    }


@qa_router.post(
    "/qa",
    response_model=RestfulModel[QAResponse | None],
    tags=["QA"],
)
def summary(request: Request, body: QABody) -> RestfulModel:
    """
    Summarize the article based on the given content and requirements
    """
    service = request.state.injector.get(QAService)
    print(body)
    return RestfulModel(data=service.qa(**body.model_dump()))
