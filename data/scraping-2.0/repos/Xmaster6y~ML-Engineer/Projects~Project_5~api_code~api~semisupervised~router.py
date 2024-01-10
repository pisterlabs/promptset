"""
Semi-supervised module router.
"""

import logging

from api.schema import ErrorResponse, TagResponse
from api.semisupervised import models
from api.semisupervised.constants import ALL_MODELS
from api.semisupervised.schema import SemisupervisedTagRequest
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from openai.error import AuthenticationError

router = APIRouter()


@router.on_event("startup")
async def startup_event():
    """
    Startup hook event.
    """
    logger = logging.getLogger("uvicorn")
    logger.info("Semi-supervised router starting up...")
    logger.info("Loading supervised models...")
    models.init_models()


@router.get("/info")
async def info():
    return {"all_models": ALL_MODELS}


@router.post(
    "/tag",
    response_model=TagResponse,
    responses={500: {"model": ErrorResponse}},
)
def tag(request: SemisupervisedTagRequest):
    """
    Tag.
    """
    title = request.title
    body_text = request.body_text
    body_code = request.body_code
    n_tag_requested = request.n_tag_requested
    model_name = request.model_name
    logger = logging.getLogger("uvicorn")

    try:
        tag_list = models.all_wrapped_models[model_name](
            title, body_text, body_code
        )
    except AuthenticationError:
        return JSONResponse(
            status_code=500, content={"error": "OpenAI API key invalid"}
        )
    except Exception as e:
        logger.error(e)
        return JSONResponse(
            status_code=500, content={"error": "Error computing tags"}
        )

    return {"tag_list": tag_list[:n_tag_requested]}
