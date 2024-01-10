from fastapi import APIRouter
from pydantic import BaseModel, Field
from service import openai_api_service

router = APIRouter(
    prefix="/api",
    tags=["api"],
    responses={404: {"description": "Not found"}},
)


class DetermineAPIOpenAIRequest(BaseModel):
    prompt: str = Field(min_length=1)


@router.post("/openai/determination")
def openai_determine_api(request: DetermineAPIOpenAIRequest):
    return openai_api_service.determine_api(request.prompt)
