import os
import sys

from fastapi import Path, Query, Body, APIRouter
from fastapi.responses import JSONResponse

sys.path.append("..")

from models.requests import OpenAICreateCompletionRequest, OpenAICreateChatRequest
from models.responses import OpenAICompletionResponse, OpenAIChatResponse
from bots._openai import OpenAIClient

openai_router = APIRouter(prefix="/openai", tags=["OpenAI"])


@openai_router.post("/completion", response_model=OpenAICompletionResponse)
def create_completion(
    data: OpenAICreateCompletionRequest = Body(..., title="请求数据"),
):
    openai_client = OpenAIClient()
    res = openai_client.completion(**data.dict())
    return JSONResponse(
        {
            "id": res["id"],
            "answer": res["choices"][0]["text"],
            "usage": res["usage"]["total_tokens"],
        }
    )


@openai_router.post("/chat", response_model=OpenAIChatResponse)
def create_chat(data: OpenAICreateChatRequest = Body(..., title="请求数据")):
    openai_client = OpenAIClient()
    res = openai_client.chat_completion(**data.dict())
    return JSONResponse(
        {
            "id": res["id"],
            "message": res["choices"][0]["message"],
            "usage": res["usage"]["total_tokens"],
        }
    )
