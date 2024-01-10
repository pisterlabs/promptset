from fastapi import FastAPI
from openai_resource import (
    ChatRequest,
    ChatResponse,
    get_chat_completion,
    get_models,
)

app = FastAPI()


@app.post("/v1/chat/completions")
async def chat_completion(chat_request: ChatRequest) -> ChatResponse:
    return await get_chat_completion(chat_request)


@app.get("/v1/models")
async def models():
    return await get_models()
