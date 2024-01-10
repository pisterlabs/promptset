from fastapi import APIRouter

from services.llm.OpenAI import OpenAI
from services.llm.Anthropic import Anthropic

router = APIRouter()

open_ai = OpenAI()
anthropic = Anthropic()

@router.get("/v1/models")
def get_models():
    return open_ai.get_models() + anthropic.get_models()

@router.post("/v1/chat/completions")
def completion(req: dict):
    model = req["model"]
    if model in anthropic.get_models():
        return anthropic.generate(req["prompt"], req["model"])

    return open_ai.generate(req["messages"], req["model"])
