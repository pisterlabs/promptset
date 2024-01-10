import json
from enum import Enum
from pathlib import Path

import openai
from fastapi import FastAPI, Request, Response
from openai.openai_object import OpenAIObject
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

from pickl import PROMPT_DIR

# TODO: add openai auth

# Create FastAPI app
app = FastAPI()

# Load system prompt
SYSTEM_PROMPT = Path(PROMPT_DIR, "api_example_system.txt").read_text(encoding="utf-8")


# Data structures
class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class Message(BaseModel):
    role: Role
    content: str


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(4))
def chat(
    messages: list[Message],
    stream: bool = False,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.8,
    max_tokens: int = 1000,
) -> OpenAIObject:
    try:
        # Prepare request arguments
        request_args = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "messages": [json.loads(x.model_dump_json()) for x in messages],
        }

        # Make request
        gpt_res: OpenAIObject = openai.ChatCompletion.create(**request_args)
        return gpt_res
    except openai.OpenAIError as e:
        raise RuntimeError from e


@app.post("/chatgpt")
async def chatgpt(request: Request) -> Response:
    """ChatGPT REST Endpoint"""
    try:
        payload = await request.json()
    except json.JSONDecodeError as e:
        return Response(status_code=400, content=str(e))

    # TODO: Validate payload
    # Build full prompt with conversation history
    messages_hist = [
        Message(role=Role.user if msg["speaker"] == "A" else Role.assistant, content=msg["message"])
        for msg in payload["messages"]
    ]

    # Build Request
    messages = [Message(role=Role.system, content=SYSTEM_PROMPT)] + messages_hist

    # Call Procedure
    response = chat(messages, model="gpt-3.5-turbo", temperature=0.8, max_tokens=200)
    sep_response = response["choices"][0]["message"]["content"]

    # Cut response if last character is not a punctuation
    valid_endings = ["!", "?", ".", "ㅎ", "ㅋ", "ㅠ", "~", ";", "ㅜ", "^", "*"]
    if sep_response[-1] not in valid_endings:
        max_idx = max([sep_response.rfind(x) for x in valid_endings])
        sep_response = sep_response[: max_idx + 1]

    sep_response = [{"speaker": "B", "message": msg} for msg in sep_response.split("<|sep|>")]
    valid_response = {"response": sep_response}

    return Response(status_code=200, content=json.dumps(valid_response), media_type="application/json")
