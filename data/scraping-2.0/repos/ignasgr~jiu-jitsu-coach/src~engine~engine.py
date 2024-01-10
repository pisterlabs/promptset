import os
import random
import time
from typing import List, Dict

import openai
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


openai.api_key = os.environ["OPENAI_API_KEY"]


class ChatHistory(BaseModel):
    history: List[Dict]


class TextInput(BaseModel):
    text: str


def get_llm_response(messages):

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True
    )

    for chunk in response:
        content = chunk.choices[0].delta.get("content", "")
        yield content


app = FastAPI()

@app.get("/")
def root():
    return {"Health Status": "OK"}


@app.post("/chat")
def chat(payload: ChatHistory):

    stream = StreamingResponse(
        content=get_llm_response(payload.history),
        media_type="text/event-stream"
    )

    return stream


@app.post("/embedding")
def create_embedding(payload: TextInput) -> list:

    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=payload.text
    )

    return response.data[0].embedding