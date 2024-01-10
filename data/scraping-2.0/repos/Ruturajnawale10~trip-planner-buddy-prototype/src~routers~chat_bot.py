from fastapi import APIRouter
from openai import OpenAI

import string
from configs.db import db
from configs.configs import settings
from utils import poi_util, user_util, prompt_util, file_util
from typing import Optional


client = OpenAI(api_key=settings.gpt_key)

router = APIRouter(
    tags=['Chat Bot']
)

@router.post("/api/gpt/response")
def generate_answer( input: str, previous_chat: list = []):
    print("Generating response for input : ", input)
    msg_list = []
    msg_list.append({"role": "system", "content": "You are a helpful chat bot which answers questions about tourist places which give response in 1-2 lines"})
    for msg in previous_chat:
        msg_list.append({"role": "user", "content": msg["user_input"]})
        msg_list.append({"role": "assistant", "content": msg["system"]})
    msg_list.append({"role": "user", "content": input})
    response = client.chat.completions.create(
        model= settings.gpt_generic_model,
        messages=msg_list
    )
    gpt_output = str(response.choices[0].message.content)
    return gpt_output