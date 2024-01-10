import os
import pprint
from typing import Literal, Optional

import openai
from dotenv import load_dotenv
from openai import ChatCompletion
from openai.embeddings_utils import cosine_similarity
from pydantic import BaseModel

import gstk.config as cfg
from gstk.models.chatgpt import Message

load_dotenv()


def model_to_openai_function_schema(name: str, model: type[BaseModel]) -> dict:
    return {
        "name": name,
        "description": model.__doc__,
        "parameters": model.model_json_schema(),
    }


async def get_chat_completion_response(
    messages: list[Message],
    tools: Optional[list[dict]] = None,
    chat_gpt_temperature: float = cfg.CHAT_GPT_TEMPERATURE,
    chat_gpt_model: str = cfg.CHAT_GPT_MODEL,
):
    """
    Implements function calling behavior as described here.
    https://platform.openai.com/docs/guides/gpt/function-calling
    """
    # import pprint
    # pprint.pprint([message.model_dump(exclude={"name"}) for message in messages])

    print("MESSAGES")
    pprint.pprint(messages)

    tool_choice: Optional[Literal["auto"] | dict]
    if not tools:
        tool_choice = None
    elif len(tools) == 1:
        # https://platform.openai.com/docs/api-reference/chat/create?lang=python
        tool_choice = {"type": "function", "function": {"name": tools[0]["function"]["name"]}}
    else:
        tool_choice = "auto"

    openai.api_key = os.environ["OPENAI_API_KEY"]
    return await ChatCompletion.acreate(
        model=chat_gpt_model,
        messages=[message.model_dump(exclude={"name"}) for message in messages],
        tools=tools,
        temperature=chat_gpt_temperature,
        tool_choice=tool_choice,
    )
    # max_tokens not used.


def get_function_tool(name: str, model: type[BaseModel]):
    return {"type": "function", "function": model_to_openai_function_schema(name, model)}


async def get_openai_vectorization(input: str) -> str:
    """
    Get the OpenAI embedding vectorization for string.
    """
    openai.api_key = os.environ["OPENAI_API_KEY"]
    return await openai.Embedding.acreate(input=input, model=cfg.CHAT_GPT_EMBEDDING_MODEL)


def get_vector_cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """
    Return the cosine similarity between two vectors.
    """
    return cosine_similarity(vector_a, vector_b)
