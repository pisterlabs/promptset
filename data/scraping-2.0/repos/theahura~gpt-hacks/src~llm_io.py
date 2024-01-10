"""
Contains base functions for calling out to the openai api.
"""
from typing import Any, cast, TypedDict, List, Optional, Tuple

import os
import openai


class Message(TypedDict):
  role: str
  content: str
  finish_reason: Optional[str]


class InputMessage(TypedDict):
  role: str
  content: str


class Response(TypedDict):
  index: int
  message: Message


class GPTChatCompletionUsage(TypedDict):
  prompt_tokens: int
  completion_tokens: int
  total_tokens: int


class GPTChatCompletionResponse(TypedDict):
  id: str
  object: str
  created: int
  model: str
  choices: List[Response]
  usage: GPTChatCompletionUsage


def get_access_token():
  # Returns a valid access token from the OS Environment
  return os.getenv("OPENAI_API_KEY")


def call_gpt(
    messages: List[InputMessage],
    model: Optional[str] = 'gpt-3.5-turbo'
) -> Tuple[Message, GPTChatCompletionUsage]:
  """Calls OpenAI API with the above messages"""
  openai.api_key = get_access_token()
  response = cast(Any,
                  openai.ChatCompletion.create(model=model, messages=messages))
  completion = cast(GPTChatCompletionResponse, response.to_dict_recursive())
  return completion['choices'][0]['message'], completion['usage']


def to_gpt_message(role: str, content: str) -> InputMessage:
  return {'role': "user", 'content': content}
