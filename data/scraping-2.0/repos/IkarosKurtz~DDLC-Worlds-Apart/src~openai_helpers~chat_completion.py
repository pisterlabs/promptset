from openai.error import ServiceUnavailableError
from ..errors import InvalidVersion
import openai
from typing import Literal as literal

Versions = ['4k', '16k']


def chat_completion(prompt: str,
                    ai_role: str = 'You are a helpful assistant.',
                    version: literal["4k", "16k"] = '4k') -> tuple[str]:
  response = None

  if version not in Versions:
    raise InvalidVersion(version)

  while response == None:
    try:
      response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo' if version == '4k' else 'gpt-3.5-turbo-16k',
        messages=[
          {'role': 'system', 'content': ai_role},
          {'role': 'user', 'content': prompt},
        ]
      )
    except ServiceUnavailableError as e:
      message = e._message
      if message != 'The server is overloaded or not ready yet.':
        raise e

      continue

  message = response.choices[0].message.content
  tokens = response['usage']['total_tokens']

  return (message, tokens)
