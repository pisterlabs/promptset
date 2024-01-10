# The AIConsole Project
#
# Copyright 2023 10Clouds
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
from typing import AsyncGenerator

import litellm
from litellm.caching import Cache
from openai import AuthenticationError

from aiconsole.api.websockets.server_messages import DebugJSONServerMessage
from aiconsole.core.gpt.partial import GPTPartialResponse
from aiconsole.core.gpt.request import GPTRequest

from .exceptions import NoOpenAPIKeyException
from .types import CLEAR_STR, CLEAR_STR_TYPE, GPTChoice, GPTResponse, GPTResponseMessage

_log = logging.getLogger(__name__)


litellm.cache = Cache()
litellm.cache.cache
litellm.set_verbose = False


class GPTExecutor:
    def __init__(self):
        self.request = {}
        self.response = GPTResponse(
            choices=[
                GPTChoice(
                    index=0,
                    message=GPTResponseMessage(role="assistant", content="Hello, how can I help you?"),
                    finnish_reason="",
                )
            ]
        )
        self.partial_response = GPTPartialResponse()

    async def execute(self, request: GPTRequest) -> AsyncGenerator[litellm.ModelResponse | CLEAR_STR_TYPE, None]:
        request.validate_request()

        request_dict = {
            "max_tokens": request.max_tokens,
            "messages": request.get_messages_dump(),
            "model": request.model,
            "temperature": request.temperature,
            "presence_penalty": request.presence_penalty,
        }

        if request.tool_choice:
            request_dict["tool_choice"] = request.tool_choice

        if request.tools:
            request_dict["tools"] = [tool.model_dump() for tool in request.tools]

        for attempt in range(3):
            try:
                _log.info("Executing GPT request:", request_dict)
                self.request = request_dict
                response = await litellm.acompletion(
                    **request_dict,
                    stream=True,
                    # caching=True,
                )

                self.partial_response = GPTPartialResponse()

                async for chunk in response:
                    self.partial_response.apply_chunk(chunk)
                    yield chunk
                    await asyncio.sleep(0)

                self.response = self.partial_response.to_final_response()

                if _log.isEnabledFor(logging.DEBUG):
                    await DebugJSONServerMessage(
                        message="GPT", object={"request": self.request, "response": self.response.model_dump()}
                    ).send_to_all()

                return
            except AuthenticationError:
                raise NoOpenAPIKeyException()
            except Exception as error:
                _log.exception(f"Error on attempt {attempt}: {error}", exc_info=error)
                if attempt == 2:
                    raise error
            _log.info("Retrying GPT request")
            yield CLEAR_STR

        raise Exception("Unable to complete GPT request.")
