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

from litellm import ModelResponse
from litellm.utils import Delta, StreamingChoices
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from pydantic import BaseModel

from aiconsole.core.gpt.parse_partial_json import parse_partial_json
from aiconsole.core.gpt.types import (
    GPTChoice,
    GPTFunctionCall,
    GPTResponse,
    GPTResponseMessage,
    GPTRole,
    GPTToolCall,
)


class GPTPartialFunctionCall(BaseModel):
    name: str = ""
    arguments_builder: list[str] = []

    @property
    def arguments(self) -> str:
        self.arguments_builder = ["".join(self.arguments_builder)]
        return self.arguments_builder[0]

    @property
    def arguments_dict(self) -> dict | None:
        return parse_partial_json(self.arguments)


class GPTPartialToolsCall(BaseModel):
    id: str = ""
    type: str = ""
    function: GPTPartialFunctionCall = GPTPartialFunctionCall()


class GPTPartialMessage(BaseModel):
    role: GPTRole | None = None
    content_builder: list[str] | None = None

    tool_calls: list[GPTPartialToolsCall] = []
    name: str | None = None

    @property
    def content(self):
        if not self.content_builder:
            return None

        self.content_builder = ["".join(self.content_builder)]
        return self.content_builder[0]


class GPTPartialChoice(BaseModel):
    index: int = 0
    message: GPTPartialMessage = GPTPartialMessage()
    role: str = ""
    finnish_reason: str = ""


class GPTPartialResponse(BaseModel):
    id: str = ""
    object: str = ""
    created: int = 0
    model: str = ""
    choices: list[GPTPartialChoice] = []

    def to_final_response(self):
        return GPTResponse(
            id=self.id,
            object=self.object,
            created=self.created,
            model=self.model,
            choices=[
                GPTChoice(
                    index=choice.index,
                    message=GPTResponseMessage(
                        role=choice.message.role or "user",  # ???
                        content=choice.message.content,
                        tool_calls=[
                            GPTToolCall(
                                id=tool_call.id,
                                type=tool_call.type,
                                function=GPTFunctionCall(
                                    name=tool_call.function.name,
                                    arguments=tool_call.function.arguments,
                                ),
                            )
                            for tool_call in choice.message.tool_calls
                        ],
                        name=choice.message.name,
                    ),
                    finnish_reason=choice.finnish_reason,
                )
                for choice in self.choices
            ],
        )

    def apply_chunk(self, chunk: ModelResponse):
        self.id = chunk.id
        self.object = chunk.object
        self.created = chunk.created

        if chunk.model is not None:
            self.model = chunk.model

        if chunk.choices:
            chunk_choices = chunk.choices

            for chunk_choice in chunk_choices:
                index = chunk_choice.index

                if index >= len(self.choices):
                    self.choices.append(GPTPartialChoice())

                choice = self.choices[index]
                choice.index = index

                if chunk_choice.finish_reason is not None:
                    choice.finnish_reason = chunk_choice.finish_reason

                message = choice.message

                if isinstance(chunk_choice, StreamingChoices):
                    chunk_delta = chunk_choice.delta

                    if "name" in chunk_delta:
                        message.name = chunk_delta["name"]

                    if "role" in chunk_delta:
                        message.role = chunk_delta["role"]

                    if "content" in chunk_delta and chunk_delta["content"] is not None:
                        if message.content_builder is None:
                            message.content_builder = []

                        message.content_builder.append(chunk_delta["content"])

                    if "tool_calls" in chunk_delta:
                        assert isinstance(chunk_delta, Delta)
                        chunk_tool_calls = chunk_delta["tool_calls"] or []

                        for tool_call in chunk_tool_calls:
                            assert isinstance(tool_call, ChoiceDeltaToolCall)

                            chunk_tool_index: int = tool_call.index
                            chunk_tool_function = tool_call.function

                            if chunk_tool_function:
                                while len(message.tool_calls) < chunk_tool_index + 1 and tool_call.id:
                                    message.tool_calls.append(
                                        GPTPartialToolsCall(
                                            id=tool_call.id,
                                        )
                                    )

                                if tool_call.type:
                                    message.tool_calls[chunk_tool_index].type = tool_call.type

                                if chunk_tool_function:
                                    if chunk_tool_function.name is not None:
                                        message.tool_calls[chunk_tool_index].function.name = chunk_tool_function.name

                                    if chunk_tool_function.arguments is not None:
                                        message.tool_calls[chunk_tool_index].function.arguments_builder.append(
                                            chunk_tool_function.arguments
                                        )
