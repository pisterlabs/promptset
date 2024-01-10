"""LLM for Development TestRun"""
import re
from typing import Any, AsyncGenerator, List, Dict, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from litellm import acompletion, get_max_tokens

from promptmodel.types.enums import ParsingType, get_pattern_by_type
from promptmodel.utils import logger
from promptmodel.utils.output_utils import convert_str_to_type, update_dict
from promptmodel.utils.token_counting import (
    num_tokens_for_messages_for_each,
    num_tokens_from_functions_input,
)
from promptmodel.types.response import LLMStreamResponse, ModelResponse


load_dotenv()


class OpenAIMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = ""
    function_call: Optional[Dict[str, Any]] = None
    name: Optional[str] = None


class LLMDev:
    def __init__(self):
        self._model: str

    def __validate_openai_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[OpenAIMessage]:
        """Validate and convert list of dictionaries to list of OpenAIMessage."""
        res = []
        for message in messages:
            res.append(OpenAIMessage(**message))
        return res

    async def dev_run(
        self,
        messages: List[Dict[str, Any]],
        parsing_type: Optional[ParsingType] = None,
        functions: Optional[List[Any]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """Parse & stream output from openai chat completion."""
        _model = model or self._model
        raw_output = ""
        if functions == []:
            functions = None
        response: AsyncGenerator[ModelResponse, None] = await acompletion(
            model=_model,
            messages=[
                message.model_dump(exclude_none=True)
                for message in self.__validate_openai_messages(messages)
            ],
            stream=True,
            functions=functions,
            **kwargs,
        )
        function_call = {"name": "", "arguments": ""}
        finish_reason_function_call = False
        async for chunk in response:
            if getattr(chunk.choices[0].delta, "content", None) is not None:
                stream_value = chunk.choices[0].delta.content
                raw_output += stream_value  # append raw output
                yield LLMStreamResponse(raw_output=stream_value)  # return raw output

            if getattr(chunk.choices[0].delta, "function_call", None) is not None:
                for key, value in (
                    chunk.choices[0].delta.function_call.model_dump().items()
                ):
                    if value is not None:
                        function_call[key] += value

            if chunk.choices[0].finish_reason == "function_call":
                finish_reason_function_call = True
                yield LLMStreamResponse(function_call=function_call)

        # parsing
        if parsing_type and not finish_reason_function_call:
            parsing_pattern: Dict[str, str] = get_pattern_by_type(parsing_type)
            whole_pattern = parsing_pattern["whole"]
            parsed_results = re.findall(whole_pattern, raw_output, flags=re.DOTALL)
            for parsed_result in parsed_results:
                key = parsed_result[0]
                type_str = parsed_result[1]
                value = convert_str_to_type(parsed_result[2], type_str)
                yield LLMStreamResponse(parsed_outputs={key: value})

    async def dev_chat(
        self,
        messages: List[Dict[str, Any]],
        functions: Optional[List[Any]] = None,
        tools: Optional[List[Any]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[LLMStreamResponse, None]:
        """Parse & stream output from openai chat completion."""
        _model = model or self._model
        raw_output = ""
        if functions == []:
            functions = None

        if model != "HCX-002":
            # Truncate the output if it is too long
            # truncate messages to make length <= model's max length
            token_per_functions = num_tokens_from_functions_input(
                functions=functions, model=model
            )
            model_max_tokens = get_max_tokens(model=model)
            token_per_messages = num_tokens_for_messages_for_each(messages, model)
            token_limit_exceeded = (
                sum(token_per_messages) + token_per_functions
            ) - model_max_tokens
            if token_limit_exceeded > 0:
                while token_limit_exceeded > 0:
                    # erase the second oldest message (first one is system prompt, so it should not be erased)
                    if len(messages) == 1:
                        # if there is only one message, Error cannot be solved. Just call LLM and get error response
                        break
                    token_limit_exceeded -= token_per_messages[1]
                    del messages[1]
                    del token_per_messages[1]

        args = dict(
            model=_model,
            messages=[
                message.model_dump(exclude_none=True)
                for message in self.__validate_openai_messages(messages)
            ],
            functions=functions,
            tools=tools,
        )

        is_stream_unsupported = model in ["HCX-002"]
        if not is_stream_unsupported:
            args["stream"] = True
        response: AsyncGenerator[ModelResponse, None] = await acompletion(**args, **kwargs)
        if is_stream_unsupported:
            yield LLMStreamResponse(raw_output=response.choices[0].message.content)
        else:
            async for chunk in response:
                yield_api_response_with_fc = False
                logger.debug(chunk)
                if getattr(chunk.choices[0].delta, "function_call", None) is not None:
                    yield LLMStreamResponse(
                        api_response=chunk,
                        function_call=chunk.choices[0].delta.function_call,
                    )
                    yield_api_response_with_fc = True

                if getattr(chunk.choices[0].delta, "tool_calls", None) is not None:
                    yield LLMStreamResponse(
                        api_response=chunk,
                        tool_calls=chunk.choices[0].delta.tool_calls,
                    )
                    yield_api_response_with_fc = True

                if getattr(chunk.choices[0].delta, "content", None) is not None:
                    raw_output += chunk.choices[0].delta.content
                    yield LLMStreamResponse(
                        api_response=chunk if not yield_api_response_with_fc else None,
                        raw_output=chunk.choices[0].delta.content,
                    )
