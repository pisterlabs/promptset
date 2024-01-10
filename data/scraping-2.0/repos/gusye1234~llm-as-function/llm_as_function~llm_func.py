import inspect
from enum import StrEnum
from pydantic import BaseModel, ValidationError
from dataclasses import dataclass, field
import json
from copy import copy
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionMessage
from functools import wraps
from .utils import generate_schema_prompt, logger, clean_output_parse, GlobalGPTBin
from .fn_calling import parse_function, function_to_name, get_argument_for_function
from .errors import InvalidFunctionParameters, InvalidLLMResponse
from .models import (
    ernie_single_create,
    openai_single_create,
    ernie_single_acreate,
    openai_single_acreate,
    JSON_SCHEMA_PROMPT,
)


def model_factory(model_name: str):
    if model_name.startswith("ernie"):
        return "ernie"
    if model_name.startswith("gpt"):
        return "openai"
    raise NotImplementedError(
        f"llm-as-function currently supports OpenAI or Ernie models, not {model_name}"
    )


@dataclass
class Final:
    pack: dict | None = None
    raw_response: str | None = None

    def ok(self):
        return self.pack is not None

    def unpack(self):
        if self.pack is not None:
            return self.pack
        return self.raw_response


@dataclass
class LLMFunc:
    """Use LLM as a function"""

    parse_mode: str = "error"
    output_schema: BaseModel | None = None
    output_json: dict | None = None
    prompt_template: str = ""
    model: str = "gpt-3.5-turbo-1106"
    temperature: float = 0.1
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    async_max_time: int | None = None
    async_wait_time: float = 0.1
    runtime_options: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.parse_mode in [
            "error",
            "accept_raw",
        ], f"Parse mode must in ['error', 'accept_raw'], not {self.parse_mode}"
        self.config: dict = dict(model=self.model, temperature=self.temperature)
        self.provider = model_factory(self.config["model"])

        self._bp_runtime_options = copy(self.runtime_options)
        self.fn_callings = {}
        self.async_models = {}
        if self.provider == "openai":
            assert (
                self.openai_api_key != ""
            ), "You must have OpenAI api key input, or set OPENAI_API_KEY in your environment."
            self.openai_client = OpenAI(
                api_key=self.openai_api_key, base_url=self.openai_base_url
            )
            self.openai_async_client = AsyncOpenAI(
                api_key=self.openai_api_key, base_url=self.openai_base_url
            )
        if self.async_max_time is None:
            self.async_models["openai"] = openai_single_acreate
            self.async_models["ernie"] = ernie_single_acreate
        else:
            self.async_models["openai"] = GlobalGPTBin(
                max_size=self.async_max_time, waiting_time=self.async_wait_time
            )(openai_single_acreate)
            self.async_models["ernie"] = GlobalGPTBin(
                max_size=self.async_max_time, waiting_time=self.async_wait_time
            )(ernie_single_acreate)

    def reset(self):
        self.prompt_template = ""
        self.output_schema = None
        self.output_json = None
        self.runtime_options = copy(self._bp_runtime_options)
        self.func_callings = []
        self.fn_callings = {}

    def prompt(self, prompt_template):
        self.prompt_template = prompt_template.strip("\n ")

        return self

    def func(self, func):
        if self.provider == "ernie":
            raise NotImplementedError("Function calling for ernie is not supported yet")
        self.fn_callings[function_to_name(func)] = func

        func_desc = parse_function(func)
        if self.runtime_options.get("tools", None) is None:
            self.runtime_options["tools"] = []
        self.runtime_options["tools"].append(func_desc)
        self.runtime_options["tool_choice"] = "auto"

        return self

    def output(self, output_schema):
        self.output_schema = output_schema
        self.output_json = generate_schema_prompt(output_schema)
        return self

    def parse_output(self, output, output_schema):
        try:
            json_str = clean_output_parse(output)
            output = output_schema(**json.loads(json_str)).model_dump()
            return Final(output)
        except:
            logger.error(f"Failed to parse output: {output}")
            if self.parse_mode == "error":
                raise InvalidLLMResponse(f"Failed to parse output: {output}")
            elif self.parse_mode == "accept_raw":
                return Final(raw_response=output)
            raise InvalidLLMResponse(f"Failed to parse output: {output}")

    def _init_setup(self, func):
        return_annotation = func.__annotations__.get("return", None)
        if return_annotation is not None:
            assert issubclass(
                return_annotation, BaseModel
            ), "Return must be a Pydantic BaseModel"
            self.output(return_annotation)
        if self.output_schema is None and return_annotation is None:
            raise ValueError(
                "You must specify the output schema or the function return annotation"
            )
        if self.prompt_template == "":
            if func.__doc__ is None:
                raise ValueError(
                    "You must specify the prompt template or the function docstring"
                )
            self.prompt(func.__doc__)

        return (
            self.prompt_template,
            self.output_json,
            self.output_schema,
            self.runtime_options,
            self.fn_callings,
        )

    def _fill_prompt(self, kwargs, local_var, prompt_template):
        if local_var is not None:
            if isinstance(local_var, Final):
                return local_var
            elif isinstance(local_var, dict):
                prompt = prompt_template.format(**kwargs, **local_var)
            else:
                raise NotImplementedError(
                    f"UnSupported branch {type(local_var)}, please use one of the branch class: Final, dict"
                )
        else:
            prompt = prompt_template.format(**kwargs)
        return prompt

    def _provider_response(self, prompt, runtime_options={}, fn_callings={}):
        logger.debug(runtime_options)
        if self.provider == "ernie":
            raw_result = ernie_single_create(
                prompt, runtime_options=runtime_options, **self.config
            )
            return raw_result
        elif self.provider == "openai":
            raw_result: ChatCompletionMessage = (
                openai_single_create(
                    prompt,
                    self.openai_client,
                    runtime_options=runtime_options,
                    **self.config,
                )
                .choices[0]
                .message
            )
            if raw_result.tool_calls is None:
                return raw_result.content
            return self._function_call_branch(
                prompt, raw_result, runtime_options, fn_callings
            )
        raise NotImplementedError(f"Provider [{self.provider}] is not supported yet")

    def _form_function_messages(
        self,
        tool_message: ChatCompletionMessage,
        fn_callings={},
        history_messages=[],
    ):
        function_messages = history_messages + [tool_message]

        tool_calls = tool_message.tool_calls
        if tool_calls is None:
            raise ValueError("tool_calls is None")
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            try:
                function_to_call = fn_callings[function_name]
            except KeyError as e:
                logger.error(f"function name is never added: {function_name}")
                raise e

            function_args_json = tool_call.function.arguments
            logger.debug(
                f"Calling function {function_name} with args {function_args_json}"
            )
            validate_type: BaseModel = get_argument_for_function(function_to_call)
            try:
                function_args_parsed = validate_type.model_validate_json(
                    function_args_json
                )
            except (ValueError, ValidationError):
                raise InvalidFunctionParameters(function_name, function_args_json)

            try:
                function_response = function_to_call(function_args_parsed)
            except Exception as e:
                logger.error(f"Occur error when running {function_name}")
                raise e

            assert isinstance(
                function_response, str
            ), f"Expect function [{function_name}] to return str, not {type(function_response)}"

            function_messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
        return function_messages

    def _function_call_branch(
        self,
        prompt,
        tool_message: ChatCompletionMessage,
        runtime_options={},
        fn_callings={},
        history_messages=[],
    ):
        function_messages = self._form_function_messages(
            tool_message, fn_callings, history_messages
        )
        logger.debug(f"Function message {function_messages}")

        if self.provider == "ernie":
            raise NotImplementedError(
                f"Function calling for provider [{self.provider}] is not supported yet"
            )
        elif self.provider == "openai":
            raw_result: ChatCompletionMessage = (
                openai_single_create(
                    prompt,
                    self.openai_client,
                    runtime_options=runtime_options,
                    function_messages=function_messages,
                    **self.config,
                )
                .choices[0]
                .message
            )
            if raw_result.tool_calls is None:
                return raw_result.content
            return self._function_call_branch(
                prompt, raw_result, runtime_options, fn_callings
            )
        raise NotImplementedError(
            f"Function calling for provider [{self.provider}] is not supported yet"
        )

    async def _provider_async_response(
        self, prompt, runtime_options={}, fn_callings={}
    ):
        if self.provider == "ernie":
            raw_result = await self.async_models["ernie"](
                prompt, runtime_options=runtime_options, **self.config
            )
        elif self.provider == "openai":
            raw_result = await self.async_models["openai"](
                prompt,
                self.openai_async_client,
                runtime_options=runtime_options,
                **self.config,
            )
            raw_result: ChatCompletionMessage = raw_result.choices[0].message
            if raw_result.tool_calls is None:
                return raw_result.content
            return await self._async_function_call_branch(
                prompt, raw_result, runtime_options, fn_callings
            )
        raise NotImplementedError(f"Provider [{self.provider}] is not supported yet")

    async def _async_function_call_branch(
        self,
        prompt,
        tool_message: ChatCompletionMessage,
        runtime_options={},
        fn_callings={},
        history_messages=[],
    ):
        function_messages = self._form_function_messages(
            tool_message, fn_callings, history_messages
        )
        logger.debug(f"Function message {function_messages}")

        if self.provider == "ernie":
            raise NotImplementedError(
                f"Function calling for provider [{self.provider}] is not supported yet"
            )
        elif self.provider == "openai":
            raw_result: ChatCompletionMessage = (
                (
                    await self.async_models["openai"](
                        prompt,
                        self.openai_async_client,
                        runtime_options=runtime_options,
                        function_messages=function_messages,
                        **self.config,
                    )
                )
                .choices[0]
                .message
            )
            if raw_result.tool_calls is None:
                return raw_result.content
            return await self._async_function_call_branch(
                prompt, raw_result, runtime_options, fn_callings
            )
        raise NotImplementedError(
            f"Function calling for provider [{self.provider}] is not supported yet"
        )

    def _append_json_schema(self, prompt, output_json):
        append_prompt = JSON_SCHEMA_PROMPT[self.provider].format(
            json_schema=output_json
        )
        return prompt + append_prompt

    def __call__(self, func):
        # parse input
        (
            prompt_template,
            output_json,
            output_schema,
            runtime_options,
            fn_callings,
        ) = self._init_setup(func)

        self.reset()

        @wraps(func)
        def new_func(**kwargs):
            local_var = func(**kwargs)
            logger.debug(f"[Variables] function args:{kwargs}, local vars: {local_var}")

            prompt = self._fill_prompt(kwargs, local_var, prompt_template)
            prompt = self._append_json_schema(prompt, output_json)
            logger.debug(prompt)

            raw_result = self._provider_response(
                prompt, runtime_options=runtime_options, fn_callings=fn_callings
            )
            result = self.parse_output(raw_result, output_schema)

            return result

        return new_func

    def async_call(self, func):
        (
            prompt_template,
            output_json,
            output_schema,
            runtime_options,
            fn_callings,
        ) = self._init_setup(func)

        self.reset()

        @wraps(func)
        async def new_func(**kwargs):
            if inspect.iscoroutinefunction(func):
                local_var = await func(**kwargs)
            else:
                local_var = func(**kwargs)
            logger.debug(f"[Variables] function args:{kwargs}, local vars: {local_var}")

            prompt = self._fill_prompt(kwargs, local_var, prompt_template)
            prompt = self._append_json_schema(prompt, output_json)
            logger.debug(prompt)

            raw_result = await self._provider_async_response(
                prompt, runtime_options=runtime_options, fn_callings=fn_callings
            )
            result = self.parse_output(raw_result, output_schema)
            logger.debug(f"Return {result}")

            return result

        return new_func

    def generate_llm_description(self, **kwargs):
        pass
        # prompt = self.prompt_template.format(input_args)
