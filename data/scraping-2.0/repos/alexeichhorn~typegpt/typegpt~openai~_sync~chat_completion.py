from __future__ import annotations

from typing import TypeVar, cast, overload

from openai import BadRequestError, resources
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    completion_create_params,
)

from ...base import BaseLLMResponse
from ...exceptions import LLMParseException
from ...prompt_definition.prompt_template import PromptTemplate
from ...utils.internal_types import _UseDefault, _UseDefaultType
from ..base_chat_completion import BaseChatCompletions
from ..exceptions import AzureContentFilterException
from ..views import AzureChatModel, OpenAIChatModel

_Output = TypeVar("_Output", bound=BaseLLMResponse)


class TypeChatCompletion(resources.chat.Completions, BaseChatCompletions):
    def generate_completion(
        self,
        model: OpenAIChatModel | AzureChatModel,
        messages: list[ChatCompletionMessageParam],
        frequency_penalty: float | None | NotGiven = NOT_GIVEN,  # [-2, 2]
        function_call: completion_create_params.FunctionCall | NotGiven = NOT_GIVEN,
        functions: list[completion_create_params.Function] | NotGiven = NOT_GIVEN,
        logit_bias: dict[str, int] | None | NotGiven = NOT_GIVEN,  # [-100, 100]
        max_tokens: int | NotGiven = 1000,
        n: int | None | NotGiven = NOT_GIVEN,
        presence_penalty: float | None | NotGiven = NOT_GIVEN,  # [-2, 2]
        response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
        seed: int | None | NotGiven = NOT_GIVEN,
        stop: str | list[str] | None | NotGiven = NOT_GIVEN,
        temperature: float | None | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: list[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_p: float | None | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        timeout: float | None | NotGiven = NOT_GIVEN,
    ) -> str:
        raw_model: OpenAIChatModel | str
        if isinstance(model, AzureChatModel):
            raw_model = model.deployment_id
            is_azure = True
        else:
            raw_model = model
            is_azure = False

        try:
            result = self.create(
                model=raw_model,
                messages=messages,
                frequency_penalty=frequency_penalty,
                function_call=function_call,
                functions=functions,
                logit_bias=logit_bias,
                max_tokens=max_tokens,
                n=n,
                presence_penalty=presence_penalty,
                response_format=response_format,
                seed=seed,
                stop=stop,
                stream=False,
                temperature=temperature,
                tool_choice=tool_choice,
                tools=tools,
                top_p=top_p,
                user=user,
                timeout=timeout,
            )

            if is_azure and result.choices[0].finish_reason == "content_filter":
                raise AzureContentFilterException(reason="completion")

            return result.choices[0].message.content or ""

        except BadRequestError as e:
            if is_azure and e.code == "content_filter":
                raise AzureContentFilterException(reason="prompt")
            elif (
                is_azure and "filtered due to the prompt triggering Azure OpenAI" in e.message
            ):  # temporary fix: OpenAI library doesn't correctly parse code into error object
                raise AzureContentFilterException(reason="prompt")
            else:
                raise e

    @overload
    def generate_output(
        self,
        model: OpenAIChatModel | AzureChatModel,
        prompt: PromptTemplate,
        max_output_tokens: int,
        output_type: type[_Output],
        max_input_tokens: int | None = None,
        frequency_penalty: float | None | NotGiven = NOT_GIVEN,  # [-2, 2]
        n: int | None | NotGiven = NOT_GIVEN,
        presence_penalty: float | None | NotGiven = NOT_GIVEN,  # [-2, 2]
        temperature: float | NotGiven = NOT_GIVEN,
        seed: int | None | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        timeout: float | None | NotGiven = NOT_GIVEN,
        retry_on_parse_error: int = 0,
    ) -> _Output:
        ...

    @overload
    def generate_output(
        self,
        model: OpenAIChatModel | AzureChatModel,
        prompt: PromptTemplate,
        max_output_tokens: int,
        output_type: _UseDefaultType = _UseDefault,
        max_input_tokens: int | None = None,
        frequency_penalty: float | None | NotGiven = NOT_GIVEN,  # [-2, 2]
        n: int | None | NotGiven = NOT_GIVEN,
        presence_penalty: float | None | NotGiven = NOT_GIVEN,  # [-2, 2]
        temperature: float | NotGiven = NOT_GIVEN,
        seed: int | None | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        timeout: float | None | NotGiven = NOT_GIVEN,
        retry_on_parse_error: int = 0,
    ) -> BaseLLMResponse:
        ...

    def generate_output(
        self,
        model: OpenAIChatModel | AzureChatModel,
        prompt: PromptTemplate,
        max_output_tokens: int,
        output_type: type[_Output] | _UseDefaultType = _UseDefault,
        max_input_tokens: int | None = None,
        frequency_penalty: float | None | NotGiven = NOT_GIVEN,  # [-2, 2]
        n: int | None | NotGiven = NOT_GIVEN,
        presence_penalty: float | None | NotGiven = NOT_GIVEN,  # [-2, 2]
        temperature: float | NotGiven = NOT_GIVEN,
        seed: int | None | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        timeout: float | None | NotGiven = NOT_GIVEN,
        retry_on_parse_error: int = 0,
    ) -> _Output | BaseLLMResponse:
        """
        Calls OpenAI Chat API, generates assistant response, and fits it into the output class

        :param model: model to use as `OpenAIChatModel` or `AzureChatModel`
        :param prompt: prompt, which is a subclass of `PromptTemplate`
        :param max_output_tokens: maximum number of tokens to generate
        :param output_type: output class used to parse the response, subclass of `BaseLLMResponse`. If not specified, the output defined in the prompt is used
        :param max_input_tokens: maximum number of tokens to use from the prompt. If not specified, the maximum number of tokens is calculated automatically
        :param request_timeout: timeout for the request in seconds
        :param retry_on_parse_error: number of retries if the response cannot be parsed (i.e. any `LLMParseException`). If set to 0, it has no effect.
        :param config: additional OpenAI/Azure config if needed (e.g. no global api key)
        """

        if isinstance(model, AzureChatModel):
            model_type = model.base_model
        else:
            model_type = model

        max_prompt_length = self.max_tokens_of_model(model_type) - max_output_tokens

        if max_input_tokens:
            max_prompt_length = min(max_prompt_length, max_input_tokens)

        messages = prompt.generate_messages(
            token_limit=max_prompt_length, token_counter=lambda messages: self.num_tokens_from_messages(messages, model=model_type)
        )

        completion = self.generate_completion(
            model=model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            max_tokens=max_output_tokens,
            frequency_penalty=frequency_penalty,
            n=n,
            presence_penalty=presence_penalty,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            timeout=timeout,
        )

        try:
            if isinstance(output_type, _UseDefaultType):
                return prompt.Output.parse_response(completion)
            else:
                return output_type.parse_response(completion)
        except LLMParseException as e:
            if retry_on_parse_error > 0:
                return self.generate_output(
                    model=model,
                    prompt=prompt,
                    max_output_tokens=max_output_tokens,
                    output_type=output_type,
                    max_input_tokens=max_input_tokens,
                    frequency_penalty=frequency_penalty,
                    n=n,
                    presence_penalty=presence_penalty,
                    temperature=temperature,
                    seed=seed,
                    top_p=top_p,
                    timeout=timeout,
                    retry_on_parse_error=retry_on_parse_error - 1,
                )
            else:
                raise e
