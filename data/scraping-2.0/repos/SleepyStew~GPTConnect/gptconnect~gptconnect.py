import copy
import json
from typing import Union

import openai

valid_models = ["gpt-3.5-turbo-0613", "gpt-4-0613", "gpt-4-32k-0613"]
functions = []


class GPTConnect:
    """
    The main class for working with GPTConnect

    :param token: The OpenAI API token
    :param model: The model to use
    """

    def __init__(self, token: str, model: str):
        openai.api_key = token
        if model not in valid_models:
            raise ValueError(
                f"Invalid model: {model}. Please use one of the following: {', '.join(valid_models)}"
            )

        self._model = model

    def call(
        self,
        prompt: Union[str, None],
        function_group: str,
        messages: Union[dict, None] = None,
    ) -> dict:
        """
        Make a ChatGPT Completion with a prompt or list of messages, and function group

        :param prompt: The prompt to provide
        :param function_group: The function group to use
        :param messages: The messages to provide as context
        :return: A dictionary containing the function if called, and the content of the response from OpenAI
        """
        if not prompt and not messages:
            raise ValueError("Prompt or messages must be provided")

        if not messages:
            messages = [{"role": "user", "content": prompt}]

        function_details = [
            f for f in copy.deepcopy(functions) if f.get("group") == function_group
        ]

        for function in function_details:
            del function["function"]
            del function["group"]

        response = openai.ChatCompletion.create(
            model=self._model, messages=messages, functions=function_details
        )

        for choice in response.choices:
            messages.append(choice.message)

        function_call = response.choices[0].message.get("function_call")

        if function_call:
            function = [
                f["function"] for f in functions if f.get("name") == function_call.name
            ][0]
            context = json.loads(function_call.arguments)

            from .gptfunctionhandler import function_handler

            if function_handler:
                call = function_handler(function, context)
            else:
                call = function(context)

            messages.append(
                {"role": "function", "name": function_call.name, "content": str(call)}
            )

            response = openai.ChatCompletion.create(
                model=self._model, messages=messages, functions=function_details
            )

        return {
            "content": response.choices[-1].message.content,
            "function_called": function_call.name if function_call else None,
        }
