"""OpenAI Functions Service"""
import json

import openai

from .schema import *


async def parse_openai_function(
    response: dict,
    functions: List[Type[F]] = OpenAIFunction.Metadata.subclasses,
    **kwargs: Any,
) -> FunctionCall:
    """Parses the response from OpenAI and returns a FunctionCall object."""
    choice = response["choices"][0]["message"]
    if "function_call" in choice:
        function_call_ = choice["function_call"]
        name = function_call_["name"]
        arguments = function_call_["arguments"]
        for i in functions:
            if i.__name__ == name:
                result = await i(**json.loads(arguments))(**kwargs)
                break
        else:
            raise ValueError(f"Function {name} not found")
        return result
    return FunctionCall(name="chat", data=choice["content"])


async def function_call(
    text: str,
    model: str = "gpt-3.5-turbo-16k-0613",
    functions: List[Type[F]] = OpenAIFunction.Metadata.subclasses,
    **kwargs,
) -> FunctionCall:
    """Calls the function orchestrator."""
    messages = [
        {"role": "user", "content": text},
        {"role": "system", "content": "You are a function Orchestrator"},
    ]
    response = await openai.ChatCompletion.acreate(
        model=model,
        messages=messages,
        functions=[i.openaischema for i in functions],
    )
    return await parse_openai_function(response, functions=functions, **kwargs)