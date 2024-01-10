"""GPT-powered workflows."""

import inspect
import os
from functools import partial, wraps
from typing import Callable, Type

import docstring_parser
import openai

from flytekit import current_context, Deck, Secret
from flytekit.experimental import eager


def define_tool(fn: Callable) -> dict:
    docstring = docstring_parser.parse(fn.__doc__)
    description = (
        docstring.short_description
        if docstring.long_description is None
        else f"{docstring.short_description}\n\n{docstring.long_description}"
    )
    signature = inspect.signature(fn)

    def get_json_datatype(type: Type):
        if type is str:
            return "string"
        if type in (int, float):
            return "number"
        return "object"

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    p.arg_name: {
                        "type": get_json_datatype(fn.__annotations__[p.arg_name]),
                        "description": p.description,
                    }
                    for p in docstring.params
                },
                "required": [
                    p.arg_name for p in docstring.params
                    if signature.parameters[p.arg_name].default is inspect._empty
                ],
            }
        }
    }


DECK_HTML_TEMPLATE = """
<h2>Auto-workflow Output</h2>
<hr>
<p>{output_contents}</p>
"""

def autoflow(
    _fn=None,
    *,
    tasks,
    remote,
    client_secret_group,
    client_secret_key,
    model,
    openai_secret_group,
    openai_secret_key,
    **kwargs,
):
    
    if _fn is None:
        return partial(
            autoflow,
            tasks=tasks,
            remote=remote,
            client_secret_group=client_secret_group,
            client_secret_key=client_secret_key,
            model=model,
            openai_secret_group=openai_secret_group,
            openai_secret_key=openai_secret_key,
            **kwargs,
        )

    fn_docstring = docstring_parser.parse(_fn.__doc__)
    secret_requests = kwargs.pop("secret_requests", None) or []
    secret_requests.append(Secret(group=openai_secret_group, key=openai_secret_key))
    
    @wraps(_fn)
    async def wrapper(prompt: str, inputs: dict):
        
        try:
            os.environ["OPENAI_API_KEY"] = current_context().secrets.get(
                openai_secret_group,
                openai_secret_key,
            )
        except ValueError:
            pass

        await _fn(prompt, inputs)

        client = openai.OpenAI()

        _globals = wrapper.__globals__
        available_tasks = {task.task_function.__name__: _globals[task.task_function.__name__] for task in tasks}
        tools = [define_tool(task.entity.task_function) for task in available_tasks.values()]

        messages = [
            {"role": "system", "content": fn_docstring.short_description},
            {"role": "user", "content": fn_docstring.long_description.format(prompt=prompt, inputs=inputs)},
        ]
        
        print("Getting task choice:")
        print(f"🧰 Tools:")
        for tool in tools:
            print(f"  - {tool}")

        print(f"💬 Messages: {messages}")
        for message in messages:
            print(f"  - {message}")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        response_message = response.choices[0].message

        # just use the first tool chosen
        tool_call = response_message.tool_calls[0]
        task_choice = available_tasks[tool_call.function.name]
        print(f"👉 Task choice: {tool_call.function.name}")
        print(f"🔢 Inputs: {inputs}")
        result = await task_choice(**inputs)

        result.download()
        try:
            with open(result.path) as f:
                output_contents = f.read()
        except Exception:
            with open(result.path, "rb") as f:
                output_contents = f.read()
        
        Deck("auto workflow output", DECK_HTML_TEMPLATE.format(output_contents=output_contents))
        return result
    
    for t in tasks:
        # hack: this makes sure the wrapper function has access to the task functions in
        # their AsyncEntity form.
        wrapper.__globals__[t.__name__] = _fn.__globals__[t.__name__]
    
    return eager(
        remote=remote,
        client_secret_group=client_secret_group,
        client_secret_key=client_secret_key,
        secret_requests=secret_requests,
        **kwargs
    )(wrapper)
