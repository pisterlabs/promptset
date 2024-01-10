import json
from openai import OpenAI
from typing import cast, Callable

from gpt_interface.log import Log, Message
from gpt_interface.options.system_message import SystemMessageOptions
from gpt_interface.tools import Tool, AnnotatedFunction


def call_modern_model(
    interface: OpenAI,
    model: str,
    log: Log,
    temperature: float,
    system_message_options: SystemMessageOptions,
    json_mode: bool,
    tools: list[Tool],
    call_again_fn: Callable,
    thinking_time: int,
) -> str:
    #================================
    # assemble log and system message
    #================================
    messages=[
        (
            {
                "role": message.role,
                "content": message.content,
                "tool_call_id": message.tool_call_id,
                "name": message.name,
            }
            if message.role == "tool"
            else
            {
                "role": message.role,
                "content": message.content,
                "tool_calls": message.tool_calls,
            }
            if (message.role == "assistant") and (len(message.tool_calls) > 0)
            else
            {
                "role": message.role,
                "content": message.content,
            }
        )
        for message in log.messages
    ]
    messages[-1]["content"] += "." * thinking_time
    if system_message_options.use_system_message:
        system_message = {
            "role": "system",
            "content": system_message_options.system_message,
        }
        if system_message_options.message_at_end:
            messages.append(system_message)
        else:
            messages.insert(0, system_message)
    #======================================
    # set arguments for completion endpoint
    #======================================
    completion_args = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    if json_mode and model in ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"]:
        completion_args["response_format"] = { "type": "json_object" }
    if len(tools) > 0:
        completion_args["tools"] = [
            tool.annotation
            for tool in tools
        ]
        completion_args["tool_choice"] = "auto"
    #=========================================================
    # get and parse response
    # https://platform.openai.com/docs/guides/function-calling
    #=========================================================
    response_message = interface.chat.completions.create(**completion_args).choices[0].message
    if response_message.tool_calls:
        log.messages.append(
            Message(
                role="assistant",
                content=cast(str, None),  # to stop complaining
                tool_calls=[
                    {
                        "id": tool_call.id,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                        "type": tool_call.type,
                    }
                    for tool_call in response_message.tool_calls
                ],
            )
        )
        available_functions = {
            tool.name: cast(AnnotatedFunction, tool).function
            for tool in tools
            if isinstance(tool, AnnotatedFunction)
        }
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_response = available_functions[function_name](
                **json.loads(tool_call.function.arguments)
            )
            log.messages.append(
                Message(
                    role="tool",
                    content=str(function_response),
                    tool_call_id=tool_call.id,
                    name=function_name,
                )
            )  # extend conversation with function response
        return_content = call_again_fn()
    else:
        return_content = response_message.content
    return return_content if return_content else "[ERROR: NO RESPONSE]"
