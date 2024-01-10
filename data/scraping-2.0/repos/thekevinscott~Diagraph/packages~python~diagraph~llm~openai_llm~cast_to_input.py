from typing import Any

from openai.types.chat import ChatCompletionMessageParam


def cast_to_input(
    prompt_str: str | list[ChatCompletionMessageParam] | dict[str, Any],
) -> tuple[list[ChatCompletionMessageParam], dict[str, Any]]:
    if isinstance(prompt_str, str):
        return [{"role": "user", "content": prompt_str}], {}
    if isinstance(prompt_str, dict):
        prompt_str = {**prompt_str}
        messages = prompt_str.get("messages")
        del prompt_str["messages"]

        return messages, prompt_str
    return prompt_str, {}
