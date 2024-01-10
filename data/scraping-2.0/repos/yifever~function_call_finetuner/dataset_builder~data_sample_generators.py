import openai
from typing import Any

# we can test some different templates
from dataset_builder.message_templates import (
    MESSAGE_TEMPLATE_2 as TEMPLATE,
    CONVERSATION_EXAMPLE_2 as PROMPT_EX,
    FUNCTION_EXAMPLE_2 as FUNCTION_EX,
    CALL_EXAMPLE_2 as CALL_EX,
)


def generate_data_sample_from(
    function_description: dict[str, Any] | str,
    model: str = "gpt-3.5-turbo-0613",
    temperature: float = 1
) -> tuple[list[dict[Any, Any] | str], dict[str, str], dict[Any, Any]]:  # messages  # completion
    """
    Generate a data sample given function description.
    func_description contains (this follows:
    https://platform.openai.com/docs/guides/gpt/function-calling):
    {
        "name": {FUNCTION_NAME},
            "description": "FUNCTION_DESCRIPTION",
            "parameters": {
                "type": "object",
                "properties": {
                    "ARG1": {
                        "type": "ARG1TYPE",
                        "description": "ARG1DESCRIPTION",
                    },
                    "ARG2:": {"type": "ARG2TYPE", "enum": ["ENUM1", "ENUM2"]},
                },
                "required": REQ_ARGS,
            },
    }
    see openai's docs for more on what the function description should be
    """
    prompt_message = TEMPLATE.format(
        FUNCTION_DESCRIPTION=str(function_description),
        PROMPT_EXAMPLE=PROMPT_EX,
        FUNCTION_EXAMPLE=FUNCTION_EX,
        CALL_EXAMPLE=CALL_EX,
    )
    meta = {
        "TEMPLATE": TEMPLATE,
        "FUNCTION_DESCRIPTION": str(function_description),
        "PROMPT_EXAMPLE": PROMPT_EX,
        "FUNCTION_EXAMPLE": FUNCTION_EX,
        "CALL_EXAMPLE": CALL_EX,
    }
    messages = [{"role": "user", "content": prompt_message}]
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=temperature,
    )

    return (messages, response["choices"][0], meta)
