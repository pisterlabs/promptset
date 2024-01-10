import openai
from typing import Any
# we can test some different templates
from description_builder.message_templates import (
    MESSAGE_TEMPLATE_1 as TEMPLATE,
    SNIPPET_EXAMPLE_2 as SNIPPET_EX,
    DESCRIPTION_EXAMPLE_2 as DESCRIPTION_EX,
)

def generate_description_from(
    function_snippet: str,
    model: str = "gpt-3.5-turbo-0613",
    temperature: float = 1
) -> tuple[list[dict[Any, Any] | str], dict[str, str], dict[Any, Any]]:  # (messages, completion)
    """
    Generates a function description compatible with openAI's function call API from a snippet of the function.
    Relies heavily on the comment in front of the snippet
    """
    prompt_message = TEMPLATE.format(
        FUNCTION_SNIPPET= str(function_snippet),
        SNIPPET_EXAMPLE = SNIPPET_EX,
        DESCRIPTION_EXAMPLE=DESCRIPTION_EX,
    )
    meta = {
        "TEMPLATE": TEMPLATE,
        "FUNCTION_SNIPPET": str(function_snippet),
        "SNIPPET_EXAMPLE": SNIPPET_EX,
        "DESCRIPTION_EXAMPLE": DESCRIPTION_EX,
    } 
    messages = [{"role": "user", "content": prompt_message}]
    response = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        messages=messages,
    )
    return (messages, response["choices"][0], meta)
