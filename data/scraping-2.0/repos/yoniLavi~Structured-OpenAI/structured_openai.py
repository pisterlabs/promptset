import json
import os
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI

_ = load_dotenv()
OPENAI_CLIENT = OpenAI()

def generate_structured_openai_call(
    prompt: str,
    response_properties: dict,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0,
    include_original_input: bool = True,
) -> Callable[[str], dict]:
    """Generate a function that calls OpenAI's API and returns a structured response.

    Args:
        prompt: The system prompt to use.
        response_properties: A dictionary of the JSON-schema properties to return.
            See https://json-schema.org/understanding-json-schema/reference/object.html#properties
        model: The model to use. Defaults to "gpt-3.5-turbo".
        temperature: The temperature to use. Defaults to 0.
        include_original_input: Whether to include the original input as one of the properties
             in the response. Defaults to True.

    Returns:
        A function that takes a user input string and returns a response dict with the properties.
    """
    if include_original_input:
        # Prepend the the original input as the first property
        response_properties = {
            "original_input": {
                "type": "string",
                "description": "The original user input",
            }
        } | response_properties

    def inner_function(user_input: str) -> dict:
        completion = OPENAI_CLIENT.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=temperature,
            functions=[
                {
                    "name": "structured_response",
                    "description": "Responses must always be structured this way",
                    "parameters": {
                        "type": "object",
                        "properties": response_properties,
                        "required": list(
                            response_properties.keys()
                        ),  # all keys are required
                    },
                }
            ],
            function_call={"name": "structured_response"},
        )
        message_response = completion.choices[0].message
        return json.loads(message_response.function_call.arguments)

    return inner_function


if __name__ == "__main__":
    # Example usage
    where_are_we = generate_structured_openai_call(
        "The name of our location, at this resolution",
        {
            "location_name": {
                "type": "string",
                "description": "The name of the location we are in",
            },
        }
    )
    print(where_are_we("Our galaxy"))
    # Expected output: {'original_input': 'Our galaxy', 'location_name': 'Milky Way'}
