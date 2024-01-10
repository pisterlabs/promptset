from openai_token_counter.format import format_function_definitions
from openai_token_counter.models import OpenAIFunction


def test_format_function_definitions() -> None:
    """Test that the format_function_definitions function works as expected."""
    functions: list[OpenAIFunction] = [
        OpenAIFunction.model_validate(
            {
                "name": "function1",
                "description": "This is function 1",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string"},
                        "param2": {"type": "number"},
                    },
                    "required": ["param1"],
                },
            }
        ),
        OpenAIFunction.model_validate(
            {
                "name": "function2",
                "description": "This is function 2",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param3": {"type": "boolean"},
                        "param4": {"type": "null"},
                    },
                },
            }
        ),
        OpenAIFunction.model_validate(
            {
                "name": "function3",
                "description": "This is function 3",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param5": {"type": "boolean"},
                        "param6": {"type": "null"},
                        "param7": {
                            "type": "object",
                            "properties": {
                                "param8": {"type": "boolean"},
                                "param9": {"type": "null"},
                            },
                            "required": ["param8"],
                        },
                    },
                },
            }
        ),
    ]

    result = format_function_definitions(functions)
    assert (
        result
        == """
namespace functions {

// This is function 1
type function1 = (_: {
param1: string,
param2?: number,
}) => any;

// This is function 2
type function2 = (_: {
param3?: boolean,
param4?: null,
}) => any;

// This is function 3
type function3 = (_: {
param5?: boolean,
param6?: null,
param7?: {
  param8: boolean,
  param9?: null,
},
}) => any;

} // namespace functions
""".strip()
    )
