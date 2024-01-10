"""Utilities for the json_fixes package."""
import ast
import json
import os.path
import re
from typing import Any, Dict

from jsonschema import Draft7Validator

from superpilot.core.configuration import Config
from superpilot.framework.helpers.logs import logger
from superpilot.framework.helpers.json_utils.json_fix_llm import auto_fix_json
from superpilot.framework.helpers.json_utils.json_fix_llm import fix_json_using_multiple_techniques

LLM_DEFAULT_RESPONSE_FORMAT = "llm_response_format_1"


def extract_json_from_response(response_content: str, schema: dict = None) -> dict:
    # Sometimes the response includes the JSON in a code block with ```
    if response_content.startswith("```") and response_content.endswith("```"):
        # Discard the first and last ```, then re-join in case the response naturally included ```
        response_content = "```".join(response_content.split("```")[1:-1])
    # response content comes from OpenAI as a Python `str(content_dict)`, literal_eval reverses this
    try:
        response_content = fix_json_using_multiple_techniques(response_content, schema).__str__()
        logger.info(f"Response after braces {response_content}")
        return ast.literal_eval(response_content)
    except BaseException as e:
        logger.info(f"Error parsing JSON response with literal_eval {e}")
        logger.debug(f"Invalid JSON received in response: {response_content}")
        # TODO: How to raise an error here without causing the program to exit?
        return {}


def extract_function_call_json_from_response(response_content: str, schema: dict = None) -> dict:
    json = extract_json_from_response(response_content, schema=schema)
    logger.info(f"Extracted json from content {json}")
    if schema is None or validate_json(json, Config(), schema=schema):
        function_call = schema["name"]
        if "function" in json:
            function_call = json["function"].split('.')[-1]
        elif "name" in json:
            function_call = json["name"]
        elif schema is not None:
            return execute_function_from_json(json, schema)
        arguments = {}
        if "arguments" in json:
            arguments = json["arguments"]
        json = {"function_call": {"name": function_call, "arguments": arguments}}
        return json
    else:
        logger.info(f"Validation failed of string {response_content} for the schema {schema}")
        return {"function_call": {"name": "retry", "arguments": {"type": "ability_extraction", "message": "validation failed"}}}


def execute_function_from_json(data, schema):
    # Extract function name from $type
    function_name = schema['name']

    # Extract parameters from schema
    params = schema['parameters']['properties']
    # Match parameters in JSON data with parameters in schema
    if "arguments" in data:
        data = data["arguments"]
    function_params = {f"{param}": f"{data[param]}" for param in params if param in data}
    print("Extract functions", function_params)
    # Execute the function with the extracted parameters
    return {"function_call": {"name": function_name, "arguments": function_params.__str__().replace("'", '"')}}


def llm_response_schema(
    schema_name: str = LLM_DEFAULT_RESPONSE_FORMAT,
) -> Dict[str, Any]:
    filename = os.path.join(os.path.dirname(__file__), f"{schema_name}.json")
    with open(filename, "r") as f:
        return json.load(f)


def validate_json(
    json_object: object, config: Config, schema_name: str = LLM_DEFAULT_RESPONSE_FORMAT, schema: dict = None
) -> bool:
    """
    :type schema_name: object
    :param schema_name: str
    :type json_object: object

    Returns:
        bool: Whether the json_object is valid or not
    """
    if schema is None:
        schema = llm_response_schema(schema_name)
    validator = Draft7Validator(schema)

    if errors := sorted(validator.iter_errors(json_object), key=lambda e: e.path):
        for error in errors:
            logger.debug(f"JSON Validation Error: {error}")

        if config.debug_mode:
            logger.error(
                json.dumps(json_object, indent=4)
            )  # Replace 'json_object' with the variable containing the JSON data
            logger.error("The following issues were found:")

            for error in errors:
                logger.error(f"Error: {error.message}")
        return False

    logger.debug("The JSON object is valid.")

    return True
