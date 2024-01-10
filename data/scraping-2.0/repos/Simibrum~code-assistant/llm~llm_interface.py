"""
This script would handle interactions with the LLM, 
such as querying the LLM to generate new code or tests.
"""
import json
import random
import time
from logging import Logger
from typing import Tuple, List

import openai
from openai import OpenAI

from config import OPENAI_API_KEY

import llm.prompts as prompts
from functions import logger, num_tokens_from_messages


client = OpenAI(api_key=OPENAI_API_KEY)


GOOD_MODEL = "gpt-4-0613"  # or whatever model you are using
QUICK_MODEL = "gpt-3.5-turbo-0613"


def load_json_string(str_in: str) -> dict:
    """
    Load a JSON string into a dictionary.

    Args:
        str_in (str): The JSON string to load.

    Returns:
        dict: The JSON string as a dictionary.
    """
    try:
        return json.loads(str_in)
    except json.JSONDecodeError as err:
        logger.debug("JSONDecodeError: %s", str(err))
        logger.debug("String to decode: %s", str_in)
        # Fix triple escaped newlines
        str_in = str_in.replace("\\\n", "\\n")
        return json.loads(str_in)


def api_request(
    messages: list[dict],
    functions: list[dict],
    function_call: str | dict = "auto",
    temperature: int = 0.7,
    model: str = GOOD_MODEL,
    max_tokens: int = None,
    gen_logger: Logger = logger,
) -> dict:
    """
    Make a request to the OpenAI API with exponential backoff.

    Args:
        messages (List[dict]): A list of message objects for the Chat API.
        temperature (int, optional): The temperature parameter for the API request. Default is 0.7.
        gen_logger (Logger, optional): Logger for logging information about the API requests.

    Returns:
        dict: The API response as a dictionary.
    """
    max_tries = 5
    initial_delay = 1
    backoff_factor = 2
    max_delay = 16
    jitter_range = (1, 3)
    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if functions:
        params["functions"] = functions

    if function_call:
        params["function_call"] = function_call

    if max_tokens:
        params["max_tokens"] = max_tokens

    for attempt in range(1, max_tries + 1):
        try:
            response = client.chat.completions.create(**params)
            return response.model_dump()
        except (
            openai.APIError,
            openai.error.Timeout,
            openai.error.RateLimitError,
            openai.error.APIConnectionError,
            openai.error.ServiceUnavailableError,
        ) as err:
            if attempt == max_tries:
                gen_logger.error(
                    f"API request failed - {attempt} attempts with final error {err}."
                )
                results = {
                    "choices": [{"message": {"content": "ERROR: API request failed."}}]
                }
                return results

            delay = min(initial_delay * (backoff_factor ** (attempt - 1)), max_delay)
            jitter = random.uniform(jitter_range[0], jitter_range[1])  # nosec B311
            sleep_time = delay + jitter
            gen_logger.error("API request failed. Error: %s.", str(err))
            gen_logger.error("Retrying in %s seconds.", sleep_time)
            time.sleep(sleep_time)


CODE_FUNCTIONS = [
    {
        "name": "add_function_to_file",
        "description": "Add a new or revised function to a Python file.",
        "parameters": {
            "type": "object",
            "properties": {
                "function_code": {
                    "type": "string",
                    "description": (
                        "Python code for the function to add, escaped. "
                        "Without imports - these are returned separately."
                    ),
                },
                "import_statements": {
                    "type": "string",
                    "description": "Python import statements for the function to add, escaped.",
                },
            },
            "required": ["function_code", "import_statements"],
        },
    }
]


def generate_from_prompt(prepare_prompt_func, prepare_prompt_args):
    """
    Use the LLM to generate Python code or a test based on a given prompt.

    Args:
        prepare_prompt_func (function): Function used to prepare the prompt.
        prepare_prompt_args (dict): Arguments to pass to the prepare prompt function.

    Returns:
        Tuple[str, str]: The generated Python code or test and the import statements.
    """
    prompt = prepare_prompt_func(**prepare_prompt_args)
    messages = prompts.build_messages(prompt)
    function_call = {"name": "add_function_to_file"}
    response = api_request(
        messages=messages, functions=CODE_FUNCTIONS, function_call=function_call
    )
    response_message = response["choices"][0]["message"]
    logger.debug("Response message: %s", response_message)
    if response_message.get("function_call"):
        arguments_string = response_message["function_call"]["arguments"]
        # Tweak to prevent malformed escape sequences
        try:
            function_args = load_json_string(arguments_string)
        except json.JSONDecodeError as err:
            logger.debug("JSONDecodeError: %s", str(err))
            return None, None
        imports = function_args.get("import_statements").split("\n")
        return function_args.get("function_code"), imports
    else:
        return response_message["content"], None


def generate_code(task_description: str, function_file: str) -> Tuple[str, List[str]]:
    """
    Use the LLM to generate Python code for a given task.

    Args:
        task_description (str): A description of the task.

    Returns:
        str: The generated Python code.
        str: The generated import statements.
    """
    return generate_from_prompt(
        prompts.create_function_prompt,
        {"task_description": task_description, "function_file": function_file},
    )


def generate_test(
    function_code: str, function_file: str, test_name: str = None
) -> Tuple[str, str]:
    """
    Use the LLM to generate a Python test based on a given prompt.

    Args:
        function_code (str): Code of function to build a test for.
        function_file (str): File containing the function to build a test for.
        test_name (str, optional): The name of the test. Defaults to None.

    Returns:
        Tuple[str, str]: A tuple containing the generated
        Python test and import statements.
    """
    return generate_from_prompt(
        prompts.create_test_prompt,
        {
            "function_code": function_code,
            "function_file": function_file,
            "test_name": test_name,
        },
    )


def revise_test(
    original_test_code: str,
    function_code: str,
    test_output: str,
) -> Tuple[str, str]:
    """
    Use the LLM to revise a Python test based on a given prompt.

    Args:
        original_test_code (str): Original generated test code.
        function_code (str): Code of function to build a test for.
        test_output (str): Output of the test.

    Returns:
        Tuple[str, str]: A tuple containing the generated
        Python test and import statements.
    """
    return generate_from_prompt(
        prompts.revise_test_prompt,
        {
            "original_test_code": original_test_code,
            "function_code": function_code,
            "test_output": test_output,
        },
    )


def generate_todo_list() -> str:
    """
    Use the LLM to generate a to-do list.

    Returns:
        str: The generated to-do list.
    """
    prompt = prompts.create_todo_list_prompt()
    messages = prompts.build_messages(prompt)
    response = api_request(
        messages=messages,
        functions=[],  # no functions required for this prompt
        function_call=None,
        model=GOOD_MODEL,
    )
    return response["choices"][0]["message"]["content"]


def generate_summary(prompt: str) -> str:
    """
    Use the LLM to generate a summary using a given prompt.

    Args:
        prompt (str): The prompt to use.

    Returns:
        str: The generated summary.
    """
    messages = prompts.build_messages(prompt, add_dir=False, add_requirements=False)
    response = api_request(
        messages=messages,
        functions=[],  # no functions required for this prompt
        function_call=None,
        model=GOOD_MODEL,
    )
    logger.debug("Response: %s", response)
    return response["choices"][0]["message"]["content"]


def generate_module_docstring(module_code: str) -> str:
    """
    Use the LLM to generate a docstring for a Python module.

    Args:
        module_code (str): The source code of the module.

    Returns:
        str: The generated docstring.
    """
    prompt = prompts.create_module_docstring_prompt(module_code)
    messages = prompts.build_messages(prompt)
    response = api_request(
        messages=messages,
        functions=[],  # no functions required for this prompt
        function_call=None,
        model=QUICK_MODEL,
        max_tokens=300,
    )
    return response["choices"][0]["message"]["content"]


def generate_function_docstring(function_code: str) -> str:
    """
    Use the LLM to generate a docstring for a Python function.

    Args:
        function_code (str): The source code of the function.

    Returns:
        str: The generated docstring.
    """
    prompt = prompts.create_function_docstring_prompt(function_code)
    messages = prompts.build_messages(prompt)
    response = api_request(
        messages=messages,
        functions=[],  # no functions required for this prompt
        function_call=None,
        model=QUICK_MODEL,
        max_tokens=600,
    )
    return response["choices"][0]["message"]["content"]


def reduce_module_descriptions(initial_description: str) -> str:
    """Reduce module descriptions to single sentence.

    Args:
        initial_description (str): string with markdown list
        of module descriptions.

    Returns:
        str: reduced markdown string list of module descriptions.
    """
    prompt = prompts.create_reduce_module_descriptions_prompt(initial_description)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    response = api_request(
        messages=messages,
        functions=[],  # no functions required for this prompt
        function_call=None,
        model=QUICK_MODEL,
    )
    return response["choices"][0]["message"]["content"]


ISSUE_REVIEW_FUNCTIONS = [
    {
        "name": "label_easiest_issue",
        "description": "Label the easiest issue.",
        "parameters": {
            "type": "object",
            "properties": {
                "issue_number": {
                    "type": "integer",
                    "description": (
                        "Number of the issue that is easiest to solve."
                        "For subsequent labelling."
                    ),
                },
            },
            "required": ["issue_number"],
        },
    }
]


def review_issues(open_issues: list, token_limit: int = 3800) -> int:
    """Review issues and assign labels.

    Args:
        open_issues (list): list of open issues.

    Returns:
        int: easiest issue number.
    """
    prompt = prompts.create_issue_review_prompt(open_issues, titles_only=False)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful Python programming assistant.",
        },
        {"role": "user", "content": prompt},
    ]
    # Check the token limit
    if num_tokens_from_messages(messages) > token_limit:
        # Rebuild the prompt using only titles if over the limit
        prompt = prompts.create_issue_review_prompt(open_issues, titles_only=True)
        messages[-1]["content"] = prompt
    # Check the token limit again
    if num_tokens_from_messages(messages) > token_limit:
        # Rebuild using the earliest issue numbers if still over the limit
        prompt = prompts.create_issue_review_prompt(open_issues[:30], titles_only=True)
        messages[-1]["content"] = prompt
    if num_tokens_from_messages(messages) > token_limit:
        # Rebuild using the earliest issue numbers if still over the limit
        prompt = prompts.create_issue_review_prompt(open_issues[:10], titles_only=True)
        messages[-1]["content"] = prompt
    response = api_request(
        messages=messages,
        functions=ISSUE_REVIEW_FUNCTIONS,
        function_call={"name": "label_easiest_issue"},
        model=QUICK_MODEL,
    )
    response_message = response["choices"][0]["message"]
    logger.debug("Response message: %s", response)
    if response_message.get("function_call"):
        arguments_string = response_message["function_call"]["arguments"]
        # Tweak to prevent malformed escape sequences
        try:
            function_args = load_json_string(arguments_string)
        except json.JSONDecodeError as err:
            logger.debug("JSONDecodeError: %s", str(err))
            return None, None
        issue_number = function_args.get("issue_number")
        return issue_number
    else:
        return response_message["content"]
