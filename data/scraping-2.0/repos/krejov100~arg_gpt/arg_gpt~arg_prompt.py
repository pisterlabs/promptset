import openai
import inspect
from pprint import pprint
import typer
import arg_gpt.prompts as prompts
import arg_gpt.gpt_helpers as gpt_helpers
from dotenv import load_dotenv
import logging
load_dotenv()

log = logging.getLogger(__name__)

_GPT_FUNCTIONS = []

def gpt_func(func):
    """Decorator for functions that should be exposed to GPT"""
    _GPT_FUNCTIONS.append(func)
    return func

def reflect_on_interface():
    # get the name of the calling function
    caller_frame = inspect.currentframe().f_back

    # list the functions in the module
    module = inspect.getmodule(caller_frame)
    functions = inspect.getmembers(module, inspect.isfunction)
    print("Found the following public functions:")
    for name, func in functions:
        print(name)
        _GPT_FUNCTIONS.append(func)


def run_conversation(prompt, functions=None):
    if functions is None:
        functions = _GPT_FUNCTIONS
    client = openai.Client()
    messages = prompts.request_detailed_result() + prompts.remain_functional() + prompts.user_prompt(prompt)

    logging.info("Starting conversation...")

    while True:
        response = gpt_helpers.call_gpt_with_function(client, functions, messages)
        logging.debug("GPT response: %s", response)

        result = gpt_helpers.interpret_response(response, functions)
        messages.extend(result)

        if response.choices[0].finish_reason in ["stop", "max_tokens", "content_filter"]:
            logging.info("Conversation finished.")
            break

    messages += prompts.summarize()
    response = gpt_helpers.call_gpt(client, messages)
    logging.debug("Final GPT response: %s", response)

    result = gpt_helpers.interpret_response(response, functions)
    final_content = result[-1].content

    logging.info("Conversation completed successfully.")
    logging.info("Final content: %s", final_content)

    return final_content

def run_arg_prompt():
    typer.run(run_conversation)
