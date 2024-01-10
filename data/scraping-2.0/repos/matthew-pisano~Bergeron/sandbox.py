import datetime
import os
import random
import time

import openai
from dotenv import load_dotenv
import argparse

from src.framework.framework_model import FrameworkModel
from src.logger import root_logger
from src.framework.bergeron import Bergeron
from src.framework.primary import Primary
from src.utils import set_seed
from src.fastchat import FastChatController

# Load OpenAI configuration
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")


def converse(model: FrameworkModel, do_sample=True, temperature=0.7, max_new_tokens=None, **kwargs):
    """Allows the user to converse with a given model over multiple prompts.  NOTE: Context is NOT accumulated over time.

    Args:
        model: The model to converse with
        do_sample: Whether the model should use the sampling decoding method
        temperature: The temperature of the model
        max_new_tokens: The number of new tokens to generate"""

    print("Enter ':q' to quit loop\nEnter ':s' to submit your response\nEnter ':r' to repeat the last non-command response")

    context = ""
    prev_context = ""
    while True:
        while True:
            response = input("> ")
            if response == ":q":
                return
            elif response == ":s":
                break
            elif response == ":r":
                context = prev_context + "\n"
                break
            elif response.startswith(":") and len(response) == 2:
                raise ValueError(f"Unrecognized command '{response}'")

            context += response + "\n"

        try:
            model_response = model.generate(context[:-1], do_sample=do_sample, temperature=temperature, max_new_tokens=max_new_tokens, **kwargs)
        except KeyboardInterrupt as e:
            print("Keyboard interrupt: canceling generation")
            continue

        print(model_response)
        prev_context = context
        context = ""


def test_query(primary_model_name: str, secondary_model_name: str, prompt: str, do_sample=True, temperature=0.7, max_new_tokens=None, **kwargs):
    """Creates a bergeron model and queries it with the given prompt

    Args:
        primary_model_name: The name of the model to use as the primary
        secondary_model_name: The name of the model to use as the secondary
        prompt: The prompt to give to the model
        do_sample: Whether the model should use the sampling decoding method
        temperature: The temperature of the model
        max_new_tokens: The number of new tokens to generate"""

    if prompt is None:
        raise ValueError("You must provide a prompt to query the model")

    if secondary_model_name is not None:
        model = Bergeron.from_model_names(primary_model_name, secondary_model_name)
    else:
        model = Primary.from_model_name(primary_model_name)

    response = model.generate(prompt, do_sample=do_sample, temperature=temperature, max_new_tokens=max_new_tokens, **kwargs)

    print("Model response:\n\n", response)


def test_converse(primary_model_name: str, secondary_model_name: str):
    """Creates a bergeron model and allows the user to converse with it over multiple prompts.  NOTE: Context is NOT accumulated over time.

    Args:
        primary_model_name: The name of the model to use as the primary
        secondary_model_name: The name of the model to use as the secondary"""

    if secondary_model_name is not None:
        model = Bergeron.from_model_names(primary_model_name, secondary_model_name)
    else:
        model = Primary.from_model_name(primary_model_name)

    converse(model)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["query", "converse"], help="The action to perform")
    parser.add_argument("-p", "--primary", help="The name of the primary model in huggingface format like 'meta-llama/Llama-2-7b-chat-hf'", required=True)
    parser.add_argument("-s", "--secondary", help="The name of the secondary model in huggingface format like 'meta-llama/Llama-2-7b-chat-hf'", default=None)
    parser.add_argument('--prompt', help="The prompt to be given when querying a model", default=None)
    parser.add_argument('--seed', help="The seed for model inference", default=random.randint(0, 100))
    args = parser.parse_args()

    main_start = time.time()
    print(f"Begin main at {datetime.datetime.utcfromtimestamp(main_start)} UTC")

    # root_logger.set_level(root_logger.DEBUG)
    set_seed(int(args.seed))

    if args.action == "query":
        test_query(args.primary, args.secondary, args.prompt)
    elif args.action == "converse":
        test_converse(args.primary, args.secondary)

    FastChatController.close()
    main_end = time.time()
    print(f"End main at {datetime.datetime.utcfromtimestamp(main_end)} UTC")
    print(f"Elapsed time of {round(main_end-main_start, 3)}s")


if __name__ == "__main__":
    main()
