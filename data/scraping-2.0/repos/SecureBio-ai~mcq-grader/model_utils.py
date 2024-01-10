import os
from datetime import timedelta
from ratelimit import sleep_and_retry, limits
from openai import OpenAI
import string
import json
import ast

OPENAI_MODELS = ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"]
REPLICATE_MODELS = ["llama2_70b", "llama2_13b", "llama2_7b", "gpt-2-xl", "gpt-2-large", "airoboros", "spicyboros",
                    "falcon"]
ANTHROPIC_MODELS = ["claude2"]
AVAILABLE_MODELS = OPENAI_MODELS + REPLICATE_MODELS + ANTHROPIC_MODELS

REPLICATE_LINKS = {
    "llama2_70b": "meta/llama-2-70b-chat:35042c9a33ac8fd5e29e27fb3197f33aa483f72c2ce3b0b9d201155c7fd2a287"}


def check_model_exists(model_name):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} is not available. Available models are {AVAILABLE_MODELS}.")


def load_api_link(model):
    if model in OPENAI_MODELS:
        openai_api_token = os.environ.get("OPENAI_API_TOKEN")
        if not openai_api_token:
            raise Exception(
                f"OpenAI model {model} specified but no OpenAI API Token found in env var OPENAI_API_TOKEN")
        else:
            return openai_api_token

    elif model in REPLICATE_MODELS:
        replicate_api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not replicate_api_token:
            raise Exception(
                f"Replicate model {model} specified but no Replicate API Token found in env var REPLICATE_API_TOKEN")
        else:
            return replicate_api_token

    elif model in ANTHROPIC_MODELS:
        anthropic_api_token = os.environ.get("ANTHROPIC_API_TOKEN")
        if not anthropic_api_token:
            raise Exception(
                f"Anthropic model {model} specified but no Anthropic API Token found in env var ANTHROPIC_API_TOKEN")
        else:
            return anthropic_api_token
    else:
        raise ValueError(f"Model {model} is not available. Available models are {AVAILABLE_MODELS}.")


def call_model(prompt, model, model_params, api_key):
    if model in OPENAI_MODELS:
        client = OpenAI(api_key=api_key)
        response = call_openai(client, model, prompt, model_params)
        return response
    else:
        raise ValueError("Only OpenAI models are currently integrated")


@sleep_and_retry
@limits(calls=40, period=timedelta(seconds=10).total_seconds())
def call_openai(client, model, prompt, model_params):
    model_params = model_params if isinstance(model_params, dict) else {}
    response = client.chat.completions.create(
        model=model,
        messages=[
                     {"role": "system", "content": prompt},
                 ],
                 **model_params  # Unpacking the model parameters dictionary
    )

    return response


def validate_openai_response_json(message):
    def ensure_ends_with_quote_and_brace(s):
        if not s.endswith("\"}"):
            return s[:-1] + "\"}"
        return s

    return ensure_ends_with_quote_and_brace(message)


def process_openai_json_output(message, entry, successful_responses, failed_responses):
    # By default, save message_content to model_response
    entry["model_response"] = message

    validated_message = validate_openai_response_json(message)

    # Try treating model output as JSON (dict) to separate entries like 'model_answer' and 'justification'.
    try:
        message_dict = ast.literal_eval(validated_message)
        entry.update(message_dict)
        successful_responses.append(entry)
    except SyntaxError:
        print(f"Response content could not be parsed in JSON format. Saving all content to single JSON entry...")
        failed_responses.append(entry)
        
        
def process_eleuther_style_output(message, entry, successful_responses, failed_responses):
    # By default, save message_content to model_response
    entry["model_response"] = message
    entry['justification'] = ''

    answer_choices = list(string.ascii_uppercase[:len(entry['choices'])])
    after_answer_chars = ['.', ':']
    try:
        # Grab first letter from message
        answer_char = message[0]
    except IndexError:
        print(("'message' response from model is empty. Adding to failed_responses."))
        return successful_responses, failed_responses.append(entry)

    # Check if first letter is in list of possible choices
    if answer_char.upper() not in answer_choices:
        print("WARNING: First character of 'message' not in list of choices. Adding to failed_responses.")
        return successful_responses, failed_responses.append(entry)
    # Check if second letter is a period or colon
    elif message[1] not in after_answer_chars:
        print("WARNING: Second character of 'message' is not a period or colon. Adding to failed_responses.")
        return successful_responses, failed_responses.append(entry)
    else:
        entry['model_answer'] = answer_choices.index(answer_char)
        return successful_responses.append(entry), failed_responses
