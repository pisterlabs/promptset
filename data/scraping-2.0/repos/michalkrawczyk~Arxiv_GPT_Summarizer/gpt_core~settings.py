import openai
import yaml

import os

PROMPTS = {}
# TODO: Think about adding test for prompt content checks


def reload_openai_key(key_yaml_path: str = "./openai_key.yaml"):
    """ Load OpenAI API Key from yaml file"""
    if os.path.isfile(key_yaml_path):
        with open('openai_key.yaml', 'r') as f:
            # Read API KEY for ChatGPT
            openai.api_key = yaml.safe_load(f)["openai_api_key"]

    if not openai.api_key or openai.api_key == "OPENAI_API_KEY":
        openai.api_key = os.environ.get("OPENAI_API_KEY")

        if not openai.api_key:
            raise ValueError("Invalid OpenAI API key - check settings.")


def reload_prompts(yaml_path: str = "./prompts.yaml"):
    """ Load yaml with prompts (commands to be executed for by GPT engine"""
    if not os.path.isfile(yaml_path):
        raise OSError("Prompts file not found")

    with open(yaml_path, 'r') as f:
        # Read prompts that will be used to make summaries
        for prompt, content in yaml.safe_load(f).items():
            # item assignment to avoid making local variable
            PROMPTS[prompt] = content
