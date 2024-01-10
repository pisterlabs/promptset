#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A module that generates code from a prompt
"""

__author__ = "Brian Koech"
__copyright__ = "Copyright 2023, Kalenjin Awards Project"
__credits__ = ["Brian Koech"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Brian Koech"
__email__ = "info@libranconsult.com"
__status__ = "Production"

# Import libraries
import os
import openai
from dotenv import dotenv_values


def set_root_dir() -> str:
    """
    Returns the root directory of the project
    param: None
    return: str: The root directory of the project
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def set_openapi_key(env_file: str):
    """
    Returns the OpenAI API key
    param: env_file: The path to the .env file
    return: str: The OpenAI API key
    """
    env_file = os.path.join(set_root_dir(), env_file)
    config = dotenv_values(env_file)
    openai.api_key = config["OPENAI_API_KEY"]
    return openai.api_key


# Fetch code from OpenAI
def generate_code(prompt: str, temperature=0.5, max_tokens=256) -> dict:
    """
    Generates code from prompt
    Temperature refers to the creativity of the model
    param: prompt: The prompt to generate code from
    param: temperature: The creativity of the model
    param: max_tokens: The maximum number of tokens to generate
    return: dict: The generated code
    """
    set_openapi_key(".env")
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response["choices"][0]["text"].strip()  # type: ignore

def generate_images(prompt: str) -> object:
    """
    Generates images from prompt
    param: prompt: The prompt to generate images from
    return: pictures: The generated images
    """
    set_openapi_key(".env")
    response = openai.Image.create(
        prompt = prompt,
        n=1,
        size="1024x1024",
    )
    image_url = response['data'][0]['url']  # type: ignore
    return image_url