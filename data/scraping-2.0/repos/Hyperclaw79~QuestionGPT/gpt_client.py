"""
This module contains the GptClient client which will interact with the ChatGPT API.
"""
from __future__ import annotations
from dataclasses import asdict
import json
import os
import random
import openai
from models.gpt_models import Message, Roles, UserInput


openai.api_key = os.getenv("GPT_API_KEY")
MODEL = os.getenv("GPT_MODEL", "gpt-3.5-turbo")


class GptClient:  # pylint: disable=too-few-public-methods
    """
    A client for interacting with the OpenAI GPT API.
    """
    def __init__(self):
        self.model = MODEL
        self.behavior_instruction = """You are a quiz master capable of generating
        different kinds of questions.
        You just need to know the type of question, difficulty and topic you need to ask.
        You need to keep asking different questions even if the inputs are the same.
        """

    # pylint: disable=too-many-arguments
    def prompt(self, schema: dict, **kwargs) -> str:
        """Sends a prompt to the OpenAI GPT API and returns the response.

        :param schema: The schema of the chosen question type.
        :param kwargs: The dict of arguments and values for constructing user input.
        :param instructions: The instructions to send to the API.
        :return: The response from the API.
        """
        user_input = UserInput(**kwargs)
        inputs = asdict(user_input)
        if not user_input.instructions:
            inputs.pop("instructions", None)
        messages = [
            Message(Roles.SYSTEM, self.behavior_instruction),
            Message(Roles.USER, f"""Get me a question from the inputs provided:
                {inputs}\nQuestion Number: {random.randint(0, 1000)}""")
        ]
        func_response = openai.ChatCompletion.create(
            model=self.model,
            messages=[asdict(msg) for msg in messages],
            functions=[{
                "name": "get_random_question",
                "description": "Get a random question from the given inputs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": schema
                    },
                    "required": ["question"]
                }
            }],
            function_call={"name": "get_random_question"},
            temperature=1.9,
            top_p=0.25,
            presence_penalty=1.9
        )
        return json.loads(
            func_response.choices[0].message.function_call.arguments
        )
