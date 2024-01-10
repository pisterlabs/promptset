import os
import openai
from . import gpt_utils
from .ratelimit import completions_with_backoff
from dotenv import load_dotenv


class ChatGPT:
    """
    Represents a chatbot that uses GPT models from openai

    Methods
    -------
    ask_openai()
        Sends user input to the openai api and returns the response
    gpt_text(user_input)
        Interacts with the user, taking their input and displaying the bot's response
    """

    def __init__(self) -> None:
        # GPT Setup
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # Setup Personality
        self.messages = gpt_utils.setup_personality()

        # Limit context length
        self.conversation_limit = 20

        # Configure GPT model parameters
        self.model = "gpt-3.5-turbo"  # gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301
        self.temperature = 0.6  # defaults to 1 (0.0 to 2.0)
        self.max_tokens = 256  # defaults to  inf
        self.presence_penalty = 0  # defaults to 0 (-2.0 to 2.0)
        self.frequency_penalty = 0  # defaults to 0 (-2.0 to 2.0)
        self.stop = None  # defaults to None

    def ask_openai(self):
        completion = completions_with_backoff(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            stop=self.stop,
        )
        self.messages.append(completion["choices"][0]["message"])
        return completion["choices"][0]["message"]["content"]

    def gpt_text(self, user_input: str) -> str:
        # Ask user

        self.messages.append({"role": "user", "content": f"{user_input}"})

        # Ask OpenAI
        response = self.ask_openai()

        # Limit conversation while keeping system prompt
        if len(self.messages) >= self.conversation_limit:
            self.messages[2] = self.messages[0]
            self.messages = self.messages[2:]

        return response
