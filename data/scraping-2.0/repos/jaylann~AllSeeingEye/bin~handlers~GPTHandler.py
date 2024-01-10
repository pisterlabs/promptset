import openai
import tiktoken
import os
from typing import List

from bin.handlers.ConfigHandler import load_env_vars
from bin.handlers.PathHandlers import get_content_root
from bin.handlers.objects.GPTModels import Model
from bin.handlers.objects.GPTResponse import GPTResponse


class GPTHandler:
    def __init__(self):
        """
        Initializes the OpenAIAPI class by setting up the API key and encoder.
        """
        env_path = os.path.join(get_content_root(), '_internal', 'credentials', 'keys.env')
        self._load_api_key(env_path)
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def _load_api_key(self, env_path: str) -> None:
        """
        Loads the OPENAIAPIKEY from the given environment path.

        Args:
            env_path (str): Path to the environment file containing the API key.

        Raises:
            ValueError: If the API key is not found in the environment file.
        """
        load_env_vars(env_path)
        self.api_key = os.getenv("OPENAIAPIKEY")
        if not self.api_key:
            raise ValueError("OPENAIAPIKEY not found in keys.env")
        openai.api_key = self.api_key

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text.

        Args:
            text (str): Text to count tokens for.

        Returns:
            int: Number of tokens in the text.
        """
        return len(self.encoder.encode(text))

    def generate_text(self, prompt: str, model: Model = Model.GPT3_5_TURBO, system_prompt: str = "",
                      max_tokens: int = 150) -> GPTResponse:
        """
        Generates text using the OpenAI API based on the given prompt and model.

        Args:
            prompt (str): The main prompt to generate text for.
            model (Model, optional): The GPT model to use. Defaults to Model.GPT3_5_TURBO.
            system_prompt (str, optional): An optional system prompt. Defaults to "".
            max_tokens (int, optional): Maximum number of tokens for the response. Defaults to 150.

        Returns:
            GPTResponse: The generated response.
        """
        messages: List[dict[str, str]] = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append({"role": "user", "content": prompt})

        response_params = {
            "model": model.value,
            "messages": messages
        }

        if max_tokens != -1:
            response_params["max_tokens"] = max_tokens

        response = openai.ChatCompletion.create(**response_params)
        gpt_response = GPTResponse().from_dict(response, prompt=prompt, system_prompt=system_prompt)

        return gpt_response
