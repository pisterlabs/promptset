from openai import OpenAI
from prompt_engineering.prompts import Prompt, SystemPrompt
from prompt_engineering.response_models import GptResponse


class GptClient:
    def __init__(self, model="gpt-3.5-turbo"):
        api_key = open("key.txt", "r").read().strip("\n")

        self.model = model

        self.client = OpenAI(
            api_key=api_key,
        )

    def complete(self, prompt: Prompt, system_prompt: SystemPrompt = None, new_context=True):
        if new_context:
            self.messages = []

        if system_prompt:
            self.messages.append(dict(system_prompt))

        self.messages.append(dict(prompt))
        resp = self.client.chat.completions.create(
            messages=self.messages,
            model=self.model,
        )

        return self._parse_response(resp)

    def _parse_response(self, resp):
        return GptResponse(**dict(resp))
