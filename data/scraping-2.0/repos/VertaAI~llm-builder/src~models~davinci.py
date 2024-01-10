import os

import openai

from .abc import Model

openai.api_key = os.environ["OPENAI_API_KEY"]


class DaVinci(Model):
    def __init__(self, id: int):
        self._id = id

    def get_id(self) -> int:
        return self._id

    def predict(self, prompt: str, input_data: str) -> str:
        return self.summarize(prompt, input_data)

    def summarize(self, prompt, text_to_summarize):
        full_prompt = f"{prompt}\n\n {text_to_summarize}"
        return self.generate_response(full_prompt)

    # TODO: allow hyperparameters to vary
    def generate_response(self, full_prompt):
        print("Calling OpenAI API...")
        completions = openai.Completion.create(
            engine="text-davinci-003",
            prompt=full_prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        print("Call complete. result: " + str(completions["choices"]))
        return completions["choices"][0]["text"]
