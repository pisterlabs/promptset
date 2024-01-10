from custom_actions import CustomActions
import json
import openai

class Action:
    def __init__(self):
        openai.api_key = "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"

    def generate_response(self, prompt):
        completions = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )

        message = completions.choices[0].text
        return message


class Actions(CustomActions):
    pass