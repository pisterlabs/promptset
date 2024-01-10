import os

from core.gpt.prompts import *
import openai


class Responder:

    def __init__(self, prompter: PromptManager, key=None):
        self.prompter = prompter
        self.__key = os.getenv('GPT_API_KEY') if key is None else key
        openai.api_key = self.__key

    def get_response(self, prompt):
        response = openai.Completion.create(
            engine="text-davinci-003",  # for GPT-3.5 Turbo, use "text-davinci-003"
            prompt=self.prompter.get_prompt_by_name(prompt),
            max_tokens=MAX_TOKENS
        )
        message = response.choices[0].text.strip()
        return message


if __name__ == '__main__':
    pm = PromptManager()
    pm.prompts['ostinato'] = StandardOstinatoPrompt('action movie sequence', 'F', 'dramatic', 8, 4)
    r = Responder(pm)

    r1 = r.get_response('ostinato')
    print(r1)
