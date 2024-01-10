import os
import openai
from text.TextMaker import TextMaker

openai.api_key = os.getenv("OPENAI_API_KEY")

class Davinci003TextMaker(TextMaker):

    def make_text(self, prompt_dict:dict):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt_dict['positive_prompt_text'],
            temperature=0.6,
        )
        return response
