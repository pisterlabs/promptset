import openai
import random
import re
import json
from config import TEXT_EMOTES
from openai.error import RateLimitError

class ChatBot:

    def __init__(self, API_KEY, bot_info, model='text-davinci-003'):
        self.bot_info = bot_info
        openai.api_key = API_KEY
        self.model = model

    def get_answer(self, question, author):
        try:
            response = openai.Completion.create(
                model=self.model,
                prompt=self.bot_info + ' text: ' + question + f' author: {author}',
                max_tokens=512,
                temperature=0.2
            )
        except RateLimitError:
            return f"Skonczyły mi się impulsy {random.choice(TEXT_EMOTES['Sad'])} pogadamy później."
        return self.parse_response(response.choices[0].text)
    
    def parse_response(self, response):
        for emote_name, emote_list in TEXT_EMOTES.items():
            response = response.replace(f'${emote_name}', random.choice(emote_list))
        pattern = r'\[.*?\]'
        info = re.search(pattern, response)
        if info:
            info = info.group(0) 
            response = re.sub(pattern, "", response)
            self.bot_info += info
            print(info)
        return response