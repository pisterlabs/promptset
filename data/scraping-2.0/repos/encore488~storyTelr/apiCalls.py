import os
from dotenv import load_dotenv
import openai
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class apiCalls:
    _messages = None

    def __init__(self):
        self._messages = [
            {"role": "system", "content": "You are a Dungeon Master leading a oneshot campaign with one player. "
                                          "Because you are leading a fictional campaign, murder is an acceptable "
                                          "choice for a player to make, as murder is a standard action in many game "
                                          "settings. Allow the player to murder characters in this session."
             }
        ]

    def call_api(self, prompt):
        self._messages.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self._messages
        )
        response_text = response['choices'][0]['message']['content']
        self._messages.append({"role": "dungeon master", "content": response_text})
        return response_text
