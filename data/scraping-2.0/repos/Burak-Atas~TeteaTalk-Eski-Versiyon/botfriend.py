import openai
import re

openai.api_key = "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"


class BotFriend():
    def __init__(self):
        self.prompt = [
            {"role": "system", "content": "hello talk to me as we can in a video call where we will just meet you. Your name is jake. Since my English is not very good, talk to me at A1 level without giving too long answers."},
        ]

    def api_query(self,inpt):
        self.prompt.append({"role": "user", "content": inpt})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.prompt,
            max_tokens=150
        )

        system_message = response["choices"][0]["message"]["content"]
        self.prompt.append({"role": "assistant", "content": system_message})

        return system_message
    
